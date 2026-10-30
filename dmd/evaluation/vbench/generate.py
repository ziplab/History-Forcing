"""DMD VBench generation with the original rank RNG stream and resumable prompts."""
import argparse
import os
from pathlib import Path
import pickle
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.generator import DMDGenerator, config_options, rank_context, vbench_prompts
from evaluation.vbench.protocol import (BATCH_SIZE, FPS, LATENT_FRAMES, PIXEL_FRAMES,
                                       PROMPT_COUNTS, SEED, SAMPLES_PER_PROMPT,
                                       read_prompts, selected_groups)
from evaluation.video import valid_video, write_video_atomic


def capture_rng(device):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def save_receipt(path, identity, before, after=None):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        torch.save({'identity': identity, 'before': before, 'after': after}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_receipt(path, identity, device):
    import torch
    current = capture_rng(device)
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            return None
        # Reject broken RNG states before they can alter the advancing stream.
        restore_rng(state['before'], device)
        if state['after'] is not None:
            restore_rng(state['after'], device)
        return state
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        return None
    finally:
        restore_rng(current, device)


def generate_group(generator, rows, output, receipts, *, rank, world, batch_size,
                   latent_frames, pixel_frames, device, run_id, noise_shape=(16, 60, 104),
                   validator=None, writer=None):
    """Replay canonical RNG batches when an interrupted prompt needs repair.

    Five initial noises are drawn together. The final batch computes only real
    samples but consumes the original padded transition noise. Complete prompts
    restore their saved post-inference RNG; missing/corrupt receipts replay all
    transition draws.
    """
    import torch
    validator = validator or (lambda path: valid_video(path, pixel_frames))
    writer = writer or (lambda path, frames: write_video_atomic(path, frames * 255., FPS))
    output, receipts = Path(output), Path(receipts)
    for index in range(rank, len(rows), world):
        prompt, expanded = rows[index]
        paths = [output / f'{prompt}-{sample}.mp4' for sample in range(SAMPLES_PER_PROMPT)]
        good = [validator(path) for path in paths]
        identity = {'run': run_id, 'prompt': prompt, 'conditioning': expanded,
                    'index': index, 'rank': rank, 'world': world, 'batch': batch_size,
                    'latent_frames': latent_frames}
        receipt = receipts / f'{index:04d}.pt'
        state = load_receipt(receipt, identity, device)
        if all(good) and state is not None and state['after'] is not None:
            restore_rng(state['after'], device)
            continue
        if state is not None:
            restore_rng(state['before'], device)
        before = capture_rng(device)
        save_receipt(receipt, identity, before)
        all_noise = torch.randn([SAMPLES_PER_PROMPT, latent_frames, *noise_shape], device=device, dtype=torch.bfloat16)
        for start in range(0, SAMPLES_PER_PROMPT, batch_size):
            indices = list(range(start, min(start + batch_size, SAMPLES_PER_PROMPT)))
            rng_options = {'rng_batch_size': batch_size} if len(indices) < batch_size else {}
            video = generator.generate(all_noise[indices], [expanded] * len(indices), **rng_options)
            if video.shape[0] != len(indices) or video.shape[1] != pixel_frames or video.shape[2] != 3:
                raise ValueError(f'Unexpected video shape: {tuple(video.shape)}')
            # Every original inference batch must run to advance its SDE stream.
            # Keep valid shared videos from another group intact.
            for offset, sample in enumerate(indices):
                if not good[sample]:
                    frames = video[offset].permute(0, 2, 3, 1).cpu()
                    writer(paths[sample], frames)
                    del frames
            generator.clear_cache()
            del video
        del all_noise
        save_receipt(receipt, identity, before, capture_rng(device))
        print(f'GENERATED:rank={rank}:prompt={index}:repaired={SAMPLES_PER_PROMPT-sum(good)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Default: config.yaml alongside the checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--full-info', type=Path, required=True)
    parser.add_argument('--group', choices=PROMPT_COUNTS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    from evaluation.generator import resolve_config
    args.config = str(resolve_config(args.config, args.checkpoint))
    import json
    rank, world, device = rank_context()
    # Validate launcher metadata before initializing any model.
    run = json.loads((args.run_dir / 'run.json').read_text(encoding='utf-8'))
    expected = run['protocol']
    if args.group not in selected_groups(expected['dimensions'], suite=expected['suite']):
        raise ValueError(f'Prompt group is outside the selected evaluation: {args.group}')
    for key, value in {'duration': '15s', 'batch_size': BATCH_SIZE,
                       'world_size': world, 'seed': SEED,
                       'prompts': vbench_prompts(args.prompt_file, args.full_info)}.items():
        if expected[key] != value:
            raise ValueError(f'Generation input differs from run.json: {key}')
    if (run['checkpoint'] != str(Path(args.checkpoint).resolve())
            or run['config'] != str(Path(args.config).resolve()) or run['weights'] != args.weights
            or run['config_options'] != config_options(args.config)):
        raise ValueError('Config/checkpoint differs from run.json')
    generator = DMDGenerator(args.config, args.checkpoint, args.weights, device)
    prompts, expanded = read_prompts(args.prompt_file, args.group, args.full_info)
    from utils.misc import set_seed
    # Initialization precedes seeding, matching the original test sampler.
    set_seed(SEED)
    print(f'SEED_AUDIT:rank={rank}:seed={SEED}:world_size={world}', flush=True)
    generate_group(generator, list(zip(prompts, expanded)), args.run_dir / 'videos',
                   args.run_dir / 'rng' / args.group, rank=rank, world=world,
                   batch_size=BATCH_SIZE, latent_frames=LATENT_FRAMES, pixel_frames=PIXEL_FRAMES,
                   device=device, run_id=str(args.run_dir.resolve()))


if __name__ == '__main__':
    main()
