"""Generate one 60s video per MovieGenBench source row and run VBench-Long."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import random
import subprocess
import time

from evaluation.generator import DMDGenerator, prepare_run, rank_context, resolve_config, write_json
from evaluation.moviegenbench.protocol import (
    FPS, LATENT_FRAMES, PIXEL_FRAMES, PROTOCOL, episode_name, generation_receipt,
    prepare, sample_seed,
)
from evaluation.video import video_info, write_video_atomic


def generate(args, manifest):
    import numpy as np
    import torch
    rank, world, device = rank_context()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = Path(args.output)
    generator = None
    for row in manifest['prompts'][rank::world]:
        if generation_receipt(directory, row) is not None:
            continue
        if generator is None:
            generator = DMDGenerator(args.config, args.checkpoint, args.weights, device)
        name = episode_name(row)
        video_path = directory / 'videos' / (name + '.mp4')
        receipt_path = directory / 'generation' / (name + '.json')
        started = time.monotonic()
        episode_seed = sample_seed(row['source_index'])
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)
        # Initial and subsequent SDE noise share the same advancing stream.
        noise = torch.randn((1, LATENT_FRAMES, 16, 60, 104), device=device, dtype=torch.bfloat16)
        video = generator.generate(noise, [row['prompt']])
        assert video.shape == (1, PIXEL_FRAMES, 3, 480, 832), video.shape
        assert torch.isfinite(video).all().item(), name
        frames = video[0].permute(0, 2, 3, 1).cpu()
        write_video_atomic(video_path, frames * 255., fps=FPS, options={'crf': '23'})
        write_json(receipt_path, {**row, 'protocol': PROTOCOL, 'seed': episode_seed,
            'video': str(video_path),
            'native_batch_size': 1, 'elapsed_seconds': time.monotonic() - started,
            'video_info': video_info(video_path), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
        generator.clear_cache()
        del video, frames, noise
        print(json.dumps({'rank': rank, 'episode': name,
                          'seconds': round(time.monotonic() - started, 2)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'generate', 'score', 'summarize'))
    parser.add_argument('--config', help='Default: config.yaml alongside the checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--output-dir')
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[2] / 'prompts/MovieGenVideoBench.txt'))
    parser.add_argument('--cache-dir', default=os.environ.get('VBENCH_CACHE_DIR',
                        str(Path(__file__).resolve().parents[3] / 'models/vbench15s_comparison/vbench_cache')),
                        help='Official VBench model cache.')
    parser.add_argument('--vbench-root', default=os.environ.get('VBENCH_ROOT',
                        str(Path(__file__).resolve().parents[1] / 'vbench/VBench')),
                        help='Official VBench checkout, including vbench2_beta_long.')
    args = parser.parse_args()
    from evaluation.moviegenbench.vbench_long import score, summarize
    # Standalone invocation follows the same repository-relative paths as shell entrypoints.
    repo = Path(__file__).resolve().parents[3]
    for name in ('config', 'checkpoint', 'source', 'cache_dir', 'output_dir', 'vbench_root'):
        value = getattr(args, name)
        if value is not None:
            path = Path(value).expanduser()
            setattr(args, name, str((path if path.is_absolute() else repo / path).resolve()))
    args.config = str(resolve_config(args.config, args.checkpoint))
    if args.stage == 'score':
        # Config loading imports DMD's top-level `utils` package. DINO's
        # released torch.hub entry also imports `utils`, so validate the run
        # in a child process and keep the scoring interpreter independent.
        args.output = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), 'prepare', *sys.argv[2:]],
            text=True, encoding='utf-8').strip()
    else:
        args.output = str(prepare_run(args.config, args.checkpoint, args.weights, 'moviegenbench',
            {'protocol': PROTOCOL, 'duration': '60s', 'source': args.source,
             'latent_frames': LATENT_FRAMES, 'samples_per_prompt': 1, 'batch_size': 1}, args.output_dir))
    manifest = prepare(args.source, args.output)
    if args.stage == 'prepare':
        print(args.output)
        return
    if args.stage == 'summarize':
        summarize(args, manifest)
    elif args.stage == 'generate':
        generate(args, manifest)
    else:
        score(args, manifest)


if __name__ == '__main__':
    main()
