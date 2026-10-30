"""Shared DMD checkpoint loading and evaluation run identity."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def resolve_config(path, checkpoint):
    """Use a checkpoint's frozen config unless an explicit config is supplied."""
    repo = Path(__file__).resolve().parents[2]
    config = Path(path).expanduser() if path else Path(checkpoint).expanduser().parent / 'config.yaml'
    config = (config if config.is_absolute() else repo / config).resolve()
    if not config.is_file():
        raise FileNotFoundError(f'Missing evaluation config: {config}; supply --config for a checkpoint without a config snapshot')
    return config


def config_options(path):
    from omegaconf import OmegaConf
    from utils.paths import load_config
    return OmegaConf.to_container(load_config(path), resolve=True)


def vbench_prompts(prompt_file, full_info):
    from evaluation.vbench.protocol import GROUPS, read_prompts
    return {group: [list(pair) for pair in zip(*read_prompts(prompt_file, group, full_info))]
            for group in GROUPS}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare_run(config, checkpoint, weights, benchmark, protocol, output=None):
    """Pin a run to its inputs before reusing any generated or scored files."""
    import fcntl
    from utils.paths import load_config
    checkpoint = Path(checkpoint).resolve()
    config = resolve_config(config, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    resolved = load_config(config)
    if not resolved.get('denoising_step_list'):
        raise ValueError('Evaluation requires a few-step DMD generator config')
    identity = {'format': 2, 'benchmark': benchmark, 'config': str(config),
                'config_options': config_options(config),
                'checkpoint': str(checkpoint), 'weights': weights,
                'protocol': protocol}
    label = protocol.get('duration', 'long')
    subset = None
    if benchmark == 'vbench':
        from evaluation.vbench.protocol import selection_id
        suite = protocol['suite']
        subset = selection_id(protocol['dimensions'], suite=suite)
        if subset is None and suite == 'semantic':
            subset = 'semantic'
    run_label = f'{label}_seed{protocol.get("seed", 0)}'
    if subset:
        run_label += '_' + subset
    directory = Path(output).resolve() if output else (
        checkpoint.parent / 'evaluation' / checkpoint.stem
        / (resolved.get('config_name') or config.stem) / weights
        / benchmark / run_label)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.init.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = directory / 'run.json'
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8')) != identity:
                raise ValueError(f'Evaluation inputs changed; choose another --output-dir: {directory}')
        else:
            if any(directory.iterdir()) and any(p.name != '.init.lock' for p in directory.iterdir()):
                raise ValueError(f'Refusing unrecorded outputs in {directory}')
            write_json(target, identity)
    return directory


def load_generator(pipeline, checkpoint_path, use_ema):
    import torch
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = state['generator_ema' if use_ema else 'generator']
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith('model._fsdp_wrapped_module.'):
            key = key.replace('model._fsdp_wrapped_module.', 'model.', 1)
        fixed[key] = value
    incompatible = pipeline.generator.load_state_dict(fixed, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f'Missing generator keys: {incompatible.missing_keys[:20]}')
    if incompatible.unexpected_keys:
        print(f'Ignoring {len(incompatible.unexpected_keys)} unexpected checkpoint keys', flush=True)


class DMDGenerator:
    def __init__(self, config, checkpoint, weights, device):
        import torch
        from pipeline import CausalInferencePipeline
        from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
        from utils.paths import load_config
        config = load_config(resolve_config(config, checkpoint))
        if not config.get('denoising_step_list'):
            raise ValueError('Evaluation requires a few-step DMD generator config')
        pipeline = CausalInferencePipeline(config, device=device)
        load_generator(pipeline, checkpoint, weights == 'ema')
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if get_cuda_free_memory_gb(gpu) < 40:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        else:
            pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        pipeline.eval()
        torch.set_grad_enabled(False)
        self.pipeline = pipeline

    def generate(self, noise, prompts, rng_batch_size=None):
        # Keep long RGB batches off GPU. A partial batch computes only real
        # samples while retaining the original padded transition RNG draws.
        output_device = 'cpu' if noise.shape[1] > 63 else None
        rng_options = {} if rng_batch_size is None else {'rng_batch_size': rng_batch_size}
        return self.pipeline.inference(noise=noise, text_prompts=prompts,
                                       return_latents=False, video_output_device=output_device,
                                       **rng_options)

    def clear_cache(self):
        self.pipeline.vae.model.clear_cache()


def rank_context():
    import torch
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    return rank, world, torch.device('cuda', local)


def main():
    from evaluation.vbench.protocol import (
        BATCH_SIZE, SEED, SAMPLES_PER_PROMPT, resolve_dimensions,
    )
    parser = argparse.ArgumentParser(description='Prepare a fixed 15s VBench evaluation run.')
    parser.add_argument('--config', help='Default: config.yaml alongside the checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--output-dir')
    parser.add_argument('--prompt-file', required=True)
    parser.add_argument('--full-info', required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--suite', choices=('quality', 'semantic'), default='quality')
    parser.add_argument('--dimensions', help='Optional comma-separated metrics within the selected suite.')
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('World size must be positive')
    protocol = {'name': 'vbench-15s-v1', 'duration': '15s', 'suite': args.suite,
                'seed': SEED, 'batch_size': BATCH_SIZE, 'world_size': args.world_size,
                'samples_per_prompt': SAMPLES_PER_PROMPT,
                'dimensions': list(resolve_dimensions(args.dimensions, suite=args.suite)),
                'prompts': vbench_prompts(args.prompt_file, args.full_info),
                'rng': 'rank-stream-with-prompt-state-v1'}
    print(prepare_run(args.config, args.checkpoint, args.weights, 'vbench', protocol, args.output_dir))


if __name__ == '__main__':
    main()
