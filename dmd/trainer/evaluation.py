"""Periodic VBench evaluation using the live, sharded DMD generator."""

from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np

import torch
import torch.distributed as dist

from evaluation.generator import write_json
from evaluation.vbench.protocol import (
    BATCH_SIZE, FPS, LATENT_FRAMES, PIXEL_FRAMES, SAMPLES_PER_PROMPT, SEED,
    read_prompts, resolve_dimensions,
)
from evaluation.vbench.score import summarize
from evaluation.video import valid_video, write_video_atomic
from utils.checkpoint import _capture_rng, _restore_rng, _collective_call
from utils.paths import resource_path


DEFAULT_DIMENSIONS = ('subject_consistency', 'dynamic_degree', 'motion_smoothness')


def due(step, interval):
    return interval > 0 and step > 0 and step % interval == 0


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def seed_evaluation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def clear_caches(trainer, pipeline=None):
    # The next training rollout initializes clean caches; no training RNG draws
    # are needed to rebuild them. Release them before allocating batch-4 KV.
    for target in (getattr(trainer.model, 'inference_pipeline', None), pipeline):
        if target is not None:
            for name in ('kv_cache1', 'crossattn_cache'):
                if hasattr(target, name):
                    setattr(target, name, None)
    vae = getattr(getattr(trainer.model, 'vae', None), 'model', None)
    if callable(getattr(vae, 'clear_cache', None)):
        vae.clear_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def preserve_training_state(trainer, weights):
    """Borrow EMA shards without casting/replacing parameters or advancing RNG."""
    rng = _capture_rng(trainer.device)
    flags = [(module, module.training) for module in trainer.model.modules()]
    gradients = [(parameter, parameter.grad) for parameter in trainer.model.parameters()]
    live = []
    ema = trainer.generator_ema if weights == 'ema' else None
    try:
        with torch.no_grad():
            def select_weights():
                nonlocal live
                if ema is not None:
                    parameters = dict(trainer.model.generator.module.named_parameters())
                    if parameters.keys() != ema.shadow.keys() or any(
                            parameters[name].shape != value.shape for name, value in ema.shadow.items()):
                        raise ValueError('EMA shards do not match the current generator')
                    live = [(parameter, parameter.detach().cpu().clone()) for parameter in parameters.values()]
                    for name, parameter in parameters.items():
                        parameter.copy_(ema.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
                trainer.model.eval()
            _collective_call(select_weights, 'Select training evaluation weights')
            yield 'ema' if ema is not None else 'raw'
    finally:
        try:
            with torch.no_grad():
                for parameter, value in live:
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                for parameter, gradient in gradients:
                    parameter.grad = gradient
            for module, training in flags:
                module.training = training
            clear_caches(trainer)
        finally:
            _restore_rng(rng, trainer.device)


def generator_fingerprint(generator):
    """Bind partial/reused outputs to the exact generator shards being evaluated."""
    def local_digest():
        digest = hashlib.sha256()
        for name, parameter in generator.named_parameters():
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    return gather(_collective_call(local_digest, 'Identify evaluation generator'))


def merge_video_records(parts, expected):
    records = [record for part in parts for record in part]
    paths = [str(Path(record['video_path']).resolve()) for record in records]
    expected = {str(Path(path).resolve()) for path in expected}
    if len(paths) != len(expected) or set(paths) != expected:
        raise ValueError('Incomplete or duplicate VBench video scores')
    if not all(math.isfinite(float(record['video_results'])) for record in records):
        raise ValueError('VBench returned nonfinite video scores')
    # This is the official VBench distributed aggregation for these dimensions.
    return [sum(float(record['video_results']) for record in records) / len(records), records]


class TrainingVBenchEvaluation:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.options = config.get('training_eval', {})
        from omegaconf import OmegaConf
        if OmegaConf.is_config(self.options):
            self.options = OmegaConf.to_container(self.options, resolve=True)
        if not isinstance(self.options, dict):
            raise ValueError('training_eval must be a mapping')
        allowed = {'interval', 'dimensions', 'max_prompts', 'weights',
                   'prompt_file', 'vbench_root', 'cache_dir'}
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(f'Unknown training_eval options: {sorted(unknown)}')
        self.interval = self.options.get('interval', 200)
        if type(self.interval) is not int or self.interval < 0:
            raise ValueError('training_eval.interval must be a non-negative integer')
        if not self.interval:
            return
        self.dimensions = resolve_dimensions(self.options.get('dimensions', DEFAULT_DIMENSIONS))
        if set(self.dimensions) - set(DEFAULT_DIMENSIONS):
            raise ValueError('Training evaluation supports subject_consistency, dynamic_degree and motion_smoothness')
        self.max_prompts = self.options.get('max_prompts')
        if self.max_prompts is not None and (type(self.max_prompts) is not int or self.max_prompts < 1):
            raise ValueError('training_eval.max_prompts must be null or a positive integer')
        self.weights = self.options.get('weights', 'ema')
        if self.weights not in ('ema', 'raw'):
            raise ValueError('training_eval.weights must be ema or raw')
        self.prompt_file = resource_path(self.options.get('prompt_file', 'dmd/prompts/vbench.txt'))
        self.vbench_root = resource_path(self.options.get('vbench_root', 'dmd/evaluation/vbench/VBench'))
        cache = self.options.get('cache_dir') or os.environ.get('VBENCH_CACHE_DIR')
        self.cache_dir = resource_path(cache) if cache else Path.home() / '.cache/vbench'
        _collective_call(self._preflight, 'Prepare training VBench evaluation')

    def _preflight(self):
        from evaluation.vbench.training import validate_metric_assets
        if not self.config.get('denoising_step_list'):
            raise ValueError('Training VBench evaluation requires a DMD few-step config')
        validate_metric_assets(self.vbench_root, self.cache_dir, self.dimensions)
        full_info = self.vbench_root / 'vbench/VBench_full_info.json'
        prompts, expanded = read_prompts(self.prompt_file, 'subject_consistency', full_info)
        full = json.loads(full_info.read_text(encoding='utf-8'))
        for dimension in self.dimensions:
            expected = {row['prompt_en'] for row in full if dimension in row['dimension']}
            if set(prompts) != expected:
                raise ValueError(f'VBench prompts do not match official metadata: {dimension}')
        self.available_prompt_count = len(prompts)
        self.rows = list(zip(prompts, expanded))[:self.max_prompts]
        selected = {prompt for prompt, _ in self.rows}
        self.full_info = [row for row in full if row['prompt_en'] in selected
                          and set(row['dimension']) & set(self.dimensions)]

    def _sample_selection(self):
        return {'max_prompts': self.max_prompts, 'prompt_count': len(self.rows),
                'available_prompt_count': self.available_prompt_count,
                'video_count': SAMPLES_PER_PROMPT * len(self.rows),
                'available_video_count': SAMPLES_PER_PROMPT * self.available_prompt_count,
                'is_subset': len(self.rows) < self.available_prompt_count}

    def due(self, step):
        return due(step, self.interval)

    def _identity(self, trainer, weights):
        fields = ('denoising_step_list', 'denoising_step_list_first_chunk', 'warp_denoising_step',
                  'model_kwargs', 'num_frame_per_block', 'independent_first_frame', 'context_noise')
        from omegaconf import OmegaConf
        options = OmegaConf.to_container(trainer.config, resolve=True)
        return {'format': 2, 'step': trainer.step, 'weights': weights,
                'generator_shards': generator_fingerprint(trainer.model.generator),
                'inference': {name: options.get(name) for name in fields},
                'duration': '15s', 'dimensions': list(self.dimensions),
                'batch_size': BATCH_SIZE, 'seed': SEED, 'world_size': trainer.world_size,
                'prompts': [list(row) for row in self.rows], 'samples_per_prompt': SAMPLES_PER_PROMPT,
                'sample_selection': self._sample_selection()}

    def _prepare_output(self, trainer, run, identity):
        run.mkdir(parents=True, exist_ok=True)
        target = run / 'run.json'
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'Training evaluation inputs changed: {run}')
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            write_json(target, identity)
        (run / 'videos').mkdir(exist_ok=True)

    def _generate(self, trainer, pipeline, run, rows):
        rank, world = dist.get_rank() if dist.is_initialized() else 0, trainer.world_size
        noise_shape = list(self.config.image_or_video_shape[2:])
        # Pad prompt rounds so all FSDP ranks execute the same actual batch
        # sequence, including on a retry and for ranks without a real prompt.
        for turn in range(math.ceil(len(rows) / world)):
            index = turn * world + rank
            real = index < len(rows)
            prompt, expanded = rows[index % len(rows)]
            noise = torch.randn([SAMPLES_PER_PROMPT, LATENT_FRAMES, *noise_shape], device=self.device, dtype=trainer.dtype)
            for start in range(0, SAMPLES_PER_PROMPT, BATCH_SIZE):
                indices = list(range(start, min(start + BATCH_SIZE, SAMPLES_PER_PROMPT)))
                # Skip padded computation while retaining its original SDE RNG budget.
                rng_options = {'rng_batch_size': BATCH_SIZE} if len(indices) < BATCH_SIZE else {}
                video = pipeline.inference(noise=noise[indices], text_prompts=[expanded] * len(indices),
                                           return_latents=False, video_output_device='cpu', **rng_options)
                def store():
                    if video.shape[:3] != (len(indices), PIXEL_FRAMES, 3):
                        raise ValueError(f'Unexpected training evaluation video shape: {tuple(video.shape)}')
                    if real:
                        for offset, sample in enumerate(indices):
                            path = run / 'videos' / f'{prompt}-{sample}.mp4'
                            if not valid_video(path, PIXEL_FRAMES):
                                frames = video[offset].permute(0, 2, 3, 1).cpu()
                                write_video_atomic(path, frames * 255., FPS)
                _collective_call(store, 'Write training evaluation videos')
                trainer.model.vae.model.clear_cache()
                del video
            del noise

    def _score(self, trainer, run, weights):
        from evaluation.vbench.training import score_local
        rank, world = dist.get_rank() if dist.is_initialized() else 0, trainer.world_size
        videos = run / 'videos'
        expected = [videos / f'{prompt}-{sample}.mp4' for prompt, _ in self.rows for sample in range(SAMPLES_PER_PROMPT)]
        owned = [videos / f'{prompt}-{sample}.mp4'
                 for index, (prompt, _) in enumerate(self.rows) if index % world == rank
                 for sample in range(SAMPLES_PER_PROMPT)]
        metadata = []
        selected_prompts = {prompt for prompt, _ in self.rows}
        for row in self.full_info:
            if row['prompt_en'] not in selected_prompts:
                continue
            entry = dict(row)
            entry['video_list'] = [str(videos / f'{row["prompt_en"]}-{sample}.mp4') for sample in range(SAMPLES_PER_PROMPT)]
            metadata.append(entry)
        info = run / f'full_info_rank{rank:05d}.json'
        _collective_call(lambda: write_json(info, metadata), 'Write VBench metadata')
        results = {}
        for dimension in self.dimensions:
            records = _collective_call(
                lambda: score_local(self.vbench_root, self.cache_dir, dimension, info, owned, self.device),
                f'Evaluate VBench {dimension}')
            results[dimension] = merge_video_records(gather(records), expected)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = summarize(results, dimensions=self.dimensions)
        selection = self._sample_selection()
        summary.update(step=trainer.step, weights=weights, sample_selection=selection)
        if selection['is_subset'] and 'subset_score' in summary:
            summary['sample_subset_score'] = summary.pop('subset_score')

        def publish():
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                write_json(run / 'combined_eval_results.json', results)
                write_json(run / 'summary.json', summary)
        _collective_call(publish, 'Publish training evaluation scores')
        return summary

    def _log_result(self, trainer, summary):
        def publish():
            if trainer.is_main_process:
                selection = summary['sample_selection']
                prefix = 'eval/vbench_sample_subset' if selection['is_subset'] else 'eval/vbench'
                metrics = {f'{prefix}/{name}': value for name, value in summary['raw'].items()}
                if not trainer.disable_wandb:
                    import wandb
                    wandb.log(metrics, step=trainer.step)
                print(json.dumps({'step': trainer.step, 'evaluation': 'vbench', 'weights': summary['weights'],
                                  'sample_selection': selection,
                                  'raw_percent': {name: value * 100 for name, value in summary['raw'].items()}}), flush=True)
        _collective_call(publish, 'Log training evaluation scores')

    def run(self, trainer):
        if not self.due(trainer.step):
            return None
        with preserve_training_state(trainer, self.weights) as weights:
            clear_caches(trainer)
            identity = self._identity(trainer, weights)
            run = (Path(trainer.output_path) / 'evaluation' / f'step_{trainer.step:06d}'
                   / weights / 'vbench' / '15s_seed0')
            if self._sample_selection()['is_subset']:
                run = run.with_name(run.name + f'_prompts{len(self.rows)}')
            _collective_call(lambda: self._prepare_output(trainer, run, identity), 'Prepare evaluation output')
            completed = run / 'summary.json'
            def read_summary():
                try:
                    value = json.loads(completed.read_text())
                    raw = value.get('raw') if isinstance(value, dict) else None
                    if (isinstance(value, dict)
                            and value.get('step') == trainer.step and value.get('weights') == weights
                            and value.get('sample_selection') == self._sample_selection()
                            and isinstance(raw, dict) and set(raw) == set(self.dimensions)
                            and all(type(score) in (int, float) and math.isfinite(score)
                                    for score in raw.values())):
                        return value
                except (OSError, ValueError, TypeError):
                    pass
                return None
            local_summary = _collective_call(read_summary, 'Read completed training evaluation')
            summaries = gather(local_summary)
            if all(value is not None and value == summaries[0] for value in summaries):
                self._log_result(trainer, summaries[0])
                return summaries[0]
            from pipeline import CausalInferencePipeline
            pipeline = _collective_call(lambda: CausalInferencePipeline(trainer.config, trainer.device,
                generator=trainer.model.generator, text_encoder=trainer.model.text_encoder, vae=trainer.model.vae),
                'Build training evaluation pipeline')
            try:
                seed_evaluation(SEED)
                self._generate(trainer, pipeline, run, self.rows)
            finally:
                clear_caches(trainer, pipeline)
                del pipeline
            summary = self._score(trainer, run, weights)
            self._log_result(trainer, summary)
            return summary
