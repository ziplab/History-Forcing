"""Opt-in CUDA training measurements, excluding evaluation and checkpoint I/O."""
import json
import hashlib
from pathlib import Path
import time

import torch
import torch.distributed as dist


def summarize_ranks(rows, effective_batch_size, generator_updated):
    """Use the slowest rank and largest per-device peak for a distributed step."""
    seconds = max(row[0] for row in rows)
    return {
        'performance/train_step_seconds': seconds,
        'performance/critic_samples_per_second': effective_batch_size / seconds,
        'performance/generator_updated': int(generator_updated),
        'performance/peak_allocated_gib': max(row[1] for row in rows) / 2**30,
        'performance/peak_reserved_gib': max(row[2] for row in rows) / 2**30,
    }


class TrainingPerformance:
    def __init__(self, config, device, model):
        self.enabled = bool(config.get('measure_training_performance', False))
        if not self.enabled:
            return
        self.device = device
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.batch_size = (self.world * config.batch_size
                           * int(config.get('gradient_accumulate', 1) or 1))
        self.output = Path(config.logdir)
        self.benchmark_seed = config.get('efficiency_benchmark_seed')
        if self.benchmark_seed is not None and type(self.benchmark_seed) is not int:
            raise ValueError('efficiency_benchmark_seed must be an integer')
        if self.benchmark_seed is not None:
            # Node-local storage: nonzero-node ranks also write workload receipts.
            self.output.mkdir(parents=True, exist_ok=True)
        teacher_present = hasattr(model, 'ar_teacher')
        if teacher_present != bool(config.get('use_history_guidance', False)):
            raise RuntimeError('History-guidance setting does not match the loaded AR teacher')
        # Initialization peaks are separate from per-step training peaks.
        initial = self._gather([0., torch.cuda.max_memory_allocated(device),
                                torch.cuda.max_memory_reserved(device)])
        if self.rank == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            metadata = {
                'world_size': self.world, 'effective_batch_size': self.batch_size,
                'torch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name(device),
                'use_history_guidance': bool(config.get('use_history_guidance', False)),
                'ar_teacher_present': teacher_present,
                'measurement': 'CUDA synchronize and wall clock around training updates',
                'excluded': ['visualization', 'metrics logging', 'garbage collection',
                             'checkpoint I/O', 'VBench evaluation'],
                'memory': 'PyTorch allocator peaks per device, maximum across ranks',
                'includes': ['data fetch', 'text encoding', 'rollouts', 'loss and backward',
                             'gradient clipping', 'optimizer updates', 'EMA update'],
                'initialization_peak_allocated_gib': max(row[1] for row in initial) / 2**30,
                'initialization_peak_reserved_gib': max(row[2] for row in initial) / 2**30,
                'generator_update_ratio': config.dfake_gen_update_ratio,
                'ema_start_step': config.ema_start_step,
                'efficiency_benchmark_seed': self.benchmark_seed,
                'per_rank_columns': ['train_step_seconds', 'peak_allocated_bytes',
                                     'peak_reserved_bytes'],
                'note': 'Compare warmed-up generator+critic and critic-only steps separately.',
            }
            (self.output / 'performance_run.json').write_text(
                json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    def _gather(self, row):
        value = torch.tensor(row, device=self.device, dtype=torch.float64)
        values = [torch.empty_like(value) for _ in range(self.world)]
        dist.all_gather(values, value)
        return torch.stack(values).cpu().tolist()

    def start(self):
        if self.enabled:
            torch.cuda.synchronize(self.device)
            # Exclude time another rank spent logging or saving the previous step.
            dist.barrier()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.started = time.perf_counter()
            self.work = []

    def begin_microbatch(self, step, generator_updated, index, prompts):
        if not self.enabled or self.benchmark_seed is None:
            return
        from utils.misc import set_seed
        phase = 'generator' if generator_updated else 'critic'
        identity = f'{self.benchmark_seed}:{step}:{phase}:{index}:{self.rank}'
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'big')
        set_seed(seed)
        self.work.append({'step': step + 1, 'phase': phase, 'microbatch': index,
            'seed': seed, 'prompt_sha256': hashlib.sha256(
                json.dumps(list(prompts), ensure_ascii=False).encode('utf-8')).hexdigest()})

    def end_microbatch(self, model):
        if self.enabled and self.benchmark_seed is not None:
            self.work[-1]['rollout_frames'] = model._last_generated_frames

    def finish(self, step, generator_updated):
        if not self.enabled:
            return {}
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        rows = self._gather([elapsed, torch.cuda.max_memory_allocated(self.device),
                             torch.cuda.max_memory_reserved(self.device)])
        metrics = summarize_ranks(rows, self.batch_size, generator_updated)
        if self.benchmark_seed is not None:
            with (self.output / f'performance_work_rank_{self.rank:02d}.jsonl').open(
                    'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': step, 'rank': self.rank,
                                         'microbatches': self.work}) + '\n')
        if self.rank == 0:
            record = {'step': step, 'kind': 'generator_and_critic' if generator_updated else 'critic_only',
                      'world_size': self.world, 'per_rank': rows, **metrics}
            with (self.output / 'performance.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
        return metrics
