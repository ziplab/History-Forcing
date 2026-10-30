"""Checkpoint callbacks that evaluate live weights without advancing training RNG."""

import argparse
import importlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

from inference import DEFAULT_INFERENCE_SEED
from utils import inference_rng
from config import PROJECT_ROOT, load_config

from .runtime import distributed_call, save_json_state, unwrap_distributed_model


@dataclass(frozen=True)
class EvaluationContext:
    model: torch.nn.Module
    backend: object
    checkpoint_dir: Path
    step: int
    device: torch.device
    rank: int
    world_size: int
    process_group: object = None


class Evaluator(ABC):
    """Task evaluation entry point, independent of the training lifecycle."""

    @abstractmethod
    def run(self, config_path, arguments):
        """Evaluate a baseline or a checkpoint using the task's own CLI."""

    def close(self):
        """Release task resources."""

    def finalize(self, output_dir, steps, *, start_step=0):
        """Aggregate the completed baseline and checkpoint reports."""

    def run_all(self, config_path, arguments=()):
        """Evaluate retained weights after training, with one shared task config."""
        from config import _path_from_repo, stage_values

        _, config = load_config(config_path)
        output = _path_from_repo(stage_values(config, 'train')['output_dir'])
        checkpoints = sorted(output.glob('checkpoint-*/model.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No saved checkpoints in {output}")
        steps = [int(path.parent.name.removeprefix('checkpoint-')) for path in checkpoints]
        self.run(config_path, ['--phase', 'baseline', *arguments])
        for checkpoint in checkpoints:
            self.run(config_path, ['--checkpoint', str(checkpoint), *arguments])
        self.finalize(output, steps)


def _score_worker(callback, output, result_path):
    # Spawn, never fork a CUDA training process; scoring owns no training RNG.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    try:
        result = dict(status='completed', metrics=callback(output))
    except BaseException:
        result = dict(status='failed', error=traceback.format_exc())
    save_json_state(result_path, result)


class EvaluationResult:
    """Completed metrics or a CPU scoring process, with deterministic cleanup."""

    def __init__(self, metrics=None, *, process=None, path=None):
        self.metrics, self.process, self.path = metrics, process, path

    @classmethod
    def background(cls, callback, output):
        path = Path(output) / 'scoring_result.json'
        path.unlink(missing_ok=True)
        process = multiprocessing.get_context('spawn').Process(
            target=_score_worker, args=(callback, Path(output), path))
        process.start()
        return cls(process=process, path=path)

    def result(self):
        if self.process is None:
            return self.metrics
        self.process.join()
        if self.process.exitcode or not self.path.is_file():
            raise RuntimeError(f'Background scoring failed: exit={self.process.exitcode}, {self.path}')
        result = json.loads(self.path.read_text(encoding='utf-8'))
        if result['status'] != 'completed':
            raise RuntimeError(result['error'])
        return result['metrics']

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.process.close()


class CheckpointEvaluator(Evaluator):
    """Subclass ``evaluate``; the engine calls ``on_save`` after a complete save.

    All ranks participate. DDP is unwrapped for independent inference; FSDP
    evaluators must keep their forward collectives aligned across ranks.
    Return numeric metrics on rank zero and write task artifacts inside the
    checkpoint directory. Evaluation failures stop training after state restore.
    """

    evaluation_seed = DEFAULT_INFERENCE_SEED
    tracker_metric_names = None  # None reports every metric; local records always retain all metrics.

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> dict[str, float]:
        """Evaluate the current student/generator, without loading a checkpoint."""

    def begin(self, context):
        """Tasks with CPU scoring can return pending results after generation."""
        return EvaluationResult(self.evaluate(context))

    def infer(self, context, config_path, expected_model):
        """Shared inference owns T5/VAE; the caller retains only live weights."""
        from inference import parse_args, run_inference

        def arguments():
            actual = f'{type(context.backend).__module__}:{type(context.backend).__name__}'
            if actual != expected_model:
                raise ValueError('Prepared camera backend differs from the training model')
            return parse_args(['--config', str(config_path)])

        args = distributed_call(arguments, context.process_group, 'Evaluation inference arguments')
        run_inference(args, backend=context.backend, generator=context.model,
                      rank=context.rank, world_size=context.world_size, device=context.device,
                      process_group=context.process_group)

    def run(self, config_path, arguments):
        """Optional task CLI; no arguments evaluates the configured baseline."""
        raise NotImplementedError(f"{type(self).__name__} has no standalone evaluation entry")

    def on_save(self, checkpoint_dir, step, *, model, backend, device, rank,
                world_size, process_group=None, tracker=None):
        modes = [(module, module.training) for module in model.modules()]
        context = EvaluationContext(
            unwrap_distributed_model(model), backend, Path(checkpoint_dir),
            step, device, rank, world_size, process_group,
        )
        path = context.checkpoint_dir / "evaluation.json"
        started = time.monotonic()
        with inference_rng(self.evaluation_seed, device=device):
            try:
                def prepare():
                    if rank == 0:
                        path.unlink(missing_ok=True)
                        path.with_suffix(".json.tmp").unlink(missing_ok=True)
                    model.eval()

                distributed_call(prepare, process_group, "Evaluation preparation")
                with torch.no_grad():
                    metrics = distributed_call(
                        lambda: self.evaluate(context), process_group,
                        f"Checkpoint {step} evaluation",
                    )

                def record():
                    if rank != 0:
                        return
                    values = {name: float(value) for name, value in metrics.items()}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("Evaluation returned non-finite metrics")
                    elapsed = time.monotonic() - started
                    result = dict(status="completed", step=step, metrics=values,
                                  elapsed_seconds=elapsed,
                                  evaluator=f"{type(self).__module__}:{type(self).__name__}")
                    save_json_state(path, result)
                    if tracker is not None:
                        reported = {**values, "seconds": elapsed}
                        names = reported if self.tracker_metric_names is None else self.tracker_metric_names
                        logged = {f"eval/{name}": reported[name] for name in names if name in reported}
                        if logged:
                            tracker.log(logged, step=step)
                    print(f"Evaluated checkpoint {step}: {values}", flush=True)
                distributed_call(record, process_group, "Evaluation metric logging")
                return metrics
            finally:
                for module, training in modes:
                    module.training = training


class CheckpointEvaluationSuite(CheckpointEvaluator):
    """Run GPU stages in order while earlier CPU scoring continues."""

    def __init__(self, entries):
        self.entries = entries
        names = [task.output_name for _, task in entries]
        if len(set(names)) != len(names):
            raise ValueError('Checkpoint evaluators require distinct output_name directories')
        self.tracker_metric_names = None if any(task.tracker_metric_names is None for _, task in entries) else tuple(
            f'{prefix}{name}' for index, (_, task) in enumerate(entries)
            for prefix in ('' if index == 0 else f'{task.output_name}/',)
            for name in (task.tracker_metric_names or ())
        )

    def evaluate(self, context):
        with ExitStack() as resources:
            pending = []
            for _, task in self.entries:
                def begin():
                    job = task.begin(context)
                    resources.callback(job.close)
                    return job
                pending.append(distributed_call(begin, context.process_group,
                                                f'{task.output_name} evaluation start'))
            metrics = {}
            for index, ((_, task), job) in enumerate(zip(self.entries, pending, strict=True)):
                values = distributed_call(job.result, context.process_group,
                                          f'{task.output_name} evaluation results')
                prefix = '' if index == 0 else f'{task.output_name}/'
                metrics.update({f'{prefix}{key}': value for key, value in values.items()})
            return metrics

    def _selected(self, arguments):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset', choices=[task.dataset_name for _, task in self.entries])
        args, arguments = parser.parse_known_args(arguments)
        return [(path, task) for path, task in self.entries
                if args.dataset is None or args.dataset == task.dataset_name], arguments

    def run(self, config_path, arguments):
        entries, arguments = self._selected(arguments)
        for path, task in entries:
            task.run(path, arguments)

    def run_all(self, config_path, arguments=()):
        entries, arguments = self._selected(arguments)
        for path, task in entries:
            task.run_all(path, arguments)

    def finalize(self, output_dir, steps, *, start_step=0):
        for _, task in self.entries:
            task.finalize(output_dir, steps, start_step=start_step)

    def close(self):
        for _, task in self.entries:
            task.close()


def load_evaluator(path, config_path=None, *, base=Evaluator):
    """Build named benchmarks from one config, or load a standalone task."""
    if path is None:
        return None
    source = OmegaConf.create({}) if config_path is None else load_config(config_path)[1]
    datasets = OmegaConf.select(source, 'evaluation.datasets')
    if datasets is not None:
        from config import _run_config, evaluation_dataset_config

        entries = []
        try:
            for name in datasets:
                selected = evaluation_dataset_config(source, name)
                selected.evaluation.inference_config = str(Path(config_path).resolve())
                selected_path = _run_config(Path(config_path).resolve(), selected, 'evaluate')
                task = load_evaluator(selected.evaluation.evaluator, selected_path, base=CheckpointEvaluator)
                task.dataset_name = name
                entries.append((selected_path, task))
            return CheckpointEvaluationSuite(entries)
        except BaseException:
            for _, task in entries:
                task.close()
            raise
    module, name = path.rsplit(":", 1)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, base):
        raise TypeError(f"{path} must subclass {base.__name__}")
    config = OmegaConf.create({}) if config_path is None else source.get('evaluation')
    if not OmegaConf.is_dict(config):
        raise TypeError("Evaluator configuration must contain an 'evaluation' mapping")
    config = {name: value for name, value in OmegaConf.to_container(config, resolve=True).items()
              if value is not None}
    config.pop('evaluator', None)
    return cls(**config)


def load_checkpoint_evaluator(path, config_path=None):
    """Training callbacks require live-weight support, beyond standalone eval."""
    return load_evaluator(path, config_path, base=CheckpointEvaluator)


def build_checkpoint_evaluator(args, *, resources=None, **context):
    """Load one optional task callback using the same module:Class convention."""
    with inference_rng(device=context.get("device")):
        evaluator = load_checkpoint_evaluator(
            getattr(args, "evaluator", None),
            getattr(args, "eval_config", None) or getattr(args, "config", None),
        )
    if evaluator is None:
        return None
    if resources is not None:
        resources.callback(evaluator.close)
    return partial(evaluator.on_save, **context)
