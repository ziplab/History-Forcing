"""PlayWorld's preset-camera evaluator for the shared checkpoint callback."""

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from engine.evaluation import CheckpointEvaluator, EvaluationResult
from engine.runtime import distributed_call
from config import _path_from_repo, load_config, stage_argument, stage_values
from pipeline import _run, launch_environment, pack_data
from .adapter import (PRESET_TRAJECTORY_PROTOCOL, SEED_LITE, preset_metadata,
                      run_preset, run_score, _write)
from .compare_oos_subset import compare, write_comparison_table


class PlayWorldEvaluator(CheckpointEvaluator):
    tracker_metric_names = ("raw_mean", "official_oos_mean")

    def __init__(self, *, inference_config, prepared_data, baseline, world_model,
                 data_root=None, selection=None, output_name="playworld_preset_oos",
                 workers=3, judge_provider="seed-lite", judge_model=SEED_LITE):
        self.inference_config = _path_from_repo(inference_config)
        self.prepared_data = _path_from_repo(prepared_data)
        self.data_root = _path_from_repo(data_root or prepared_data)
        self.selection = _path_from_repo(selection or prepared_data)
        self.baseline = _path_from_repo(baseline)
        self.world_model = world_model
        self.output_name = output_name
        self.workers = workers
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        metadata = preset_metadata(self.prepared_data)
        summary = metadata["summary"]
        self.model = summary["model"]
        self.fps = summary["fps"]
        selected = ((metadata if self.selection == self.prepared_data else preset_metadata(self.selection))["selection"]
                    if self.selection.suffix == ".parquet"
                    else json.loads(self.selection.read_text(encoding="utf-8")))
        ids = selected["task_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Evaluation task IDs must be nonempty and unique")
        if Path(output_name).name != output_name or output_name in {".", ".."}:
            raise ValueError("Evaluation output_name must be a directory name")
        if workers <= 0:
            raise ValueError("Scoring workers must be positive")
        self.task_ids = ",".join(ids)

    def _generate(self, checkpoint, output, inference_runner, *, rank=0, world_size=1,
                  process_group=None):
        def invalidate():
            if rank == 0:
                self._clear_reports(output)

        if world_size > 1:
            distributed_call(invalidate, process_group, "Evaluation report invalidation")
        else:
            invalidate()
        run_preset(SimpleNamespace(
            inference_config=self.inference_config, prepared_data=self.prepared_data,
            checkpoint=checkpoint, data_root=self.data_root, task_ids=self.task_ids,
            output_dir=output, rank=rank, world_size=world_size, process_group=process_group,
        ), inference_runner=inference_runner)

    def generate(self, context):
        output = context.checkpoint_dir / self.output_name

        def infer(config_path):
            self.infer(context, config_path, self.model)

        # Shared inference owns and releases T5/VAE on every rank before scoring.
        self._generate(context.checkpoint_dir / "model.pt", output, infer, rank=context.rank,
                       world_size=context.world_size, process_group=context.process_group)
        return output

    def begin(self, context):
        output = self.generate(context)
        return (EvaluationResult.background(self.score_metrics, output) if context.rank == 0
                else EvaluationResult({}))

    def evaluate(self, context):
        output = self.generate(context)
        if context.rank != 0:
            return {}
        return self.score_metrics(output)

    def score_metrics(self, output):
        report = self.score(output)
        summary = report["after"]
        metrics = {name: summary[name] for name in (
            "official_oos_mean", "raw_mean", "gate_pass", "count",
            "evolution_quality_mean",
        ) if summary[name] is not None}
        metrics.update({f"{name}/mean": value["mean"]
                        for name, value in summary["raw_category_means"].items()})
        if report["paired_delta_mean"] is not None:
            metrics["paired_delta_mean"] = report["paired_delta_mean"]
        return metrics

    def run(self, config_path, arguments):
        """Use the callback's configuration and report format for standalone runs."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--phase', choices=('baseline', 'after', 'compare'))
        parser.add_argument('--mode', choices=('player', 'score', 'both'), default='both')
        parser.add_argument('--checkpoint', type=Path)
        args = parser.parse_args(arguments)
        phase = args.phase or ('after' if args.checkpoint is not None else 'baseline')
        if phase == 'baseline' and args.checkpoint is not None:
            parser.error('--checkpoint selects a trained model; omit --phase baseline')
        inference_config = None
        if phase == 'baseline':
            output = self.baseline
        else:
            if args.checkpoint is not None:
                checkpoint = _path_from_repo(args.checkpoint)
            else:
                source, config = load_config(config_path)
                train = stage_values(config, 'train')
                checkpoint = (_path_from_repo(train['output_dir']) /
                              f"checkpoint-{int(train['max_train_steps']):06d}" / 'model.pt')
                if source == self.inference_config:
                    inference_config = config
            output = checkpoint.parent / self.output_name
        if phase == 'compare':
            report = compare(self.selection, self.baseline, output, model=self.model,
                             fps=self.fps, protocol=PRESET_TRAJECTORY_PROTOCOL)
            self._save_report(output, report)
            return
        if args.mode in {'player', 'both'}:
            if inference_config is None:
                _, inference_config = load_config(self.inference_config)
            if stage_argument(inference_config, 'inference', 'model') != self.model:
                raise ValueError("Prepared camera backend differs from inference model")
            if phase == 'baseline':
                checkpoint = _path_from_repo(stage_argument(inference_config, 'inference', 'checkpoint'))
            env = launch_environment(inference_config)
            pack_data(inference_config, ['--replace-existing'], env,
                      section='inference', manifest=self.prepared_data)

            def infer(path):
                processes = min(len(self.task_ids.split(',')), int(env['NPROC_PER_NODE']))
                _run('inference.py', path, [], env, processes)

            self._generate(checkpoint, output, infer)
        if args.mode in {'score', 'both'}:
            self.score(output)

    def finalize(self, output_dir, steps, *, start_step=0):
        output = Path(output_dir)
        self._clear_comparison(output)
        reports = {}
        for step in steps:
            name = f'checkpoint-{step:06d}'
            report, archived = self._report_paths(output / name / self.output_name)
            path = archived if not (output / name).exists() and archived.is_file() else report
            if step > start_step or path.is_file():
                reports[f'{step} step'] = path
        if not reports:
            print("No checkpoint evaluation reports to aggregate.", flush=True)
            return
        write_comparison_table(reports, output / 'evaluation_comparison.json')
        print(f"Evaluation comparison: {output / 'evaluation_comparison.md'}", flush=True)

    @staticmethod
    def _clear_comparison(output_dir):
        for suffix in ('.json', '.md'):
            (output_dir / f'evaluation_comparison{suffix}').unlink(missing_ok=True)

    def _report_paths(self, output):
        output = Path(output)
        path = output.with_name(output.name + '_comparison.json')
        yield path
        # Retention removes checkpoint weights and videos, but final summaries
        # still need the small reports from every evaluated step.
        if re.fullmatch(r'checkpoint-\d{6,}', output.parent.name):
            yield output.parent.parent / 'evaluation_reports' / output.parent.name / path.name

    def _clear_reports(self, output):
        output = Path(output)
        current, *archives = self._report_paths(output)
        for path in (current, *archives, output.with_name(output.name + '_score') / 'summary.json'):
            path.unlink(missing_ok=True)
            path.with_suffix('.md').unlink(missing_ok=True)
        if archives:
            self._clear_comparison(output.parent.parent)

    def _save_report(self, output, report):
        output = Path(output)
        self._clear_reports(output)
        for path in self._report_paths(output):
            _write(path, report)
        _write(output.with_name(output.name + '_score') / 'summary.json', report['after'])

    def score(self, output):
        """Score unchanged videos and verify all selected tasks against the baseline."""
        output = Path(output)
        self._clear_reports(output)
        score_output = output.with_name(output.name + "_score")
        scoring = SimpleNamespace(
            data_root=self.data_root, task_ids=self.task_ids,
            videos_dir=output, output_dir=score_output, world_model=self.world_model,
            workers=int(os.environ.get('PLAYWORLD_SCORE_WORKERS', self.workers)), judge_provider=self.judge_provider,
            judge_model=self.judge_model, video_manifest=None, dry_run=False,
        )
        run_score(scoring)
        report = compare(
            self.selection, self.baseline, output,
            model=self.model, fps=self.fps,
            protocol=PRESET_TRAJECTORY_PROTOCOL,
        )
        self._save_report(output, report)
        return report
