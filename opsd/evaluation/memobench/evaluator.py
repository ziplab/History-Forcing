"""MemoBench Real evaluation with shared standalone and live-weight inference."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.io import write_json, write_text
from dataset_hub import download, release_root
from engine.evaluation import CheckpointEvaluator
from engine.runtime import distributed_call
from evaluation.trajectory import (
    _configured_packed_trajectory,
    _infer_and_register_trajectory,
    preset_metadata,
)
from config import _path_from_repo, load_config
from pipeline import _run, launch_environment, pack_data

from .adapter import CAMERA_METRICS, CAMERA_PROTOCOL, METRICS, PHASES, merge_scores, require_metrics, summarize
from .prepare import CAMERA_BLOCKS, PROTOCOL, gt_paths


class MemoBenchEvaluator(CheckpointEvaluator):
    """Own benchmark I/O and aggregation; generation remains shared inference."""

    tracker_metric_names = ()  # Only PlayWorld's two means are uploaded during training.

    def __init__(self, *, inference_config, prepared_data, baseline, world_model,
                 data_root, output_name='memobench_real', camera=None):
        self.inference_config = _path_from_repo(inference_config)
        self.prepared_data = _path_from_repo(prepared_data)
        self.baseline = _path_from_repo(baseline)
        self.data_root = _path_from_repo(data_root)
        self.world_model, self.output_name = world_model, output_name
        self.camera = camera or {'movement_blocks': CAMERA_BLOCKS,
                                 'initial_hold_blocks': 1, 'return_hold_blocks': 3}
        if Path(output_name).name != output_name or output_name in {'.', '..'}:
            raise ValueError('Evaluation output_name must be a directory name')

    def _metadata(self, config):
        if not self.prepared_data.is_file():
            download(release_root(self.prepared_data), datasets=('memobench_real',), train=False)
        metadata = preset_metadata(self.prepared_data)
        if (metadata['protocol'] != PROTOCOL or metadata['summary']['model'] != config.model
                or metadata.get('camera', self.camera) != self.camera or len(metadata['tasks']) != 164):
            raise ValueError('Prepared MemoBench camera/model contract changed; prepare the Parquet again')
        if len({task['task_id'] for task in metadata['tasks']}) != 164:
            raise ValueError('MemoBench task IDs must be unique')
        if any(not path.is_file() for task in metadata['tasks']
               for path in gt_paths(self.data_root, task['task_id'])):
            download(release_root(self.prepared_data), datasets=('memobench_real',), train=False)
        return metadata

    def _generate(self, checkpoint, output, config, metadata, env, *, inference_runner=None,
                  rank=0, world_size=1, process_group=None):
        from backbones import load_backend
        from dataset import read_packing_source

        if inference_runner is None:
            pack_data(config, ['--replace-existing'], env, section='inference', manifest=self.prepared_data)
        def prepare_inputs():
            args = SimpleNamespace(inference_config=self.inference_config, checkpoint=checkpoint,
                                   output_dir=output, rank=rank, world_size=world_size, process_group=process_group)
            config, options, _, seed = _configured_packed_trajectory(
                args, self.prepared_data, metadata['summary']['fps'])
            manifests, media = read_packing_source(self.prepared_data, load_backend(config.model),
                                                   config.inference.data_path)
            if len(manifests) != len(metadata['tasks']):
                raise ValueError('Packed and official MemoBench task counts differ')
            inputs = []
            for index, (row, task) in enumerate(zip(manifests, metadata['tasks'], strict=True)):
                if row['prompt'] != task['model_prompt']:
                    raise ValueError(f"Prepared official input changed: {task['task_id']}")
                inputs.append({'task_id': task['task_id'], 'row_index': index, 'seed': seed + index,
                               'student_prompt': row['prompt'],
                               'video_frames': task['video_frames'], 'actions': task['actions']})
            return args, config, options, inputs

        args, config, options, inputs = distributed_call(prepare_inputs, process_group, 'MemoBench generation inputs')

        def infer(path):
            _run('inference.py', path, [], env, min(len(inputs), int(env['NPROC_PER_NODE'])))

        def invalidate():
            if rank == 0:
                output.mkdir(parents=True, exist_ok=True)
                (output / 'summary.json').unlink(missing_ok=True)

        distributed_call(invalidate, process_group, 'MemoBench report invalidation')
        _infer_and_register_trajectory(args, config, options, inputs, PROTOCOL, inference_runner or infer)

    def evaluate(self, context):
        def prepare_inputs():
            _, config = load_config(self.inference_config)
            return config, self._metadata(config), launch_environment(config)

        config, metadata, env = distributed_call(prepare_inputs, context.process_group, 'MemoBench inputs')
        output = context.checkpoint_dir / self.output_name

        def generate():
            self._generate(context.checkpoint_dir / 'model.pt', output, config, metadata, env,
                           inference_runner=lambda path: self.infer(context, path, metadata['summary']['model']),
                           rank=context.rank, world_size=context.world_size,
                           process_group=context.process_group)

        distributed_call(generate, context.process_group, 'MemoBench generation')
        if context.rank != 0:
            return {}
        report = self.score(output, metadata, env)
        return {**report['metrics'], **report['camera_metrics']}

    def _verify_generated(self, output, metadata):
        inputs = json.loads((output / 'trajectory_inputs.json').read_text(encoding='utf-8'))
        if [item['task_id'] for item in inputs] != [task['task_id'] for task in metadata['tasks']]:
            raise ValueError('Generated MemoBench task IDs differ from the prepared inputs')
        for item, task in zip(inputs, metadata['tasks'], strict=True):
            result = json.loads((output / task['task_id'] / 'result.json').read_text(encoding='utf-8'))
            video = output / task['task_id'] / f"{task['task_id']}.mp4"
            sampling = result['sampling_manifest']
            if (result['status'] != 'completed' or result['player_protocol'] != PROTOCOL
                    or result['trajectory_inputs'] != item or not video.is_file()
                    or item['actions'] != task['actions']
                    or item['student_prompt'] != task['model_prompt'] or item['video_frames'] != task['video_frames']
                    or sampling.get('seed') != item['seed'] or sampling.get('status') != 'observed'
                    or sampling.get('schema_version') != 1 or not sampling.get('effective', {}).get('schedules')):
                raise ValueError(f"Unverified MemoBench video: {task['task_id']}")
        return inputs

    def score(self, output, metadata, env, *, camera_only=False):
        self._verify_generated(output, metadata)
        (output / 'summary.json').unlink(missing_ok=True)
        for phase in (('camera',) if camera_only else (*PHASES, 'camera')):
            # Metric subprocesses release their GPU models between phases;
            # shared inference already released T5/VAE, retaining live weights.
            processes = min(int(env['NPROC_PER_NODE']), len(metadata['tasks']))
            _run('evaluation.memobench.adapter', None, [phase,
                            '--prepared-data', str(self.prepared_data), '--data-root', str(self.data_root),
                            '--output', str(output)], env, processes if processes > 1 else None)
            if processes > 1:
                merge_scores(metadata, output, phase, processes)
        report = summarize(metadata, output)
        report['world_model'] = self.world_model
        report['sampling_signatures'] = {
            task['task_id']: json.loads((output / task['task_id'] / 'result.json').read_text(
                encoding='utf-8'))['sampling_manifest']['effective']['schedules'] for task in metadata['tasks']}
        write_json(output / 'summary.json', report)
        return report

    def run(self, config_path, arguments):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--phase', choices=('baseline', 'after'))
        parser.add_argument('--mode', choices=('player', 'score', 'both', 'camera'), default='both')
        parser.add_argument('--checkpoint', type=Path)
        args = parser.parse_args(arguments)
        phase = args.phase or ('after' if args.checkpoint is not None else 'baseline')
        if phase == 'baseline' and args.checkpoint is not None:
            parser.error('--checkpoint selects a trained model; omit --phase baseline')
        if phase == 'after' and args.checkpoint is None:
            parser.error('--phase after requires --checkpoint')
        _, config = load_config(self.inference_config)
        metadata, env = self._metadata(config), launch_environment(config)
        checkpoint = (_path_from_repo(config.paths.initial_checkpoint)
                      if phase == 'baseline' else _path_from_repo(args.checkpoint))
        if phase != 'baseline':
            from backbones import load_backend
            checkpoint = load_backend(config.model).checkpoint_file(checkpoint)
        output = self.baseline if phase == 'baseline' else checkpoint.parent / self.output_name
        if args.mode in {'player', 'both'}:
            self._generate(checkpoint, output, config, metadata, env)
        if args.mode in {'score', 'both', 'camera'}:
            self.score(output, metadata, env, camera_only=args.mode == 'camera')

    def finalize(self, output_dir, steps, *, start_step=0):
        outputs = {'Base': self.baseline, **{f'{step} step': Path(output_dir) /
                                           f'checkpoint-{step:06d}' / self.output_name for step in steps
                                           if step > start_step or (Path(output_dir) /
                                           f'checkpoint-{step:06d}' / self.output_name / 'summary.json').is_file()}}
        reports = {}
        for name, path in outputs.items():
            report = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
            reports[name] = {key: report[key] for key in (
                'status', 'protocol', 'count', 'metrics', 'records', 'gt_alignment',
                'camera_protocol', 'camera_metrics', 'world_model', 'sampling_signatures') if key in report}
        baseline = reports['Base']
        for name, report in reports.items():
            if (report.get('status') != 'completed'
                    or report['count'] != len(report['records'])
                    or len({record['id'] for record in report['records']}) != report['count']
                    or any(report[key] != baseline[key] for key in ('count',
                               'sampling_signatures'))):
                raise ValueError(f'Unpaired or incomplete MemoBench report: {name}')
            require_metrics([{'id': name, **report['metrics']}], METRICS)
            # Compare only the configured automatic metrics.
            report['metrics'] = {metric: report['metrics'][metric] for metric in METRICS}
            for record, base_record in zip(report['records'], baseline['records'], strict=True):
                if record['id'] != base_record['id']:
                    raise ValueError(f'MemoBench task order differs: {name}')
        result = Path(output_dir) / 'memobench_comparison.json'
        lines = ['# MemoBench Real: Base / checkpoints', '',
                 'Custom native block cameras; upstream automatic metrics; no VQA.', '']
        lines += ['## Full Real (164)', '', '| Metric | ' + ' | '.join(reports) + ' |',
                  '|---|' + '---:|' * len(reports)]
        for metric in METRICS:
            lines.append('| ' + metric + ' | ' + ' | '.join(f'{report["metrics"][metric]:.4f}'
                                                           for report in reports.values()) + ' |')
        lines += ['']
        lines += ['PSNR: dB; SSIM/LPIPS: native units. Remaining metrics: 0–100. LPIPS is lower-is-better.', '']
        if any('camera_metrics' in report for report in reports.values()):
            if any(report.get('camera_protocol') != CAMERA_PROTOCOL or
                   'camera_metrics' not in report for report in reports.values()):
                raise ValueError('Camera return comparisons require the current protocol for every model')
            require_metrics([{'id': name, **report['camera_metrics']} for name, report in reports.items()], CAMERA_METRICS)
            lines += ['## Camera return rotation error (custom metric)', '',
                      '| Metric | ' + ' | '.join(reports) + ' |', '|---|' + '---:|' * len(reports)]
            for metric in CAMERA_METRICS:
                lines.append('| ' + metric + ' | ' + ' | '.join(f'{report["camera_metrics"][metric]:.4f}'
                                                               for report in reports.values()) + ' |')
            lines += ['', 'ReturnRotError_deg: degrees, lower is better; final control timestamp after first-frame alignment.', '']
        write_json(result, reports)
        write_text(result.with_suffix('.md'), '\n'.join(lines))
        print(f'MemoBench comparison: {result.with_suffix(".md")}', flush=True)
