"""Localize MemoBench paths; reuse its automatic metric functions."""

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path

from evaluation.io import write_json
from evaluation.trajectory import preset_metadata

from .prepare import HERE, PROTOCOL, gt_paths
from .camera import METRICS as CAMERA_METRICS, PROTOCOL as CAMERA_PROTOCOL

UPSTREAM = HERE / 'upstream'
ROOT = HERE.parents[2]
AUTOMATED = ('VisualQuality', 'MotionSmoothness', 'ObjIdentityConsistency',
             'Geo3DConsistency', 'CameraControllability', 'ImageRewardScore',
             'GT_ALL_PSNR', 'GT_ALL_SSIM', 'GT_ALL_LPIPS')
METRICS = (*AUTOMATED[:6], 'ORS', *AUTOMATED[6:])
PHASES = ('automated', 'ors')


@cache
def upstream(name):
    paths = {'automated': 'evaluation/run_eval.py', 'ors': 'evaluation/compute_ors.py'}
    path = UPSTREAM / paths[name]
    spec = importlib.util.spec_from_file_location(f'memobench_upstream_{name}', path)
    module = importlib.util.module_from_spec(spec)
    # The frozen scripts use local absolute imports. Keep those imports scoped
    # to their package; do not change process CWD or their metric implementations.
    sys.path.insert(0, str(UPSTREAM / 'evaluation'))
    previous = logging.root.manager.disable, logging.root.level
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(UPSTREAM / 'evaluation'))
        logging.disable(previous[0])
        logging.root.setLevel(previous[1])
    return module


def require_metrics(records, names):
    for record in records:
        missing = [name for name in names if record.get(name) is None or
                   not math.isfinite(float(record[name]))]
        if missing:
            raise ValueError(f"Incomplete MemoBench metrics for {record['id']}: {missing}")


@contextmanager
def local_torch_hub():
    """Resolve Torch Hub repositories from disk without querying GitHub."""
    import torch

    load = torch.hub.load

    def load_local(repo_or_dir, model, *args, source='github', **kwargs):
        if source == 'github':
            repo, _, ref = repo_or_dir.partition(':')
            candidates = (Path(torch.hub.get_dir()) / f"{repo.replace('/', '_')}_{branch.replace('/', '_')}"
                          for branch in ([ref] if ref else ('main', 'master')))
            directory = next((path for path in candidates if (path / 'hubconf.py').is_file()), None)
            if directory is None:
                raise FileNotFoundError(f'Torch Hub repository is not cached: {repo_or_dir}; prepare it before scoring')
            repo_or_dir, source = str(directory), 'local'
        return load(repo_or_dir, model, *args, source=source, **kwargs)

    torch.hub.load = load_local
    try:
        yield
    finally:
        torch.hub.load = load


def metric_cache(visual):
    """Redirect upstream download paths, preserving model identities and math."""
    root = ROOT / 'models/.cache'
    os.environ.setdefault('HF_HOME', str(root / 'huggingface'))
    os.environ.setdefault('TORCH_HOME', str(root / 'torch'))
    cache_dir = Path(os.environ['TORCH_HOME'])
    import ImageReward
    import open_clip
    visual._AESTHETIC_CACHE = str(cache_dir / 'aesthetic_predictor.pth')
    open_clip.create_model_and_transforms = partial(open_clip.create_model_and_transforms,
                                                   cache_dir=str(cache_dir / 'clip'))
    ImageReward.load = partial(ImageReward.load, download_root=str(cache_dir / 'ImageReward'))


@local_torch_hub()
def score_automated(metadata, data_root, output, *, score_dir=None):
    import pandas as pd
    official = upstream('automated')
    from automated.metrics import camera_controllability as camera
    from automated.metrics import visual_quality
    metric_cache(visual_quality)
    camera.GT_REAL_ROOT = str(data_root / 'Real')
    camera.GT_REAL_POSES = str(data_root / 'gt_cameras')
    from .camera import cache_upstream_estimates
    rows = []
    for task in metadata['tasks']:
        clip = task['task_id']
        video = output / clip / f'{clip}.mp4'
        gt, poses = gt_paths(data_root, clip)
        with cache_upstream_estimates(camera, video, output / f'scores/camera_poses/{clip}.npz'):
            rows.append(official._eval_clip({
                'id': clip, 'scene': 'real', 'data_type': 'real',
                'video_path': str(video), 'frames_dir': None,
                'gt_mp4': str(gt),
                'gt_intrinsics_npz': str(poses),
                'h_start_gt': task['h_start_gt'], 'r_start_gt': task['r_start_gt'],
                'prompt': task['model_prompt']}, 640, 4, 'cuda', camera_ctrl=True))
    frame = pd.DataFrame(rows)
    for metric, (low, high) in official._RAW_NORM.items():
        if metric in frame:
            frame[f'{metric}_pct'] = ((frame[metric] - low) / (high - low) * 100).clip(0, 100).round(1)
    frame = official._compute_composite(frame)
    score_dir = score_dir or output / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(score_dir / 'automated.csv', index=False)
    # Pandas turns absent optional diagnostics into NaN. Keep JSON portable,
    # while required metrics must still pass the finite-value check below.
    records = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    for record in records:
        record['image_quality_backend'] = visual_quality._iqa_backend
    require_metrics(records, AUTOMATED)
    write_json(score_dir / 'automated.json', records)


@local_torch_hub()
def score_ors(metadata, output, *, score_dir=None):
    import torch

    official = upstream('ors')
    automated = json.loads((output / 'scores/automated.json').read_text())
    by_id = {row['id']: row for row in automated}
    if len(by_id) != len(automated):
        raise ValueError('Automated clip IDs must be unique')
    automated = [by_id[task['task_id']] for task in metadata['tasks']]
    checkpoint = os.environ.get('SAM3_CHECKPOINT')
    cached = ROOT / 'models/SAM3/sam3.pt'
    if not checkpoint and cached.is_file():
        checkpoint = str(cached)
    if checkpoint:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        processor = Sam3Processor(build_sam3_image_model(
            checkpoint_path=checkpoint, load_from_HF=False))
    else:
        processor = official.init_sam3()
    records = []
    for task, row in zip(metadata['tasks'], automated, strict=True):
        clip = task['task_id']
        frames = official.extract_frames_from_mp4(str(output / clip / f'{clip}.mp4'))
        if len(frames) != task['video_frames']:
            raise ValueError(f'Incomplete generated video: {clip}')
        # SAM3's released image examples use BF16 autocast for the full forward.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = official.compute_ors_for_clip(processor, frames, row['r_start'],
                                                   task['subject'], is_bgr_arrays=True)
        records.append({'id': clip, **result, 'ORS': result['ors'] * 100})
    require_metrics(records, ('ORS',))
    score_dir = score_dir or output / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    write_json(score_dir / 'ors.json', records)


@local_torch_hub()
def score_camera(metadata, prepared_data, output, *, score_dir=None):
    """Score existing videos against verified packed controls; never regenerate."""
    import numpy as np
    import torch
    from backbones import load_backend
    from dataset import read_packing_source
    from .camera import cached_poses, return_rotation_error

    official = upstream('automated')
    from automated.metrics import camera_controllability as camera
    # Use the generation snapshot, not today's config or action defaults.
    from config import load_config, _path_from_repo
    _, config = load_config(output / 'inference.yaml')
    manifests, media = read_packing_source(prepared_data, load_backend(config.model),
                                         _path_from_repo(config.inference.data_path))
    inputs = {item['task_id']: item for item in json.loads((output / 'trajectory_inputs.json').read_text())}
    records = []
    for task in metadata['tasks']:
        clip, item = task['task_id'], inputs[task['task_id']]
        controls = media / manifests[item['row_index']]['camera']
        video = output / clip / f'{clip}.mp4'
        frames = official.VideoReader(str(video), max_side=640)
        if frames.num_frames != item['video_frames']:
            raise ValueError(f'Incomplete camera-scoring video: {clip}')
        torch.manual_seed(item['seed'])
        poses = cached_poses(camera, frames, video, output / f'scores/camera_poses/{clip}.npz', 'cuda')
        with np.load(controls) as command:
            result = return_rotation_error(poses, command['viewmats'], command['rgb_frame_indices'])
        records.append({'id': clip, **result})
        # Persist each completed clip; interrupted workers can reuse pose caches.
        score_dir = score_dir or output / 'scores'
        score_dir.mkdir(parents=True, exist_ok=True)
        write_json(score_dir / 'camera.json', records)
        print(f"Camera {clip}: return rotation error {result['ReturnRotError_deg']:.2f} deg", flush=True)
    require_metrics(records, CAMERA_METRICS)


def merge_scores(metadata, output, phase, processes):
    """Merge independent GPU workers in the official task order."""
    records = [row for rank in range(processes) for row in json.loads(
        (output / f'scores/rank-{rank}/{phase}.json').read_text())]
    by_id = {row['id']: row for row in records}
    expected = [task['task_id'] for task in metadata['tasks']]
    if len(by_id) != len(records) or set(by_id) != set(expected):
        raise ValueError('GPU scoring shards contain duplicate, missing, or unknown clips')
    records = [by_id[ident] for ident in expected]
    require_metrics(records, {'automated': AUTOMATED, 'ors': ('ORS',), 'camera': CAMERA_METRICS}[phase])
    write_json(output / f'scores/{phase}.json', records)
    if phase == 'automated':
        import pandas as pd
        pd.DataFrame(records).to_csv(output / 'scores/automated.csv', index=False)


def summarize(metadata, output):
    parts = [json.loads((output / f'scores/{name}.json').read_text(encoding='utf-8'))
             for name in PHASES]
    expected = [task['task_id'] for task in metadata['tasks']]
    if any([row['id'] for row in records] != expected for records in parts):
        raise ValueError('MemoBench scoring phases must cover exactly the same selected clips')
    records = [{**automatic, **ors} for automatic, ors in zip(*parts, strict=True)]
    require_metrics(records, METRICS)

    def means(rows):
        return {name: sum(float(row[name]) for row in rows) / len(rows) for name in METRICS}

    report = {'status': 'completed', 'protocol': PROTOCOL, 'count': len(records),
              'metrics': means(records), 'records': records,
              'gt_alignment': 'Official GT phases/poses and upstream resampling; camera inputs use custom native blocks'}
    camera_path = output / 'scores/camera.json'
    if camera_path.is_file():
        camera_rows = json.loads(camera_path.read_text())
        if ([row['id'] for row in camera_rows] != expected or
                any(row.get('camera_protocol') != CAMERA_PROTOCOL for row in camera_rows)):
            raise ValueError('Camera return errors must cover the same clips and protocol')
        require_metrics(camera_rows, CAMERA_METRICS)
        report['camera_protocol'] = CAMERA_PROTOCOL
        report['camera_metrics'] = {name: sum(row[name] for row in camera_rows) / len(camera_rows)
                                    for name in CAMERA_METRICS}
        report['records'] = [{**record, **camera} for record, camera in zip(records, camera_rows, strict=True)]
    write_json(output / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=(*PHASES, 'camera'))
    for name in ('prepared_data', 'data_root', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    metadata = preset_metadata(args.prepared_data)
    (args.output / 'scores').mkdir(parents=True, exist_ok=True)
    processes = int(os.environ.get('WORLD_SIZE', 1))
    score_dir = None
    if processes > 1:
        import torch
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        metadata = {**metadata, 'tasks': metadata['tasks'][rank::processes]}
        score_dir = args.output / f'scores/rank-{rank}'
    if args.phase == 'automated':
        score_automated(metadata, args.data_root, args.output, score_dir=score_dir)
    elif args.phase == 'ors':
        score_ors(metadata, args.output, score_dir=score_dir)
    else:
        score_camera(metadata, args.prepared_data, args.output, score_dir=score_dir)


if __name__ == '__main__':
    main()
