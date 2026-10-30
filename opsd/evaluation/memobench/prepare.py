"""Prepare all MemoBench Real inputs as the framework's image/action Parquet."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.io import read_json, write_parquet

HERE = Path(__file__).resolve().parent
METADATA = HERE / 'upstream/data/sam3_metadata/real_metadata.csv'
DATASET = 'tonyc54/MemoBench'
PROTOCOL = 'memobench_real_native_blocks_v1'
CAMERA_BLOCKS = {'pan_left_right': 5, 'pan_right_left': 4, 'tilt_up_down': 3}
DIRECTIONS = {'pan_left_right': ('LEFT', 'RIGHT'),
              'pan_right_left': ('RIGHT', 'LEFT'), 'tilt_up_down': ('UP', 'DOWN')}



VERSION = 'memobench-image-scene-process-v1'


def normalize_prompt(prompt: str) -> str:
    style = "Style: Photorealistic. "
    view = "First-person view."
    if not isinstance(prompt, str) or not prompt.startswith(style):
        raise ValueError(f"Scene prompt must begin exactly {style!r}")
    scene = prompt[len(style):]
    if scene.startswith(view) and (len(scene) == len(view) or scene[len(view)].isspace()):
        if not scene[len(view):].strip():
            raise ValueError("First-person prompt must contain nonempty scene text")
        return prompt
    if not scene.strip():
        raise ValueError("Scene prompt must contain nonempty scene text")
    return style + view + " " + scene


def validate_captions(captions, tasks):
    rows = captions.get('records', [])
    if (captions.get('version') != VERSION or not captions.get('model_signature') or
            len(rows) != len(tasks) or [row['task_id'] for row in rows] != [t['task_id'] for t in tasks]):
        raise ValueError('Qwen captions must cover every official MemoBench task in order')
    for row in rows:
        if (row['model_prompt'] != normalize_prompt(row['model_prompt'])):
            raise ValueError(f"Incomplete Qwen caption provenance: {row['task_id']}")


def gt_paths(root, clip):
    """Locate GT in the organized release or the original upstream download."""
    root = Path(root)
    if (root / 'Real').is_dir() and not (root / 'gt_videos').is_dir():
        return root / f'Real/{clip}/{clip}.mp4', root / f'Real/{clip}/raw/{clip}_mapanything.npz'
    return root / f'gt_videos/{clip}.mp4', root / f'gt_cameras/{clip}_mapanything.npz'


def real_tasks():
    with METADATA.open(encoding='utf-8') as handle:
        tasks = list(csv.DictReader(handle))
    for task in tasks:
        task['task_id'] = f"{int(task['video_id']):03d}"
        prompt = task['prompt']
        patterns = {'pan_left_right': r'pans to the left.*pans back to the right',
                    'pan_right_left': r'pans to the right.*pans back to the left',
                    'tilt_up_down': r'tilts upward.*tilts back down'}
        matches = [name for name, pattern in patterns.items() if re.search(pattern, prompt, re.I)]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized official camera instruction: {task['task_id']}")
        task['camera_type'] = matches[0]
    tasks.sort(key=lambda task: task['task_id'])
    if (len({task['task_id'] for task in tasks}) != 164 or
            Counter(task['camera_type'] for task in tasks) !=
            {'pan_left_right': 56, 'pan_right_left': 51, 'tilt_up_down': 57}):
        raise ValueError('The pinned MemoBench Real task list changed')
    return tasks


def action_sequence(camera_type, movement_blocks=CAMERA_BLOCKS, *,
                    initial_hold_blocks=1, return_hold_blocks=3):
    if set(movement_blocks) != set(DIRECTIONS):
        raise ValueError('Specify movement_blocks for all three Real camera types')
    counts = [*movement_blocks.values(), initial_hold_blocks, return_hold_blocks]
    if any(type(value) is not int or value < 1 for value in counts):
        raise ValueError('Camera block counts must be positive integers')
    outward, returning = DIRECTIONS[camera_type]
    count = movement_blocks[camera_type]
    return ['WAIT'] * initial_hold_blocks + [outward] * count + [returning] * count + ['WAIT'] * return_hold_blocks


def initial_image(video):
    """Return the unchanged first decoded GT frame as lossless PNG bytes."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = capture.read()
    finally:
        capture.release()
    if not success:
        raise ValueError(f'Cannot decode GT first frame: {video}')
    encoded, payload = cv2.imencode('.png', image)
    if not encoded:
        raise ValueError(f'Cannot encode GT first frame: {video}')
    return frames, payload.tobytes()


def prepare(data_root, output, model, *, camera=None, scene_prompts=None, reference_parquet=None):
    """CPU-only preparation; the backend owns every angle and temporal offset."""
    import pandas as pd
    from omegaconf import OmegaConf

    from backbones import load_backend
    from inference import DEFAULT_OUTPUT_FPS

    data_root, output = Path(data_root), Path(output)
    if OmegaConf.is_config(camera):
        camera = OmegaConf.to_container(camera, resolve=True)
    backend = load_backend(model)
    defaults = argparse.ArgumentParser(add_help=False)
    backend.add_inference_arguments(defaults)
    fps = defaults.get_default('fps') or DEFAULT_OUTPUT_FPS
    options = {key: value for key, value in backend.interactive_input_options().items()
               if key in ('size', 'resize_mode', 'resize_resample')}
    reference = reference_metadata = {}
    if reference_parquet is not None:
        from evaluation.trajectory import preset_metadata
        reference_metadata = preset_metadata(Path(reference_parquet))
        reference = {task['task_id']: task for task in reference_metadata['tasks']}
        phases = {ident: {key: task[key] for key in ('h_start_gt', 'r_start_gt')}
                  for ident, task in reference.items()}
    else:
        keyframes = pd.read_excel(data_root / 'Real/Real_ExitReenter.xlsx')
        phases = {f"{int(row['id']):03d}": {'h_start_gt': int(row['exit']),
                                           'r_start_gt': int(row['reenter'])}
                  for _, row in keyframes.iterrows()}
    records, entries, tasks = [], [], real_tasks()
    captions = read_json(scene_prompts) if scene_prompts else None
    caption_rows = {row['task_id']: row for row in captions['records']} if captions else {}
    if captions:
        validate_captions(captions, tasks)
    for index, task in enumerate(tasks):
        clip = task['task_id']
        gt, poses = gt_paths(data_root, clip)
        if not gt.is_file() or not poses.is_file() or clip not in phases:
            raise FileNotFoundError(f'Incomplete official GT assets: {clip}')
        gt_frames, image_bytes = initial_image(gt)
        if not 0 < phases[clip]['h_start_gt'] < phases[clip]['r_start_gt'] < gt_frames:
            raise ValueError(f'Invalid GT video or phase annotations: {clip}')
        caption = caption_rows.get(clip)
        task['model_prompt'] = (caption['model_prompt'] if caption else
                               reference.get(clip, {}).get('model_prompt', task['prompt']))
        actions = action_sequence(task['camera_type'], **(camera or {}))
        controls = backend.interactive_trajectory(actions, condition_frames=1)
        frames = len(controls['viewmats'])
        relative_camera = f'generation_cameras/{clip}.npz'
        camera_path = output.parent / relative_camera
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(camera_path, **{key: controls[key] for key in ('viewmats', 'intrinsics', 'rgb_frame_indices')})
        if not np.allclose(controls['viewmats'][-1], controls['viewmats'][0], atol=2e-4):
            raise ValueError(f'Camera does not return to its initial pose: {clip}')
        match = {'dataset': 'MemoBench', 'split': 'real', 'sample_id': clip,
                 'relation': 'official_evaluation_sample'}
        records.append({'image': {'bytes': image_bytes, 'path': f'{clip}.png'},
                        'prompt': task['model_prompt'], 'actions': [actions], 'source': {
                            'sample_id': clip, 'dataset': 'MemoBench', 'asset_id': clip,
                            'url': f'https://huggingface.co/datasets/{DATASET}/blob/main/Real/{clip}/{clip}.mp4',
                            'category': task['camera_type'],
                            'dataset_matches': [match], 'benchmark_matches': [match],
                            'camera_task_template': {'benchmark': 'MemoBench', 'split': 'real',
                                                     'sample_id': clip, 'relation': PROTOCOL}}})
        entries.append({'row_index': index, 'image_index': index, 'action_index': 0,
                        'options': {**options, 'condition_frames': 1, 'num_latent_frames': frames},
                        'camera': relative_camera})
        task.update(phases[clip], gt_frames=gt_frames, video_frames=backend.decoded_frames(frames), actions=actions)
    packing = {'version': 1, 'backend': model, 'rows': entries, 'evaluation': {
        'protocol': PROTOCOL, 'dataset': DATASET,
        'tasks': tasks, 'summary': {'model': model, 'fps': fps},
        'camera': camera or {'movement_blocks': CAMERA_BLOCKS,
                            'initial_hold_blocks': 1, 'return_hold_blocks': 3},
        'phase_policy': 'upstream GT keyframes linearly mapped to generated length',
        'reference_image': 'first decoded frame of the official GT video',
        'prompt_protocol': (captions['version'] if captions else
                            reference_metadata.get('prompt_protocol', 'official_prompt'))}}
    write_parquet(records, packing, output)
    return packing['evaluation']


def main():
    from config import _path_from_repo, evaluation_dataset_config, load_config
    from dataset_hub import download, release_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/world_model/dreamx/ar_few_step.yaml'))
    parser.add_argument('--download', action='store_true', help='Download Real GT videos, poses and annotations')
    parser.add_argument('--output', type=Path, help='Write the prepared Parquet and generation cameras')
    parser.add_argument('--scene-prompts', type=Path, help='Optional new Qwen annotations')
    args = parser.parse_args()
    _, config = load_config(args.config)
    config = evaluation_dataset_config(config, 'memobench_real')
    root = _path_from_repo(config.evaluation.data_root)
    if args.download:
        download(release_root(_path_from_repo(config.evaluation.prepared_data)),
                 datasets=('memobench_real',), train=False)
    metadata = prepare(root, args.output or _path_from_repo(config.evaluation.prepared_data), config.model,
                       camera=config.evaluation.get('camera'),
                       scene_prompts=args.scene_prompts,
                       reference_parquet=_path_from_repo(config.evaluation.prepared_data))
    print(f"Prepared {len(metadata['tasks'])} MemoBench Real inputs.")


if __name__ == '__main__':
    main()
