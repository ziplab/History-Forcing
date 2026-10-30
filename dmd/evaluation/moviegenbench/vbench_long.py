"""Official VBench-Long scoring for all MovieGenBench prompts."""
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

from evaluation.generator import rank_context, write_json
from evaluation.moviegenbench.protocol import (
    FPS, PIXEL_FRAMES, PROTOCOL, episode_name, generation_receipt,
)


# VBench-Long README, section 3.2: all supported custom-video dimensions.
DIMENSIONS = ('subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality')


@contextmanager
def local_torch_hub():
    """Use prepared auxiliary repositories without GitHub branch discovery."""
    import torch
    original_load, original_directory = torch.hub.load, torch.hub.get_dir()
    repositories = {
        'facebookresearch/dino': 'facebookresearch_dino_main',
        'facebookresearch/dino:main': 'facebookresearch_dino_main',
        'facebookresearch/dinov2': 'facebookresearch_dinov2_main',
        'facebookresearch/dinov2:main': 'facebookresearch_dinov2_main',
    }

    def load(repo_or_dir, model, *args, source='github', **kwargs):
        if source == 'github' and repo_or_dir in repositories:
            repository = Path(torch.hub.get_dir()) / repositories[repo_or_dir]
            if not (repository / 'hubconf.py').is_file():
                raise FileNotFoundError(f'Missing local Torch Hub repository: {repository}')
            return original_load(str(repository), model, *args, source='local', **kwargs)
        return original_load(repo_or_dir, model, *args, source=source, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original_load
        # DreamSim changes this process-wide setting while loading its backbone.
        torch.hub.set_dir(original_directory)


def result_directory(output):
    return Path(output) / 'results' / 'vbench_long'


def scoring_protocol(args):
    root = Path(args.vbench_root)
    required = [root / 'vbench/VBench_full_info.json',
                root / 'vbench2_beta_long/__init__.py',
                root / 'vbench2_beta_long/configs/slow_fast_params.yaml',
                root / 'vbench2_beta_long/configs/subject_mapping_table.yaml',
                root / 'vbench2_beta_long/configs/background_mapping_table.yaml',
                root / 'vbench2_beta_long/configs/clip_length_mix.yaml',
                *[root / f'vbench2_beta_long/{dim}.py' for dim in DIMENSIONS]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing official VBench-Long source: {path}')
    sources = sorted({*required, *root.glob('vbench/*.py'),
                      *root.glob('vbench2_beta_long/*.py'),
                      *root.glob('vbench2_beta_long/configs/*.yaml')})
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {'benchmark': 'vbench_long', 'protocol': PROTOCOL, 'mode': 'long_custom_input',
            'dimensions': list(DIMENSIONS), 'dev_flag': True,
            'use_semantic_splitting': False, 'clip_length_seconds': 2,
            'sb_clip2clip_feat_extractor': 'dinov2', 'bg_clip2clip_feat_extractor': 'dreamsim',
            'slow_fast_weights': 'official configs/slow_fast_params.yaml',
            'imaging_quality_preprocessing_mode': 'longer',
            'vbench_root': str(root), 'source_sha256': digest.hexdigest(),
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cache': str(Path(args.cache_dir)),
            'torch_home': os.environ.get('TORCH_HOME',
                str(Path(__file__).resolve().parents[3] / 'models/.cache/torch')),
            'score_scale': 'raw and percentage'}


def protocol_id(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def video_stamp(directory, row):
    if generation_receipt(directory, row) is None:
        raise ValueError(f'Missing or invalid generation: {episode_name(row)}; rerun generate')
    name = episode_name(row)
    video = (directory / 'videos' / (name + '.mp4')).stat()
    receipt = (directory / 'generation' / (name + '.json')).stat()
    return {'video_size': video.st_size, 'video_mtime_ns': video.st_mtime_ns,
            'generation_mtime_ns': receipt.st_mtime_ns}


def score_receipt(output, row, dimension, identity):
    directory = Path(output)
    path = result_directory(output) / 'scores' / dimension / (episode_name(row) + '.json')
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        if (result['protocol_id'] != identity or result['source_index'] != row['source_index']
                or result['prompt'] != row['prompt']
                or result['dimension'] != dimension
                or result['generation'] != video_stamp(directory, row)
                or type(result['count']) is not int or result['count'] < 1
                or not math.isfinite(result['sum'])):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def official_options(root):
    config = Path(root) / 'vbench2_beta_long/configs'
    # Match eval_long.py --mode long_custom_input --dev_flag. In this mode the
    # official preprocessor uses 2s clips and retains the source frame rate.
    return {'use_semantic_splitting': False, 'clip_length_config': 'clip_length_mix.yaml',
            'dev_flag': True, 'static_filter_flag': False,
            'sb_clip2clip_feat_extractor': 'dinov2', 'bg_clip2clip_feat_extractor': 'dreamsim',
            'w_inclip': 1.0, 'w_clip2clip': 0.0,
            'slow_fast_eval_config': str(config / 'slow_fast_params.yaml'),
            'sb_mapping_file_path': str(config / 'subject_mapping_table.yaml'),
            'bg_mapping_file_path': str(config / 'background_mapping_table.yaml'),
            'imaging_quality_preprocessing_mode': 'longer', 'num_of_samples_per_prompt': 1}


def collect_scores(result, dimension, selected):
    """Retain upstream aggregation units, including its imaging /100 scaling."""
    records = result[1] if dimension in DIMENSIONS[:2] else result[2]
    grouped = defaultdict(list)
    expected = {episode_name(row) for row in selected}
    scale = 100.0 if dimension == 'imaging_quality' else 1.0
    for record in records:
        name = Path(record['video_path']).stem
        if name not in expected:
            raise ValueError(f'Unexpected VBench-Long video: {record["video_path"]}')
        value = float(record['video_results']) / scale
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite {dimension} score: {name}')
        grouped[name].append(value)
    if set(grouped) != expected:
        raise ValueError(f'Incomplete VBench-Long coverage: {dimension}')
    values = [value for group in grouped.values() for value in group]
    if not math.isclose(math.fsum(values) / len(values), float(result[0]), rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f'Unexpected upstream aggregation: {dimension}')
    return {name: {'sum': math.fsum(values), 'count': len(values)}
            for name, values in grouped.items()}


def validate_clips(inputs, selected):
    # Upstream's cache check only counts folders. Also reject an interrupted
    # split (one complete 2s clip for every ceil(source_frames / clip_frames)).
    from evaluation.video import valid_video
    count = math.ceil(PIXEL_FRAMES / (2 * FPS))
    for row in selected:
        name = episode_name(row)
        folder = inputs / 'split_clip' / name
        expected = {f'{name}_{index:03d}.mp4' for index in range(count)}
        clips = list(folder.glob('*.mp4'))
        if {p.name for p in clips} != expected:
            raise ValueError(f'Incomplete official split: {folder}; remove this worker cache and retry')
        if any(not valid_video(p, 2 * FPS, FPS) for p in clips):
            raise ValueError(f'Incomplete official clip in {folder}')


def prepare_clips(benchmark, inputs, selected, options):
    try:
        validate_clips(inputs, selected)
        return
    except ValueError:
        pass
    # An interrupted upstream split leaves directories that its folder-count
    # shortcut treats as complete. Discard only this rank's derived cache.
    cache = inputs / 'split_clip'
    if cache.exists():
        if cache.is_symlink() or cache.resolve().parent != inputs.resolve():
            raise ValueError(f'Unexpected split cache path: {cache}')
        shutil.rmtree(cache)
    benchmark.preprocess(str(inputs), 'long_custom_input', **options)
    validate_clips(inputs, selected)


def score(args, manifest):
    (result_directory(args.output) / 'COMPLETE').unlink(missing_ok=True)
    rank, world, device = rank_context()
    selected = manifest['prompts'][rank::world]
    if not selected:
        return
    protocol = scoring_protocol(args)
    identity = protocol_id(protocol)
    directory = Path(args.output)
    stamps = {episode_name(row): video_stamp(directory, row) for row in selected}
    if all(score_receipt(directory, row, dim, identity) is not None
           for dim in DIMENSIONS for row in selected):
        return
    # The workspace is private to a rank and a particular set of inputs, so
    # official preprocessing/model helpers never race on shared output files.
    signature = hashlib.sha256(json.dumps(stamps, sort_keys=True).encode()).hexdigest()[:16]
    work = result_directory(directory) / 'work' / identity[:16] / f'world{world}-rank{rank}-{signature}'
    inputs = work / 'videos'
    inputs.mkdir(parents=True, exist_ok=True)
    for row in selected:
        source = directory / 'videos' / (episode_name(row) + '.mp4')
        target = inputs / source.name
        if not target.is_symlink():
            target.symlink_to(source.resolve())
        elif target.resolve() != source.resolve():
            raise ValueError(f'Unexpected scoring link: {target}')

    os.environ['VBENCH_CACHE_DIR'] = protocol['cache']
    os.environ.setdefault('TORCH_HOME', protocol['torch_home'])
    sys.path.insert(0, str(Path(args.vbench_root)))
    from evaluation.vbench.evaluate import prepare_runtime
    prepare_runtime(DIMENSIONS)
    from vbench2_beta_long import VBenchLong
    import torch
    torch.set_grad_enabled(False)
    benchmark = VBenchLong(device, str(Path(args.vbench_root) / 'vbench/VBench_full_info.json'), str(work))
    options = official_options(args.vbench_root)
    # Split once before any metric runs, then verify every expected clip.
    prepare_clips(benchmark, inputs, selected, options)
    for dim in DIMENSIONS:
        if all(score_receipt(directory, row, dim, identity) is not None for row in selected):
            continue
        with local_torch_hub():
            benchmark.evaluate(str(inputs), dim, dimension_list=[dim], local=True,
                               read_frame=False, mode='long_custom_input', **options)
        raw_path = work / (dim + '_eval_results.json')
        result = json.loads(raw_path.read_text(encoding='utf-8'))[dim]
        scores = collect_scores(result, dim, selected)
        for row in selected:
            name = episode_name(row)
            if video_stamp(directory, row) != stamps[name]:
                raise ValueError(f'Generation changed while scoring: {name}')
            write_json(result_directory(directory) / 'scores' / dim / (name + '.json'),
                       {**row, 'dimension': dim,
                        'protocol_id': identity, 'protocol': protocol,
                        'generation': stamps[name], **scores[name]})
        print(json.dumps({'rank': rank, 'dimension': dim, 'videos': len(selected)}), flush=True)
        torch.cuda.empty_cache()


def summarize(args, manifest):
    (result_directory(args.output) / 'COMPLETE').unlink(missing_ok=True)
    directory = Path(args.output)
    protocol = scoring_protocol(args)
    identity = protocol_id(protocol)
    raw, records = {}, {}
    for dim in DIMENSIONS:
        totals, counts = [], []
        for row in manifest['prompts']:
            result = score_receipt(directory, row, dim, identity)
            if result is None:
                raise ValueError(f'Missing or invalid {dim} score: {episode_name(row)}; rerun score')
            totals.append(result['sum'])
            counts.append(result['count'])
            name = episode_name(row)
            records.setdefault(name, dict(row))[dim] = result['sum'] / result['count']
        # Weight the actual upstream units, never average rank-level means.
        raw[dim] = math.fsum(totals) / sum(counts)
    summary = {'status': 'completed', 'benchmark': 'vbench_long', 'videos': len(records),
               'prompts': len(manifest['prompts']), 'samples_per_prompt': 1,
               'raw': raw, 'percentage': {dim: value * 100 for dim, value in raw.items()},
               'protocol': protocol, 'run': json.loads((directory / 'run.json').read_text(encoding='utf-8'))}
    output = result_directory(directory)
    write_json(output / 'summary.json', summary)
    temporary = output / 'per_video.csv.partial'
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_index', 'prompt', *DIMENSIONS])
        writer.writeheader()
        writer.writerows(records.values())
    temporary.replace(output / 'per_video.csv')
    (output / 'COMPLETE').touch()
    print(json.dumps(summary['percentage'], indent=2))
