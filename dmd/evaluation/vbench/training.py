"""Official VBench metrics on one training rank's locally generated videos.

This module never starts a process group or gathers results. The training hook
owns synchronization, merges the returned per-video records, and averages those
records in the same way as VBench's distributed metric entry points.
"""

from contextlib import ExitStack, contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import sys

from .evaluate import prepare_runtime
from .protocol import resolve_dimensions


TRAINING_DIMENSIONS = (
    'subject_consistency', 'dynamic_degree', 'motion_smoothness',
)
DINO_WEIGHTS_URL = ('https://dl.fbaipublicfiles.com/dino/'
                    'dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth')


def validate_dimensions(dimensions=None):
    selected = resolve_dimensions(TRAINING_DIMENSIONS if dimensions is None else dimensions)
    unsupported = set(selected) - set(TRAINING_DIMENSIONS)
    if unsupported:
        raise ValueError('Training VBench evaluation supports only '
                         + ', '.join(TRAINING_DIMENSIONS)
                         + f'; got {sorted(unsupported)}')
    return selected


def validate_metric_assets(vbench_root, cache_dir, dimensions=None):
    """Reject missing local code/weights before official initializers can download."""
    root, cache = Path(vbench_root).resolve(), Path(cache_dir).resolve()
    required = {
        'subject_consistency': (
            cache / 'dino_model/facebookresearch_dino_main/hubconf.py',
            cache / 'dino_model/facebookresearch_dino_main/vision_transformer.py',
            cache / 'dino_model/facebookresearch_dino_main/utils.py',
            cache / 'dino_model/dino_vitbase16_pretrain.pth',
        ),
        'dynamic_degree': (cache / 'raft_model/models/raft-things.pth',),
        'motion_smoothness': (
            root / 'vbench/third_party/amt/cfgs/AMT-S.yaml',
            cache / 'amt_model/amt-s.pth',
        ),
    }
    selected = validate_dimensions(dimensions)
    common = [root / 'vbench' / name
              for name in ('__init__.py', 'utils.py', 'distributed.py')]
    files = common + [root / 'vbench' / f'{name}.py' for name in selected]
    files += [path for name in selected for path in required[name]]
    missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Missing local VBench evaluation assets: ' + ', '.join(missing))
    return {name: [str(path) for path in required[name]] for name in selected}


@contextmanager
def _patch(module, **attributes):
    absent = object()
    previous = {name: getattr(module, name, absent) for name in attributes}
    try:
        for name, value in attributes.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                delattr(module, name)
            else:
                setattr(module, name, value)


@contextmanager
def _repo_imports(directory):
    """Scope vendored top-level imports, including DINO's generic ``utils``."""
    directory = Path(directory)
    names = {path.stem for path in directory.glob('*.py') if path.name != '__init__.py'}
    names.update(path.name for path in directory.iterdir()
                 if path.is_dir() and (path / '__init__.py').is_file())

    def belongs(name):
        return name.split('.', 1)[0] in names

    saved = {name: module for name, module in list(sys.modules.items()) if belongs(name)}
    original_path = sys.path[:]
    try:
        for name in saved:
            del sys.modules[name]
        sys.path.insert(0, str(directory))
        yield
    finally:
        for name in list(sys.modules):
            if belongs(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = original_path


@contextmanager
def _local_dino_weights(path):
    """Keep the unmodified DINO hub entry point entirely on local storage."""
    import torch

    def load(url, *args, **kwargs):
        if url != DINO_WEIGHTS_URL:
            raise ValueError(f'Unexpected DINO weight request: {url}')
        return torch.load(Path(path), map_location='cpu', weights_only=True)

    with _patch(torch.hub, load_state_dict_from_url=load):
        yield


@contextmanager
def _runtime(vbench_root, cache_dir, dimension):
    import torch

    root = Path(vbench_root).resolve()
    previous_path = sys.path[:]
    previous_cache = os.environ.get('VBENCH_CACHE_DIR')
    package = sys.modules.get('vbench')
    if package is not None:
        source = getattr(package, '__file__', None)
        if source is None or Path(source).resolve().parent != root / 'vbench':
            raise RuntimeError('A different VBench checkout is already imported')
    safe_globals = None
    serialization = None
    try:
        os.environ['VBENCH_CACHE_DIR'] = str(Path(cache_dir).resolve())
        sys.path.insert(0, str(root))
        utils = importlib.import_module('vbench.utils')
        if dimension == 'motion_smoothness':
            serialization = torch.serialization
            safe_globals = list(serialization.get_safe_globals())
        prepare_runtime([dimension])
        # Training enables TF32; independent official VBench uses FP32 matmul.
        with _patch(torch.backends.cuda.matmul, allow_tf32=False), _patch(
                    utils, CACHE_DIR=str(Path(cache_dir).resolve()),
                    get_rank=lambda: 0, barrier=lambda: None):
            yield utils
    finally:
        if serialization is not None and safe_globals is not None:
            serialization.clear_safe_globals()
            serialization.add_safe_globals(safe_globals)
        sys.path[:] = previous_path
        if previous_cache is None:
            os.environ.pop('VBENCH_CACHE_DIR', None)
        else:
            os.environ['VBENCH_CACHE_DIR'] = previous_cache


def _owned_metadata(full_info_json, dimension, owned_paths):
    metadata = json.loads(Path(full_info_json).read_text(encoding='utf-8'))
    if not isinstance(metadata, list):
        raise ValueError('VBench evaluation metadata must be a list')
    paths = []
    for row in metadata:
        if dimension in row['dimension']:
            values = row.get('video_list', [])
            paths.extend(values if isinstance(values, list) else [values])
    canonical = {str(Path(path).resolve()): str(path) for path in paths}
    if len(canonical) != len(paths):
        raise ValueError('VBench evaluation metadata contains duplicate video paths')
    owned = [str(Path(path).resolve()) for path in owned_paths]
    if len(set(owned)) != len(owned):
        raise ValueError('Owned VBench video paths contain duplicates')
    missing = set(owned) - set(canonical)
    if missing:
        raise ValueError(f'Owned videos are absent from {dimension} metadata: {sorted(missing)}')
    for path in owned:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return owned, canonical


def score_local(vbench_root, cache_dir, dimension, full_info_json, owned_paths, device):
    """Return official per-video scores for this rank, without nested collectives."""
    validate_dimensions([dimension])
    owned, canonical = _owned_metadata(full_info_json, dimension, owned_paths)
    if not owned:
        return []
    validate_metric_assets(vbench_root, cache_dir, [dimension])
    root, cache = Path(vbench_root).resolve(), Path(cache_dir).resolve()
    with _runtime(root, cache, dimension) as utils, ExitStack() as stack:
        if dimension == 'subject_consistency':
            stack.enter_context(_repo_imports(cache / 'dino_model/facebookresearch_dino_main'))
            stack.enter_context(_local_dino_weights(cache / 'dino_model/dino_vitbase16_pretrain.pth'))
        elif dimension == 'motion_smoothness':
            stack.enter_context(_repo_imports(root / 'vbench/third_party/amt'))
        metric = importlib.import_module('vbench.' + dimension)

        def local_paths(available):
            available = {str(Path(path).resolve()) for path in available}
            if not set(owned).issubset(available):
                raise ValueError('Official VBench loaded different evaluation metadata')
            return [canonical[path] for path in owned]

        stack.enter_context(_patch(metric, get_world_size=lambda: 1,
                                   distribute_list_to_rank=local_paths))
        submodules = utils.init_submodules([dimension], local=True, read_frame=False)
        _, records = getattr(metric, 'compute_' + dimension)(
            str(full_info_json), device, submodules[dimension])
    if not isinstance(records, list):
        raise ValueError('Official VBench did not return per-video records')
    result = []
    for record in records:
        path = str(Path(record['video_path']).resolve())
        value = float(record['video_results'])
        if not math.isfinite(value):
            raise ValueError(f'Non-finite {dimension} score for {path}')
        result.append({'video_path': path, 'video_results': value})
    returned = [row['video_path'] for row in result]
    if len(returned) != len(owned) or set(returned) != set(owned):
        raise ValueError(f'Incomplete or duplicate local {dimension} score records')
    return result
