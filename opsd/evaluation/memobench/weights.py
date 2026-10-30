"""Prepare MemoBench weights with the same loaders and caches used by scoring."""

import argparse
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LOADERS = {
    'aesthetic': ('visual_quality', '_load_aesthetic'),
    'clip_iqa': ('visual_quality', '_load_iqa'),
    'image_reward': ('visual_quality', '_load_image_reward'),
    'dino': ('geometry', '_load_dino'),
    'depth': ('geometry', '_load_depth_anything'),
    'raft': ('temporal', '_load_raft'),
    'clip': ('clip_metrics', '_load_clip'),
    'lpips': ('reference_fidelity', '_get_lpips'),
    'mapanything': ('camera_controllability', '_load_mapanything'),
}


def prepare(component):
    if component == 'sam3':
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError

        checkpoint = Path(os.environ.get('SAM3_CHECKPOINT') or ROOT / 'models/SAM3/sam3.pt')
        if not checkpoint.is_file():
            try:
                downloaded = hf_hub_download('facebook/sam3', 'sam3.pt', local_dir=checkpoint.parent)
            except GatedRepoError as error:
                raise RuntimeError('SAM3 requires access at https://huggingface.co/facebook/sam3; '
                                   'set an authorized HF_TOKEN and rerun python -m evaluation.memobench.weights '
                                   'with the PYTHONPATH and cache settings in the MemoBench README') from error
            if Path(downloaded) != checkpoint:
                Path(downloaded).replace(checkpoint)
        return

    from .adapter import upstream, metric_cache

    upstream('automated')
    visual = import_module('automated.metrics.visual_quality')
    metric_cache(visual)
    module, loader = LOADERS[component]
    getattr(import_module(f'automated.metrics.{module}'), loader)('cpu')
    if component == 'clip_iqa' and visual._iqa_backend != 'clipiqa+':
        raise RuntimeError('MemoBench requires CLIP-IQA+; its weights did not load successfully')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=(*LOADERS, 'sam3'))
    args = parser.parse_args(argv)
    if args.component:
        prepare(args.component)
    else:
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
        for component in (*LOADERS, 'sam3'):
            print(f'[MemoBench weights] {component}', flush=True)
            subprocess.run([sys.executable, '-m', 'evaluation.memobench.weights',
                            '--component', component], env=environment, check=True)


if __name__ == '__main__':
    main()
