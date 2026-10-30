"""Download prepared training data and named benchmarks."""

import argparse
from pathlib import Path

DATASET_REPO = 'xzxxntxdy/History-Forcing-Data'
DATASET_REVISION = 'v1.0'


def release_root(parquet):
    path = Path(parquet)
    return path.parent.parent.parent if path.parent.parent.name == 'benchmarks' else path.parent.parent


def download(root, *, datasets=('playworld_oos', 'memobench_real'), train=True):
    from huggingface_hub import snapshot_download
    if set(datasets) - {'playworld_oos', 'memobench_real'}:
        raise ValueError('Unknown evaluation dataset')
    patterns = [f'benchmarks/{name}/**' for name in datasets]
    if train:
        patterns.append('train/**')
    return snapshot_download(DATASET_REPO, repo_type='dataset', revision=DATASET_REVISION,
                             allow_patterns=patterns, local_dir=root,
                             cache_dir=Path(root) / '.hf-cache', max_workers=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    download(args.root)


if __name__ == '__main__':
    main()
