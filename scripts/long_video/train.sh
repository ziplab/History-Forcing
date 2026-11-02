#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT="$REPO_ROOT/dmd"
CONFIG_ROOT="$REPO_ROOT/configs/long_video"
cd "$REPO_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/models/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/.cache/torch}"
export WANDB_MODE="${WANDB_MODE:-online}"

PYTHON="${PYTHON:-python}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
[[ "$NNODES" =~ ^[1-9][0-9]*$ && "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || {
  echo 'NNODES and NPROC_PER_NODE must be positive integers' >&2
  exit 2
}
if (( NNODES > 1 )); then
  NODE_RANK="${NODE_RANK:-${RANK:?Set NODE_RANK or RANK on each node}}"
  MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
else
  NODE_RANK="${NODE_RANK:-${RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < NNODES )) || {
  echo 'NODE_RANK must be between 0 and NNODES - 1' >&2
  exit 2
}

"$PYTHON" - "$CONFIG_ROOT/dmd_ar10s_bi1p3b.yaml" "$NNODES" "$NPROC_PER_NODE" "$@" <<'PY'
import argparse
import sys
from utils.paths import load_config

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config_path', default=sys.argv[1])
args, _ = parser.parse_known_args(sys.argv[4:])
config = load_config(args.config_path)
world = int(sys.argv[2]) * int(sys.argv[3])
effective = world * int(config.batch_size) * int(config.get('gradient_accumulate', 1))
if effective != int(config.total_batch_size):
    raise SystemExit(f'WORLD_SIZE_GLOBAL_BATCH_MISMATCH: actual={effective}, config={config.total_batch_size}')
PY

exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="${MASTER_PORT:-19001}" \
  "$PROJECT_ROOT/train.py" \
  --config_path "$CONFIG_ROOT/dmd_ar10s_bi1p3b.yaml" "$@"
