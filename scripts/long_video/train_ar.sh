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
CONFIG="${CONFIG:-$CONFIG_ROOT/ar_teacher_10s.yaml}"
LOGDIR="${LOGDIR:-output/dmd/ar_teacher_10s}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if (( NNODES > 1 )); then
  NODE_RANK="${NODE_RANK:-${RANK:?Set NODE_RANK or RANK on each node}}"
  MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
else
  NODE_RANK="${NODE_RANK:-${RANK:-0}}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
"$PYTHON" - "$CONFIG" "$NNODES" "$NPROC_PER_NODE" "$@" <<'PY'
import argparse
import sys
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config_path', default=sys.argv[1])
args, _ = parser.parse_known_args(sys.argv[4:])
config = OmegaConf.load(args.config_path)
frames = int(config.num_training_frames)
required = {'teacher_forcing': True, 'num_frame_per_block': 3,
            'model_kwargs.max_num_frames': frames,
            'model_kwargs.local_attn_size': frames, 'model_kwargs.sink_size': 3}
for name, expected in required.items():
    actual = OmegaConf.select(config, name)
    if actual != expected:
        raise SystemExit(f'AR_TEACHER_CONFIG_MISMATCH: {name}={actual}, expected {expected}')
world = int(sys.argv[2]) * int(sys.argv[3])
effective = world * int(config.batch_size) * int(config.get('gradient_accumulate', 1))
if effective != int(config.total_batch_size):
    raise SystemExit(f'WORLD_SIZE_GLOBAL_BATCH_MISMATCH: actual={effective}, config={config.total_batch_size}')
print(f'AR_TEACHER_CONFIG_OK:world_size={world}:global_batch={effective}')
PY

exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="${MASTER_PORT:-19002}" \
  "$PROJECT_ROOT/train.py" \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" "$@"
