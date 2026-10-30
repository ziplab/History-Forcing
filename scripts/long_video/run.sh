#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT="$REPO_ROOT/dmd"
CONFIG_ROOT="$REPO_ROOT/configs/long_video"
cd "$REPO_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/models/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/.cache/torch}"
stage="${1:-train}"
if (( $# )); then shift; fi
case "$stage" in
  train) exec "${PYTHON:-python}" "$PROJECT_ROOT/train.py" "$@" ;;
  infer) exec "${PYTHON:-python}" "$PROJECT_ROOT/inference.py" "$@" ;;
  *) echo "Usage: bash scripts/long_video/run.sh {train|infer} [arguments]" >&2; exit 2 ;;
esac
