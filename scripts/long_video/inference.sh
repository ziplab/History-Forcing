#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export HF_HOME="${HF_HOME:-$PWD/models/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/models/.cache/torch}"

exec "${PYTHON:-python}" dmd/long_video/inference_memrope.py \
  --config configs/long_video/dmd_ar10s_bi1p3b.yaml \
  --checkpoint models/History-Forcing/dmd/model.pt "$@"
