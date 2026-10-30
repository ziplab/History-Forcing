#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/opsd/src:${PWD}/opsd${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-configs/world_model/dreamx/ar_few_step.yaml}"

stage="${1:-experiment}"
if (( $# )); then shift; fi
exec "${PYTHON:-python}" opsd/src/pipeline.py "$stage" \
  --config "$CONFIG" "$@"
