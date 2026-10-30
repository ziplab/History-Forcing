#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DMD_ROOT="$REPO_ROOT/dmd"
cd "$REPO_ROOT"
PYTHON=${PYTHON:-python}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
ARGS=()
usage() {
  cat <<'EOF'
Usage: bash scripts/long_video/eval_moviegenbench.sh --checkpoint MODEL [options]
  --config YAML        Default: config.yaml alongside the checkpoint.
  --weights ema|raw     Default: ema.
  --output-dir PATH     Default: checkpoint-parent/evaluation/..., named by config.
  --source PATH         Default: dmd/prompts/MovieGenVideoBench.txt (original prompts).
  --cache-dir PATH      VBench model cache (default: VBENCH_CACHE_DIR or models/vbench15s_comparison/vbench_cache).
  --vbench-root PATH    Official VBench checkout (same default as eval_vbench.sh).
  --score-only          Score existing videos without invoking generation.
  --gpus LIST           Default: CUDA_VISIBLE_DEVICES or 0,1,2,3,4,5,6,7.
  --python PATH         Python from the active environment.

All nonempty source rows, one video per prompt, including repeated prompt text.
252 latent/1005 RGB frames at 16 FPS, batch 1.
Official VBench-Long: all six dimensions supported for custom videos.
Re-run to resume generation and scoring; GPU count may change.
All relative paths start at the History-Forcing repository root.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|--checkpoint|--weights|--output-dir|--source|--cache-dir|--vbench-root) ARGS+=("$1" "${2:?}"); shift 2 ;;
    --score-only) SCORE_ONLY=1; shift ;;
    --gpus) GPU_LIST=${2:?}; shift 2 ;;
    --python) PYTHON=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU list' >&2; exit 2; }
PYTHON=$(command -v "$PYTHON")
[[ "$PYTHON" = /* ]] || PYTHON="$(pwd)/$PYTHON"
export PYTHONPATH="$DMD_ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_LIST"
BENCHMARK="$DMD_ROOT/evaluation/moviegenbench/evaluate.py"
RUN_DIR=$("$PYTHON" "$BENCHMARK" prepare "${ARGS[@]}")
exec 9>"$RUN_DIR/.evaluation.lock"
flock -n 9 || { echo "Another evaluation is using $RUN_DIR" >&2; exit 1; }
rm -f "$RUN_DIR/results/vbench_long/COMPLETE"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NPROC=${#GPU_IDS[@]}
STAGES=(score)
[[ "${SCORE_ONLY:-0}" = 1 ]] || STAGES=(generate score)
for stage in "${STAGES[@]}"; do
  OMP_NUM_THREADS=1 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$BENCHMARK" "$stage" "${ARGS[@]}"
done
"$PYTHON" "$BENCHMARK" summarize "${ARGS[@]}"
echo "VBENCH_LONG_COMPLETE:$RUN_DIR/results/vbench_long/summary.json"
