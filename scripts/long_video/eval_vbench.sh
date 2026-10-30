#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DMD_ROOT="$REPO_ROOT/dmd"
cd "$REPO_ROOT"
SUITE=quality
DIMENSIONS=""
CONFIG=""
CHECKPOINT=""
PROMPT_FILE="$DMD_ROOT/prompts/vbench.txt"
OUTPUT_DIR=""
VBENCH_ROOT=${VBENCH_ROOT:-$DMD_ROOT/evaluation/vbench/VBench}
PYTHON=${PYTHON:-python}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WEIGHTS=ema
usage() {
  cat <<'EOF'
Usage: bash scripts/long_video/eval_vbench.sh --checkpoint MODEL [options]

  --config YAML            Default: config.yaml alongside the checkpoint.
  --suite quality|semantic  Default: quality (seven metrics); semantic has nine.
  --dimensions CSV          Optional metric subset within the selected suite.
  --prompt-file PATH        Default: dmd/prompts/vbench.txt.
  --vbench-root PATH        Official VBench checkout.
  --output-dir PATH         Default: checkpoint-parent/evaluation/..., named by config.
  --gpus LIST               Default: CUDA_VISIBLE_DEVICES or 0,1,2,3,4,5,6,7.
  --weights ema|raw         Default: ema.
  --python PATH             Python from the active environment.

15s VBench: quality7 uses 326 prompts; semantic9 uses 799 prompt entries.
Five videos each, 63 latent/249 RGB frames, 832x480, 16 FPS, batch 4, seed 0
on every rank. The final batch computes one
video while retaining the batch-4 noise draws. Re-run to resume with the same
GPU count. All relative paths start at the History-Forcing repository root.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE=${2:?}; shift 2 ;;
    --dimensions) DIMENSIONS=${2:?}; shift 2 ;;
    --config) CONFIG=${2:?}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --prompt-file) PROMPT_FILE=${2:?}; shift 2 ;;
    --vbench-root) VBENCH_ROOT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --gpus) GPU_LIST=${2:?}; shift 2 ;;
    --weights) WEIGHTS=${2:?}; shift 2 ;;
    --python) PYTHON=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$SUITE" in quality|semantic) ;; *) echo 'Expected --suite quality|semantic' >&2; exit 2 ;; esac
case "$WEIGHTS" in ema|raw) ;; *) echo 'Expected --weights ema|raw' >&2; exit 2 ;; esac
[[ -n "$CHECKPOINT" ]] || { usage >&2; exit 2; }
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU list' >&2; exit 2; }
resolve_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s\n' "$REPO_ROOT/$1"; fi
}
CHECKPOINT=$(resolve_path "$CHECKPOINT")
if [[ -n "$CONFIG" ]]; then CONFIG=$(resolve_path "$CONFIG"); else CONFIG="$(dirname -- "$CHECKPOINT")/config.yaml"; fi
PROMPT_FILE=$(resolve_path "$PROMPT_FILE")
VBENCH_ROOT=$(resolve_path "$VBENCH_ROOT")
FULL_INFO="$VBENCH_ROOT/vbench/VBench_full_info.json"
for path in "$CONFIG" "$CHECKPOINT" "$PROMPT_FILE" "$VBENCH_ROOT/evaluate.py" "$FULL_INFO"; do
  [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
PYTHON=$(command -v "$PYTHON")
[[ "$PYTHON" = /* ]] || PYTHON="$(pwd)/$PYTHON"
export PYTHONPATH="$DMD_ROOT" PYTHONUNBUFFERED=1
BENCHMARK="$DMD_ROOT/evaluation/vbench"
DIMENSION_ARGS=()
[[ -z "$DIMENSIONS" ]] || DIMENSION_ARGS=(--dimensions "$DIMENSIONS")
PLAN=$("$PYTHON" "$BENCHMARK/protocol.py" plan --suite "$SUITE" --prompt-file "$PROMPT_FILE" \
  --full-info "$FULL_INFO" "${DIMENSION_ARGS[@]}")
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NPROC=${#GPU_IDS[@]}
OUTPUT_ARGS=()
[[ -z "$OUTPUT_DIR" ]] || OUTPUT_ARGS+=(--output-dir "$(resolve_path "$OUTPUT_DIR")")
RUN_DIR=$("$PYTHON" "$DMD_ROOT/evaluation/generator.py" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --weights "$WEIGHTS" --suite "$SUITE" --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" \
  --world-size "$NPROC" "${OUTPUT_ARGS[@]}" "${DIMENSION_ARGS[@]}")
exec 9>"$RUN_DIR/.evaluation.lock"
flock -n 9 || { echo "Another evaluation is using $RUN_DIR" >&2; exit 1; }
RESULT_DIR="$RUN_DIR/results/$SUITE"
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR/COMPLETE"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
matching_result() {
  "$PYTHON" "$BENCHMARK/score.py" find-result --dimension "$1" --group "$2" \
    --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" --videos "$RUN_DIR/videos" --output-dir "$3"
}
RESULTS=()
while IFS=$'\t' read -r group expected dimension_list; do
  missing=$("$PYTHON" "$BENCHMARK/protocol.py" missing-videos \
    --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" --group "$group" --videos "$RUN_DIR/videos")
  IFS=',' read -r -a group_dimensions <<< "$dimension_list"
  if [[ "$missing" -gt 0 ]]; then
    # Repairing videos invalidates scores computed from the old files.
    for dimension in "${group_dimensions[@]}"; do
      out="$RESULT_DIR/by_dimension/$dimension"
      if [[ -d "$out" ]]; then
        find "$out" -maxdepth 1 -type f -name '*_eval_results.json' -delete
      fi
    done
    echo "GENERATE:$group:videos=$expected:duration=15s"
    OMP_NUM_THREADS=8 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      "$BENCHMARK/generate.py" --config "$CONFIG" --checkpoint "$CHECKPOINT" --weights "$WEIGHTS" \
      --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" --group "$group" --run-dir "$RUN_DIR"
  fi
  "$PYTHON" "$BENCHMARK/protocol.py" validate-videos \
    --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" --group "$group" --videos "$RUN_DIR/videos"
  for dimension in "${group_dimensions[@]}"; do
    out="$RESULT_DIR/by_dimension/$dimension"
    mkdir -p "$out"
    result=$(matching_result "$dimension" "$group" "$out")
    if [[ -z "$result" ]]; then
      echo "EVALUATE:$SUITE:$dimension"
      (cd "$VBENCH_ROOT" && PYTHONPATH="$VBENCH_ROOT" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VBENCH_TORCH_THREADS=1 \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        "$BENCHMARK/evaluate.py" --vbench-root "$VBENCH_ROOT" \
        --videos_path "$RUN_DIR/videos" --output_path "$out" --dimension "$dimension" \
        --load_ckpt_from_local True --mode vbench_standard)
      result=$(matching_result "$dimension" "$group" "$out")
    fi
    [[ -n "$result" ]] || { echo "Missing/incomplete result: $dimension" >&2; exit 1; }
    RESULTS+=("$result")
  done
done <<< "$PLAN"
"$PYTHON" "$BENCHMARK/score.py" summarize --suite "$SUITE" --prompt-file "$PROMPT_FILE" --full-info "$FULL_INFO" \
  --videos "$RUN_DIR/videos" --output-dir "$RESULT_DIR" "${DIMENSION_ARGS[@]}" "${RESULTS[@]}"
touch "$RESULT_DIR/COMPLETE"
echo "VBENCH_COMPLETE:$SUITE:$RESULT_DIR/summary.json"
