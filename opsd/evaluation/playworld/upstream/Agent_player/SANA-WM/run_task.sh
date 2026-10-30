#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

KEY_FILE="${AGENT_PLAYER_KEYS_FILE:-${PLAYER_API_KEYS_FILE:-${SCRIPT_DIR}/../api_keys.sh}}"
if [[ -f "$KEY_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$KEY_FILE"
fi

PYTHON_BIN="${SANA_WM_PYTHON_BIN:-python3}"

# Match the other Agent Player adapters: a positional, comma-separated task-ID
# list is accepted. Advanced callers can still pass player.py flags directly.
if [[ $# -gt 0 && "${1}" != -* ]]; then
  TASK_IDS_VALUE="$1"
  shift
  IFS=',' read -r -a TASK_IDS_ARRAY <<< "$TASK_IDS_VALUE"

  DATA_ROOT="${SANA_WM_DATA_ROOT:-${REPO_ROOT}/data}"
  MAPPING_JSON="${SANA_WM_MAPPING_JSON:-}"
  if [[ -z "$MAPPING_JSON" ]]; then
    CATEGORY="${TASK_IDS_ARRAY[0]%%[0-9]*}"
    CATEGORY_UPPER="$(printf '%s' "$CATEGORY" | tr '[:lower:]' '[:upper:]')"
    case "$CATEGORY_UPPER" in
      GC) DATA_SPLIT="gc" ;;
      IF) DATA_SPLIT="if" ;;
      OE) DATA_SPLIT="${SANA_WM_OE_SPLIT:-insight}" ;;
      *)
        echo "Cannot infer dataset split from task ID: ${TASK_IDS_ARRAY[0]}" >&2
        exit 2
        ;;
    esac
    MAPPING_JSON="${DATA_ROOT}/${DATA_SPLIT}/data.json"
  fi
  if [[ ! -f "$MAPPING_JSON" ]]; then
    echo "SANA-WM mapping JSON not found: $MAPPING_JSON" >&2
    echo "Set SANA_WM_DATA_ROOT/SANA_WM_OE_SPLIT, or pass --mapping-json explicitly." >&2
    exit 2
  fi

  exec "$PYTHON_BIN" "$SCRIPT_DIR/player.py" \
    --mapping-json "$MAPPING_JSON" \
    --images-dir "$DATA_ROOT" \
    --output-dir "${SANA_WM_OUT_ROOT:-${REPO_ROOT}/outputs/sana_wm}" \
    --tasks "${TASK_IDS_ARRAY[@]}" \
    "$@"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: ./run_task.sh GC001[,GC002]" >&2
  echo "Or: ./run_task.sh --mapping-json /path/to/GC.json --tasks GC001" >&2
  exit 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/player.py" "$@"
