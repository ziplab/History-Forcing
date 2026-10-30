#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TASK_ID="${1:-GC002}"
shift || true

# Reuse the repository-local ignored credential file. Override
# HYWORLD2_API_KEYS_FILE when credentials are stored elsewhere.
API_KEYS_FILE="${HYWORLD2_API_KEYS_FILE:-$HERE/../api_keys.sh}"
if [[ -f "$API_KEYS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$API_KEYS_FILE"
fi

exec python3 "$HERE/player.py" "$TASK_ID" "$@"
