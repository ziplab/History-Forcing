#!/usr/bin/env bash
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
data_root="${HYWORLD2_DATA_ROOT:-$repo_root/data}"
result_root="${HYWORLD2_VIDEO_OUTPUT_ROOT:-$repo_root/outputs/hyworld2}"
log_root="${HYWORLD2_BATCH_LOG_ROOT:-$result_root/_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$log_root"

data_files=(
  "$data_root/gc/data.json"
  "$data_root/if/data.json"
  "$data_root/insight/data.json"
  "$data_root/outsight/data.json"
)

failures=0
skipped=0
for json_path in "${data_files[@]}"; do
  if [[ ! -f "$json_path" ]]; then
    printf 'missing data file: %s\n' "$json_path" >&2
    failures=$((failures + 1))
    continue
  fi
  while IFS= read -r task_id; do
    [[ -n "$task_id" ]] || continue
    if [[ -s "$result_root/$task_id/result.json" ]]; then
      skipped=$((skipped + 1))
      printf '[%s] SKIP  %s existing_result\n' "$(date '+%F %T')" "$task_id"
      continue
    fi
    printf '[%s] START %s\n' "$(date '+%F %T')" "$task_id"
    if HYWORLD2_DATA_ROOT="$data_root" HYWORLD2_VIDEO_OUTPUT_ROOT="$result_root" \
      "$here/run_task.sh" "$task_id" --video-source both \
      </dev/null >"$log_root/$task_id.log" 2>&1; then
      printf '[%s] DONE  %s\n' "$(date '+%F %T')" "$task_id"
    else
      status=$?
      failures=$((failures + 1))
      printf '[%s] FAIL  %s exit=%s log=%s\n' \
        "$(date '+%F %T')" "$task_id" "$status" "$log_root/$task_id.log" >&2
    fi
  done < <(python3 -c 'import json,sys; print("\n".join(str(x["task_id"]) for x in json.load(open(sys.argv[1]))))' "$json_path")
done

printf 'batch complete: failures=%s skipped=%s logs=%s\n' "$failures" "$skipped" "$log_root"
exit "$((failures > 0))"
