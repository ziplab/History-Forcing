#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT="${ROOT:-$REPO_ROOT/dmd}"
cd "$REPO_ROOT"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
MANIFEST_IDS="$ROOT/tools/ultravideo/selection_ids.csv"
ACTION="${1:-all}"
DATA_ROOT="${DATA_ROOT:-$ROOT/../dataset/UltraVideo-Long-5K}"
FULL_MANIFEST="$DATA_ROOT/manifests/train.csv"

download_args=(
  --output-dir "$DATA_ROOT"
  --selection-ids "$MANIFEST_IDS"
  --count 5000
  --seed 42
  --resolution 960
  --min-duration 15.6
  --val-count 100
)

case "$ACTION" in
  all)
    for step in download-range check encode encode-validation; do
      echo "[prepare] $step"
      bash "$SCRIPT" "$step"
    done
    for step in build-10s build-validation; do
      echo "[prepare] $step"
      bash "$SCRIPT" "$step" --skip-existing
    done
    echo "[COMPLETE] 42-latent training and validation LMDBs: $DATA_ROOT"
    ;;
  metadata)
    "$PYTHON" "$ROOT/tools/ultravideo/download.py" \
      "${download_args[@]}" --metadata-only
    ;;
  download)
    "$PYTHON" "$ROOT/tools/ultravideo/download.py" \
      "${download_args[@]}"
    ;;
  download-range)
    "$PYTHON" "$ROOT/tools/ultravideo/download.py" \
      "${download_args[@]}" --range-download --workers "${DOWNLOAD_WORKERS:-8}"
    ;;
  check)
    "$PYTHON" "$ROOT/tools/ultravideo/encode_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_63" \
      --num-frames 249 --limit "${LIMIT:-8}" --dry-run
    ;;
  encode)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
      "$ROOT/tools/ultravideo/encode_latents.py" \
      --manifest "$FULL_MANIFEST" \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_63" \
      --num-frames 249 --fps 16 --height 480 --width 832 --seed 42
    ;;
  encode-validation)
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
      "$ROOT/tools/ultravideo/encode_latents.py" \
      --manifest "$DATA_ROOT/manifests/validation.csv" --split validation \
      --videos-dir "$DATA_ROOT/videos" \
      --output-dir "$DATA_ROOT/latents_63_validation" \
      --num-frames 249 --fps 16 --height 480 --width 832 --seed 42
    ;;
  build-10s)
    "$PYTHON" "$ROOT/tools/ultravideo/build_lmdb.py" \
      --latents-dir "$DATA_ROOT/latents_63" \
      --num-latent-frames 42 --output "$DATA_ROOT/clean_data_42" "${@:2}"
    ;;
  build-validation)
    "$PYTHON" "$ROOT/tools/ultravideo/build_lmdb.py" \
      --latents-dir "$DATA_ROOT/latents_63_validation" \
      --num-latent-frames 42 --output "$DATA_ROOT/clean_data_42_validation" "${@:2}"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash scripts/long_video/prepare_ultravideo.sh [ACTION]

Actions (each is safe to rerun unless noted):
  all          Run download-range, check, encode and all builds (default)
  metadata     Download official metadata and regenerate the locked manifests
  download     Resume official ZIP downloads and extract the locked 5K clips
  download-range  Fetch the same locked clips directly from ZIP byte ranges
  check        Validate the first LIMIT=8 videos without loading a GPU model
  encode       Encode 249 pixel frames -> 63 Wan latent frames on 8 GPUs
  encode-validation  Encode the 100 held-out clips as 63-frame Wan latents
  build-10s    Build 42-latent prefixes from the existing 63-latent shards
  build-validation  Build the separate 100-clip, 42-latent validation LMDB

The fixed 5K selection ships in tools/ultravideo/selection_ids.csv (4900 train/100 validation).
Downloads and latent encoding resume; all validates and skips existing LMDBs.
Individual builds refuse existing outputs unless --skip-existing is passed.
Optional environment variables: ROOT, PYTHON, DATA_ROOT, NPROC_PER_NODE, LIMIT, DOWNLOAD_WORKERS.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
