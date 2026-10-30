#!/usr/bin/env python3
"""Reproducibly download a 5K subset of UltraVideo-Long.

The official dataset stores the 960p videos in eight large ZIP archives.  This
script selects clip IDs deterministically from ``long.csv``, downloads one ZIP
at a time with HTTP Range resume, extracts only selected clips, validates their
ZIP CRC/size, then removes the ZIP by default to keep peak disk usage bounded.

Raw videos remain sourced from APRIL-AIGC/UltraVideo-Long.  The generated
manifest and this downloader can be shared without republishing the videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


REPO_ID = "APRIL-AIGC/UltraVideo-Long"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_BASE = f"{HF_ENDPOINT}/datasets/{REPO_ID}"
DEFAULT_FILES = ("README.md", "license-april-lab.txt", "long.csv")
CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_EVERY = 512 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a deterministic, resumable UltraVideo-Long subset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/UltraVideo-Long-5K"),
        help="Dataset output directory (default: dataset/UltraVideo-Long-5K).",
    )
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-ids",
        type=Path,
        help=(
            "Optional locked selection_ids.csv from the code repository. "
            "When supplied, no new random sample is drawn."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=("960", "1920"),
        default="960",
        help="Official archive variant. Use 960 for training unless 1920 is required.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.6,
        help=(
            "Only sample clips at least this many seconds long. The 15.6s default "
            "supports a full 249-frame/16fps crop. Set 5.1 for 81-frame training."
        ),
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=100,
        help="Number of selected clips reserved for validation.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download metadata and create manifests, but do not download video ZIPs.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIPs. Default deletes each ZIP after verified extraction.",
    )
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--range-download", action="store_true",
                        help="Fetch only selected ZIP members using HTTP Range (requires remotezip).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent archive readers for --range-download (default: 8).")
    return parser.parse_args()


def auth_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "huggingface_hub/0.35.3"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_url(path: str) -> str:
    return f"{HF_BASE}/resolve/main/{path}?download=true"


def api_tree(path: str, timeout: int) -> List[dict]:
    quoted = urllib.parse.quote(path, safe="/")
    suffix = f"/{quoted}" if quoted else ""
    url = f"{HF_ENDPOINT}/api/datasets/{REPO_ID}/tree/main{suffix}?limit=1000"
    request = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Hugging Face API response for {path!r}")
    return payload


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def download_with_resume(
    repo_path: str,
    destination: Path,
    *,
    expected_size: int | None,
    retries: int,
    timeout: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if expected_size is None or destination.stat().st_size == expected_size:
            print(f"[skip] {destination} ({human_bytes(destination.stat().st_size)})")
            return
        raise RuntimeError(
            f"Existing file has wrong size: {destination} "
            f"({destination.stat().st_size} != {expected_size})"
        )

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if expected_size is not None and offset == expected_size:
            os.replace(part, destination)
            return
        if expected_size is not None and offset > expected_size:
            raise RuntimeError(f"Partial file is larger than expected: {part}")

        headers = {**auth_headers(), "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(resolve_url(repo_path), headers=headers)
        mode = "ab" if offset else "wb"
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                if offset and status != 206:
                    print(f"[resume] server ignored Range for {repo_path}; restarting file")
                    offset = 0
                    mode = "wb"
                elif offset and not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        f"Resume range mismatch for {repo_path}: {content_range!r}"
                    )

                print(
                    f"[download] {repo_path}: resume at {human_bytes(offset)}"
                    + (f" / {human_bytes(expected_size)}" if expected_size else "")
                )
                written = offset
                next_report = ((written // PROGRESS_EVERY) + 1) * PROGRESS_EVERY
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            print(
                                f"  {destination.name}: {human_bytes(written)}"
                                + (
                                    f" / {human_bytes(expected_size)} "
                                    f"({100.0 * written / expected_size:.1f}%)"
                                    if expected_size
                                    else ""
                                )
                            )
                            next_report += PROGRESS_EVERY
                    handle.flush()
                    os.fsync(handle.fileno())

            final_size = part.stat().st_size
            if expected_size is not None and final_size != expected_size:
                raise IOError(
                    f"Incomplete download for {repo_path}: {final_size} != {expected_size}"
                )
            os.replace(part, destination)
            print(f"[done] {destination} ({human_bytes(final_size)})")
            return
        except (OSError, TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == retries:
                raise
            delay = min(60, 2 ** min(attempt, 6))
            print(f"[retry {attempt}/{retries}] {repo_path}: {exc}; waiting {delay}s")
            time.sleep(delay)




def load_rows(metadata_csv: Path) -> tuple[List[dict], List[str]]:
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "clip_id" not in fieldnames or "duration" not in fieldnames:
        raise RuntimeError(
            f"Expected clip_id and duration columns in {metadata_csv}; got {fieldnames}"
        )
    return rows, fieldnames


def caption_for(row: Mapping[str, str]) -> str:
    for key in ("Summarized Description", "Detailed Description", "Brief Description"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""




def build_selection(
    rows: Sequence[dict], *, count: int, seed: int, min_duration: float, val_count: int
) -> List[dict]:
    by_id: MutableMapping[str, dict] = {}
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if not clip_id:
            continue
        try:
            duration = float(row.get("duration") or 0)
        except ValueError:
            continue
        if duration >= min_duration:
            by_id.setdefault(clip_id, row)
    if len(by_id) < count:
        raise RuntimeError(
            f"Only {len(by_id)} clips satisfy duration >= {min_duration}s; "
            f"cannot select {count}. Lower --min-duration."
        )
    if not 0 <= val_count < count:
        raise ValueError(f"--val-count must be in [0, count), got {val_count}")

    population = sorted(by_id)
    chosen = random.Random(seed).sample(population, count)
    val_ids = set(random.Random(seed + 1).sample(chosen, val_count))
    selected: List[dict] = []
    for sample_index, clip_id in enumerate(chosen):
        row = dict(by_id[clip_id])
        row["sample_index"] = str(sample_index)
        row["split"] = "validation" if clip_id in val_ids else "train"
        row["local_path"] = f"videos/{clip_id}"
        row["text"] = caption_for(row)
        selected.append(row)
    return selected


def load_locked_selection(
    rows: Sequence[dict],
    selection_path: Path,
    *,
    count: int,
    min_duration: float,
) -> List[dict]:
    by_id = {(row.get("clip_id") or "").strip(): row for row in rows}
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        locked = list(csv.DictReader(handle))
    if len(locked) != count:
        raise RuntimeError(
            f"Locked manifest has {len(locked)} rows but --count is {count}"
        )
    selected: List[dict] = []
    seen: set[str] = set()
    for expected_index, item in enumerate(locked):
        clip_id = (item.get("clip_id") or "").strip()
        if not clip_id or clip_id in seen:
            raise RuntimeError(f"Invalid/duplicate locked clip_id: {clip_id!r}")
        seen.add(clip_id)
        if clip_id not in by_id:
            raise RuntimeError(f"Locked clip is absent from official long.csv: {clip_id}")
        row = dict(by_id[clip_id])
        if float(row.get("duration") or 0) < min_duration:
            raise RuntimeError(f"Locked clip is shorter than requested: {clip_id}")
        sample_index = int(item.get("sample_index") or -1)
        if sample_index != expected_index:
            raise RuntimeError(
                f"Locked sample_index mismatch at row {expected_index}: {sample_index}"
            )
        split = item.get("split")
        if split not in {"train", "validation"}:
            raise RuntimeError(f"Invalid split for {clip_id}: {split!r}")
        row["sample_index"] = str(sample_index)
        row["split"] = split
        row["local_path"] = f"videos/{clip_id}"
        row["text"] = caption_for(row)
        selected.append(row)
    return selected


def write_manifests(
    selected: Sequence[dict], original_fields: Sequence[str], manifest_dir: Path
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    extra = ["sample_index", "split", "local_path", "text"]
    fields = extra + [field for field in original_fields if field not in extra]

    def write_csv(path: Path, subset: Sequence[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
        os.replace(tmp, path)

    write_csv(manifest_dir / "ultravideo_long_5k.csv", selected)
    write_csv(
        manifest_dir / "train.csv", [row for row in selected if row["split"] == "train"]
    )
    write_csv(
        manifest_dir / "validation.csv",
        [row for row in selected if row["split"] == "validation"],
    )

    # This compact file is the only manifest intended for a public code repo.
    # Full descriptions/captions are regenerated from the official metadata on
    # the training server and must not be redistributed independently.
    public_path = manifest_dir / "selection_ids.csv"
    public_tmp = public_path.with_suffix(public_path.suffix + ".tmp")
    with public_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_index", "split", "clip_id"]
        )
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    os.replace(public_tmp, public_path)


def ensure_selection_state(
    state_path: Path,
    *,
    count: int,
    seed: int,
    resolution: str,
    min_duration: float,
    selected_ids: list[str],
) -> None:
    desired = {
        "count": count,
        "seed": seed,
        "resolution": resolution,
        "min_duration": min_duration,
        "selected_ids": sorted(selected_ids),
        "source": REPO_ID,
    }
    if state_path.exists():
        current = json.loads(state_path.read_text(encoding="utf-8"))
        if any(current.get(key) != value for key, value in desired.items() if key in current):
            raise RuntimeError(
                f"Selection settings do not match existing {state_path}. "
                "Use the original arguments or a new --output-dir."
            )
    atomic_json(state_path, desired)


def disk_preflight(root: Path, archive_size: int) -> None:
    usage = shutil.disk_usage(root)
    reserve = 8 * 1024**3
    if usage.free < archive_size + reserve:
        raise RuntimeError(
            f"Not enough free space under {root}: {human_bytes(usage.free)} free, "
            f"need at least {human_bytes(archive_size + reserve)}."
        )


def marker_valid(
    marker: Path, *, selected_ids: list[str], archive: Mapping[str, object], videos_dir: Path
) -> bool:
    if not marker.exists():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("selected_ids", selected_ids) != selected_ids:
            return False
        for clip_id, expected_size in state.get("extracted", {}).items():
            path = videos_dir / clip_id
            if not path.is_file() or path.stat().st_size != expected_size:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def extract_selected(
    archive_path: Path | zipfile.ZipFile,
    videos_dir: Path,
    selected_ids: set[str],
) -> Dict[str, int]:
    videos_dir.mkdir(parents=True, exist_ok=True)
    extracted: Dict[str, int] = {}
    context = nullcontext(archive_path) if isinstance(archive_path, zipfile.ZipFile) else zipfile.ZipFile(archive_path)
    with context as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            clip_id = Path(info.filename).name
            if clip_id not in selected_ids:
                continue
            destination = videos_dir / clip_id
            if destination.exists() and destination.stat().st_size == info.file_size:
                extracted[clip_id] = info.file_size
                continue
            part = destination.with_suffix(destination.suffix + ".part")
            part.unlink(missing_ok=True)
            print(f"  [extract] {clip_id} ({human_bytes(info.file_size)})")
            try:
                with archive.open(info, "r") as source, part.open("wb") as target:
                    shutil.copyfileobj(source, target, length=CHUNK_SIZE)
                    target.flush()
                    os.fsync(target.fileno())
                if part.stat().st_size != info.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {clip_id}: "
                        f"{part.stat().st_size} != {info.file_size}"
                    )
                os.replace(part, destination)
                extracted[clip_id] = info.file_size
            except Exception:
                part.unlink(missing_ok=True)
                raise
    return extracted


def download_selected_ranges(archives, videos_dir, state_dir, selected_ids, args):
    """Read every archive index, retaining the identical locked subset and splits."""
    import requests
    from remotezip import RemoteZip

    if args.workers < 1:
        raise ValueError("--workers must be positive")

    def extract(entry):
        repo_path = str(entry["path"])
        marker = state_dir / f"{Path(repo_path).name}.done.json"
        if marker_valid(marker, selected_ids=sorted(selected_ids), archive=entry, videos_dir=videos_dir):
            print(f"[range skip] {repo_path}", flush=True)
            return
        for attempt in range(1, args.retries + 1):
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    with RemoteZip(resolve_url(repo_path), session=session, headers=auth_headers(),
                                   timeout=args.timeout, initial_buffer_size=1024**2) as archive:
                        extracted = extract_selected(archive, videos_dir, selected_ids)
                atomic_json(marker, {"archive": repo_path,
                    "archive_size": int(entry["size"]), "selected_ids": sorted(selected_ids),
                    "extracted": extracted, "download_mode": "http_range"})
                print(f"[range complete] {repo_path}: {len(extracted)} clips", flush=True)
                return
            except Exception as exc:
                if attempt == args.retries:
                    raise
                print(f"[range retry {attempt}/{args.retries}] {repo_path}: {type(exc).__name__}", flush=True)
                time.sleep(min(60, 2 ** min(attempt, 6)))

    with ThreadPoolExecutor(max_workers=min(args.workers, len(archives))) as pool:
        list(pool.map(extract, archives))


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    root = args.output_dir.expanduser().resolve()
    metadata_dir = root / "metadata"
    manifest_dir = root / "manifests"
    archives_dir = root / "archives"
    videos_dir = root / "videos"
    state_dir = root / ".state"
    for directory in (metadata_dir, manifest_dir, archives_dir, videos_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)

    top_tree = {entry["path"]: entry for entry in api_tree("", args.timeout)}
    for name in DEFAULT_FILES:
        entry = top_tree.get(name)
        expected = int(entry["size"]) if entry else None
        download_with_resume(
            name,
            metadata_dir / name,
            expected_size=expected,
            retries=args.retries,
            timeout=args.timeout,
        )

    rows, original_fields = load_rows(metadata_dir / "long.csv")
    if args.selection_ids:
        selected = load_locked_selection(
            rows,
            args.selection_ids.expanduser().resolve(),
            count=args.count,
            min_duration=args.min_duration,
        )
        print(f"[locked selection] {args.selection_ids}")
    else:
        selected = build_selection(
            rows,
            count=args.count,
            seed=args.seed,
            min_duration=args.min_duration,
            val_count=args.val_count,
        )
    selected_ids = {row["clip_id"] for row in selected}
    ensure_selection_state(
        state_dir / "selection.json",
        count=args.count,
        seed=args.seed,
        resolution=args.resolution,
        min_duration=args.min_duration,
        selected_ids=sorted(selected_ids),
    )
    write_manifests(selected, original_fields, manifest_dir)
    print(
        f"[selection] {len(selected_ids)} clips, seed={args.seed}, "
        f"duration>={args.min_duration}s"
    )
    print(f"[manifest] {manifest_dir / 'ultravideo_long_5k.csv'}")
    print(f"[public manifest] {manifest_dir / 'selection_ids.csv'}")

    if args.metadata_only:
        print("[metadata-only] Video download intentionally skipped.")
        return 0

    archive_dir_name = f"clips_long_{args.resolution}"
    archives = [
        entry
        for entry in api_tree(archive_dir_name, args.timeout)
        if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".zip")
    ]
    archives.sort(key=lambda entry: natural_key(str(entry["path"])))
    if not archives:
        raise RuntimeError(f"No ZIP archives found in {archive_dir_name}")
    total_network = sum(int(entry["size"]) for entry in archives)
    if args.range_download:
        print(f"[plan] read {len(archives)} ZIP indexes and Range-download only the locked clips", flush=True)
        download_selected_ranges(archives, videos_dir, state_dir, selected_ids, args)
    else:
        print(
            f"[plan] scan {len(archives)} official ZIPs ({human_bytes(total_network)} network); "
            "only selected MP4s will be retained"
        )

    for index, entry in enumerate([] if args.range_download else archives, start=1):
        repo_path = str(entry["path"])
        archive_name = Path(repo_path).name
        archive_path = archives_dir / archive_name
        marker = state_dir / f"{archive_name}.done.json"
        if marker_valid(
            marker, selected_ids=sorted(selected_ids), archive=entry, videos_dir=videos_dir
        ):
            print(f"[archive {index}/{len(archives)} skip] {archive_name}")
            continue

        expected_size = int(entry["size"])
        disk_preflight(root, expected_size)
        download_with_resume(
            repo_path,
            archive_path,
            expected_size=expected_size,
            retries=args.retries,
            timeout=args.timeout,
        )
        print(f"[archive {index}/{len(archives)}] extracting selected clips")
        extracted = extract_selected(archive_path, videos_dir, selected_ids)
        atomic_json(
            marker,
            {
                "archive": repo_path,

                "archive_size": expected_size,
                "selected_ids": sorted(selected_ids),
                "extracted": extracted,
            },
        )
        if not args.keep_archives:
            archive_path.unlink()
            print(f"[space] removed scanned ZIP {archive_name}")

    missing = sorted(
        clip_id
        for clip_id in selected_ids
        if not (videos_dir / clip_id).is_file() or (videos_dir / clip_id).stat().st_size == 0
    )
    if missing:
        missing_path = state_dir / "missing_clip_ids.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Only {len(selected_ids) - len(missing)}/{len(selected_ids)} clips found. "
            f"Missing IDs written to {missing_path}"
        )

    atomic_json(
        root / "COMPLETE.json",
        {
            "source": REPO_ID,
            "count": len(selected_ids),
            "resolution": args.resolution,
            "seed": args.seed,
            "min_duration": args.min_duration,
            "selected_ids": sorted(selected_ids),
            "videos_dir": "videos",
            "manifest": "manifests/ultravideo_long_5k.csv",
        },
    )
    print(f"[COMPLETE] verified {len(selected_ids)} videos under {videos_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
