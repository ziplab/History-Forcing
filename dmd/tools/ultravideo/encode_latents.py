#!/usr/bin/env python3
"""Convert an UltraVideo-Long manifest into resumable Wan VAE latent shards.

Run with ``torchrun`` to split clips deterministically across GPUs.  Every
completed clip is an independent atomic ``.pt`` shard, so an interrupted run
can be restarted without touching finished work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode, crop and VAE-encode UltraVideo-Long clips."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Pixel frames per clip: use 81 for 21 latents or 249 for 63 latents.",
    )
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefetch-clips", type=int, choices=(0, 1), default=1,
        help="Decode the next clip on a CPU thread during GPU encoding (default: 1).",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "all"), default="train"
    )
    parser.add_argument(
        "--caption-field",
        default="text",
        help="Manifest caption column (default: text).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N clips (0 means all)."
    )
    parser.add_argument("--available-only", action="store_true",
                        help="Encode completed downloads now; keep original row indices for later resume.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the plan without importing GPU models.",
    )
    return parser.parse_args()


def stable_fraction(seed: int, clip_id: str) -> float:
    payload = f"{seed}:{clip_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64 - 1)


def load_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "local_path"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"Manifest lacks required columns: {sorted(missing)}")
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    return rows


def resolve_video(videos_dir: Path, row: dict[str, str]) -> Path:
    candidates = [
        videos_dir / Path(row["local_path"]).name,
        videos_dir / row["clip_id"],
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found for {row['clip_id']}: {candidates}")


def output_name(index: int, clip_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in clip_id)
    return f"{index:06d}_{safe}.pt"


def probe_duration_seconds(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / 1_000_000.0
    if stream.frames and stream.average_rate:
        return float(stream.frames / stream.average_rate)
    raise RuntimeError("Video duration is unavailable")


def resize_center_crop(rgb, target_height: int, target_width: int):
    import cv2

    source_height, source_width = rgb.shape[:2]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    return resized[top : top + target_height, left : left + target_width]


def decode_window(
    path: Path,
    *,
    clip_id: str,
    seed: int,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
):
    import av
    import numpy as np
    import torch

    window_seconds = (num_frames - 1) / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = probe_duration_seconds(container, stream)
        if duration + 1e-3 < window_seconds:
            raise RuntimeError(
                f"{path.name} is too short: {duration:.3f}s < {window_seconds:.3f}s"
            )
        slack = max(0.0, duration - window_seconds - 1.0 / fps)
        start = stable_fraction(seed, clip_id) * slack
        target_times = [start + index / fps for index in range(num_frames)]
        if start > 0 and stream.time_base is not None:
            container.seek(int(start / stream.time_base), stream=stream,
                           backward=True, any_frame=False)
        selected = []
        next_index = 0
        last_rgb = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            last_rgb = rgb
            while next_index < num_frames and frame.time + 1e-6 >= target_times[next_index]:
                selected.append(resize_center_crop(rgb, height, width))
                next_index += 1
            if next_index == num_frames:
                break
        if last_rgb is not None:
            while len(selected) < num_frames:
                selected.append(resize_center_crop(last_rgb, height, width))
        if len(selected) != num_frames:
            raise RuntimeError(f"Decoded only {len(selected)}/{num_frames} frames")

    frames = np.stack(selected, axis=0)
    pixel = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    return pixel.div_(127.5).sub_(1.0), start, duration


def iter_rank_rows(rows: list[dict[str, str]], rank: int, world_size: int) -> Iterable:
    for index, row in enumerate(rows):
        if index % world_size == rank:
            yield index, row


def atomic_torch_save(torch, payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def iter_decoded_clips(pending, args):
    """Keep one decoded CPU clip ahead without changing preprocessing or order."""
    import cv2

    # Eight ranks each own a decoder; avoid eight full-machine OpenCV pools.
    cv2.setNumThreads(1)

    def decode(entry):
        _, row, _ = entry
        return decode_window(
            resolve_video(args.videos_dir.resolve(), row),
            clip_id=row["clip_id"], seed=args.seed, num_frames=args.num_frames,
            fps=args.fps, height=args.height, width=args.width,
        )

    if not args.prefetch_clips:
        for entry in pending:
            yield (*entry, *decode(entry))
        return
    entries = iter(pending)
    entry = next(entries, None)
    if entry is None:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(decode, entry)
        while entry is not None:
            current = entry
            decoded = future.result()
            entry = next(entries, None)
            future = executor.submit(decode, entry) if entry is not None else None
            yield (*current, *decoded)


def main() -> int:
    args = parse_args()
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must have the form 4k+1 (for example 81 or 249)")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16")

    rows = load_manifest(args.manifest.resolve(), args.split)
    if args.limit:
        rows = rows[: args.limit]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assigned = list(iter_rank_rows(rows, rank, world_size))
    if args.available_only:
        ready = []
        for index, row in assigned:
            try:
                resolve_video(args.videos_dir.resolve(), row)
            except FileNotFoundError:
                continue
            ready.append((index, row))
        assigned = ready
    print(
        f"[rank {rank}/{world_size}] {len(assigned)} of {len(rows)} clips; "
        f"{args.num_frames} frames at {args.fps:g}fps -> "
        f"{(args.num_frames - 1) // 4 + 1} Wan latents"
    )
    for index, row in assigned:
        resolve_video(args.videos_dir.resolve(), row)
        if not (row.get(args.caption_field) or "").strip():
            raise RuntimeError(f"Empty caption for {row['clip_id']}")
    if args.dry_run:
        print("[dry-run] Manifest, captions and assigned video paths are valid.")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Wan VAE encoding requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    from utils.wan_wrapper import WanVAEWrapper

    device = torch.device("cuda", local_rank)
    vae = WanVAEWrapper().to(device)
    pending = []
    for index, row in assigned:
        destination = args.output_dir / output_name(index, row["clip_id"])
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        pending.append((index, row, destination))
    completed = len(assigned) - len(pending)
    for index, row, destination, pixel, start, duration in iter_decoded_clips(pending, args):
        clip_id = row["clip_id"]
        video_path = resolve_video(args.videos_dir.resolve(), row)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = vae.encode_to_latent(pixel.unsqueeze(0).to(device))[0]
        latent = latent.to(device="cpu", dtype=torch.float16).contiguous()
        expected_latents = (args.num_frames - 1) // 4 + 1
        if tuple(latent.shape) != (expected_latents, 16, args.height // 8, args.width // 8):
            raise RuntimeError(f"Unexpected latent shape for {clip_id}: {tuple(latent.shape)}")
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"Non-finite latent values for {clip_id}")
        atomic_torch_save(
            torch,
            {
                "clip_id": clip_id,
                "prompt": row[args.caption_field].strip(),
                "latent": latent,
                "source_video": video_path.name,
                "preprocess": {
                    "seed": args.seed,
                    "start_seconds": start,
                    "source_duration_seconds": duration,
                    "pixel_frames": args.num_frames,
                    "fps": args.fps,
                    "height": args.height,
                    "width": args.width,
                },
            },
            destination,
        )
        completed += 1
        print(f"[rank {rank}] {completed}/{len(assigned)} {clip_id} -> {destination.name}")
    print(f"[rank {rank}] COMPLETE ({completed} shards)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
