#!/usr/bin/env python3
"""Build the clean-latent LMDB consumed by LatentLMDBDataset.

Input shards come from ``encode_latents.py``.  The operation
is atomic at the directory level: it writes a sibling ``.building`` LMDB and
renames it only after all shapes and metadata have been committed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack UltraVideo latent shards into LMDB")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-latent-frames", type=int, default=None,
                        help="Keep this many leading frames from the causal VAE latents.")
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--commit-every", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Validate and skip an existing LMDB instead of overwriting it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import lmdb
    import numpy as np
    import torch

    files = sorted(args.latents_dir.resolve().glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt shards found under {args.latents_dir}")
    destination = args.output.resolve()
    building = destination.with_name(destination.name + ".building")
    if destination.exists():
        if not args.skip_existing:
            raise FileExistsError(f"Refusing to overwrite existing LMDB: {destination}")
        latent = torch.load(files[0], map_location="cpu", weights_only=False)["latent"]
        if args.num_latent_frames is not None:
            if not 0 < args.num_latent_frames <= latent.shape[0]:
                raise ValueError(f"Invalid latent prefix length: {args.num_latent_frames}")
            latent = latent[:args.num_latent_frames]
        expected_shape = " ".join(map(str, (len(files), *latent.shape))).encode("utf-8")
        with lmdb.open(str(destination), readonly=True, lock=False, readahead=False) as existing:
            with existing.begin() as txn:
                if (txn.get(b"latents_shape") != expected_shape
                        or txn.get(b"prompts_shape") != str(len(files)).encode("utf-8")
                        or txn.get(b"clip_ids_shape") != str(len(files)).encode("utf-8")
                        or txn.stat()["entries"] != 3 * len(files) + 4):
                    raise RuntimeError(f"Existing LMDB does not match the shards: {destination}")
        print(f"[skip] Existing LMDB has the expected count and shape: {destination}")
        return 0
    if building.exists():
        raise FileExistsError(
            f"Incomplete build already exists: {building}. Inspect/remove it before retrying."
        )
    building.mkdir(parents=True)
    env = lmdb.open(
        str(building),
        map_size=args.map_size_gb * 1024**3,
        subdir=True,
        map_async=True,
        writemap=False,
    )
    latent_shape = None
    txn = env.begin(write=True)
    try:
        for index, path in enumerate(files):
            item = torch.load(path, map_location="cpu", weights_only=False)
            latent = item["latent"]
            if args.num_latent_frames is not None:
                if not 0 < args.num_latent_frames <= latent.shape[0]:
                    raise ValueError(f"Invalid latent prefix length: {args.num_latent_frames}")
                latent = latent[:args.num_latent_frames]
            latent = latent.to(dtype=torch.float16).contiguous().numpy()
            prompt = str(item["prompt"])
            if latent_shape is None:
                latent_shape = tuple(latent.shape)
            elif tuple(latent.shape) != latent_shape:
                raise RuntimeError(
                    f"Mixed latent shapes: {path} has {tuple(latent.shape)}, "
                    f"expected {latent_shape}"
                )
            txn.put(f"latents_{index}_data".encode(), latent.tobytes())
            txn.put(f"prompts_{index}_data".encode(), prompt.encode("utf-8"))
            txn.put(f"clip_ids_{index}_data".encode(), str(item["clip_id"]).encode("utf-8"))
            if (index + 1) % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"[pack] {index + 1}/{len(files)}")
        assert latent_shape is not None
        txn.put(
            b"latents_shape",
            " ".join(map(str, (len(files), *latent_shape))).encode("utf-8"),
        )
        txn.put(b"prompts_shape", f"{len(files)}".encode("utf-8"))
        txn.put(b"clip_ids_shape", f"{len(files)}".encode("utf-8"))
        txn.put(
            b"dataset_metadata",
            json.dumps(
                {
                    "count": len(files),
                    "latent_shape": latent_shape,
                    "source": "APRIL-AIGC/UltraVideo-Long",
                    "format": "float16 Wan VAE latents",
                    "temporal_slice": [0, args.num_latent_frames],
                },
                sort_keys=True,
            ).encode("utf-8"),
        )
        txn.commit()
        txn = None
        env.sync()
        env.close()
        os.replace(building, destination)
    except Exception:
        if txn is not None:
            txn.abort()
        env.close()
        raise
    print(f"[COMPLETE] {len(files)} clips -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
