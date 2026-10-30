#!/usr/bin/env python3
"""Launch the SANA-WM backend with chunk-level PlayWorld agent control."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RUNNER = Path(
    os.environ.get(
        "SANA_WM_RUNNER",
        str(HERE / "run_agent.py"),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description="PlayWorld adapter for agent-controlled SANA-WM")
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--images-dir")
    parser.add_argument("--output-dir", default=str(HERE.parents[1] / "outputs" / "sana_wm"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--action-source", choices=["dataset", "agent", "dataset_then_agent", "refined"], default="agent")
    parser.add_argument("--num-chunks", type=int, default=30)
    parser.add_argument("--chunk-frames", type=int, default=33)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args, extra = parser.parse_known_args()

    if not args.runner.is_file():
        parser.error(f"SANA-WM runner not found: {args.runner}; set SANA_WM_RUNNER")
    if args.action_source != "dataset":
        missing = [
            name
            for name in ("PLAYER_API_URL", "PLAYER_API_KEY")
            if not os.environ.get(name)
        ]
        if missing:
            parser.error(
                f"{', '.join(missing)} required for agent control; "
                "source Agent_player/api_keys.sh"
            )

    cmd = [
        sys.executable,
        str(args.runner),
        "--mapping-json", args.mapping_json,
        "--output-dir", args.output_dir,
        "--action-source", args.action_source,
        "--num-chunks", str(args.num_chunks),
        "--chunk-frames", str(args.chunk_frames),
        "--agent-base-url", os.environ.get("PLAYER_API_URL", ""),
        "--agent-model", os.environ.get("PLAYER_MODEL", "claude-haiku-4-5-20251001"),
        "--agent-api-key", os.environ.get("PLAYER_API_KEY", ""),
        "--agent-user-key", os.environ.get("PLAYER_USER_KEY") or os.environ.get("PLAYER_API_KEY", ""),
        "--agent-biz-scene", os.environ.get("PLAYER_BIZ_SCENE", "offline"),
    ]
    if args.images_dir:
        cmd += ["--images-dir", args.images_dir]
    if args.tasks:
        cmd += ["--tasks", *args.tasks]
    cmd += extra
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
