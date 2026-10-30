#!/usr/bin/env python3
"""Convert PlayWorld image tasks into a selected backend's packed input format.

The same packed image/condition rows feed the generic closed-loop Player.  The
backend capability decides whether rows carry discrete actions, camera metadata,
or no native control.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]


from evaluation.playworld.adapter import _write, load_tasks, select_tasks


ACTION_HINTS = {
    "w", "a", "s", "d", "i", "j", "k", "l",
    "up", "left", "down", "right", "wait", "wait and observe",
}


def parse_action_timing(
    value: str, *, latent_fps: float = 3.0, action_unit_ms: int = 450,
) -> tuple[int, list[str]]:
    """Return the hinted horizon in latent frames, independent of native controls."""
    unsupported: list[str] = []
    elapsed_latent_frames = 0.0
    emitted = 0
    for raw_part in value.split("->"):
        part = raw_part.strip()
        match = re.fullmatch(r"(.+?)\*(\d+)", part)
        if match:
            name, count = match.group(1).strip().lower(), int(match.group(2))
            duration_ms = count * action_unit_ms
        else:
            seconds = re.fullmatch(r"wait\*(\d+(?:\.\d+)?)s", part.lower())
            if seconds:
                name = "wait"
                duration_ms = float(seconds.group(1)) * 1000
            else:
                name, duration_ms = part.lower(), action_unit_ms
        if name not in ACTION_HINTS:
            unsupported.append(part)
        elapsed_latent_frames += duration_ms * latent_fps / 1000
        stop = round(elapsed_latent_frames)
        emitted += max(1, stop - emitted)
    return emitted, unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="backbones.minwm:MinWM")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-ids", default=None,
        help="Optional comma-separated exact task IDs; only these rows are packed",
    )
    parser.add_argument(
        "--target-latent-frames", type=int,
        help=(
            "Optional fixed generated horizon. By default each task is rounded "
            "up to the selected backend's native interactive block."
        ),
    )
    parser.add_argument(
        "--latent-fps", type=float, default=None,
        help="Latent control rate; defaults to one native block per second",
    )
    args = parser.parse_args()
    from backbones import load_backend

    backend = load_backend(args.model)
    native_options = backend.interactive_input_options()
    default_intrinsics = native_options.pop('intrinsics',
        ((1., 0., .5), (0., 1., .5), (0., 0., 1.)))
    spec = backend.interactive_spec()
    if spec is None:
        parser.error("Selected backend does not implement interactive inference")
    block = spec.block_frames
    latent_fps = float(block if args.latent_fps is None else args.latent_fps)
    condition_in_block = spec.condition_consumes_block_slot
    if latent_fps <= 0:
        parser.error("--latent-fps must be positive")
    if args.target_latent_frames is not None:
        target = args.target_latent_frames
        aligned = (target + int(condition_in_block)) % block == 0
        if target <= 0 or not aligned:
            rule = (
                f"a multiple of {block}" if not condition_in_block
                else f"one less than a multiple of {block}"
            )
            parser.error(f"--target-latent-frames must be positive and {rule}")

    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    actions_dir = output / "actions"
    cameras_dir = output / "cameras"
    discrete_actions = spec.control_mode == "discrete"
    if discrete_actions:
        actions_dir.mkdir(parents=True, exist_ok=True)
    if spec.control_mode == "camera":
        cameras_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(data_root)
    source_indices = {task["task_id"]: index for index, task in enumerate(tasks)}
    selected = select_tasks(tasks, args.task_ids)
    manifest = []
    metadata = []
    for packed_index, task in enumerate(selected):
        action_frames, unsupported = parse_action_timing(
            str(task.get("action") or "wait*1"), latent_fps=latent_fps,
        )
        aligned_frames = action_frames + int(condition_in_block)
        episode_steps = max(1, (aligned_frames + block - 1) // block)
        native_target = episode_steps * block - int(condition_in_block)
        target_latent_frames = args.target_latent_frames or native_target
        episode_steps = (
            target_latent_frames + int(condition_in_block)
        ) // block
        if action_frames > target_latent_frames:
            raise ValueError(
                f"{task['task_id']} has {action_frames} action frames, exceeding "
                f"the {target_latent_frames}-frame horizon"
            )
        controls = {}
        if discrete_actions:
            full_actions = np.zeros(
                (target_latent_frames + 1, spec.control_features),
                dtype=np.int8,
            ).tolist()
            action_path = actions_dir / f"{task['task_id']}.json"
            action_path.write_text(
                json.dumps(full_actions, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            controls["actions"] = str(action_path)
        if spec.control_mode == "camera":
            camera_path = cameras_dir / f"{task['task_id']}.npz"
            np.savez_compressed(
                camera_path,
                viewmats=np.eye(4, dtype=np.float64)[None],
                intrinsics=np.asarray([default_intrinsics], dtype=np.float64),
                rgb_frame_indices=np.asarray([0], dtype=np.int32),
            )
            controls["camera"] = str(camera_path)
        manifest.append({
            **native_options,
            "prompt": backend.interactive_prompt(task["prompt"]),
            "image": str(task["_image"]),
            **controls,
            "num_latent_frames": target_latent_frames + 1,
            "condition_frames": 1,
        })
        metadata.append({
            "row_index": packed_index,
            "source_row_index": source_indices[task["task_id"]],
            "task_id": task["task_id"],
            "split": task["_split"],
            "source_category": task.get("source_category") or task.get("category"),
            "source_task_id": task.get("source_task_id") or task["task_id"],
            "action": task.get("action"),
            "hint_action_latent_frames": action_frames,
            "target_latent_frames": target_latent_frames,
            "episode_steps": episode_steps,
            "horizon_padding_latent_frames": target_latent_frames - action_frames,
            "unsupported_action_hints": unsupported,
        })
    _write(output / "manifest.json", manifest)
    _write(output / "task_index.json", metadata)
    _write(output / "protocol.json", {
        "protocol": "interactive_player_inputs",
        "official_protocol": "official_agent_player",
        "action_timing": (
            "public counts are 450 ms; durations are cumulatively rounded "
            f"to the {latent_fps:g} Hz backend latent timeline"
        ),
        "model": args.model,
        "control_type": (
            spec.control_mode
        ),
        "native_block_frames": spec.block_frames,
        "condition_consumes_block_slot": spec.condition_consumes_block_slot,
        "control_features": spec.control_features,
        "target_latent_frames": (
            args.target_latent_frames or "per_task_native_block_rounding"
        ),
        "task_count": len(manifest),
        "task_ids": [row["task_id"] for row in metadata],
    })
    print(f"Prepared {len(manifest)} {type(backend).__name__} inputs in {output}")


if __name__ == "__main__":
    main()
