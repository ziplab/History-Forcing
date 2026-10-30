#!/usr/bin/env python3
"""Connect framework InteractiveSession videos to vendored PlayWorld code.

The Agent decision and VQA/aggregation implementations are imported from the
pinned ``upstream/`` snapshot with localized package imports and a per-call VQA
client factory hook. This file owns model I/O and paths, and uses the shared
Seed Lite API transport.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
import re
import sys
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace



HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = HERE / "upstream"
if not __package__:
    sys.path.insert(0, str(ROOT))
from evaluation.io import write_text  # noqa: E402
from evaluation.vlm import (  # noqa: E402
    ARK_BASE_URL, SEED_LITE, SeedJudge as _SeedJudge, _ark_completion,
)
from evaluation.trajectory import (  # noqa: E402
    _configured_packed_trajectory, _infer_and_register_trajectory, preset_metadata,
)

SPLITS = ("gc", "if", "insight", "outsight")
PLAYER_PROTOCOL = "upstream_sana_refined_2s_chunks_v4"
OFFICIAL_CHUNK_SECONDS = (33 - 1) / 16  # SANA-WM defaults: 33 frames at 16 FPS.
OFFICIAL_VIDEO_SECONDS = (961 - 1) / 16
TRAINING_TRAJECTORY_PROTOCOL = "training_camera_trajectory_v1"
PRESET_TRAJECTORY_PROTOCOL = "preset_camera_trajectory_v1"


def training_camera_instruction(row, *, fps, rotation, translation):
    """Describe the prepared RGB controls, including shortened task angles."""
    phases = []
    for phase in row["phases"]:
        key, frames = phase["key"], phase["rgb_frames"]
        seconds = frames / fps
        if key in "jl":
            text = f"Turn the camera {'left' if key == 'j' else 'right'} {frames * rotation:g} degrees"
        elif key in "ik":
            text = f"Tilt the camera {'up' if key == 'i' else 'down'} {frames * rotation:g} degrees"
        elif key in "ad":
            text = f"Move the camera {'left' if key == 'a' else 'right'} {frames * translation:g} native distance units"
        elif key == " ":
            text = "Hold the camera still"
        else:
            raise ValueError(f"Unsupported prepared camera control: {key!r}")
        if phase["purpose"] == "return":
            text += " to return to the original camera pose"
        phases.append(f"{text} for {seconds:g} seconds")
    return "; then ".join(phases) + ".", phases


def run_trajectory(args):
    """Use shared inference on packed training cameras; localize only task controls."""
    data = args.training_data.resolve()
    rows = json.loads((data / "training_samples.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    selected = select_tasks(load_tasks(args.data_root), args.task_ids)
    ids = [row["task_id"] for row in rows]
    if ids != [task["task_id"] for task in selected]:
        raise ValueError("Prepared training order and selected evaluation tasks differ")
    manifests = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if len(manifests) != len(rows):
        raise ValueError("Prepared training manifest length differs")
    config, options, fps, seed = _configured_packed_trajectory(args, data, summary["fps"])
    inputs = []
    localized = {split: [] for split in SPLITS}
    for index, (row, task, manifest) in enumerate(zip(rows, selected, manifests, strict=True)):
        if manifest["prompt"] != row["student_prompt"]:
            raise ValueError(f"Prepared image/prompt mismatch: {row['task_id']}")
        instruction, steps = training_camera_instruction(
            row, fps=fps, rotation=summary["rotation_degrees_per_rgb"],
            translation=summary["translation_units_per_rgb"])
        localized[task["_split"]].append({
            **{key: value for key, value in task.items() if not key.startswith("_")},
            "image_path": str(task["_image"]), "prompt": instruction,
            "action_sequence": "; ".join(steps), "action_sequence_steps": steps,
            "action": "; ".join(steps),
        })
        inputs.append(dict(
            task_id=row["task_id"], row_index=index, seed=seed + index,
            student_prompt=row["student_prompt"],
            camera_instruction=instruction, video_frames=row["video_frames"],
            revisit_rgb_frame=row["revisit_rgb_frame"],
        ))
    for split, tasks in localized.items():
        _write(args.output_dir / "scoring_dataset" / split / "data.json", tasks)
    _infer_and_register_trajectory(args, config, options, inputs,
                                   TRAINING_TRAJECTORY_PROTOCOL)
    print(f"Registered {len(inputs)} unlabelled training-camera videos: {args.output_dir}", flush=True)


def _prepare_preset(args):
    """Validate every input before any rank enters model inference."""
    data = args.prepared_data.absolute()
    metadata = preset_metadata(data)
    plans, summary = metadata["native_plans"], metadata["summary"]
    config, options, fps, seed = _configured_packed_trajectory(args, data, summary["fps"])
    if config.model != summary["model"]:
        raise ValueError("Prepared camera backend differs from inference model")
    if data.suffix == ".parquet":
        from backbones import load_backend
        from dataset import read_packing_source

        manifests, media = read_packing_source(data, load_backend(config.model), config.inference.data_path)
    else:
        manifests = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        media = data
    selected = select_tasks(load_tasks(args.data_root), args.task_ids)
    if (len(manifests) != len(plans) or len(plans) != len(selected)
            or [plan["task_id"] for plan in plans] != [task["task_id"] for task in selected]):
        raise ValueError("Prepared trajectory and selected PlayWorld tasks differ")
    inputs = []
    for index, (manifest, plan, task) in enumerate(zip(manifests, plans, selected, strict=True)):
        control = manifest.get("camera") or manifest.get("actions")
        if not control or plan["num_latent_frames"] != manifest["num_latent_frames"]:
            raise ValueError(f"Prepared control mismatch: {task['task_id']}")
        inputs.append(dict(
            task_id=task["task_id"], row_index=index, seed=seed + index,
            student_prompt=manifest["prompt"],
            camera_instruction=plan.get("actual_instruction"),
            video_frames=plan["video_frames"],
            revisit_rgb_frame=plan["revisit_rgb_frame"],
        ))
    return config, options, inputs


def run_preset(args, *, inference_runner=None):
    """Run every prepared task with the framework's packed-data inference path."""
    if getattr(args, "world_size", 1) > 1:
        from engine.runtime import distributed_call
        config, options, inputs = distributed_call(
            lambda: _prepare_preset(args), args.process_group, "Evaluation input preparation")
    else:
        config, options, inputs = _prepare_preset(args)
    _infer_and_register_trajectory(args, config, options, inputs,
                                   PRESET_TRAJECTORY_PROTOCOL, inference_runner)
    if getattr(args, "rank", 0) == 0:
        print(f"Registered {len(inputs)} preset-camera PlayWorld videos: {args.output_dir}", flush=True)






def _seed_agent_action(args, prompt: str, images_b64: list[str]) -> str:
    def image_url(encoded: str) -> str:
        magic = base64.b64decode(encoded[:24])
        if magic.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif magic.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        else:
            raise ValueError("PlayWorld Agent image is not JPEG or PNG")
        return f"data:{mime_type};base64,{encoded}"

    content = [
        {"type": "image_url", "image_url": {"url": image_url(image)}}
        for image in images_b64
    ]
    content.append({"type": "text", "text": prompt})
    started = time.perf_counter()
    thinking = getattr(args, 'agent_thinking', 'disabled')
    answer = _ark_completion(args.agent_base_url, args.agent_api_key,
                             args.agent_model, content,
                             max_tokens=4096 if thinking == 'enabled' else 128,
                             timeout=args.agent_timeout_s, thinking=thinking)
    # Upstream expects one action word. Seed can instead explain alternatives;
    # upstream's fixed-priority search would pick RIGHT before a final LEFT.
    mentions = re.findall(r"\b(?:LEFT|RIGHT|WAIT|DOWN|UP|NEXT|STOP|END|W|S|A|D)\b",
                          answer.upper())
    if len(set(mentions)) > 1:
        final_line = answer.upper().strip().splitlines()[-1].strip(" .:;!`'\"")
        if final_line in mentions:
            answer = final_line
        else:
            raise ValueError("Seed Agent returned conflicting actions without a final action line")
    if os.environ.get("PLAYWORLD_PROFILE") == "1":
        print(
            f"[seed-profile] model={args.agent_model} images={len(images_b64)} "
            f"api_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return answer


def _seed_camera_task(task: dict) -> dict:
    """Tell the Seed Agent how the native model interprets camera keys."""
    guidance = (
        "Camera controls: LEFT/RIGHT rotate the view in place; A/D move sideways "
        "left/right without turning; W/S move forward/backward; UP/DOWN rotate "
        "the view vertically; WAIT holds the camera still. Choose the control "
        "for the requested type of movement."
    )
    return {**task, "prompt": f"{task['prompt']}\n\n{guidance}"}


def _player_protocol(provider: str, spec, max_seconds: float | None = None) -> str:
    if provider == "seed-lite":
        protocol = (PLAYER_PROTOCOL + "+seed-camera-guidance-v4"
                    if spec.control_mode == "camera"
                    else PLAYER_PROTOCOL + "+seed-action-normalization-v1")
    else:
        protocol = PLAYER_PROTOCOL
    return protocol if max_seconds is None else f"{protocol}+max{max_seconds:g}s"


def _player_block_schedule(spec, fps: int, max_seconds: float | None = None) -> list[int]:
    """Keep official two-second decision boundaries within a native frame cap."""
    if fps <= 0 or (max_seconds is not None and
                    (not math.isfinite(max_seconds) or max_seconds <= 0)):
        raise ValueError("FPS and max video seconds must be positive and finite")
    frames_per_block = 4 * spec.block_frames
    # reset() renders only the image. The first action generates its native
    # block, reusing one latent slot when the image belongs inside that block.
    # After B blocks: RGB = 1 + 4*(B*block_frames - reused_image_slots).
    initial_frames = 1 - 4 * int(spec.condition_consumes_block_slot)
    chunks = round(OFFICIAL_VIDEO_SECONDS / OFFICIAL_CHUNK_SECONDS)
    if max_seconds is not None:
        chunks = min(chunks, math.ceil(max_seconds / OFFICIAL_CHUNK_SECONDS))
        max_blocks = math.floor((math.floor(max_seconds * fps) - initial_frames)
                                / frames_per_block)
        if max_blocks < 1:
            raise ValueError("Video cap cannot fit a native model block")
    schedule = []
    previous = 0
    for chunk in range(1, chunks + 1):
        target_frames = 1 + round(chunk * OFFICIAL_CHUNK_SECONDS * fps)
        cumulative = round((target_frames - initial_frames) / frames_per_block)
        if max_seconds is not None:
            cumulative = min(cumulative, max_blocks)
        if cumulative <= previous:
            if max_seconds is not None and previous == max_blocks:
                break
            raise ValueError("Native video block is too long for PlayWorld chunks")
        schedule.append(cumulative - previous)
        previous = cumulative
    return schedule






def load_tasks(data_root: Path) -> list[dict]:
    if data_root.suffix == ".parquet":
        import pyarrow.parquet as pq
        from filelock import FileLock

        metadata = preset_metadata(data_root)
        official = metadata["official_tasks"]
        root = data_root.parent / f".{data_root.stem}.playworld"
        with pq.ParquetFile(data_root) as source, FileLock(str(root) + ".lock"):
            if source.metadata.num_rows != len(official):
                raise ValueError("Parquet images and official task order differ")
            batches = source.iter_batches(batch_size=1, columns=["image", "source"], use_threads=False)
            for task, batch in zip(official, batches, strict=True):
                record = batch.to_pylist()[0]
                if record["source"]["sample_id"] != task["task_id"]:
                    raise ValueError("Parquet images and official task order differ")
                image = (root / task["image_path"]).resolve()
                if not image.is_relative_to(root.resolve()):
                    raise ValueError("Official image path must stay within the benchmark cache")
                payload = record["image"]["bytes"]
                if image.exists():
                    if image.read_bytes() != payload:
                        raise ValueError(f"Cached official image changed: {task['task_id']}")
                else:
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(payload)
            for split in SPLITS:
                manifest = root / split / "data.json"
                expected = official if split == "outsight" else []
                if manifest.exists():
                    if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                        raise ValueError(f"Cached official task manifest changed: {manifest}")
                else:
                    _write(manifest, expected)
        data_root = root
    tasks = []
    seen = set()
    for split in SPLITS:
        dataset = data_root / split / "data.json"
        if not dataset.is_file():
            dataset = data_root / split / "annotations.json"
        for row in json.loads(dataset.read_text(encoding="utf-8")):
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError(f"Duplicate PlayWorld task ID: {task_id}")
            seen.add(task_id)
            image = data_root / str(row["image_path"])
            if not image.is_file():
                image = data_root / split / "images" / Path(row["image_path"]).name
            if not image.is_file():
                raise FileNotFoundError(image)
            tasks.append({**row, "_dataset": dataset, "_split": split,
                          "_image": image.resolve()})
    return tasks


def select_tasks(tasks: list[dict], text: str | None) -> list[dict]:
    if not text or text == "all":
        return tasks
    by_id = {row["task_id"]: row for row in tasks}
    requested = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    missing = set(requested) - by_id.keys()
    if missing:
        raise ValueError(f"Unknown PlayWorld tasks: {', '.join(sorted(missing))}")
    return [by_id[task_id] for task_id in requested]


def _official_agent():
    path = UPSTREAM / "Agent_player" / "SANA-WM" / "run_agent.py"
    spec = importlib.util.spec_from_file_location("playworld_sana_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_player(args, backend) -> None:
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from inference import InferenceResources

    with torch.no_grad(), InferenceResources() as resources:
        _run_player(args, backend, resources)


def _player_input_identity(task, sample):
    """Record the selected task and packed-condition layout."""
    import torch

    fields = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
              if torch.is_tensor(value) else value for name, value in sample.items()}
    return dict(task={key: value for key, value in task.items() if not key.startswith('_')},
                split=task['_split'], conditions=fields)


def _run_player(args, backend, resources) -> None:
    import numpy as np
    import torch

    from inference import resolve_inference_sampling, sampling_manifest, write_mp4
    from scheduler import observe_sampling

    if not torch.cuda.is_available():
        raise RuntimeError("PlayWorld model interaction requires CUDA")
    args, backend = resolve_inference_sampling(args, backend)
    provider = getattr(args, "agent_provider", "official")
    key_env = args.agent_api_key_env or ("ARK_API_KEY" if provider == "seed-lite" else "PLAYER_API_KEY")
    key = os.environ.get(key_env)
    base_url = args.agent_base_url or (ARK_BASE_URL if provider == "seed-lite" else os.environ.get("PLAYER_API_URL"))
    agent_model = args.agent_model or (SEED_LITE if provider == "seed-lite" else "claude-haiku-4-5-20251001")
    agent_thinking = getattr(args, 'agent_thinking', 'disabled')
    if not key or not base_url:
        raise ValueError(f"Set {key_env} and --agent-base-url")
    official = _official_agent()
    # The official player re-reads every completed chunk at each decision.
    # Chunk MP4s are immutable, so reuse the exact extracted JPEGs and timestamps.
    official.extract_video_frames_with_timestamps_b64 = lru_cache(maxsize=128)(
        official.extract_video_frames_with_timestamps_b64)
    if provider == "seed-lite":
        official.request_agent_action = _seed_agent_action
    tasks = select_tasks(load_tasks(args.data_root), args.task_ids)
    from dataset import LatentSequenceDataset
    if args.data_path is None or args.task_index is None:
        raise ValueError("Packed models require --data-path and --task-index")
    packed = {
        row["task_id"]: row["row_index"]
        for row in json.loads(args.task_index.read_text(encoding="utf-8"))
    }
    missing = [row["task_id"] for row in tasks if row["task_id"] not in packed]
    if missing:
        raise ValueError(f"Tasks missing from packed input: {missing}")
    spec = backend.interactive_spec()
    fps = args.fps or 4 * spec.block_frames
    max_seconds = (args.max_video_seconds if args.max_video_seconds is not None
                   else getattr(backend, "playworld_max_video_seconds", None))
    block_schedule = _player_block_schedule(spec, fps, max_seconds)
    player_protocol = _player_protocol(provider, spec, max_seconds)
    native_steps = sum(block_schedule)
    steps = len(block_schedule)
    backend.validate(args.model_dir, args.checkpoint, codec=True)
    checkpoint = Path(backend.checkpoint_file(args.checkpoint))
    files = [checkpoint]
    if checkpoint.is_dir():
        from model import safetensors_files
        files = safetensors_files(checkpoint)
        for name in ('config.json', 'diffusion_pytorch_model.safetensors.index.json'):
            path = checkpoint / name
            if path.is_file():
                files.append(path)
    checkpoint_files = {}
    for path in files:
        stat = path.stat()
        checkpoint_files[str(path.resolve())] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    options = backend.inference_options(args)
    device = torch.device("cuda")
    model = backend.load(args.model_dir, args.checkpoint, device, torch.bfloat16).eval()
    text_encoder, codec = resources.load(backend, args.model_dir, device, options)
    session = backend.interactive_session(
        model, text_encoder, codec, device, dtype=torch.bfloat16, **options,
    )
    with LatentSequenceDataset(args.data_path) as dataset:
        agent_args = SimpleNamespace(
            agent_base_url=base_url,
            agent_api_key=key,
            agent_user_key=key,
            agent_model=agent_model,
            agent_biz_scene=args.agent_biz_scene,
            agent_timeout_s=args.agent_timeout_s,
            agent_thinking=agent_thinking,
        )
        videos = {}
        for task in tasks:
            official.extract_video_frames_with_timestamps_b64.cache_clear()
            task_id = task["task_id"]
            video = args.output_dir / task_id / f"{task_id}.mp4"
            result_file = video.parent / "result.json"
            row_index = packed[task_id]
            sample = dataset[row_index]
            stationary = official.is_oe_insight_wait_only({
                **task, "_dataset_split": task["_split"],
            })
            episode_inputs = dict(model=args.model, checkpoint=str(args.checkpoint.resolve()),
                                  checkpoint_files=checkpoint_files,
                                  seed=args.seed + row_index, fps=fps,
                                  max_video_seconds=max_seconds,
                                  trajectory_inputs=_player_input_identity(task, sample))
            if video.is_file() and result_file.is_file() and not args.overwrite:
                prior = json.loads(result_file.read_text(encoding="utf-8"))
                if prior.get("status") != "completed" or Path(prior["video_path"]) != video.resolve():
                    raise ValueError(f"Invalid existing PlayWorld episode: {task_id}")
                if prior.get("player_protocol") != player_protocol:
                    raise ValueError(f"Existing {task_id} used a different Player protocol; pass --overwrite")
                if (prior.get("agent_provider", "official") != provider or
                        (prior.get("agent_model") is not None and
                         prior["agent_model"] != agent_model)):
                    raise ValueError(f"Existing {task_id} used a different Agent; pass --overwrite")
                if (provider == 'seed-lite' and not stationary
                        and prior.get('agent_thinking', 'disabled') != agent_thinking):
                    raise ValueError(f"Existing {task_id} used different Agent thinking; pass --overwrite")
                if any(prior.get(key) != value for key, value in episode_inputs.items()):
                    raise ValueError(f"Existing {task_id} used different inference inputs; pass --overwrite")
                prior_sampling = prior.get("sampling_manifest", {})
                expected = sampling_manifest(args, options, [])
                if (prior_sampling.get("schema_version") != 1 or prior_sampling.get("status") != "observed"
                        or not prior_sampling.get("effective", {}).get("schedules")
                        or prior_sampling.get("seed") != episode_inputs['seed']
                        or prior_sampling.get("resolved") != expected["resolved"]
                        or {key: value for key, value in prior_sampling.get("effective", {}).items()
                            if key != "schedules"} != {
                                key: value for key, value in expected["effective"].items() if key != "schedules"}):
                    raise ValueError(f"Existing {task_id} has unverified or different sampling; pass --overwrite")
                videos[task_id] = str(video.resolve())
                continue
            recommendations = (None if stationary else
                               official.dataset_actions_for_chunks(task, steps, 3))
            result_file.unlink(missing_ok=True)
            with observe_sampling() as schedules:
                frame = session.reset(sample, seed=episode_inputs['seed'],
                                      max_steps=native_steps)
                history = []
                chunk_videos = []
                for step in range(steps):
                    agent_started = time.perf_counter()
                    if stationary:
                        action = "WAIT"
                    else:
                        agent_task = {**task, "image_path": str(task["_image"])}
                        if provider == "seed-lite" and spec.control_mode == "camera":
                            agent_task = _seed_camera_task(agent_task)
                        action = official.decide_action(
                            agent_task,
                            np.asarray(frame), history, step, steps, agent_args,
                            dataset_action=recommendations[step],
                            allow_stop=True, allow_next=False,
                            chunk_video_paths=chunk_videos,
                        )
                        if action in {"STOP", "END"}:
                            break
                    agent_seconds = time.perf_counter() - agent_started
                    if action not in spec.actions:
                        raise ValueError(f"{type(backend).__name__} cannot execute {action}")
                    previous_frames = len(session.frames()) if not stationary else None
                    block_count = block_schedule[step]
                    rollout_started = time.perf_counter()
                    for block in range(block_count):
                        frame = session.step(action, observe=not stationary and
                                             block == block_count - 1)
                    rollout_seconds = time.perf_counter() - rollout_started
                    write_started = time.perf_counter()
                    if not stationary:
                        rendered = session.frames()
                        chunk = video.parent / "chunks" / f"chunk_{step + 1:02d}.mp4"
                        chunk.parent.mkdir(parents=True, exist_ok=True)
                        write_mp4(chunk, rendered[max(0, previous_frames - 1):], fps)
                        chunk_videos.append(str(chunk.resolve()))
                    if os.environ.get("PLAYWORLD_PROFILE") == "1":
                        print(
                            f"[playworld-profile] task={task_id} chunk={step + 1} "
                            f"action={action} agent_s={agent_seconds:.3f} "
                            f"rollout_s={rollout_seconds:.3f} "
                            f"mp4_s={time.perf_counter() - write_started:.3f}",
                            flush=True,
                        )
                    history.append(action)
            video.parent.mkdir(parents=True, exist_ok=True)
            final_write_started = time.perf_counter()
            final_frames = session.frames()
            if max_seconds is not None and len(final_frames) / fps > max_seconds:
                raise RuntimeError(f"PlayWorld video exceeds {max_seconds:g}s: "
                                   f"{len(final_frames)} frames at {fps} FPS")
            write_mp4(video, final_frames, fps)
            if os.environ.get("PLAYWORLD_PROFILE") == "1":
                print(
                    f"[playworld-profile] task={task_id} final_mp4_s="
                    f"{time.perf_counter() - final_write_started:.3f}",
                    flush=True,
                )
            _write(result_file, {
                **episode_inputs, "task_id": task_id, "status": "completed",
                "player_protocol": player_protocol,
                "agent_provider": provider,
                "agent_model": agent_model if not stationary else None,
                "agent_thinking": agent_thinking if provider == 'seed-lite' and not stationary else None,
                "agent_source": "upstream/Agent_player/SANA-WM/run_agent.py",
                "actions": history, "player_steps": len(history),
                "native_blocks": sum(block_schedule[:len(history)]),
                "completion_reason": "wait_and_observe" if stationary else (
                    "agent_stop" if len(history) < steps else "step_budget"),
                "video_path": str(video.resolve()),
                "sampling_manifest": sampling_manifest(args, options, schedules,
                                                        seed=episode_inputs['seed']),
            })
            videos[task_id] = str(video.resolve())
            print(f"{task_id}: {len(history)} actions -> {video}", flush=True)
        _write(args.output_dir / "videos_manifest.json", videos)


def _video(task: dict, root: Path, manifest: dict[str, str]) -> Path:
    task_id = task["task_id"]
    if task_id in manifest:
        video = Path(manifest[task_id])
        if not video.is_absolute():
            video = root / video
    else:
        matches = [path for path in (
            root / task_id / f"{task_id}.mp4",
            root / task_id / f"{task_id}_full_process.mp4",
            root / f"{task_id}.mp4",
        ) if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one video for {task_id}, found {matches}")
        video = matches[0]
    if not video.is_file():
        raise FileNotFoundError(video)
    return video.resolve()


def run_score(args) -> None:
    provider = getattr(args, "judge_provider", "official")
    key_env = getattr(args, "judge_api_key_env", None) or (
        "ARK_API_KEY" if provider == "seed-lite" else "GEMINI_API_KEY")
    key = os.environ.get(key_env)
    if not args.dry_run and not key:
        raise ValueError(f"Set {key_env} for the PlayWorld judge")
    judge_model = args.judge_model or (SEED_LITE if provider == "seed-lite" else "gemini-3.1-pro")
    from evaluation.playworld.upstream.metrics.aggregate import (
        OE_INSIGHT_TASKS, aggregate_fail_one_oe_split,
        markdown_table, score_key, normalized_score, embedded_review, _finite,
    )
    from evaluation.playworld.upstream.metrics.vqa import score as official_scorer
    base_url = ((getattr(args, 'judge_base_url', None) or ARK_BASE_URL).rstrip('/')
                if provider == 'seed-lite' else None)
    client_factory = None
    if provider == 'seed-lite' and not args.dry_run:
        client_factory = lambda: _SeedJudge(base_url, key)

    def score_cached(options, path, item):
        task = official_scorer.load_task(options.dataset, options.task_id)
        identity = dict(provider=provider, model=judge_model, mode=options.mode,
                        world_model=options.world_model, evidence_transport=options.evidence_transport,
                        base_url=base_url, task=task)
        video = Path(options.video)
        if video.is_file():
            stat = video.stat()
            identity['video'] = [str(video.resolve()), stat.st_size, stat.st_mtime_ns]
        if options.reference_image is not None:
            reference = Path(options.reference_image)
            stat = reference.stat()
            identity['reference'] = [str(reference.resolve()), stat.st_size, stat.st_mtime_ns]
        def valid(value):
            result = value.get('score')
            if not isinstance(result, dict) or result.get('error'):
                return False
            if options.mode == 'gate':
                return (result.get('verdict') in ('pass','partial','fail')
                        and _finite(result.get('instruction_following_score')) in (0,1))
            needs_gate = ((item['group'] == 'OE' and item['task'] not in OE_INSIGHT_TASKS)
                          or (item['group'] == 'IF' and item['view'] == 'third-person'))
            return (normalized_score(score_key(item), result) is not None and
                    (not needs_gate or embedded_review(result, item['group'], item['view']) is not None))
        if path.is_file() and not args.dry_run:
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                cached = None
            if isinstance(cached, dict) and cached.get('_cache_identity') == identity and valid(cached):
                return cached
        result = official_scorer.score(options, client_factory=client_factory)
        response = result.get('score')
        if response is not None and not isinstance(response, dict):
            result['score'] = dict(error='Judge response must be a JSON object', response=response)
        result['_cache_identity'] = identity
        _write(path, result)
        return result

    tasks = load_tasks(args.data_root)
    selected = select_tasks(tasks, args.task_ids)
    manifest = (json.loads(args.video_manifest.read_text(encoding="utf-8"))
                if args.video_manifest else {})
    workers = getattr(args, 'workers', 3)
    if workers < 1:
        raise ValueError('Scoring workers must be positive')
    player_pid = getattr(args, 'follow_player_pid', None)
    if player_pid is not None and player_pid <= 0:
        raise ValueError('Player PID must be positive')
    # Aggregate artifacts describe this attempt; per-task evidence caches can
    # still be reused independently after a failed call or a subset change.
    for name in ('videos_manifest.json', 'scores.json', 'gc_task_completion.json',
                 'judge.json', 'summary.json', 'summary.md'):
        (args.output_dir / name).unlink(missing_ok=True)

    def score_task(task):
        task_id = task["task_id"]
        if player_pid is not None:
            result_file = args.videos_dir / task_id / 'result.json'
            video_path = args.videos_dir / task_id / f'{task_id}.mp4'
            def completed():
                try:
                    episode = json.loads(result_file.read_text(encoding='utf-8'))
                    return (episode.get('status') == 'completed' and video_path.is_file() and
                            Path(episode.get('video_path', '')).resolve() == video_path.resolve())
                except (FileNotFoundError, json.JSONDecodeError):
                    return False
            while not completed():
                try:
                    os.kill(player_pid, 0)
                except ProcessLookupError:
                    if completed():  # The last record can arrive just before the Player exits.
                        break
                    raise RuntimeError(f'Player {player_pid} stopped before completing {task_id}')
                time.sleep(5)
        video = _video(task, args.videos_dir, manifest)
        metric_id = task.get("source_task_id") or task_id
        item = {
            "model": args.world_model, "task": metric_id,
            "group": task.get("source_category") or task.get("category"),
            "view": str(task.get("perspective") or "").replace(" ", "-"),
            "path": f"{args.world_model}/{metric_id}/{video.name}",
            "isFullProcess": False, "isPlaceholder": False,
        }
        output = args.output_dir / "tasks" / task_id
        common = dict(
            dataset=task["_dataset"], task_id=task_id, video=video,
            world_model=args.world_model, model=judge_model,
            evidence_transport="files" if provider == "seed-lite" else "auto",
            inline_limit_mb=18.0,
            sampling_output_dir=None, dry_run=args.dry_run,
        )
        scored = score_cached(argparse.Namespace(
            **common, mode="score", reference_image=task["_image"],
            context_output=output / "score_context.json",
        ), output / 'score.json', item)
        gated = None
        if item["group"] == "GC":
            gated = score_cached(argparse.Namespace(
                **common, mode="gate", reference_image=None,
                context_output=output / "gate_context.json",
            ), output / 'gate.json', item)
        print(f"scored {task_id}", flush=True)
        return item, scored['score'], None if gated is None else gated['score']

    videos, scores, gates = [], {}, {"results": {}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item, scored, gated in pool.map(score_task, selected):
            videos.append(item)
            if scored is not None:
                scores[score_key(item)] = scored
            if gated is not None:
                gates['results'][score_key(item)] = gated
    _write(args.output_dir / "videos_manifest.json", videos)
    _write(args.output_dir / "scores.json", scores)
    _write(args.output_dir / "gc_task_completion.json", gates)
    _write(args.output_dir / "judge.json", {
        "provider": provider, "model": judge_model,
        "rubric_source": "upstream/metrics/vqa",
        "aggregation_source": "upstream/metrics/aggregate.py",
    })
    if len(tasks) == 171 and len(selected) == 171 and not args.dry_run:
        rows = aggregate_fail_one_oe_split(
            videos, scores, gates, models=((args.world_model, args.world_model),),
        )
        _write(args.output_dir / "summary.json", rows)
        write_text(args.output_dir / "summary.md", markdown_table(rows))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in {"player", "trajectory", "preset", "score"}:
        raise SystemExit("Usage: adapter.py {player|trajectory|preset|score} [options]")
    command = argv[0]
    backend = None
    if command == "player":
        sys.path.insert(0, str(ROOT / "src"))
        from backbones import load_backend

        bootstrap = argparse.ArgumentParser(add_help=False)
        bootstrap.add_argument("--model", default="backbones.abot:ABotWorld")
        known, _ = bootstrap.parse_known_args(argv[1:])
        backend = load_backend(known.model)
        if backend.interactive_spec() is None:
            raise ValueError(f"{known.model} has no InteractiveSession")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    if command == "player":
        parser.add_argument("--model", default=known.model)
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--data-path", type=Path)
        parser.add_argument("--task-index", type=Path)
        parser.add_argument("--fps", type=int)
        parser.add_argument("--max-video-seconds", type=float,
                            help="Cap generated video while retaining official action decisions")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--agent-provider", choices=("official", "seed-lite"), default="official")
        parser.add_argument("--agent-base-url")
        parser.add_argument("--agent-api-key-env")
        parser.add_argument("--agent-model")
        parser.add_argument("--agent-thinking", choices=("disabled", "enabled"), default="disabled")
        parser.add_argument("--agent-biz-scene", default="offline")
        parser.add_argument("--agent-timeout-s", type=float, default=90.0)
        backend.add_inference_arguments(parser)
    elif command in {"trajectory", "preset"}:
        parser.add_argument("--training-data" if command == "trajectory" else "--prepared-data",
                            type=Path, required=True)
        parser.add_argument("--inference-config", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
    else:
        parser.add_argument("--videos-dir", type=Path, required=True)
        parser.add_argument("--video-manifest", type=Path)
        parser.add_argument("--world-model", required=True)
        parser.add_argument("--judge-provider", choices=("official", "seed-lite"), default="official")
        parser.add_argument("--judge-base-url")
        parser.add_argument("--judge-api-key-env")
        parser.add_argument("--judge-model")
        parser.add_argument("--workers", type=int, default=3,
                            help="Parallel independent official scoring tasks")
        parser.add_argument("--follow-player-pid", type=int,
                            help="Wait for each completed episode while this Player runs")
        parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv[1:])
    if command == "player":
        run_player(args, backend)
    elif command == "trajectory":
        run_trajectory(args)
    elif command == "preset":
        run_preset(args)
    else:
        from filelock import FileLock
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output_dir / 'score.lock')):
            run_score(args)


if __name__ == "__main__":
    main()
