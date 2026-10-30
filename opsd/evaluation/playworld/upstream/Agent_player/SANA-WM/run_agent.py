#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import httpx


THIS_DIR = Path(__file__).resolve().parent
PLAYWORLD_ROOT = THIS_DIR.parents[1]
DEFAULT_DATA_ROOT = PLAYWORLD_ROOT / "data"
BENCHMARK_DIR = Path(
    os.environ.get(
        "WORLDMODEL_BENCHMARK_DIR",
        "/m2v_intern/public_datasets/worldmodel_benchmark",
    )
)
WORLDPLAY_DIR = Path(
    os.environ.get("WORLDPLAY_DATA_ROOT", str(DEFAULT_DATA_ROOT))
)
DEFAULT_SANA_ROOT = BENCHMARK_DIR / "experiment" / "SANA_WM" / "Sana"

VALID_ACTIONS = ["W", "S", "A", "D", "LEFT", "RIGHT", "UP", "DOWN", "WAIT", "STOP", "END"]
TOKEN_NORMALIZE = {"←": "LEFT", "→": "RIGHT", "↑": "UP", "↓": "DOWN", "wait": "WAIT"}
ACTION_SOURCE = "refined"
DATASET_SPLIT_FILES = {
    "gc": Path("gc/data.json"),
    "if": Path("if/data.json"),
    "insight": Path("insight/data.json"),
    "outsight": Path("outsight/data.json"),
}


def normalize_action_str(action_str: str) -> str:
    s = action_str or ""
    s = re.sub(r"interact\([^)]*\)", "W", s)
    s = re.sub(r"\bu(?=\*)", "UP", s)
    s = re.sub(r"\bj(?=\*)", "DOWN", s)
    for raw, std in TOKEN_NORMALIZE.items():
        s = s.replace(raw, std)
    return s


def parse_segments(task: dict) -> List[tuple]:
    segments = []
    action_str = normalize_action_str(task.get("action") or "")
    for seg in re.split(r"->|→", action_str):
        m = re.match(r"\s*([A-Za-z]+)\s*\*\s*(\d+)", seg)
        if m:
            act, n = m.group(1).upper(), int(m.group(2))
            segments.append((act, max(1, n)))
    if not segments:
        for step in task.get("action_sequence_steps") or []:
            m = re.match(r"hold\(\s*([A-Za-z]+)\s*,\s*(\d+)\s*ms\s*\)", step)
            if m:
                segments.append((m.group(1).upper(), max(1, round(int(m.group(2)) / 700))))
    return segments or [("W", 80)]


def compress_frame_jpg(frame_np, max_size: int = 512) -> bytes:
    from PIL import Image
    img = Image.fromarray(frame_np.astype("uint8"))
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def image_file_to_b64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def extract_video_frames_with_timestamps_b64(
    video_path: str, num_frames: int = 8
) -> List[dict]:
    """Sample JPEG observations from a generated chunk in chronological order."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return []
    count = min(num_frames, total)
    indices = [int(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
    samples = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            continue
        samples.append({
            "b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "frame_index": index,
            "timestamp_s": round(index / fps, 3) if fps > 0 else None,
        })
    cap.release()
    return samples


def request_agent_action(args, prompt: str, images_b64: List[str]) -> str:
    """Call the repository-standard OpenAI-compatible Agent API."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        }
        for image_b64 in images_b64
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": args.agent_model,
        "max_completion_tokens": 128,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": args.agent_api_key,
        "x-ks-user-key": args.agent_user_key,
        "x-ks-llm-model": args.agent_model,
        "x-ks-biz-scene": args.agent_biz_scene,
    }
    with httpx.Client(trust_env=False) as client:
        response = client.post(
            f"{args.agent_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=args.agent_timeout_s,
        )
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        raise RuntimeError("Agent API response has no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def decide_action(task: dict, frame, recent_actions: List[str], chunk_idx: int,
                  num_chunks: int, args, dataset_action: str = "W",
                  prev_video_path: str = None,
                  allow_stop: bool = True,
                  allow_next: bool = False,
                  chunk_video_paths: List[str] | None = None,
                  stationary_only: bool = False) -> str:
    """Ask the VLM for the next SANA action using the latest rollout state."""
    images_b64 = []
    frame_labels = []
    init_path = task.get("image_path")
    if init_path and os.path.exists(init_path):
        images_b64.append(image_file_to_b64(init_path))
        frame_labels.append("initial frame at t=0.000s")
    history_paths = [str(path) for path in (chunk_video_paths or []) if path and os.path.exists(path)]
    if not history_paths and prev_video_path and os.path.exists(prev_video_path):
        history_paths = [prev_video_path]
    if history_paths:
        # Chronological visual context: initial frame first, four samples from
        # every older chunk, then eight samples from the latest chunk.
        for history_idx, video_path in enumerate(history_paths[:-1], 1):
            for sample in extract_video_frames_with_timestamps_b64(video_path, 4):
                images_b64.append(sample["b64"])
                frame_labels.append(
                    f"chunk {history_idx:02d} at local t={sample['timestamp_s']}s"
                )
        latest_idx = len(history_paths)
        for sample in extract_video_frames_with_timestamps_b64(history_paths[-1], 8):
            images_b64.append(sample["b64"])
            frame_labels.append(
                f"chunk {latest_idx:02d} at local t={sample['timestamp_s']}s"
            )
    elif frame is not None:
        images_b64.append(base64.b64encode(compress_frame_jpg(frame)).decode("ascii"))

    choices = "WAIT" if stationary_only else "W, S, A, D, LEFT, RIGHT, UP, DOWN, WAIT"
    action_history_text = " -> ".join(recent_actions) if recent_actions else "none"
    frame_timeline_text = "; ".join(frame_labels) if frame_labels else "current generated frame"
    stationary_instruction = ""
    if stationary_only:
        criteria = " ".join(
            str(item.get("question", "")).strip()
            for item in (task.get("questions") or [])
            if item.get("question")
        )
        stationary_instruction = (
            "Keep the camera completely stationary. WAIT generates another observation chunk with no camera motion. "
            "Return END only after the frames show enough natural scene evolution to satisfy the goal. "
            f"Use this evidence when judging completion: {criteria}\n"
        )
    prompt = (
        f"Our goal is: {task.get('prompt', '')}\n"
        f"We are now at step {chunk_idx + 1} of at most {num_chunks}. "
        f"We have completed {chunk_idx} chunks and the actions already taken are: "
        f"{action_history_text}.\n"
        f"The attached frames are chronological and have these timestamps in the same order: "
        f"{frame_timeline_text}.\n"
        f"{stationary_instruction}"
        "Look at the actual visual progress in these frames and independently decide what action is needed now "
        "to accomplish the goal.\n"
        f"Choose one action from {choices}{', NEXT' if allow_next else ''}. "
        + ("Return NEXT only when the current action segment is visually complete and we should move to the next segment. "
           if allow_next else "")
        + "If the full goal in the prompt is already visually complete, return END. "
        "Return exactly one action word and nothing else."
    )
    for attempt in range(1, 4):
        try:
            raw = request_agent_action(args, prompt, images_b64)
            text = str(raw or "").strip().upper()
            print(f"    [agent] raw_response={text!r}")
            candidates = (["END", "WAIT"] if stationary_only else
                          ["END", "RIGHT", "LEFT", "DOWN", "WAIT", "UP", "W", "S", "A", "D"])
            if allow_next:
                candidates.insert(0, "NEXT")
            if allow_stop:
                candidates.insert(4, "STOP")
            for action in candidates:
                if re.search(rf"\b{action}\b", text):
                    return action
            print(f"    [agent] invalid response attempt {attempt}: {text!r}")
        except Exception as exc:
            print(f"    [agent] error attempt {attempt}: {exc}")
    if allow_next:
        print("    [agent] fallback=NEXT")
        return "NEXT"
    if stationary_only:
        print("    [agent] fallback=WAIT")
        return "WAIT"
    print(f"    [agent] fallback={dataset_action}")
    return dataset_action if dataset_action in VALID_ACTIONS else "W"


def grouped_dataset_actions(task: dict, actions_per_chunk: int = 3) -> List[str]:
    """Map every N repeated dataset tokens to one native SANA chunk."""
    grouped = []
    unit = max(1, int(actions_per_chunk))
    for action, count in parse_segments(task):
        action = action if action in VALID_ACTIONS else "W"
        grouped.extend([action] * max(1, (int(count) + unit - 1) // unit))
    return grouped or ["W"]


def mixed_dataset_agent_plan(task: dict, dataset_fraction: float = 0.6,
                             actions_per_chunk: int = 3) -> List[dict]:
    """Build per-segment dataset minima; agent extensions are scheduled live."""
    unit = max(1, int(actions_per_chunk))
    fraction = min(1.0, max(0.0, float(dataset_fraction)))
    plan = []
    for action, count in parse_segments(task):
        action = action if action in VALID_ACTIONS else "W"
        dataset_count = min(count, int(math.ceil(count * fraction)))
        dataset_chunks = []
        remaining = dataset_count
        while remaining > 0:
            token_count = min(unit, remaining)
            dataset_chunks.append(token_count)
            remaining -= token_count
        plan.append({
            "dataset_action": action,
            "original_action_count": count,
            "dataset_action_count": dataset_count,
            "dataset_chunks": dataset_chunks,
        })
    return plan or [{
        "dataset_action": "W",
        "original_action_count": 1,
        "dataset_action_count": 1,
        "dataset_chunks": [1],
    }]


def dataset_actions_for_chunks(task: dict, num_chunks: int,
                               actions_per_chunk: int = 3) -> List[str]:
    grouped = grouped_dataset_actions(task, actions_per_chunk)
    if len(grouped) >= num_chunks:
        return grouped[:num_chunks]
    return grouped + [grouped[-1]] * (num_chunks - len(grouped))


def to_sana_key(action: str) -> str:
    action = action.upper()
    mp = {
        "W": "w",
        "S": "s",
        "A": "a",
        "D": "d",
        "LEFT": "j",
        "RIGHT": "l",
        "UP": "i",
        "DOWN": "k",
        "WAIT": "none",
        "STOP": "none",
        "END": "none",
    }
    return mp.get(action, "w")


def is_insight_evolution(task: dict) -> bool:
    """Recognize the insight split and its equivalent task metadata."""
    if str(task.get("category", "")).upper() != "OE":
        return False
    split = str(task.get("_dataset_split", "")).strip().lower()
    sub_category = re.sub(
        r"[\s_-]+", "", str(task.get("sub_category", "")).strip().lower()
    )
    question_categories = {
        re.sub(r"[\s_-]+", "", str(item.get("category", "")).strip().lower())
        for item in (task.get("questions") or [])
        if isinstance(item, dict)
    }
    return (
        split == "insight"
        or sub_category == "insightevolution"
        or "insightevolution" in question_categories
    )


def is_oe_insight_wait_only(task: dict) -> bool:
    if not is_insight_evolution(task):
        return False
    action = str(task.get("action") or "").strip().lower().replace(" ", "")
    steps = [str(x).strip().lower().replace(" ", "") for x in (task.get("action_sequence_steps") or [])]
    return bool(
        re.fullmatch(r"wait\*\d+", action)
        or action in {"wait", "waitandobserve"}
        or (steps and all(x.startswith("wait(") for x in steps))
    )


def build_full_sana_action(task: dict, frame_unit: int, tail_frames: int, max_total_frames: int) -> tuple[str, list]:
    if is_oe_insight_wait_only(task):
        frames = 34 * 33 - 1
        return "none-{}".format(frames), [{"dataset_action": "WAIT", "count": 34, "sana_action": "none-{}".format(frames), "frames": frames, "raw_frames": frames, "special": "oe_insight_wait_only_34chunk_no_action"}]
    segments = parse_segments(task)
    base = []
    for action, count in segments:
        key = to_sana_key(action)
        frames = max(1, int(count) * int(frame_unit))
        base.append({"dataset_action": action, "count": int(count), "key": key, "raw_frames": frames})
    action_budget = max_total_frames - int(tail_frames) - 1 if max_total_frames > 0 else None
    raw_total = sum(item["raw_frames"] for item in base)
    if action_budget is not None and raw_total > action_budget and action_budget > 0:
        scale = action_budget / float(raw_total)
        scaled = [max(1, int(round(item["raw_frames"] * scale))) for item in base]
        diff = action_budget - sum(scaled)
        order = sorted(range(len(scaled)), key=lambda i: base[i]["raw_frames"], reverse=True)
        idx = 0
        while diff != 0 and order:
            i = order[idx % len(order)]
            if diff > 0:
                scaled[i] += 1
                diff -= 1
            elif scaled[i] > 1:
                scaled[i] -= 1
                diff += 1
            idx += 1
            if idx > 100000:
                break
    else:
        scaled = [item["raw_frames"] for item in base]
        scale = 1.0
    parts = []
    records = []
    for item, frames in zip(base, scaled):
        parts.append(f"{item['key']}-{frames}")
        records.append({
            "dataset_action": item["dataset_action"],
            "count": item["count"],
            "sana_action": f"{item['key']}-{frames}",
            "frames": int(frames),
            "raw_frames": int(item["raw_frames"]),
        })
    if tail_frames > 0:
        parts.append(f"none-{int(tail_frames)}")
        records.append({"dataset_action": "TAIL", "count": 1, "sana_action": f"none-{int(tail_frames)}", "frames": int(tail_frames), "raw_frames": int(tail_frames)})
    return ",".join(parts), records


def prepare_input_image(image_path: str, task_dir: Path, max_size: int, target_area: int) -> str:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if target_area > 0:
        scale = min(scale, (float(target_area) / float(w * h)) ** 0.5)
    if max_size > 0:
        scale = min(scale, max_size / float(max(w, h)))
    if scale >= 0.999:
        return image_path
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    out = task_dir / "input_resized.jpg"
    img.resize(new_size, Image.LANCZOS).save(out, format="JPEG", quality=95)
    return str(out)


def make_default_intrinsics(image_path: str, task_dir: Path, num_frames: int) -> str:
    import numpy as np
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    f = float(max(w, h))
    arr = np.array([[f, f, w / 2.0, h / 2.0]] * int(num_frames), dtype=np.float32)
    path = task_dir / "intrinsics.npy"
    np.save(path, arr)
    return str(path)


def save_video(frames: List, path: str, fps: int) -> None:
    import imageio
    import numpy as np
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to save")
    h, w = frames[0].shape[:2]
    fixed = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.shape[:2] != (h, w):
            arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS))
        fixed.append(arr)
    imageio.mimsave(path, fixed, fps=fps)


def read_video(path: Path) -> List:
    import imageio
    import numpy as np
    return [np.asarray(frame) for frame in imageio.mimread(str(path))]


class SanaWMEngine:
    def __init__(self, args):
        if str(BENCHMARK_DIR) not in sys.path:
            sys.path.insert(0, str(BENCHMARK_DIR))
        self.args = args
        self.sana_root = Path(args.sana_root or os.environ.get("SANA_ROOT") or DEFAULT_SANA_ROOT).resolve()
        self.script = self.sana_root / "inference_video_scripts" / "wm" / "inference_sana_wm.py"
        if not self.script.exists():
            raise FileNotFoundError(f"Sana inference script not found: {self.script}")
        self._wm = None
        self._pipeline = None
        self._config = None
        if not args.sana_subprocess:
            self._load_inprocess_pipeline()

    def _load_inprocess_pipeline(self) -> None:
        """Load the official reusable pipeline once instead of once per chunk."""
        import torch
        import pyrallis

        if str(self.sana_root) not in sys.path:
            sys.path.insert(0, str(self.sana_root))
        from inference_video_scripts.wm import inference_sana_wm as wm

        config_path = self.args.sana_config or wm.HF_DEFAULTS["config"]
        model_path = self.args.sana_checkpoint or wm.HF_DEFAULTS["model_path"]
        config = pyrallis.parse(
            config_class=wm.InferenceConfig,
            config_path=wm.resolve_hf_path(config_path),
            args=[],
        )
        refiner = wm.RefinerSettings(
            root=wm.HF_DEFAULTS["refiner_root"],
            gemma_root=wm.HF_DEFAULTS["refiner_gemma_root"],
        )
        self._wm = wm
        self._config = config
        self._pipeline = wm.SanaWMPipeline(
            config=config,
            model_path=wm.resolve_hf_path(model_path),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            refiner=refiner,
            logger=wm.get_root_logger(),
        )
        # Cache connector embeddings for the prompt shared by a task's chunks.
        refiner_module = getattr(self._pipeline, "refiner", None)
        if refiner_module is not None:
            original_encode_prompt = refiner_module._encode_prompt
            prompt_cache = {}

            def cached_encode_prompt(prompt: str):
                if prompt not in prompt_cache:
                    prompt_cache[prompt] = tuple(
                        tensor.detach() for tensor in original_encode_prompt(prompt)
                    )
                return prompt_cache[prompt]

            refiner_module._encode_prompt = cached_encode_prompt

    def _generate_inprocess(self, image_path: str, prompt: str, action: str,
                            output_dir: Path, name: str, num_frames: int,
                            intrinsics_path: str = None) -> Path:
        from PIL import Image

        wm = self._wm
        image = Image.open(image_path).convert("RGB")
        c2w_full = wm.action_string_to_c2w(
            action,
            translation_speed=wm.DEFAULT_TRANSLATION_SPEED,
            rotation_speed_deg=wm.DEFAULT_ROTATION_SPEED_DEG,
        )
        num_frames = min(num_frames, c2w_full.shape[0])
        num_frames = wm._snap_num_frames(num_frames, stride=8, upper_bound=c2w_full.shape[0])
        c2w = c2w_full[:num_frames]
        cropped, src_size, resized_size, crop_offset = wm.resize_and_center_crop(image)
        if intrinsics_path:
            intr_src = wm.load_intrinsics(Path(intrinsics_path), num_frames)
        else:
            raise ValueError("in-process SANA generation requires intrinsics")
        intrinsics = wm.transform_intrinsics_for_crop(
            intr_src, src_size, resized_size, crop_offset
        )
        sampling_algo = self._config.scheduler.vis_sampler
        if sampling_algo not in {"chunk_flow_euler", "self_forcing"}:
            sampling_algo = "flow_euler_ltx"
        params = wm.GenerationParams(
            num_frames=num_frames,
            fps=self.args.fps,
            step=self.args.num_inference_steps,
            cfg_scale=self.args.guidance_scale,
            seed=self.args.seed,
            sampling_algo=sampling_algo,
        )
        result = self._pipeline.generate(cropped, prompt, c2w, intrinsics, params)
        return wm.write_video(output_dir, name, result["video"], params.fps, self._pipeline.logger)

    def generate(self, image_path: str, prompt: str, action: str, output_dir: Path, name: str, num_frames: int, intrinsics_path: str = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._pipeline is not None:
            return self._generate_inprocess(
                image_path, prompt, action, output_dir, name, num_frames,
                intrinsics_path=intrinsics_path,
            )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(prompt or "A photo-realistic game scene with rich details.")
            prompt_file = f.name
        cmd = [
            sys.executable,
            str(self.script),
            "--image", image_path,
            "--prompt", prompt_file,
            "--action", action,
            "--num_frames", str(num_frames),
            "--output_dir", str(output_dir),
            "--name", name,
            "--fps", str(self.args.fps),
            "--step", str(self.args.num_inference_steps),
            "--cfg_scale", str(self.args.guidance_scale),
        ]
        if intrinsics_path:
            cmd += ["--intrinsics", intrinsics_path]
        if self.args.sana_config:
            cmd += ["--config", self.args.sana_config]
        if self.args.sana_checkpoint:
            cmd += ["--model_path", self.args.sana_checkpoint]
        env = os.environ.copy()
        env.setdefault("DISABLE_XFORMERS", "1")
        env["PYTHONPATH"] = f"{self.sana_root}:{BENCHMARK_DIR}:" + env.get("PYTHONPATH", "")
        try:
            subprocess.run(cmd, cwd=str(self.sana_root), env=env, check=True)
        finally:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass
        candidates = list(output_dir.glob(f"{name}*.mp4")) + list(output_dir.rglob(f"{name}*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"Sana output mp4 not found in {output_dir}")
        return sorted(set(candidates), key=lambda p: p.stat().st_mtime)[-1]


def run_task_closed_loop(task: dict, engine: SanaWMEngine, output_dir: Path,
                         args, img_path: str, prompt: str, task_dir: Path) -> dict:
    """Generate SANA chunks and request a fresh agent decision between chunks."""
    import numpy as np
    from PIL import Image

    task_id = task["task_id"]
    stationary_only = is_oe_insight_wait_only(task)
    # Adjacent chunks share their boundary frame, so each additional chunk
    # contributes chunk_frames - 1 frames to the stitched rollout.
    max_chunks_by_frames = max(
        1, (args.max_total_frames - 1) // max(1, args.chunk_frames - 1)
    )
    if stationary_only:
        num_chunks = min(args.num_chunks, max_chunks_by_frames)
        print(f"  [{task_id}] stationary wait-only mode: fixed WAIT chunks={num_chunks}, agent disabled")
    elif ACTION_SOURCE == "dataset_then_agent":
        mixed_segments = mixed_dataset_agent_plan(
            task, args.mixed_dataset_fraction, args.dataset_actions_per_chunk
        )
        num_chunks = min(args.num_chunks, max_chunks_by_frames)
        segment_idx = 0
        segment_dataset_chunk_idx = 0
        segment_extension_chunks = 0
    else:
        num_chunks = min(args.num_chunks, max_chunks_by_frames)
        recommendations = dataset_actions_for_chunks(
            task, num_chunks, args.dataset_actions_per_chunk
        )
        control_plan = [{
            "source": "agent",
            "dataset_action": recommendations[i],
            "action_token_count": args.dataset_actions_per_chunk,
        } for i in range(num_chunks)]
    recent_actions: List[str] = []
    action_records = []
    all_frames = []
    current_image = img_path
    current_frame = np.asarray(Image.open(current_image).convert("RGB"))
    prev_video = None
    generated_chunk_videos: List[str] = []
    t0 = time.time()

    chunk_idx = 0
    while chunk_idx < num_chunks:
        predecided_action = None
        if stationary_only:
            planned = {
                "source": "dataset",
                "dataset_action": "WAIT",
                "action_token_count": 1,
                "is_extension": False,
                "extension_reason": None,
                "segment_idx": 0,
            }
        elif ACTION_SOURCE == "dataset_then_agent":
            while True:
                if segment_idx < len(mixed_segments):
                    segment = mixed_segments[segment_idx]
                    dataset_chunks = segment["dataset_chunks"]
                    if segment_dataset_chunk_idx < len(dataset_chunks):
                        planned = {
                            "source": "dataset",
                            "dataset_action": segment["dataset_action"],
                            "action_token_count": dataset_chunks[segment_dataset_chunk_idx],
                            "is_extension": False,
                            "segment_idx": segment_idx,
                        }
                        segment_dataset_chunk_idx += 1
                        break

                    future_dataset_chunks = sum(
                        len(item["dataset_chunks"])
                        for item in mixed_segments[segment_idx + 1:]
                    )
                    remaining_slots = num_chunks - chunk_idx
                    if segment_extension_chunks >= args.max_segment_extension_chunks:
                        print(
                            f"    [scheduler] NEXT forced after segment {segment_idx + 1}; "
                            f"agent extension reached {args.max_segment_extension_chunks} chunks"
                        )
                        segment_idx += 1
                        segment_dataset_chunk_idx = 0
                        segment_extension_chunks = 0
                        continue
                    if remaining_slots <= future_dataset_chunks:
                        print(
                            f"    [scheduler] NEXT forced after segment {segment_idx + 1}; "
                            "reserving the 60% dataset minimum for later segments"
                        )
                        segment_idx += 1
                        segment_dataset_chunk_idx = 0
                        segment_extension_chunks = 0
                        continue

                    predecided_action = decide_action(
                        task,
                        current_frame,
                        recent_actions,
                        chunk_idx,
                        num_chunks,
                        args,
                        dataset_action=segment["dataset_action"],
                        prev_video_path=str(prev_video) if prev_video else None,
                        allow_stop=False,
                        allow_next=True,
                        chunk_video_paths=generated_chunk_videos,
                        stationary_only=False,
                    )
                    if predecided_action == "NEXT":
                        print(f"    [agent] NEXT after segment {segment_idx + 1}")
                        segment_idx += 1
                        segment_dataset_chunk_idx = 0
                        segment_extension_chunks = 0
                        continue
                    planned = {
                        "source": "agent",
                        "dataset_action": segment["dataset_action"],
                        "action_token_count": args.dataset_actions_per_chunk,
                        "is_extension": True,
                        "extension_reason": "agent_extended_current_segment",
                        "segment_idx": segment_idx,
                    }
                    break

                # All original segments have received their dataset minimum.
                # The agent freely fills the remaining fixed-length rollout.
                recommendation = mixed_segments[-1]["dataset_action"]
                planned = {
                    "source": "agent",
                    "dataset_action": recommendation,
                    "action_token_count": args.dataset_actions_per_chunk,
                    "is_extension": True,
                    "extension_reason": "agent_filled_remaining_rollout",
                    "segment_idx": None,
                }
                break
        else:
            planned = control_plan[chunk_idx]

        recommended = planned["dataset_action"]
        if planned["source"] == "dataset":
            action = recommended
            source = "dataset"
        elif predecided_action is not None:
            action = predecided_action
            source = "agent"
        else:
            action = decide_action(
                task,
                current_frame,
                recent_actions,
                chunk_idx,
                num_chunks,
                args,
                dataset_action=recommended,
                prev_video_path=str(prev_video) if prev_video else None,
                allow_stop=ACTION_SOURCE != "dataset_then_agent",
                chunk_video_paths=generated_chunk_videos,
                stationary_only=stationary_only,
            )
            source = "agent"
        if action in {"STOP", "END"}:
            action_records.append({
                "chunk_idx": chunk_idx,
                "action": action,
                "source": source,
                "dataset_action": recommended,
                "generated": False,
            })
            print(f"  [{task_id}] agent {action} at chunk {chunk_idx + 1}")
            break

        action_frames = max(1, args.chunk_frames)
        sana_action = f"{to_sana_key(action)}-{action_frames}"
        chunk_dir = task_dir / "chunks" / f"chunk_{chunk_idx + 1:02d}"
        chunk_name = f"{task_id}_chunk_{chunk_idx + 1:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        intrinsics = make_default_intrinsics(current_image, chunk_dir, args.chunk_frames)
        print(
            f"  [{task_id}] chunk {chunk_idx + 1}/{num_chunks}: "
            f"action={action} source={source} sana={sana_action}"
        )
        generated = engine.generate(
            current_image,
            prompt,
            sana_action,
            chunk_dir,
            chunk_name,
            args.chunk_frames,
            intrinsics_path=intrinsics,
        )
        canonical_chunk = chunk_dir / f"{chunk_name}.mp4"
        if Path(generated) != canonical_chunk:
            shutil.copy2(generated, canonical_chunk)
        frames = read_video(canonical_chunk)
        if not frames:
            raise RuntimeError(f"empty SANA chunk: {canonical_chunk}")
        all_frames.extend(frames if not all_frames else frames[1:])
        current_frame = frames[-1]
        current_image = str(task_dir / f"last_frame_chunk{chunk_idx + 1:02d}.jpg")
        Image.fromarray(current_frame.astype("uint8")).save(current_image, quality=95)
        prev_video = canonical_chunk
        generated_chunk_videos.append(str(canonical_chunk))
        recent_actions.append(action)
        action_records.append({
            "chunk_idx": chunk_idx,
            "action": action,
            "source": source,
            "dataset_action": recommended,
            "action_token_count": planned["action_token_count"],
            "is_extension": planned.get("is_extension", False),
            "extension_reason": planned.get("extension_reason"),
            "segment_idx": planned.get("segment_idx"),
            "sana_action": sana_action,
            "frames": len(frames),
            "video_path": str(canonical_chunk),
            "generated": True,
        })
        if (ACTION_SOURCE == "dataset_then_agent"
                and planned.get("extension_reason") == "agent_extended_current_segment"):
            segment_extension_chunks += 1
        chunk_idx += 1

    if not all_frames:
        all_frames = [np.asarray(Image.open(img_path).convert("RGB"))]
    video_path = task_dir / f"{task_id}.mp4"
    save_video(all_frames, str(video_path), args.fps)
    result = {
        "task_id": task_id,
        "prompt": prompt,
        "image_path": img_path,
        "actions": action_records,
        "action_source": ACTION_SOURCE,
        "total_frames": len(all_frames),
        "elapsed": round(time.time() - t0, 1),
        "video_path": str(video_path),
        "success": True,
        "model": "SANA_WM",
        "closed_loop_agent": True,
    }
    with open(task_dir / f"{task_id}_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(task_dir / f"{task_id}.json", "w") as f:
        json.dump({
            "task_id": task_id,
            "prompt": prompt,
            "actions": action_records,
            "total_frames": len(all_frames),
            "closed_loop_agent": True,
        }, f, indent=2, ensure_ascii=False)
    print(f"  [{task_id}] DONE | chunks={len(recent_actions)} frames={len(all_frames)} -> {video_path}")
    return result


def run_task(task: dict, engine: SanaWMEngine, output_dir: Path, args) -> dict:
    import numpy as np
    from PIL import Image
    task_id = task["task_id"]
    prompt = task.get("prompt", "")
    img_path = task.get("image_path", "")
    if not os.path.exists(img_path):
        print(f"  [SKIP] image not found: {img_path}")
        return {"task_id": task_id, "success": False, "error": "image not found"}

    task_dir = output_dir / task_id
    if is_oe_insight_wait_only(task) and task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    img_path = prepare_input_image(img_path, task_dir, args.max_image_size, args.target_area)
    if ACTION_SOURCE != "dataset":
        return run_task_closed_loop(task, engine, output_dir, args, img_path, prompt, task_dir)
    actions_log = []
    recent_actions = []
    t0 = time.time()
    sana_action, actions_log = build_full_sana_action(task, args.action_frame_unit, args.tail_frames, args.max_total_frames)
    total_frames = sum(item["frames"] for item in actions_log) + 1
    print(f"  [{task_id}] sana action: {sana_action}")

    intrinsics_path = make_default_intrinsics(img_path, task_dir, total_frames)
    mp4 = engine.generate(img_path, prompt, sana_action, task_dir / "chunks", f"{task_id}_full", total_frames, intrinsics_path=intrinsics_path)
    video_path = task_dir / f"{task_id}.mp4"
    if Path(mp4) != video_path:
        shutil.copy2(mp4, video_path)
    output_frames = total_frames
    result = {
        "task_id": task_id,
        "prompt": prompt,
        "image_path": img_path,
        "actions": actions_log,
        "sana_action": sana_action,
        "total_frames": output_frames,
        "elapsed": round(time.time() - t0, 1),
        "video_path": str(video_path),
        "success": True,
        "model": "SANA_WM",
    }
    with open(task_dir / f"{task_id}_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(task_dir / f"{task_id}.json", "w") as f:
        json.dump({"task_id": task_id, "prompt": prompt, "actions": actions_log, "sana_action": sana_action, "total_frames": result["total_frames"]}, f, indent=2, ensure_ascii=False)
    print(f"  [{task_id}] DONE | frames={result['total_frames']} → {video_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="SANA-WM benchmark runner")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--skip-done", action="store_true", default=True)
    parser.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    parser.add_argument("--mapping-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--action-source", choices=["dataset", "agent", "dataset_then_agent", "refined"], default="refined")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--data-base", default=None)
    parser.add_argument("--images-dir", default=None, help="Base directory for relative image_path values")
    parser.add_argument("--sana-root", default=None)
    parser.add_argument("--sana-config", default=os.environ.get("SANA_CONFIG"))
    parser.add_argument("--sana-checkpoint", default=os.environ.get("SANA_CHECKPOINT"))
    parser.add_argument("--sana-subprocess", action="store_true", help="Reload the official CLI for every chunk instead of reusing one in-process pipeline")
    parser.add_argument("--num-inference-steps", type=int, default=60)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-chunks", type=int, default=30)
    parser.add_argument("--chunk-frames", type=int, default=33)
    parser.add_argument("--dataset-actions-per-chunk", type=int, default=3)
    parser.add_argument("--mixed-dataset-fraction", type=float, default=0.6)
    parser.add_argument("--max-segment-extension-chunks", type=int, default=5,
                        help="Maximum agent-added chunks for one original action segment")
    parser.add_argument("--action-frame-unit", type=int, default=20)
    parser.add_argument("--tail-frames", type=int, default=50)
    parser.add_argument("--max-image-size", type=int, default=0)
    parser.add_argument("--target-area", type=int, default=1280 * 720)
    parser.add_argument("--max-total-frames", type=int, default=961)
    parser.add_argument("--agent-sample-frames", type=int, default=8)
    parser.add_argument("--agent-base-url", default=os.environ.get("PLAYER_API_URL", ""))
    parser.add_argument("--agent-model", default=os.environ.get("PLAYER_MODEL", "claude-haiku-4-5-20251001"))
    parser.add_argument("--agent-api-key", default=os.environ.get("PLAYER_API_KEY", ""))
    parser.add_argument("--agent-user-key", default=os.environ.get("PLAYER_USER_KEY") or os.environ.get("PLAYER_API_KEY", ""))
    parser.add_argument("--agent-biz-scene", default=os.environ.get("PLAYER_BIZ_SCENE", "offline"))
    parser.add_argument("--agent-timeout-s", type=float, default=float(os.environ.get("PLAYER_API_TIMEOUT_S", "60")))
    args = parser.parse_args()

    global ACTION_SOURCE
    ACTION_SOURCE = args.action_source

    if args.categories:
        data_base = Path(args.data_base or WORLDPLAY_DIR)
        cat_jsons = {}
        for category in args.categories:
            key = category.strip().lower()
            relative_path = DATASET_SPLIT_FILES.get(key, Path(f"{category}.json"))
            cat_jsons[category] = str(data_base / relative_path)
    else:
        cat_jsons = {"default": args.mapping_json}
    if not all(cat_jsons.values()):
        raise SystemExit("--mapping-json or --categories is required")

    print("[SANA_WM] Loading engine...")
    engine = SanaWMEngine(args)

    for cat_name, input_json in cat_jsons.items():
        print("\n" + "=" * 60)
        print(f"[SANA_WM] Category: {cat_name} | JSON: {input_json}")
        print("=" * 60)
        output_dir = Path(args.output_dir or (PLAYWORLD_ROOT / "outputs" / "sana_wm"))
        if args.categories:
            output_dir = output_dir / cat_name
        output_dir.mkdir(parents=True, exist_ok=True)
        input_parent = Path(input_json).resolve().parent
        image_base = str(
            input_parent.parent
            if input_parent.name.lower() in DATASET_SPLIT_FILES
            else input_parent
        )
        with open(input_json) as f:
            tasks = json.load(f)
        dataset_split = Path(input_json).resolve().parent.name.lower()
        for task in tasks:
            task["_dataset_split"] = dataset_split
        for t in tasks:
            p = t.get("image_path", "")
            if p and not os.path.isabs(p):
                t["image_path"] = os.path.join(args.images_dir or image_base, p)
        if args.tasks:
            wanted = set(args.tasks)
            tasks = [t for t in tasks if t["task_id"] in wanted]
        if args.start_from:
            idx = next((i for i, t in enumerate(tasks) if t["task_id"] == args.start_from), 0)
            tasks = tasks[idx:]
        if args.num_shards > 1:
            tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
            print(f"[SANA_WM] shard {args.shard}/{args.num_shards}: {len(tasks)} tasks")
        if args.skip_done:
            before = len(tasks)
            tasks = [t for t in tasks if is_oe_insight_wait_only(t) or not (output_dir / t["task_id"] / f"{t['task_id']}.mp4").exists()]
            print(f"[SANA_WM] skip-done: {before} -> {len(tasks)} remaining")
        results = []
        suffix = f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
        results_path = output_dir / f"results_so_far{suffix}.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results = json.load(f)
            except Exception:
                results = []
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] [{task['task_id']}] {task.get('prompt', '')[:80]}")
            try:
                result = run_task(task, engine, output_dir, args)
            except Exception as e:
                print(f"  [{task['task_id']}] fatal: {e}, skipping")
                result = {"task_id": task["task_id"], "success": False, "error": str(e)}
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        sc = sum(1 for r in results if r.get("success"))
        print(f"\n[SANA_WM] [{cat_name}] DONE: {sc}/{len(results)}")


if __name__ == "__main__":
    main()
