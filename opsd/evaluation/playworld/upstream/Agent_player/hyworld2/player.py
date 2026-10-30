#!/usr/bin/env python3
"""Generate and control Tencent HY WorldPlay scenes with Playwright."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps
from playwright.sync_api import Frame, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

try:
    from .agent_ablation import (
        ablation_mode,
        align_action_sources,
        configured_agent,
        observation_package,
        request_agent_decision,
        validate_agent_config,
    )
except ImportError:  # Direct execution: python Agent_player/hyworld2/player.py
    from agent_ablation import (
        ablation_mode,
        align_action_sources,
        configured_agent,
        observation_package,
        request_agent_decision,
        validate_agent_config,
    )


HERE = Path(__file__).resolve().parent
WORLDPLAY_ROOT = HERE.parents[1]
DEFAULT_DATA_ROOT = WORLDPLAY_ROOT / "data"
DEFAULT_OUTPUT_ROOT = WORLDPLAY_ROOT / "outputs" / "hyworld2" / "runs"
DEFAULT_VIDEO_OUTPUT_ROOT = WORLDPLAY_ROOT / "outputs" / "hyworld2"
DEFAULT_URL = "https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay"

KEY_MAP = {
    "w": "KeyW",
    "a": "KeyA",
    "s": "KeyS",
    "d": "KeyD",
    "u": "ArrowUp",
    "j": "ArrowDown",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "←": "ArrowLeft",
    "→": "ArrowRight",
    "↑": "ArrowUp",
    "↓": "ArrowDown",
    "<-": "ArrowLeft",
    "e": "KeyE",
    "space": "Space",
    "jump": "Space",
}

AUTH_MARKERS = (
    "绑定手机号",
    "获取验证码",
    "扫码登录",
    "微信登录",
    "登录后体验",
    "请登录",
    "身份验证已过期",
    "重新登录",
)
GENERATING_MARKERS = ("排队中", "生成中", "预计还需", "请耐心等待")
FAIL_MARKERS = ("生成失败", "任务失败", "当前生成次数已达到上限", "账号异常")
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 2304
MIN_IMAGE_WIDTH = 1280
MIN_IMAGE_HEIGHT = 720
ACTION_UNIT_SECONDS = 1.5
ROTATION_EXTRA_SECONDS = 1.5
LS_ROTATION_HOLD_SECONDS = 50.0
ARROW_KEYS = {"ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}
TARGET_IMAGE_ASPECT = 16 / 9
CLAUDE_ADAPTIVE_KEY_MAP = {
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "←": "ArrowLeft",
    "→": "ArrowRight",
    "↑": "ArrowUp",
    "↓": "ArrowDown",
}

_CLAUDE_CLIENT: httpx.Client | None = None
AGENT_ABLATION_MODE = ablation_mode()


class ImageRejectedError(RuntimeError):
    """The site rejected an uploaded image under its content rules."""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def claude_headers(args: argparse.Namespace) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-api-key": args.claude_api_key,
        "x-ks-user-key": args.claude_user_key,
        "x-ks-llm-model": args.claude_model,
        "x-ks-biz-scene": args.claude_biz_scene,
    }


def claude_client(args: argparse.Namespace) -> httpx.Client:
    global _CLAUDE_CLIENT
    if _CLAUDE_CLIENT is None:
        _CLAUDE_CLIENT = httpx.Client(trust_env=args.claude_trust_env)
    return _CLAUDE_CLIENT


def post_claude(args: argparse.Namespace, payload: dict[str, Any], timeout_s: float) -> httpx.Response:
    return claude_client(args).post(
        f"{args.claude_base_url.rstrip('/')}/chat/completions",
        headers=claude_headers(args),
        json=payload,
        timeout=timeout_s,
    )


def extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Claude response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Claude response JSON is not an object")
    return parsed


def claude_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.claude_controller:
        return {"enabled": False, "reason": "disabled_by_argument"}
    missing = [
        name
        for name, value in (
            ("KIGRESS_BASE_URL", args.claude_base_url),
            ("KIGRESS_API_KEY", args.claude_api_key),
            ("KIGRESS_USER_KEY", args.claude_user_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("Claude controller missing configuration: " + ", ".join(missing))
    payload = {
        "model": args.claude_model,
        "max_completion_tokens": 12,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    }
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        try:
            response = post_claude(args, payload, args.claude_timeout_s)
            attempt_info = {"attempt": attempt, "http_status": response.status_code}
            attempts.append(attempt_info)
            if response.status_code < 400:
                content = extract_chat_text(response.json())
                if not content:
                    raise RuntimeError("Claude preflight returned empty assistant content")
                return {
                    "enabled": True,
                    "model": args.claude_model,
                    "base_url": args.claude_base_url,
                    "http_status": response.status_code,
                    "assistant_content": content[:80],
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "attempt_count": attempt,
                    "attempts": attempts,
                    "minimum_key_hold_ms": args.claude_min_hold_ms,
                    "adaptive_direction_hold_ms": 1000,
                }
            attempt_info["error"] = response.text[:300] or f"HTTP {response.status_code}"
            retryable = response.status_code >= 500 or response.status_code == 429
            if not retryable or attempt == 3:
                raise RuntimeError(f"Claude preflight HTTP {response.status_code}: {response.text[:300]}")
        except RuntimeError:
            raise
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": repr(exc)})
            if attempt == 3:
                raise RuntimeError(f"Claude preflight failed after 3 attempts: {exc!r}") from exc
        page_delay_s = attempt * 2
        time.sleep(page_delay_s)
    raise RuntimeError("Claude preflight failed")


def claude_viewer_screenshot(target: Any, destination: Path) -> bytes:
    raw = target.screenshot(type="jpeg", quality=65, timeout=5000)
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=45, optimize=True)
        compact = output.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(compact)
    return compact


def request_claude_hold_decision(
    args: argparse.Namespace,
    task: dict[str, Any],
    action: dict[str, Any],
    action_position: int,
    elapsed_hold_ms: int,
    planned_hold_ms: int,
    total_extension_ms: int,
    upcoming_actions: list[dict[str, Any]],
    image_bytes: bytes,
) -> dict[str, Any]:
    started = time.perf_counter()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    upcoming = [item.get("raw") for item in upcoming_actions[:5]]
    is_rotation_hold = any(key in ARROW_KEYS for key in action.get("keys", []))
    rotation_policy = (
        "This is an arrow-key rotation action. Similar-looking objects can reappear before a full 360-degree "
        "rotation, so never request early release; the runner must complete its fixed padded duration. "
        if is_rotation_hold
        else ""
    )
    prompt = (
        "You are the visual closed-loop controller for a Tencent HY WorldPlay benchmark. "
        "The runner is currently physically holding the stated keyboard key and will keep holding while you answer. "
        "Use the screenshot and task goal to decide conservatively whether to release early, extend the same hold, "
        "or add one small arrow-key camera correction after release. A key must never be released before it has been "
        f"held for {args.claude_min_release_ms}ms in this run. Before that threshold, release_now MUST be false. "
        "Do not replace the dataset plan or add movement keys. Adaptive correction may only be one "
        "of left/right/up/down and it will be held for exactly 1000ms. Use none when unnecessary. Extension is for "
        "insufficient progress of the SAME current key; release_now is for clear completion or likely overshoot. "
        "For spatial-entry goals, being beside, near, or aligned with the target boundary is NOT completion. "
        "Do not release while the character's feet/body are visibly on dry ground or outside the target region. "
        "Only declare completion when the character is unambiguously inside the requested region. "
        "At or after the original planned hold, if the goal is not unambiguously complete, request another "
        f"extension of up to {args.claude_max_extension_ms}ms for the same key. Repeat this on later checks when "
        f"progress is still insufficient. If uncertain, extend rather than release. {rotation_policy}\n\n"
        f"Task id: {task.get('task_id')}\n"
        f"Environment: {task.get('image_caption', '')}\n"
        f"Goal: {task.get('prompt', '')}\n"
        f"Perspective: {task.get('perspective', '')}\n"
        f"Original action sequence: {task.get('action', '')}\n"
        f"Current action index: {action_position}\n"
        f"Current held key(s): {action.get('keys')} ({action.get('raw')})\n"
        f"Elapsed hold: {elapsed_hold_ms}ms; original planned hold: {planned_hold_ms}ms; "
        f"extension already granted: {total_extension_ms}ms.\n"
        f"Upcoming dataset actions: {upcoming}\n\n"
        "Return only strict JSON: "
        '{"release_now":false,"extend_ms":0,"adaptive_direction":"none","reason":"short reason"}'
    )
    payload = {
        "model": args.claude_model,
        "max_completion_tokens": 60,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
    }
    info: dict[str, Any] = {
        "enabled": True,
        "model": args.claude_model,
        "action_index": action_position,
        "action": action.get("raw"),
        "elapsed_hold_ms": elapsed_hold_ms,
        "planned_hold_ms": planned_hold_ms,
        "release_now": False,
        "extend_ms": 0,
        "adaptive_direction": None,
    }
    try:
        response = post_claude(args, payload, args.claude_timeout_s)
        info["http_status"] = response.status_code
        info["wall_elapsed_s"] = round(time.perf_counter() - started, 3)
        if response.status_code >= 400:
            info["error"] = response.text[:500] or f"HTTP {response.status_code}"
            return info
        raw = extract_chat_text(response.json())
        info["raw_response"] = raw[:1000]
        parsed = extract_json_object(raw)
        info["release_now"] = bool(parsed.get("release_now"))
        try:
            requested_extension = int(float(parsed.get("extend_ms") or 0))
        except (TypeError, ValueError):
            requested_extension = 0
        info["extend_ms"] = max(0, min(requested_extension, args.claude_max_extension_ms))
        direction = str(parsed.get("adaptive_direction") or "none").strip().lower()
        info["adaptive_direction"] = CLAUDE_ADAPTIVE_KEY_MAP.get(direction)
        info["reason"] = str(parsed.get("reason") or "")[:500]
        return info
    except Exception as exc:
        info["wall_elapsed_s"] = round(time.perf_counter() - started, 3)
        info["error"] = repr(exc)
        return info


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def body_text(page: Page) -> str:
    try:
        return page.locator("body").inner_text(timeout=3000)
    except Exception:
        return ""


def auth_reason(page: Page) -> str | None:
    text = body_text(page)
    if any(part in page.url for part in ("/scan", "/login", "/verification")):
        return f"认证页: {page.url}"
    return next((marker for marker in AUTH_MARKERS if marker in text), None)


def load_task(data_root: Path, task_id: str) -> dict[str, Any]:
    group = re.match(r"[A-Za-z]+", task_id)
    candidates = []
    if group:
        group_name = group.group(0).upper()
        candidates.append(data_root / f"{group_name}.json")
        candidates.extend(sorted((data_root / group_name).glob("*.json")))
        if group_name == "GC":
            candidates.append(data_root / "gc" / "data.json")
        elif group_name == "IF":
            candidates.append(data_root / "if" / "data.json")
        elif group_name == "OE":
            candidates.extend(
                (data_root / split / "data.json")
                for split in ("insight", "outsight")
            )
    candidates.extend(data_root / name for name in ("GC.json", "OE.json", "IF.json", "GC_landscape.json"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item.get("task_id") == task_id:
                task = dict(item)
                task["source_file"] = str(path)
                image_path = Path(task["image_path"])
                if not image_path.is_absolute():
                    image_path = data_root / image_path
                task["resolved_image_path"] = str(image_path.resolve())
                if not image_path.exists():
                    raise FileNotFoundError(f"任务图片不存在: {image_path}")
                return task
    raise ValueError(f"在 PlayWorld 数据文件中找不到任务: {task_id}")


def constrain_source_image(image_path: Path) -> dict[str, Any]:
    """Center-crop to 16:9, resize if needed, and replace the source in place."""
    with Image.open(image_path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_size = image.size
        original_aspect = image.width / image.height
        crop_box = [0, 0, image.width, image.height]
        if abs(original_aspect - TARGET_IMAGE_ASPECT) > 1 / max(image.width, image.height):
            if original_aspect > TARGET_IMAGE_ASPECT:
                crop_width = max(2, round(image.height * TARGET_IMAGE_ASPECT))
                left = max(0, (image.width - crop_width) // 2)
                crop_box = [left, 0, left + crop_width, image.height]
            else:
                crop_height = max(2, round(image.width / TARGET_IMAGE_ASPECT))
                top = max(0, (image.height - crop_height) // 2)
                crop_box = [0, top, image.width, top + crop_height]
            image = image.crop(tuple(crop_box))
        cropped = crop_box != [0, 0, original_size[0], original_size[1]]
        pre_resize_size = image.size
        image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        downscaled = image.size != pre_resize_size
        if image.width < MIN_IMAGE_WIDTH or image.height < MIN_IMAGE_HEIGHT:
            image = image.resize((MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            upscaled = True
        else:
            upscaled = False
        final_size = image.size
        resized = downscaled or upscaled
        if not cropped and not resized:
            return {
                "cropped_to_16_9": False,
                "resized": False,
                "upscaled_to_minimum": False,
                "original_size": list(original_size),
                "final_size": list(final_size),
                "final_aspect": round(final_size[0] / final_size[1], 6),
            }
        temporary = image_path.with_name(f".{image_path.stem}.resizing{image_path.suffix}")
        save_format = "JPEG" if image_path.suffix.lower() in {".jpg", ".jpeg"} else source.format
        save_options = {"quality": 95, "optimize": True} if save_format == "JPEG" else {}
        image.save(temporary, format=save_format, **save_options)
    temporary.replace(image_path)
    return {
        "cropped_to_16_9": cropped,
        "crop_box": crop_box,
        "resized": resized,
        "downscaled_to_maximum": downscaled,
        "upscaled_to_minimum": upscaled,
        "original_size": list(original_size),
        "pre_resize_size": list(pre_resize_size),
        "final_size": list(final_size),
        "original_aspect": round(original_aspect, 6),
        "final_aspect": round(final_size[0] / final_size[1], 6),
        "source_replaced": True,
    }


def parse_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    task_id = str(task.get("task_id") or "")
    action_field = str(task.get("action") or "").strip()
    if action_field:
        actions: list[dict[str, Any]] = []
        for index, expression in enumerate(re.split(r"\s*->\s*", action_field)):
            expression = expression.strip()
            if not expression:
                continue
            repeat = re.fullmatch(r"(.+?)\s*\*\s*([0-9.]+)\s*(ms|s)?", expression, re.I)
            token = (repeat.group(1) if repeat else expression).strip()
            if repeat:
                value = float(repeat.group(2))
                explicit_unit = (repeat.group(3) or "").lower()
                if explicit_unit == "ms":
                    duration_ms = round(value)
                elif explicit_unit == "s":
                    duration_ms = round(value * 1000)
                else:
                    # Dataset action counts use a 1.5-second unit: w*3 = 4.5s.
                    duration_ms = round(value * ACTION_UNIT_SECONDS * 1000)
            else:
                duration_ms = round(ACTION_UNIT_SECONDS * 1000)
            normalized = token.lower()
            if normalized in {"wait", "wait and observe"}:
                actions.append({"type": "wait", "duration_ms": duration_ms, "raw": expression, "index": index})
                continue
            interact = re.fullmatch(r"interact\((.*)\)", normalized)
            if interact:
                intent = interact.group(1)
                key = "Space" if "jump" in intent or "climb" in intent else "KeyE"
            else:
                if normalized not in KEY_MAP:
                    raise ValueError(f"不支持的 action token: {token}")
                key = KEY_MAP[normalized]
            base_duration_ms = duration_ms
            if key in ARROW_KEYS and task_id.startswith("LS"):
                duration_ms = round(LS_ROTATION_HOLD_SECONDS * 1000)
                rotation_extra_ms = duration_ms - base_duration_ms
            else:
                rotation_extra_ms = round(ROTATION_EXTRA_SECONDS * 1000) if key in ARROW_KEYS else 0
                duration_ms += rotation_extra_ms
            actions.append(
                {
                    "type": "hold",
                    "keys": [key],
                    "duration_ms": duration_ms,
                    "base_duration_ms": base_duration_ms,
                    "rotation_extra_ms": rotation_extra_ms,
                    "raw": expression,
                    "index": index,
                    "source": (
                        "action_field_ls_fixed_50s_rotation"
                        if key in ARROW_KEYS and task_id.startswith("LS")
                        else "action_field_1_5s_unit_with_fixed_rotation_extra"
                    ),
                }
            )
        return actions

    steps = task.get("action_sequence_steps") or []
    if not steps and task.get("action_sequence"):
        steps = [part.strip() for part in task["action_sequence"].split(";") if part.strip()]
    actions: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        raw = str(step).strip()
        wait = re.fullmatch(r"wait\(\s*([0-9.]+)\s*(ms|s)?\s*\)", raw, re.I)
        if wait:
            value = float(wait.group(1))
            duration_ms = round(value * 1000 if (wait.group(2) or "ms").lower() == "s" else value)
            actions.append({"type": "wait", "duration_ms": duration_ms, "raw": raw, "index": index})
            continue
        hold = re.fullmatch(r"hold\(\s*([^,]+?)\s*,\s*([0-9.]+)\s*(ms|s)?\s*\)", raw, re.I)
        if not hold:
            raise ValueError(f"不支持的 action_sequence_steps 格式: {raw}")
        value = float(hold.group(2))
        duration_ms = round(value * 1000 if (hold.group(3) or "ms").lower() == "s" else value)
        keys = []
        for token in re.split(r"[+&]", hold.group(1)):
            normalized = token.strip().lower()
            if normalized not in KEY_MAP:
                raise ValueError(f"不支持的按键: {token} ({raw})")
            keys.append(KEY_MAP[normalized])
        base_duration_ms = duration_ms
        is_rotation = any(key in ARROW_KEYS for key in keys)
        if is_rotation and task_id.startswith("LS"):
            target_duration_ms = round(LS_ROTATION_HOLD_SECONDS * 1000)
            rotation_extra_ms = target_duration_ms - base_duration_ms
        else:
            rotation_extra_ms = round(ROTATION_EXTRA_SECONDS * 1000) if is_rotation else 0
        actions.append(
            {
                "type": "hold",
                "keys": keys,
                "duration_ms": duration_ms + rotation_extra_ms,
                "base_duration_ms": base_duration_ms,
                "rotation_extra_ms": rotation_extra_ms,
                "raw": raw,
                "index": index,
            }
        )
    return actions


def click_first(page: Page, labels: tuple[str, ...], timeout_ms: int = 1500) -> str | None:
    for label in labels:
        for locator in (
            page.get_by_role("button", name=label, exact=True),
            page.get_by_text(label, exact=True),
            page.locator("button,[role=button]").filter(has_text=label),
        ):
            try:
                if locator.count() and locator.first.is_visible(timeout=300):
                    locator.first.click(timeout=timeout_ms)
                    return label
            except Exception:
                pass
    return None


def dismiss_dialogs(page: Page) -> list[str]:
    clicked = []
    for labels in (("我同意", "同意", "I agree"), ("知道了", "我知道了", "Got it")):
        label = click_first(page, labels, 1000)
        if label:
            clicked.append(label)
            page.wait_for_timeout(500)
    return clicked


def leave_creation_asset_view(page: Page) -> list[str]:
    """Return from a creation-asset detail page before opening image-to-scene."""
    clicked = []
    for _ in range(3):
        has_input_panel = page.locator("textarea, input[type=file]").count() > 0
        if has_input_panel:
            break
        label = click_first(page, ("返回", "Back"), 3000)
        if label:
            clicked.append(label)
            page.wait_for_timeout(2000)
            continue
        body_text = page.locator("body").inner_text(timeout=3000)
        if "生成完成，点击可查看结果" not in body_text:
            break
        page.reload(wait_until="domcontentloaded", timeout=60000)
        clicked.append("reload_asset_view")
        page.wait_for_timeout(5000)
    return clicked


def screenshot(page: Page, path: Path) -> None:
    try:
        page.screenshot(path=str(path), type="jpeg", quality=88, full_page=False, timeout=15000)
    except Exception as exc:
        print(f"截图失败 {path.name}: {exc!r}", flush=True)


def choose_image_mode(page: Page) -> str | None:
    # The query parameter selects WorldPlay; this tab chooses image-to-scene.
    # The bottom input panel is collapsed until its text field is focused.
    try:
        page.locator("textarea").last.click(force=True, timeout=3000)
        page.wait_for_timeout(500)
    except Exception:
        pass
    for selector in (
        '.row-tab .tab-item:has-text("图生场景")',
        '.tab-item:has-text("图生场景")',
    ):
        try:
            locator = page.locator(selector).last
            locator.wait_for(state="visible", timeout=3000)
            locator.click(timeout=3000)
            page.wait_for_timeout(500)
            return "图生场景"
        except Exception:
            pass
    return click_first(page, ("图生场景", "Image to Scene"), 3000)


def upload_image(page: Page, image_path: Path) -> dict[str, Any]:
    def verify_upload() -> None:
        page.wait_for_timeout(2000)
        failed = page.locator(".uploadSuccess.fail, .scene-image-upload .errorTips")
        for failure_index in range(failed.count()):
            failure = failed.nth(failure_index)
            try:
                if failure.is_visible(timeout=300):
                    message = (failure.inner_text(timeout=1000) or "图片上传被页面拒绝").strip()
                    raise ImageRejectedError(f"图片上传被页面拒绝: {message}")
            except PlaywrightTimeoutError:
                pass

    inputs = page.locator("input[type=file]")
    errors = []
    for index in range(inputs.count() - 1, -1, -1):
        locator = inputs.nth(index)
        try:
            accept = locator.get_attribute("accept") or ""
            if accept and "image" not in accept and not any(ext in accept.lower() for ext in ("jpg", "jpeg", "png", "webp")):
                continue
            locator.set_input_files(str(image_path), timeout=20000)
            verify_upload()
            return {"method": "file_input", "input_index": index, "accept": accept}
        except Exception as exc:
            errors.append({"input_index": index, "error": repr(exc)})

    upload = page.get_by_text("点击或拖拽图片至此处上传", exact=False).first
    try:
        with page.expect_file_chooser(timeout=5000) as chooser:
            upload.click(timeout=3000)
        chooser.value.set_files(str(image_path), timeout=20000)
        verify_upload()
        return {"method": "file_chooser"}
    except Exception as exc:
        errors.append({"file_chooser": repr(exc)})
    raise RuntimeError(f"找不到可用的图片上传控件: {errors}")


def fill_prompt(page: Page, prompt: str) -> dict[str, Any]:
    candidates = page.locator("textarea, [contenteditable=true], [role=textbox]")
    for index in range(candidates.count()):
        locator = candidates.nth(index)
        try:
            if not locator.is_visible(timeout=300):
                continue
            active = locator.evaluate(
                """el => {
                  const host = el.closest('.text-comp');
                  if (!host) return true;
                  const style = getComputedStyle(host);
                  return Number(style.opacity) > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                }"""
            )
            if not active:
                continue
            placeholder = locator.get_attribute("placeholder") or ""
            if placeholder and not any(x in placeholder for x in ("描述", "提示", "场景", "prompt", "Prompt")):
                continue
            locator.fill(prompt, timeout=3000)
            return {"filled": True, "index": index, "placeholder": placeholder}
        except Exception:
            try:
                locator.click(timeout=1000)
                page.keyboard.insert_text(prompt)
                return {"filled": True, "index": index, "method": "insert_text"}
            except Exception:
                pass
    return {"filled": False, "reason": "image-to-scene 页面无可见提示词输入框"}


def submit_generation(page: Page) -> str:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        rejected = page.locator(".uploadSuccess.fail, .scene-image-upload .errorTips")
        for index in range(rejected.count()):
            failure = rejected.nth(index)
            if failure.is_visible(timeout=300):
                message = (failure.inner_text(timeout=1000) or "图片上传被页面拒绝").strip()
                raise ImageRejectedError(f"图片上传被页面拒绝: {message}")
        button = page.locator(".scene-generate-btn").last
        try:
            classes = button.get_attribute("class") or ""
            enabled = button.is_visible(timeout=300) and button.get_attribute("disabled") is None and "t-is-disabled" not in classes
            if enabled:
                label = (button.inner_text(timeout=1000) or "实时生成").strip()
                button.click(timeout=5000)
                return label
        except Exception:
            pass
        page.wait_for_timeout(500)
    raise RuntimeError("图片上传后 60 秒内“实时生成”按钮未启用")


def large_visuals(frame: Frame) -> list[dict[str, float]]:
    try:
        return frame.locator("canvas,video").evaluate_all(
            """els => els.map((el, index) => {
              const r = el.getBoundingClientRect();
              const s = getComputedStyle(el);
              return {index, width:r.width, height:r.height, area:r.width*r.height,
                visible:s.display!=='none' && s.visibility!=='hidden' && r.width>240 && r.height>160};
            }).filter(x => x.visible).sort((a,b) => b.area-a.area)"""
        )
    except Exception:
        return []


def find_world_target(page: Page) -> tuple[Frame, Any] | None:
    best = None
    for frame in page.frames:
        # HY WorldPlay streams frames into an <img>, not a canvas/video.
        for selector in (".hy-world-play-viewer__img", ".hy-world-play-viewer"):
            locators = frame.locator(selector)
            for index in range(locators.count()):
                locator = locators.nth(index)
                try:
                    if not locator.is_visible(timeout=200):
                        continue
                    box = locator.bounding_box(timeout=500)
                    if not box or box["width"] < 240 or box["height"] < 160:
                        continue
                    area = box["width"] * box["height"]
                    candidate = (area, frame, locator)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                except Exception:
                    pass
        visuals = large_visuals(frame)
        if not visuals:
            continue
        candidate = (visuals[0]["area"], frame, frame.locator("canvas,video").nth(visuals[0]["index"]))
        if best is None or candidate[0] > best[0]:
            best = candidate
    return (best[1], best[2]) if best else None


def wait_for_world(page: Page, timeout_s: int, result: dict[str, Any]) -> tuple[Frame, Any]:
    deadline = time.monotonic() + timeout_s
    clicked_explore = False
    last_log = 0.0
    while time.monotonic() < deadline:
        reason = auth_reason(page)
        if reason:
            raise PermissionError(reason)
        rejected = page.locator(".uploadSuccess.fail, .scene-image-upload .errorTips")
        for index in range(rejected.count()):
            failure = rejected.nth(index)
            if failure.is_visible(timeout=300):
                message = (failure.inner_text(timeout=1000) or "图片上传被页面拒绝").strip()
                raise ImageRejectedError(f"图片上传被页面拒绝: {message}")
        text = body_text(page)
        failed = next((x for x in FAIL_MARKERS if x in text), None)
        if failed:
            raise RuntimeError(f"页面报告失败: {failed}")
        if not clicked_explore:
            label = click_first(page, ("探索", "Explore", "即刻体验"), 1500)
            if label:
                clicked_explore = True
                result["explore_button"] = label
                page.wait_for_timeout(1500)
        target = find_world_target(page)
        stream_connecting = "连接中" in text or "初始化成功" in text
        if target and "单次生成时长已达上限" not in text and not stream_connecting:
            try:
                box = target[1].bounding_box(timeout=1000)
                if box and box["width"] > 240 and box["height"] > 160:
                    return target
            except Exception:
                pass
        if time.monotonic() - last_log > 15:
            marker = next((x for x in GENERATING_MARKERS if x in text), "等待可交互画布")
            print(f"[{now_text()}] {marker} url={page.url}", flush=True)
            last_log = time.monotonic()
        page.wait_for_timeout(2000)
    raise TimeoutError(f"等待可交互世界超时 ({timeout_s}s)，最后页面: {page.url}")


def world_target_box(locator: Any) -> dict[str, float] | None:
    try:
        locator.scroll_into_view_if_needed(timeout=3000)
        return locator.bounding_box(timeout=3000)
    except Exception:
        return None


def world_has_ended(page: Page) -> bool:
    text = body_text(page)
    return "单次生成时长已达上限" in text or bool(re.search(r"01:00\s*/\s*01:00", text))


def wait_for_live_session_timer(page: Page, timeout_s: int = 300) -> dict[str, Any]:
    """Reject a stale 00:00 viewer and wait until the newly submitted world advances."""
    started = time.monotonic()
    deadline = started + timeout_s
    last_log = 0.0
    while time.monotonic() < deadline:
        reason = auth_reason(page)
        if reason:
            raise PermissionError(reason)
        text = body_text(page)
        timer = re.search(r"(\d{2}):(\d{2})\s*/\s*01:00", text)
        if timer:
            timer_seconds = int(timer.group(1)) * 60 + int(timer.group(2))
            if 1 <= timer_seconds < 60 and "单次生成时长已达上限" not in text:
                return {
                    "timer": timer.group(0),
                    "timer_seconds": timer_seconds,
                    "waited_ms": round((time.monotonic() - started) * 1000),
                    "started_at": now_text(),
                }
        if time.monotonic() - last_log >= 15:
            print(
                f"WAIT_LIVE_SESSION timer={timer.group(0) if timer else 'unknown'}",
                flush=True,
            )
            last_log = time.monotonic()
        page.wait_for_timeout(500)
    raise TimeoutError(f"等待新实时世界计时开始超时 ({timeout_s}s)")


def execute_adaptive_direction(page: Page, key: str, source_action: str) -> dict[str, Any]:
    """Physically hold one Claude-added arrow key for exactly one second."""
    started = time.monotonic()
    down_started = started
    repeats = 0
    ended = False
    page.keyboard.down(key)
    try:
        deadline = down_started + 1.0
        last_end_check = down_started
        while time.monotonic() < deadline:
            page.wait_for_timeout(min(90, max(1, round((deadline - time.monotonic()) * 1000))))
            if time.monotonic() < deadline:
                page.keyboard.down(key)
                repeats += 1
            if time.monotonic() - last_end_check >= 0.25:
                last_end_check = time.monotonic()
                if world_has_ended(page):
                    ended = True
                    break
    finally:
        page.keyboard.up(key)
    actual_ms = round((time.monotonic() - down_started) * 1000)
    return {
        "type": "hold",
        "keys": [key],
        "duration_ms": 1000,
        "raw": f"claude:{key}",
        "source": "claude_adaptive_direction",
        "claude_source_action": source_action,
        "started_at": now_text(),
        "actual_hold_ms": actual_ms,
        "actual_duration_ms": round((time.monotonic() - started) * 1000),
        "auto_repeat_keydown_count": repeats,
        "gap_before_keydown_ms": 0,
        "keyboard_events": [f"down:{key}", f"up:{key}"],
        "interrupted_by_natural_end": ended,
    }


def execute_agent_only_actions(
    page: Page,
    target: Any,
    task: dict[str, Any],
    args: argparse.Namespace,
    outdir: Path,
    result: dict[str, Any],
    config: Any,
) -> None:
    """Plan and execute from observations without accepting any base-action input."""
    max_decisions = max(1, int(os.environ.get("HYWORLD2_AGENT_MAX_DECISIONS", "16")))
    max_actions = max(1, int(os.environ.get("HYWORLD2_AGENT_MAX_ACTIONS", "3")))
    timeout_s = float(os.environ.get("HYWORLD2_AGENT_TIMEOUT_S", "60"))
    result["executed_actions"] = []
    result["agent_decisions"] = []
    result["observation_path"] = []
    executed_dsl: list[str] = []
    observation_dir = outdir / "agent_observations"
    initial_path = observation_dir / "observation_000.jpg"
    claude_viewer_screenshot(target, initial_path)
    result["observation_path"].append(str(initial_path))
    page.bring_to_front()
    target.evaluate(
        """element => { window.focus(); if (!element.hasAttribute('tabindex')) element.setAttribute('tabindex','-1'); element.focus({preventScroll:true}); }"""
    )
    client = httpx.Client(trust_env=config.trust_env, verify=config.verify_tls)
    try:
        for decision_index in range(max_decisions):
            if world_has_ended(page):
                break
            latest_path = initial_path
            if decision_index:
                latest_path = observation_dir / f"observation_{decision_index:03d}.jpg"
                claude_viewer_screenshot(target, latest_path)
                result["observation_path"].append(str(latest_path))
            package = observation_package(task.get("prompt", ""), initial_path, latest_path, executed_dsl)
            decision, messages = request_agent_decision(
                client,
                config,
                package,
                max_actions=max_actions,
                timeout_s=timeout_s,
            )
            semantic_request = {
                "objective": package["objective"],
                "initial_observation": package["initial_observation"],
                "latest_observation": package["latest_observation"],
                "executed_actions": package["executed_actions"],
            }
            audit = {
                "decision_index": decision_index,
                "base_sequence_exposed": False,
                "observation_package_fields": sorted(package),
                "observation_package": semantic_request,
                "request_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                **decision,
            }
            result["agent_decisions"].append(audit)
            if decision["status"] == "done":
                break
            for action in decision["actions"]:
                if world_has_ended(page):
                    break
                record = dict(action)
                record.update(
                    source="inserted",
                    observation_path=str(latest_path),
                    decision_latency_ms=decision["decision_latency_ms"],
                    started_at=now_text(),
                )
                started = time.monotonic()
                if action["type"] == "wait":
                    deadline = started + action["duration_ms"] / 1000
                    while time.monotonic() < deadline and not world_has_ended(page):
                        page.wait_for_timeout(min(100, max(1, round((deadline - time.monotonic()) * 1000))))
                    record["keyboard_events"] = []
                else:
                    key = KEY_MAP[action["keys"][0]]
                    repeats = 0
                    page.keyboard.down(key)
                    try:
                        deadline = started + action["duration_ms"] / 1000
                        while time.monotonic() < deadline and not world_has_ended(page):
                            page.wait_for_timeout(min(90, max(1, round((deadline - time.monotonic()) * 1000))))
                            if time.monotonic() < deadline:
                                page.keyboard.down(key)
                                repeats += 1
                    finally:
                        page.keyboard.up(key)
                    record["keys"] = [key]
                    record["auto_repeat_keydown_count"] = repeats
                    record["keyboard_events"] = [f"down:{key}", f"up:{key}"]
                    record["actual_hold_ms"] = round((time.monotonic() - started) * 1000)
                record["actual_duration_ms"] = round((time.monotonic() - started) * 1000)
                result["executed_actions"].append(record)
                executed_dsl.append(action["raw"])
                print(f"AGENT_ACTION {action['raw']} actual={record['actual_duration_ms']}ms", flush=True)
            atomic_json(outdir / "result.json", result)
    finally:
        client.close()
    result["executed_action_sequence"] = executed_dsl
    result["decision_latency_ms"] = [item["decision_latency_ms"] for item in result["agent_decisions"]]
    result["decision_latency_total_ms"] = sum(result["decision_latency_ms"])


def execute_actions(
    page: Page,
    target: Any,
    actions: list[dict[str, Any]],
    result: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    outdir: Path,
) -> None:
    result["executed_actions"] = []
    result["action_sequence_interrupted"] = False
    result["claude_decisions"] = []
    result["claude_predicted_action"] = []
    # Focus the actual viewer/canvas in its owning frame once. This matters for
    # ArrowRight/ArrowLeft/ArrowUp/ArrowDown because WorldPlay listens on the
    # viewer frame's window; focusing the top-level body can send them elsewhere.
    if any(action["type"] != "wait" for action in actions):
        page.bring_to_front()
        target.evaluate(
            """element => {
              window.focus();
              if (!element.hasAttribute('tabindex')) element.setAttribute('tabindex', '-1');
              element.focus({preventScroll: true});
            }"""
        )
        result["keyboard_focus"] = target.evaluate(
            "element => ({tag: element.tagName, focused: document.activeElement === element})"
        )
    previous_key_release: float | None = None
    executor = ThreadPoolExecutor(max_workers=1) if args.claude_controller else None
    try:
        for action_position, action in enumerate(actions):
            started = time.monotonic()
            record = dict(action)
            record["started_at"] = now_text()
            adaptive_directions: list[str] = []
            if action["type"] == "wait":
                # Important: wait means observation only. Never focus, call Claude,
                # or emit any key during the dataset wait itself.
                wait_deadline = time.monotonic() + action["duration_ms"] / 1000
                ended = False
                while True:
                    remaining_ms = max(0, round((wait_deadline - time.monotonic()) * 1000))
                    if remaining_ms <= 0:
                        break
                    page.wait_for_timeout(min(250, remaining_ms))
                    if world_has_ended(page):
                        ended = True
                        break
                record["keyboard_events"] = []
                record["claude_checks"] = []
            else:
                # HY WorldPlay listens for real document/window keyboard events.
                # Do not click its on-screen W/A/S/D or arrow controls.
                down_keys: list[str] = []
                ended = False
                stopped_by_claude = False
                key_down_started = time.monotonic()
                hold_started = key_down_started
                planned_hold_ms = max(args.claude_min_hold_ms, int(action["duration_ms"]))
                record["planned_duration_ms_before_claude"] = int(action["duration_ms"])
                record["gap_before_keydown_ms"] = (
                    None if previous_key_release is None else round((key_down_started - previous_key_release) * 1000)
                )
                decisions: list[dict[str, Any]] = []
                pending = None
                pending_screenshot: str | None = None
                checks_started = 0
                total_extension_ms = 0
                is_rotation_hold = any(key in ARROW_KEYS for key in action["keys"])
                record["rotation_early_release_disabled"] = is_rotation_hold
                # Start the first visual request immediately after keydown, as
                # in the Genie3 live controller. The key remains held while the
                # model responds, but an early-release decision is never applied
                # before the hard 1000ms minimum.
                next_check_at = hold_started
                check_spacing_s = max(
                    args.claude_check_interval_ms / 1000,
                    planned_hold_ms / 1000 / max(1, args.claude_max_checks),
                )
                try:
                    for key in action["keys"]:
                        page.keyboard.down(key)
                        down_keys.append(key)
                    hold_started = time.monotonic()
                    hold_deadline = hold_started + planned_hold_ms / 1000
                    maximum_deadline = hold_deadline + args.claude_max_total_extension_ms / 1000
                    repeat_keys = list(down_keys)
                    repeat_keydown_count = 0
                    last_end_check = hold_started
                    while True:
                        now = time.monotonic()
                        elapsed_hold_ms = round((now - hold_started) * 1000)

                        if pending is not None and pending.done():
                            decision = pending.result()
                            decision["screenshot"] = pending_screenshot
                            pending = None
                            pending_screenshot = None
                            decisions.append(decision)
                            result["claude_decisions"].append(decision)
                            release_requested = bool(decision.get("release_now"))
                            release_applied = release_requested and not is_rotation_hold
                            if release_requested and is_rotation_hold:
                                decision["release_ignored_for_fixed_rotation"] = True
                            if release_applied and elapsed_hold_ms < args.claude_min_release_ms:
                                release_applied = False
                                decision["release_deferred_until_ms"] = args.claude_min_release_ms
                            applied_extension = 0 if release_requested else min(
                                int(decision.get("extend_ms") or 0),
                                max(0, args.claude_max_total_extension_ms - total_extension_ms),
                            )
                            if applied_extension:
                                hold_deadline = min(maximum_deadline, hold_deadline + applied_extension / 1000)
                                total_extension_ms += applied_extension
                                decision["applied_extension_ms"] = applied_extension
                            adaptive_key = decision.get("adaptive_direction")
                            if (
                                adaptive_key
                                and adaptive_key not in adaptive_directions
                                and len(adaptive_directions) < args.claude_max_adaptive_actions
                            ):
                                adaptive_directions.append(adaptive_key)
                            if release_applied:
                                stopped_by_claude = True
                                decision["release_applied_at_ms"] = elapsed_hold_ms
                                break
                            next_check_at = hold_started + checks_started * check_spacing_s

                        remaining_ms = max(0, round((hold_deadline - time.monotonic()) * 1000))
                        if remaining_ms <= 0:
                            break
                        page.wait_for_timeout(min(90, remaining_ms))
                        now = time.monotonic()
                        if now < hold_deadline:
                            for key in repeat_keys:
                                page.keyboard.down(key)
                                repeat_keydown_count += 1

                        if (
                            executor is not None
                            and not is_rotation_hold
                            and pending is None
                            and checks_started < args.claude_max_checks
                            and now >= next_check_at
                        ):
                            shot_path = outdir / "claude" / f"action_{action_position:02d}_check_{checks_started:02d}.jpg"
                            try:
                                image_bytes = claude_viewer_screenshot(target, shot_path)
                                elapsed_for_request = round((time.monotonic() - hold_started) * 1000)
                                pending = executor.submit(
                                    request_claude_hold_decision,
                                    args,
                                    task,
                                    action,
                                    action_position,
                                    elapsed_for_request,
                                    planned_hold_ms,
                                    total_extension_ms,
                                    actions[action_position + 1 :],
                                    image_bytes,
                                )
                                pending_screenshot = str(shot_path)
                                checks_started += 1
                                next_check_at = hold_started + checks_started * check_spacing_s
                            except Exception as exc:
                                decisions.append({"enabled": True, "screenshot_error": repr(exc)})
                                checks_started += 1
                                next_check_at = hold_started + checks_started * check_spacing_s

                        if time.monotonic() - last_end_check >= 0.25:
                            last_end_check = time.monotonic()
                            if world_has_ended(page):
                                ended = True
                                break
                    record["auto_repeat_keydown_count"] = repeat_keydown_count
                finally:
                    for key in reversed(down_keys):
                        page.keyboard.up(key)
                    previous_key_release = time.monotonic()
                record["actual_hold_ms"] = round((previous_key_release - hold_started) * 1000)
                record["keyboard_events"] = [f"down:{x}" for x in down_keys] + [f"up:{x}" for x in reversed(down_keys)]
                record["claude_checks"] = decisions
                record["claude_release_early"] = stopped_by_claude
                record["claude_extension_ms"] = total_extension_ms
                record["claude_adaptive_directions"] = adaptive_directions
            record["actual_duration_ms"] = round((time.monotonic() - started) * 1000)
            result["executed_actions"].append(record)
            print(
                f"ACTION {record['raw']} actual={record['actual_duration_ms']}ms "
                f"claude_checks={len(record.get('claude_checks', []))}",
                flush=True,
            )

            if not ended:
                for adaptive_key in adaptive_directions:
                    adaptive_record = execute_adaptive_direction(page, adaptive_key, record["raw"])
                    result["executed_actions"].append(adaptive_record)
                    result["claude_predicted_action"].append(
                        {
                            "source_action": record["raw"],
                            "adaptive_direction": adaptive_key,
                            "duration_ms": 1000,
                            "applied": True,
                        }
                    )
                    print(f"CLAUDE_ADAPTIVE {adaptive_record['raw']} actual={adaptive_record['actual_duration_ms']}ms", flush=True)
                    if adaptive_record.get("interrupted_by_natural_end"):
                        ended = True
                        break

            for decision in record.get("claude_checks", []):
                prediction = {
                    "source_action": record.get("raw"),
                    "release_now": decision.get("release_now", False),
                    "extend_ms": decision.get("applied_extension_ms", 0),
                    "adaptive_direction": decision.get("adaptive_direction"),
                    "reason": decision.get("reason"),
                }
                if prediction not in result["claude_predicted_action"]:
                    result["claude_predicted_action"].append(prediction)

            if ended:
                record["interrupted_by_natural_end"] = True
                result["action_sequence_interrupted"] = True
                result["action_interruption_reason"] = "world_reached_01:00"
                result["unexecuted_actions"] = actions[action_position + 1 :]
                print("ACTION_SEQUENCE_INTERRUPTED world_reached_01:00", flush=True)
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def wait_for_natural_world_end(page: Page, timeout_s: int = 90) -> dict[str, Any]:
    """Keep the WorldPlay session open until its one-minute runtime ends naturally."""
    started = time.monotonic()
    deadline = started + timeout_s
    last_log = 0.0
    max_timer_seconds = -1
    while time.monotonic() < deadline:
        text = body_text(page)
        if "单次生成时长已达上限" in text or re.search(r"01:00\s*/\s*01:00", text):
            return {
                "natural_end": True,
                "matched": "单次生成时长已达上限" if "单次生成时长已达上限" in text else "01:00/01:00",
                "waited_ms": round((time.monotonic() - started) * 1000),
                "ended_at": now_text(),
            }
        timer = re.search(r"(\d{2}):(\d{2})\s*/\s*01:00", text)
        if timer:
            timer_seconds = int(timer.group(1)) * 60 + int(timer.group(2))
            # Tencent may replace the finished live viewer with the next/current
            # asset immediately, resetting the visible timer to 00:00 before we
            # can observe 01:00. A reset after meaningful progress is therefore
            # the natural end of the session we were recording.
            if max_timer_seconds >= 10 and timer_seconds <= 2 and timer_seconds < max_timer_seconds - 5:
                return {
                    "natural_end": True,
                    "matched": "timer_reset_after_asset_switch",
                    "max_timer_seconds": max_timer_seconds,
                    "reset_timer_seconds": timer_seconds,
                    "waited_ms": round((time.monotonic() - started) * 1000),
                    "ended_at": now_text(),
                }
            max_timer_seconds = max(max_timer_seconds, timer_seconds)
        if time.monotonic() - last_log >= 15:
            print(f"WAIT_NATURAL_END timer={timer.group(0) if timer else 'unknown'}", flush=True)
            last_log = time.monotonic()
        page.wait_for_timeout(1000)
    raise TimeoutError(f"等待实时世界自然结束超时 ({timeout_s}s)")


def download_native_video(page: Page, destination: Path, not_before_epoch: int | None = None) -> dict[str, Any]:
    """Download the site's official clean MP4 variant through its download menu."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    selected_asset: dict[str, Any] = {}
    try:
        if not_before_epoch is not None:
            current_creation = None
            for poll_index in range(6):
                response = page.request.post(
                    "https://3d.hunyuan.tencent.com/api/3d/creations/list",
                    data={
                        "limit": 20,
                        "offset": 0,
                        "sceneTypeList": ["interaction"],
                        "modelTypeList": ["worldPlay"],
                    },
                    timeout=15000,
                )
                if not response.ok:
                    raise RuntimeError(f"查询腾讯创作资产失败: HTTP {response.status}")
                creations = response.json().get("creations") or []
                candidates = [
                    creation
                    for creation in creations
                    if int(creation.get("createdAt") or 0) >= not_before_epoch - 5
                    and creation.get("status") == "success"
                    and creation.get("result")
                    and ((creation["result"][0].get("urlResult") or {}).get("videoInfo") or {}).get("video_url")
                ]
                if candidates:
                    current_creation = max(candidates, key=lambda creation: int(creation.get("createdAt") or 0))
                    break
                if poll_index < 5:
                    page.wait_for_timeout(2000)
            if current_creation is None:
                raise RuntimeError(
                    "腾讯创作资产列表中没有本次会话的新 WorldPlay 资产；拒绝下载旧资产作为 native.mp4"
                )
            result_item = current_creation["result"][0]
            video_info = (result_item.get("urlResult") or {}).get("videoInfo") or {}
            direct_video_url = video_info["video_url"]
            with httpx.stream("GET", direct_video_url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            if temporary.stat().st_size < 1024:
                raise RuntimeError(f"当前创作资产视频文件过小: {temporary.stat().st_size} bytes")
            temporary.replace(destination)
            return {
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "downloaded_at": now_text(),
                "source_type": "official_creations_api_current_session",
                "variant": "mp4（纯享版）",
                "creation_id": current_creation.get("id"),
                "creation_created_at": current_creation.get("createdAt"),
                "asset_id": result_item.get("assetId"),
            }

        # Explicitly select the newest/current item in the right-side 创作资产
        # panel. Do this even when a download menu is already visible, because
        # that menu may still be bound to the previously viewed creation.
        panel = page.locator(".display-panel").filter(has_text="创作资产").last
        panel.wait_for(state="visible", timeout=15000)
        newest_asset = panel.locator(".assets-list .list-item").first
        newest_asset.wait_for(state="visible", timeout=15000)
        selected_asset = {
            "index": 0,
            "thumbnail_url": newest_asset.locator("img").first.get_attribute("src"),
            "class_before_click": newest_asset.get_attribute("class"),
        }
        newest_asset.click(timeout=5000)
        page.wait_for_timeout(2000)
        selected_asset["class_after_click"] = newest_asset.get_attribute("class")

        menu = page.locator(".download-assets-select").last
        if not menu.count() or not menu.is_visible(timeout=500):
            raise RuntimeError("已选择当前创作资产，但下载菜单未出现")
        menu.wait_for(state="visible", timeout=15000)
        option = page.locator(".t-select-option").filter(has_text="mp4（纯享版）").last
        download = None
        download_attempts = 0
        last_download_error: Exception | None = None
        for download_attempts in range(1, 2):
            if not option.is_visible(timeout=500):
                menu.click(timeout=5000)
            option.wait_for(state="visible", timeout=5000)
            try:
                with page.expect_download(timeout=10000) as download_info:
                    option.click(timeout=5000, force=True)
                download = download_info.value
                break
            except PlaywrightTimeoutError as exc:
                last_download_error = exc
                print(f"DOWNLOAD_DIRECT_FALLBACK after_menu_attempt={download_attempts}", flush=True)
                page.wait_for_timeout(500)
        if download is None:
            direct_video_url = page.evaluate(
                r"""() => {
                    const urls = [...document.querySelectorAll('video')]
                        .flatMap(el => [el.currentSrc, el.src])
                        .filter(Boolean);
                    return urls.find(url => /\.mp4(?:\?|$)/i.test(url)) || null;
                }"""
            )
            if direct_video_url:
                with httpx.stream("GET", direct_video_url, follow_redirects=True, timeout=120) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
                if temporary.stat().st_size < 1024:
                    raise RuntimeError(f"直链下载到的视频文件过小: {temporary.stat().st_size} bytes")
                temporary.replace(destination)
                return {
                    "path": str(destination),
                    "size_bytes": destination.stat().st_size,
                    "downloaded_at": now_text(),
                    "source_type": "official_video_element_direct_fallback",
                    "variant": "mp4（纯享版）",
                    "download_attempts": download_attempts,
                    "selected_creation_asset": selected_asset,
                }
            diagnostics = page.evaluate(
                """() => {
                    const values = [];
                    for (const el of document.querySelectorAll('video,source,a')) {
                        for (const key of ['src', 'currentSrc', 'href', 'download']) {
                            const value = el[key] || el.getAttribute?.(key);
                            if (value) values.push({tag: el.tagName, key, value});
                        }
                    }
                    const resources = performance.getEntriesByType('resource')
                        .map(e => ({name: e.name, initiatorType: e.initiatorType, duration: e.duration}))
                        .filter(e => /mp4|m3u8|video|download|media|vod/i.test(e.name) || e.initiatorType === 'video');
                    return {dom: values, resources: resources.slice(-500), capturedAt: new Date().toISOString()};
                }"""
            )
            (destination.parent / "download_debug.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise last_download_error or RuntimeError("腾讯原生视频下载未触发")
        download.save_as(str(temporary))
        if temporary.stat().st_size < 1024:
            raise RuntimeError(f"下载到的视频文件过小: {temporary.stat().st_size} bytes")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "downloaded_at": now_text(),
        "source_type": "official_download_menu",
        "variant": "mp4（纯享版）",
        "download_attempts": download_attempts,
        "selected_creation_asset": selected_asset,
    }


def start_full_session_recording(page: Page, frame_dir: Path) -> dict[str, Any]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.jpg"):
        old_frame.unlink()
    session = page.context.new_cdp_session(page)
    state: dict[str, Any] = {
        "session": session,
        "frame_dir": frame_dir,
        "count": 0,
        "started_monotonic": time.monotonic(),
        "started_at": now_text(),
    }

    def on_frame(event: dict[str, Any]) -> None:
        try:
            index = state["count"]
            (frame_dir / f"frame_{index:06d}.jpg").write_bytes(base64.b64decode(event["data"]))
            state["count"] = index + 1
            session.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
        except Exception as exc:
            state.setdefault("errors", []).append(repr(exc))

    state["handler"] = on_frame
    session.on("Page.screencastFrame", on_frame)
    session.send(
        "Page.startScreencast",
        {
            "format": "jpeg",
            "quality": 85,
            "maxWidth": 1200,
            "maxHeight": 823,
            "everyNthFrame": 2,
        },
    )
    return state


def stop_and_encode_recording(
    page: Page,
    state: dict[str, Any],
    world_box: dict[str, float],
    destination: Path,
) -> dict[str, Any]:
    session = state["session"]
    try:
        session.send("Page.stopScreencast")
        page.wait_for_timeout(500)
    finally:
        try:
            session.detach()
        except Exception:
            pass
    duration_s = max(0.1, time.monotonic() - state["started_monotonic"])
    frame_dir: Path = state["frame_dir"]
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        raise RuntimeError(f"完整录制帧数不足: {len(frames)}")

    with Image.open(frames[0]) as first_frame:
        frame_width, frame_height = first_frame.size
    viewport = page.evaluate("() => ({width: innerWidth, height: innerHeight})")
    scale_x = frame_width / viewport["width"]
    scale_y = frame_height / viewport["height"]
    crop_x = max(0, round(world_box["x"] * scale_x))
    crop_y = max(0, round(world_box["y"] * scale_y))
    crop_w = min(frame_width - crop_x, round(world_box["width"] * scale_x)) // 2 * 2
    crop_h = min(frame_height - crop_y, round(world_box["height"] * scale_y)) // 2 * 2
    # Preserve wall-clock duration. CDP may emit more than 30 frames/second;
    # capping the input rate would incorrectly slow a 60s session to ~96s.
    fps = max(1.0, min(60.0, len(frames) / duration_s))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part.mp4")
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:.6f}",
        "-i", str(frame_dir / "frame_%06d.jpg"),
        # CDP may change JPEG dimensions by one or two pixels while browser UI
        # geometry settles. Normalize every frame to the first frame's exact
        # dimensions before applying the viewer crop. Keeping the aspect-ratio
        # option here can round an odd target width up by one pixel, making the
        # following fixed-size pad smaller than the scaled frame.
        "-vf", (
            f"scale={frame_width}:{frame_height},"
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, timeout=300)
        if temporary.stat().st_size < 1024:
            raise RuntimeError("编码后的完整视频文件过小")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(frame_dir, ignore_errors=True)
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "frame_count": len(frames),
        "recording_wall_duration_s": round(duration_s, 3),
        "encoding_fps": round(fps, 6),
        "crop": [crop_x, crop_y, crop_w, crop_h],
        "recorded_at": now_text(),
        "source_type": "cdp_full_session_screencast",
    }


def probe(page: Page, outdir: Path) -> dict[str, Any]:
    text = body_text(page)
    screenshot(page, outdir / "probe.jpg")
    return {
        "url": page.url,
        "title": page.title(),
        "auth_reason": auth_reason(page),
        "body_text": text[:6000],
        "buttons": page.locator("button").all_inner_texts(),
        "file_inputs": page.locator("input[type=file]").count(),
        "visible_visuals": sum(len(large_visuals(frame)) for frame in page.frames),
    }


def run(args: argparse.Namespace) -> int:
    if AGENT_ABLATION_MODE in {"preset_only", "agent_only"}:
        # These modes use separate action loops from the preset-aware controller.
        args.claude_controller = False
    if AGENT_ABLATION_MODE in {"preset_only", "agent_only"} and args.action_override is not None:
        raise ValueError(f"--action-override is forbidden in {AGENT_ABLATION_MODE}")
    args.claude_min_hold_ms = max(1000, int(args.claude_min_hold_ms))
    args.claude_min_release_ms = max(
        args.claude_min_hold_ms,
        int(args.claude_min_release_ms),
    )
    args.claude_max_extension_ms = max(0, int(args.claude_max_extension_ms))
    args.claude_max_total_extension_ms = max(
        args.claude_max_extension_ms,
        int(args.claude_max_total_extension_ms),
    )
    args.claude_check_interval_ms = max(250, int(args.claude_check_interval_ms))
    args.claude_max_checks = max(0, int(args.claude_max_checks))
    args.claude_max_adaptive_actions = max(0, int(args.claude_max_adaptive_actions))
    data_root = Path(args.data_root).expanduser().resolve()
    task = load_task(data_root, args.task_id)
    original_action = task.get("action")
    if args.action_override is not None:
        task["action"] = args.action_override
    run_id = args.run_id or f"{task['task_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = Path(args.output_root).expanduser().resolve() / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    original_image_path = Path(task["resolved_image_path"])
    working_image_path = outdir / f"{task['task_id']}_input{original_image_path.suffix}"
    shutil.copy2(original_image_path, working_image_path)
    task["resolved_image_path"] = str(working_image_path)
    source_image_info = constrain_source_image(working_image_path)
    actions = parse_actions(task)
    agent_config = configured_agent() if AGENT_ABLATION_MODE == "agent_only" else None
    if agent_config is not None:
        validate_agent_config(agent_config)
    result_path = outdir / "result.json"
    public_task_dir = Path(args.video_output_root).expanduser().resolve() / task["task_id"]
    public_task_dir.mkdir(parents=True, exist_ok=True)
    public_result_path = public_task_dir / "result.json"
    result: dict[str, Any] = {
        "status": "started",
        "started_at": now_text(),
        "run_id": run_id,
        "task_id": task["task_id"],
        "source_file": task["source_file"],
        "original_image_path": str(original_image_path),
        "image_path": task["resolved_image_path"],
        "source_image_constraint": source_image_info,
        "prompt": task.get("prompt", ""),
        "action": task.get("action"),
        "original_action": original_action,
        "ablation_mode": AGENT_ABLATION_MODE or "preset_agent",
        "agent_provider": agent_config.provider if agent_config else (
            "claude" if AGENT_ABLATION_MODE == "preset_agent" and args.claude_controller else None
        ),
        "agent_model": agent_config.model if agent_config else (
            args.claude_model if AGENT_ABLATION_MODE == "preset_agent" and args.claude_controller else None
        ),
        "original_action_sequence": original_action,
        "executed_action_sequence": [],
        "action_provenance": [],
        "deleted_actions": [],
        "observation_path": [],
        "decision_latency_ms": [],
        "action_override": args.action_override,
        "claude_predicted_action": [],
        "claude_controller": {
            "enabled": args.claude_controller,
            "model": args.claude_model,
            "base_url": args.claude_base_url,
            "minimum_key_hold_ms": args.claude_min_hold_ms,
            "minimum_release_ms": args.claude_min_release_ms,
            "credentials_recorded": False,
        },
        "action_sequence_steps": task.get("action_sequence_steps", []),
        "parsed_actions": actions,
        "cdp_url": args.cdp_url,
        "target_url": args.url,
        "output_dir": str(outdir),
        "video_source": args.video_source,
    }
    atomic_json(result_path, result)

    if args.dry_run:
        result.update(status="dry_run_ok", finished_at=now_text())
        atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(args.cdp_url)
        if not browser.contexts:
            raise RuntimeError("CDP 浏览器没有可用 context；请用 launch_chrome_cdp.sh 启动")
        context = browser.contexts[0]
        page = next((p for p in context.pages if "3d.hunyuan.tencent.com" in p.url), None) or context.new_page()
        page.bring_to_front()
        try:
            if not page.url.startswith("https://3d.hunyuan.tencent.com/") or "/scan" in page.url:
                page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)
            dismiss_dialogs(page)
            result["initial_probe"] = probe(page, outdir)
            reason = auth_reason(page)
            if reason:
                result.update(status="auth_required", auth_reason=reason, page_url=page.url)
                print(f"AUTH_REQUIRED: {reason}", flush=True)
                return 3
            if args.probe_only:
                result.update(status="probe_ok", page_url=page.url)
                return 0

            result["asset_view_recovery"] = leave_creation_asset_view(page)
            result["claude_controller_preflight"] = claude_preflight(args)
            result["claude_controller"].update(result["claude_controller_preflight"])
            atomic_json(result_path, result)
            if args.claude_controller:
                print(
                    f"CLAUDE_PREFLIGHT model={args.claude_model} "
                    f"http={result['claude_controller_preflight'].get('http_status')} ",
                    flush=True,
                )

            result["selected_mode"] = choose_image_mode(page)
            page.wait_for_timeout(700)
            result["upload"] = upload_image(page, Path(task["resolved_image_path"]))
            result["prompt_fill"] = fill_prompt(page, task.get("prompt", ""))
            screenshot(page, outdir / "after_upload.jpg")
            result["submit_button"] = submit_generation(page)
            result["submitted_at"] = now_text()
            atomic_json(result_path, result)

            _, target = wait_for_world(page, args.generation_timeout, result)
            result["live_session_started"] = wait_for_live_session_timer(
                page,
                min(args.generation_timeout, 300),
            )
            refreshed_target = find_world_target(page)
            if refreshed_target is not None:
                _, target = refreshed_target
            result["world_ready_at"] = now_text()
            result["world_url"] = page.url
            result["world_box"] = world_target_box(target)
            if not result["world_box"]:
                raise RuntimeError("无法取得实时世界画面区域")
            recording = (
                start_full_session_recording(page, outdir / "screencast_frames")
                if args.video_source in ("recording", "both")
                else None
            )
            try:
                screenshot(page, outdir / "before_actions.jpg")
                page.wait_for_timeout(args.initial_wait_ms)
                if AGENT_ABLATION_MODE == "agent_only":
                    execute_agent_only_actions(page, target, task, args, outdir, result, agent_config)
                else:
                    execute_actions(page, target, actions, result, task, args, outdir)
                page.wait_for_timeout(args.post_wait_ms)
                screenshot(page, outdir / "after_actions.jpg")
                result["natural_world_end"] = wait_for_natural_world_end(page, args.natural_end_timeout)
                screenshot(page, outdir / "natural_end.jpg")
            finally:
                if recording is not None:
                    result["recording_video"] = stop_and_encode_recording(
                        page,
                        recording,
                        result["world_box"],
                        public_task_dir / "recording.mp4",
                    )
            if args.video_source in ("official", "both"):
                result["native_video"] = download_native_video(
                    page,
                    public_task_dir / "native.mp4",
                    int(datetime.fromisoformat(result["started_at"]).timestamp()),
                )
            result.update(status="completed", finished_at=now_text(), page_url=page.url)
            return 0
        except PermissionError as exc:
            result.update(status="auth_required", auth_reason=str(exc), page_url=page.url)
            screenshot(page, outdir / "auth_required.jpg")
            print(f"AUTH_REQUIRED: {exc}", flush=True)
            return 3
        except ImageRejectedError as exc:
            result.update(
                status="skipped_content_rejected",
                skip_reason=str(exc),
                page_url=page.url,
            )
            screenshot(page, outdir / "content_rejected.jpg")
            print(f"SKIP_CONTENT_REJECTED: {exc}", flush=True)
            return 0
        except Exception as exc:
            result.update(status="failed", error=repr(exc), failed_at=now_text(), page_url=page.url)
            screenshot(page, outdir / "failure.jpg")
            raise
        finally:
            result["finished_at"] = result.get("finished_at", now_text())
            provenance, deleted = align_action_sources(
                actions,
                result.get("executed_actions", []),
                AGENT_ABLATION_MODE or "preset_agent",
            )
            result["action_provenance"] = provenance
            result["deleted_actions"] = deleted
            for action, source in zip(result.get("executed_actions", []), provenance):
                action["source"] = source["source"]
            result["actual_executed_action"] = [
                {
                    "raw": action.get("raw"),
                    "keys": action.get("keys", []),
                    "base_duration_ms": action.get("base_duration_ms"),
                    "rotation_extra_ms": action.get("rotation_extra_ms", 0),
                    "planned_duration_ms": action.get("duration_ms"),
                    "actual_duration_ms": action.get("actual_duration_ms"),
                    "actual_hold_ms": action.get("actual_hold_ms"),
                    "auto_repeat_keydown_count": action.get("auto_repeat_keydown_count", 0),
                    "source": action.get("source"),
                    "claude_release_early": action.get("claude_release_early", False),
                    "rotation_early_release_disabled": action.get("rotation_early_release_disabled", False),
                    "claude_extension_ms": action.get("claude_extension_ms", 0),
                    "claude_adaptive_directions": action.get("claude_adaptive_directions", []),
                    "gap_before_keydown_ms": action.get("gap_before_keydown_ms"),
                    "keyboard_events": action.get("keyboard_events", []),
                    "interrupted_by_natural_end": action.get("interrupted_by_natural_end", False),
                }
                for action in result.get("executed_actions", [])
            ]
            if AGENT_ABLATION_MODE != "agent_only":
                result["executed_action_sequence"] = [
                    action.get("raw") for action in result.get("executed_actions", [])
                ]
            atomic_json(result_path, result)
            atomic_json(public_result_path, result)
            print(f"RESULT={result_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", nargs="?", default="GC002")
    parser.add_argument("--data-root", default=os.environ.get("HYWORLD2_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("HYWORLD2_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument(
        "--video-output-root",
        default=os.environ.get("HYWORLD2_VIDEO_OUTPUT_ROOT", str(DEFAULT_VIDEO_OUTPUT_ROOT)),
    )
    parser.add_argument("--cdp-url", default=os.environ.get("HYWORLD2_CDP_URL", "http://127.0.0.1:9220"))
    parser.add_argument("--url", default=os.environ.get("HYWORLD2_URL", DEFAULT_URL))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--action-override",
        help="仅覆盖本次运行的 action，不修改源 JSON；例如 'd*25s -> a*25s'",
    )
    parser.add_argument(
        "--claude-controller",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("HYWORLD2_CLAUDE_CONTROLLER", "1").lower() not in {"0", "false", "no"},
        help="启用 Claude Haiku 视觉闭环；用 --no-claude-controller 禁用",
    )
    parser.add_argument("--claude-base-url", default=os.environ.get("PLAYER_API_URL") or os.environ.get("KIGRESS_BASE_URL", ""))
    parser.add_argument("--claude-api-key", default=os.environ.get("PLAYER_API_KEY") or os.environ.get("KIGRESS_API_KEY", ""), help=argparse.SUPPRESS)
    parser.add_argument("--claude-user-key", default=os.environ.get("PLAYER_API_KEY") or os.environ.get("KIGRESS_USER_KEY", ""), help=argparse.SUPPRESS)
    parser.add_argument(
        "--claude-model",
        default=os.environ.get("PLAYER_MODEL") or os.environ.get("KIGRESS_MODEL", "claude-haiku-4-5-20251001"),
    )
    parser.add_argument("--claude-biz-scene", default=os.environ.get("KIGRESS_BIZ_SCENE", "offline"))
    parser.add_argument(
        "--claude-trust-env",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("KIGRESS_TRUST_ENV", "0").lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--claude-timeout-s", type=float, default=float(os.environ.get("HYWORLD2_CLAUDE_TIMEOUT_S", "30")))
    parser.add_argument("--claude-min-hold-ms", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_MIN_HOLD_MS", "1000")))
    parser.add_argument(
        "--claude-min-release-ms",
        type=int,
        default=int(os.environ.get("HYWORLD2_CLAUDE_MIN_RELEASE_MS", "1000")),
        help="Claude 不得在该实际长按时长前提前释放当前按键",
    )
    parser.add_argument("--claude-check-interval-ms", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_CHECK_INTERVAL_MS", "1500")))
    parser.add_argument("--claude-max-checks", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_MAX_CHECKS", "3")))
    parser.add_argument("--claude-max-extension-ms", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_MAX_EXTENSION_MS", "5000")))
    parser.add_argument("--claude-max-total-extension-ms", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_MAX_TOTAL_EXTENSION_MS", "10000")))
    parser.add_argument("--claude-max-adaptive-actions", type=int, default=int(os.environ.get("HYWORLD2_CLAUDE_MAX_ADAPTIVE_ACTIONS", "2")))
    parser.add_argument(
        "--video-source",
        choices=("both", "recording", "official"),
        default=os.environ.get("HYWORLD2_VIDEO_SOURCE", "both"),
        help="both=同时保存录屏和下载；recording=仅完整会话录屏；official=仅腾讯 mp4（纯享版）下载",
    )
    parser.add_argument("--generation-timeout", type=int, default=int(os.environ.get("HYWORLD2_GENERATION_TIMEOUT", "1800")))
    parser.add_argument("--initial-wait-ms", type=int, default=int(os.environ.get("HYWORLD2_INITIAL_WAIT_MS", "0")))
    parser.add_argument("--post-wait-ms", type=int, default=int(os.environ.get("HYWORLD2_POST_WAIT_MS", "0")))
    parser.add_argument(
        "--natural-end-timeout",
        type=int,
        default=int(os.environ.get("HYWORLD2_NATURAL_END_TIMEOUT", "90")),
        help="动作完成后等待实时世界自然到达 1 分钟的最长秒数",
    )
    parser.add_argument("--dry-run", action="store_true", help="只验证 JSON、图片和 action 解析")
    parser.add_argument("--probe-only", action="store_true", help="只检查当前登录和页面状态，不提交任务")
    return parser


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("已中断；finally 会释放当前按住的键。", file=sys.stderr)
        sys.exit(130)
