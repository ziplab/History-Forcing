"""Provider-neutral agent interface for HYWorld2 ablations."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_MODES = {"preset_only", "agent_only", "preset_agent"}
VALID_PROVIDERS = {"claude", "gemini"}
ACTION_RE = re.compile(r"^hold\((w|a|s|d|left|right|up|down|space|e),([1-9]\d{1,4})ms\)$", re.I)
WAIT_RE = re.compile(r"^wait\(([1-9]\d{1,4})ms\)$", re.I)


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    base_url: str
    headers: dict[str, str]
    transport: str
    verify_tls: bool = True
    trust_env: bool = False


def ablation_mode() -> str:
    mode = os.environ.get("AGENT_ABLATION_MODE", "").strip().lower()
    if mode and mode not in VALID_MODES:
        raise ValueError(f"AGENT_ABLATION_MODE must be one of {sorted(VALID_MODES)}, got {mode!r}")
    return mode


def configured_agent() -> AgentConfig:
    provider = os.environ.get("AGENT_PROVIDER", "claude").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"AGENT_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}")
    if provider == "gemini":
        model = os.environ.get("AGENT_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-3.1-pro")
        url = os.environ.get("AGENT_BASE_URL") or os.environ.get("GEMINI_BASE_URL", "")
        return AgentConfig(
            provider="gemini",
            model=model,
            base_url=url,
            transport="gemini_generate_content",
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ.get("GEMINI_API_KEY", ""),
                "x-ks-user-key": os.environ.get("GEMINI_USER_KEY", ""),
                "x-ks-llm-model": model,
                "x-ks-biz-scene": os.environ.get("GEMINI_BIZ_SCENE", "offline"),
            },
        )
    transport = os.environ.get("AGENT_TRANSPORT", "openai").strip().lower()
    model = os.environ.get("AGENT_MODEL") or os.environ.get("PLAYER_MODEL") or os.environ.get("KIGRESS_MODEL", "claude-haiku-4-5-20251001")
    if transport == "anthropic":
        return AgentConfig(
            provider="claude",
            model=model,
            base_url=os.environ.get("AGENT_BASE_URL") or os.environ.get(
                "WQ_MESSAGES_URL", "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/messages"
            ),
            transport="anthropic",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('WQ_API_KEY', '')}",
            },
        )
    return AgentConfig(
        provider="claude",
        model=model,
        base_url=(os.environ.get("AGENT_BASE_URL") or os.environ.get("PLAYER_API_URL") or os.environ.get("KIGRESS_BASE_URL", "")).rstrip("/"),
        transport="openai",
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("PLAYER_API_KEY") or os.environ.get("KIGRESS_API_KEY", ""),
            "x-ks-user-key": os.environ.get("PLAYER_API_KEY") or os.environ.get("KIGRESS_USER_KEY", ""),
            "x-ks-llm-model": model,
            "x-ks-biz-scene": os.environ.get("KIGRESS_BIZ_SCENE", "offline"),
        },
        trust_env=os.environ.get("KIGRESS_TRUST_ENV", "0").lower() in {"1", "true", "yes"},
    )


def validate_agent_config(config: AgentConfig) -> None:
    missing: list[str] = []
    if not config.base_url:
        missing.append("agent base URL")
    if config.transport == "anthropic":
        if not config.headers.get("Authorization", "").removeprefix("Bearer "):
            missing.append("WQ_API_KEY")
    else:
        if not config.headers.get("x-api-key"):
            missing.append("agent API key")
        if not config.headers.get("x-ks-user-key"):
            missing.append("agent user key")
    if missing:
        raise RuntimeError("Missing agent configuration: " + ", ".join(missing))


def observation_package(objective: str, initial: Path, latest: Path, executed_actions: list[str]) -> dict[str, Any]:
    """The complete and exclusive semantic context for Agent Only."""
    return {
        "objective": objective,
        "initial_observation": str(initial),
        "latest_observation": str(latest),
        "executed_actions": list(executed_actions),
    }


def _data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def build_agent_only_messages(package: dict[str, Any], max_actions: int) -> list[dict[str, Any]]:
    system = (
        "You control an interactive 3D world from visual observations. Choose actions from scratch to achieve "
        "the objective. You have no preset plan. Use only the objective, initial observation, latest observation, "
        "and already executed actions supplied in this request. Return strict JSON only with schema "
        "{\"status\":\"continue|done\",\"actions\":[\"hold(w,1500ms)\"],\"reason\":\"short\"}. "
        f"Return at most {max_actions} actions. Allowed DSL is "
        "hold(w|a|s|d|left|right|up|down|space|e,100..10000ms) or wait(100..10000ms)."
    )
    text = json.dumps(
        {"objective": package["objective"], "executed_actions": package["executed_actions"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "text", "text": "Initial observation:"},
        {"type": "image_url", "image_url": {"url": _data_url(package["initial_observation"])}},
        {"type": "text", "text": "Latest observation:"},
        {"type": "image_url", "image_url": {"url": _data_url(package["latest_observation"])}},
    ]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def parse_agent_decision(raw: str, max_actions: int = 3) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    status = str(data.get("status", "continue")).lower()
    if status not in {"continue", "done"}:
        raise ValueError(f"invalid status: {status}")
    actions: list[dict[str, Any]] = []
    for raw_action in data.get("actions") or []:
        token = re.sub(r"\s+", "", str(raw_action)).lower()
        match = ACTION_RE.fullmatch(token)
        wait = WAIT_RE.fullmatch(token)
        if match:
            ms = max(100, min(10000, int(match.group(2))))
            actions.append({"type": "hold", "keys": [match.group(1).lower()], "duration_ms": ms, "raw": token})
        elif wait:
            ms = max(100, min(10000, int(wait.group(1))))
            actions.append({"type": "wait", "duration_ms": ms, "raw": token})
        else:
            raise ValueError(f"invalid action DSL: {token!r}")
        if len(actions) >= max_actions:
            break
    if status == "continue" and not actions:
        raise ValueError("continue requires at least one action")
    return {"status": status, "actions": actions, "reason": str(data.get("reason") or "")[:500]}


def _anthropic_payload(messages: list[dict[str, Any]], model: str, max_tokens: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in messages[1]["content"]:
        if part["type"] == "text":
            content.append(part)
        else:
            media, encoded = part["image_url"]["url"].removeprefix("data:").split(";base64,", 1)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": encoded}})
    return {"model": model, "max_tokens": max_tokens, "system": messages[0]["content"], "messages": [{"role": "user", "content": content}]}


def _gemini_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": messages[0]["content"]}]
    for part in messages[1]["content"]:
        if part["type"] == "text":
            parts.append({"text": part["text"]})
        else:
            media, encoded = part["image_url"]["url"].removeprefix("data:").split(";base64,", 1)
            parts.append({"inlineData": {"mimeType": media, "data": encoded}})
    return {"contents": [{"role": "user", "parts": parts}]}


def request_agent_decision(client: Any, config: AgentConfig, package: dict[str, Any], max_actions: int = 3,
                           max_tokens: int = 192, timeout_s: float = 60) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    messages = build_agent_only_messages(package, max_actions)
    if config.transport == "anthropic":
        url, payload = config.base_url, _anthropic_payload(messages, config.model, max_tokens)
    elif config.transport == "gemini_generate_content":
        url, payload = config.base_url, _gemini_payload(messages)
    else:
        url = f"{config.base_url}/chat/completions"
        payload = {"model": config.model, "max_completion_tokens": max_tokens, "messages": messages}
    started = time.perf_counter()
    response = client.post(url, headers=config.headers, json=payload, timeout=timeout_s)
    latency_ms = round((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    data = response.json()
    if config.transport == "gemini_generate_content":
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        content = "".join(str(part.get("text") or "") for part in parts)
    elif config.transport == "anthropic":
        content = "".join(str(part.get("text") or "") for part in data.get("content") or [])
    else:
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        if isinstance(content, list):
            content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    decision = parse_agent_decision(content, max_actions)
    decision.update(raw_response=content, decision_latency_ms=latency_ms, response_model=data.get("model") or config.model)
    return decision, messages


def action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    if action.get("type") == "wait":
        return ("wait", (), round(int(action.get("duration_ms") or 0) / 250))
    keys = tuple(sorted(str(key).lower() for key in action.get("keys", [])))
    duration = int(action.get("actual_hold_ms") or action.get("duration_ms") or 0)
    return ("key", keys, round(duration / 250))


def align_action_sources(base: list[dict[str, Any]], executed: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode == "preset_only":
        return ([{"index": i + 1, "source": "retained", "base_index": i + 1} for i in range(len(executed))], [])
    if mode == "agent_only":
        return (
            [{"index": i + 1, "source": "inserted", "base_index": None} for i in range(len(executed))],
            [{"base_index": i + 1, "source": "deleted"} for i in range(len(base))],
        )
    # Align executed Preset+Agent actions with the base sequence by position.
    records = []
    for i, action in enumerate(executed):
        same = i < len(base) and action_signature(base[i]) == action_signature(action)
        records.append({"index": i + 1, "source": "retained" if same else ("replaced" if i < len(base) else "inserted"),
                        "base_index": i + 1 if i < len(base) else None})
    deleted = [{"base_index": i + 1, "source": "deleted"} for i in range(len(executed), len(base))]
    return records, deleted
