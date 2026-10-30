"""Shared Seed transport for benchmark rubrics written against Gemini's client."""

import base64
import mimetypes
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import local
from types import SimpleNamespace

import httpx

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SEED_LITE = "doubao-seed-2-0-lite-260428"


def _ark_client(timeout):
    return httpx.Client(trust_env=False, timeout=timeout,
                        transport=httpx.HTTPTransport(retries=3, trust_env=False))


class JudgeClient:
    """Track temporary API evidence separately for every scoring worker."""

    def __init__(self, provider):
        if provider == 'seed-lite':
            key = os.environ.get('ARK_API_KEY') or os.environ.get('SEED_API_KEY')
            if not key:
                raise ValueError('Seed evaluation requires ARK_API_KEY')
            self.client = SeedJudge(os.environ.get('ARK_BASE_URL', ARK_BASE_URL), key)
        elif provider == 'gemini':
            from google import genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise ValueError('Gemini evaluation requires GEMINI_API_KEY')
            self.client = genai.Client(api_key=key)
        else:
            raise ValueError(f'Unknown judge provider: {provider}')
        self.models, self.files = self.client.models, self
        self._local = local()

    def upload(self, **kwargs):
        file = self.client.files.upload(**kwargs)
        self._local.uploads.append(file.name)
        return file

    def get(self, **kwargs):
        return self.client.files.get(**kwargs)

    @contextmanager
    def scope(self):
        self._local.uploads = []
        try:
            yield
        finally:
            for name in self._local.uploads:
                try:
                    self.client.files.delete(name=name)
                except Exception as error:
                    warnings.warn(f'Could not remove temporary judge evidence: {type(error).__name__}')
            self._local.uploads = []


def _ark_completion(base_url: str, api_key: str, model: str, content: list[dict],
                    *, max_tokens: int, temperature: float | None = None,
                    json_mode: bool = False, timeout: float = 300.0,
                    thinking: str = "disabled") -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "thinking": {"type": thinking},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    with _ark_client(timeout) as client:
        for attempt in range(8):
            response = client.post(url, json=payload, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })
            if response.status_code != 429 or attempt == 7:
                break
            delay = min(60, 15 * 2 ** attempt)
            try:
                seconds = float(response.headers.get('Retry-After', ''))
                if 0 < seconds <= 60:
                    delay = seconds
            except ValueError:
                pass
            warnings.warn(f'Seed HTTP 429; retrying in {delay}s (attempt {attempt + 1}/8).')
            time.sleep(delay)
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        raise RuntimeError("Seed Lite response has no choices")
    value = (choices[0].get("message") or {}).get("content")
    if isinstance(value, list):
        return "".join(item.get("text", "") for item in value if isinstance(item, dict))
    return str(value or "")


def _ark_files_url(base_url: str, name: str = "") -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        root = root[:-len("/chat/completions")]
    return f"{root}/files" + (f"/{name}" if name else "")


class SeedJudge:
    """Only translate the official Gemini evidence call to Ark Chat Completions."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.models = self
        self.files = self
        self._mimes = {}

    def _file(self, data: dict, mime_type: str | None = None):
        file_id = data.get("id")
        if not file_id:
            raise RuntimeError("Ark Files API response has no file ID")
        mime_type = (data.get("mime_type") or data.get("content_type") or
                     mime_type or self._mimes.get(file_id, "image/jpeg"))
        self._mimes[file_id] = mime_type
        state = str(data.get("status") or data.get("state") or "active")
        if state.lower() in {"processed", "ready", "succeeded"}:
            state = "active"
        return SimpleNamespace(name=file_id, uri=f"ark-file://{file_id}",
                               mime_type=mime_type, state=state.upper())

    def upload(self, *, file):
        path = Path(file)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        with path.open("rb") as handle, _ark_client(300) as client:
            response = client.post(
                _ark_files_url(self.base_url),
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"purpose": "user_data"},
                files={"file": (path.name, handle, mime_type)},
            )
        response.raise_for_status()
        return self._file(response.json(), mime_type)

    def get(self, *, name):
        with _ark_client(90) as client:
            response = client.get(
                _ark_files_url(self.base_url, name),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        response.raise_for_status()
        return self._file(response.json())

    def delete(self, *, name):
        with _ark_client(90) as client:
            response = client.delete(
                _ark_files_url(self.base_url, name),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        response.raise_for_status()

    def generate_content(self, *, model, contents, config=None):
        content = []
        for part in getattr(contents, 'parts', contents):
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
                continue
            if getattr(part, 'uri', '').startswith('ark-file://'):
                kind = "video_url" if part.mime_type.startswith("video/") else "image_url"
                content.append({"type": kind, kind: {"file_id": part.name}})
                continue
            if part.text is not None:
                content.append({"type": "text", "text": part.text})
            elif part.inline_data is not None:
                inline = part.inline_data
                encoded = base64.b64encode(inline.data).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{inline.mime_type};base64,{encoded}"},
                })
            elif part.file_data is not None:
                uri = part.file_data.file_uri
                if not uri.startswith("ark-file://"):
                    raise ValueError(f"Unknown Ark evidence URI: {uri}")
                kind = ("video_url" if getattr(part.file_data, 'mime_type', 'image/jpeg').startswith("video/")
                        else "image_url")
                content.append({"type": kind, kind: {"file_id": uri[len("ark-file://"):]}})
            else:
                raise ValueError("Unsupported multimodal evidence part")
        return SimpleNamespace(text=_ark_completion(
            self.base_url, self.api_key, model, content, max_tokens=8192,
            temperature=getattr(config, "temperature", 0.0), json_mode=hasattr(contents, 'parts'),
        ))

