"""Minimal OpenAI-compatible chat client for vLLM (GB10) and Ollama (laptop).

Uses only the standard library so the project runs with almost no installs.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class VLMError(RuntimeError):
    pass


def image_to_data_url(image: Any) -> str:
    """Accepts a path or a PIL image."""
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
        mime = "image/png" if str(image).lower().endswith(".png") else "image/jpeg"
    else:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data, mime = buf.getvalue(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates <think> blocks and ``` fences)."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # opening tag was in the prompt template
        text = text.split("</think>", 1)[1]
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise VLMError(f"No JSON object in model reply: {text[:300]!r}")


def yes_no(value: Any) -> str:
    """Normalise to 'yes' / 'no' / 'uncertain'."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    v = str(value or "").strip().lower()
    if v in {"yes", "y", "true", "1"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    return "uncertain"


@dataclass
class ChatResult:
    text: str
    model: str
    raw: dict


class VLMClient:
    def __init__(self, settings: Settings, model: str | None = None):
        self.s = settings
        self.base_url = settings.vlm_base_url.rstrip("/")
        self._model = model or settings.vlm_model or None

    # -- plumbing ----------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method="POST" if data is not None else "GET",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.s.vlm_api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.s.vlm_timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise VLMError(f"HTTP {e.code} from {self.base_url}{path}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise VLMError(f"Cannot reach model server at {self.base_url}: {e}") from e

    @property
    def model(self) -> str:
        if not self._model:
            models = self._request("/models", timeout=15).get("data", [])
            if not models:
                raise VLMError(f"Server at {self.base_url} lists no models; set VLM_MODEL")
            self._model = models[0]["id"]
        return self._model

    def chat(self, messages: list[dict], json_schema: dict | None = None, max_tokens: int | None = None,
             temperature: float = 0.0, retries: int = 2) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.vlm_max_tokens,
        }
        if json_schema is not None and self.s.vlm_json_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": json_schema}}
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                raw = self._request("/chat/completions", payload)
                msg = raw["choices"][0]["message"]
                return ChatResult(text=msg.get("content") or "", model=raw.get("model", self.model), raw=raw)
            except (VLMError, KeyError, IndexError) as e:
                last = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise VLMError(str(last))

    def ask_json(self, prompt: str, images: list[Any] = (), json_schema: dict | None = None,
                 system: str | None = None, retries: int = 2) -> tuple[dict, ChatResult]:
        content: list[dict] = []
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        last: Exception | None = None
        for attempt in range(retries + 1):
            result = self.chat(messages, json_schema=json_schema)
            try:
                return extract_json(result.text), result
            except VLMError as e:  # model rambled; try again
                last = e
        raise VLMError(str(last))

    def ask_text(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.chat(messages, max_tokens=max_tokens).text
        text = _THINK_RE.sub("", text)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()
