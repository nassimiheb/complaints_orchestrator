"""Shared Mistral request helpers used by multiple agents."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable
from urllib import error, request

MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"


def resolve_mistral_api_key(explicit_api_key: str | None, missing_key_error: str) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    env_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError(missing_key_error)


def resolve_mistral_model(explicit_model: str | None, default_model: str = "mistral-small-latest") -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    return os.getenv("CCO_MODEL_NAME", default_model)


def _extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_json_object(raw_text: str) -> dict[str, object] | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def request_chat_json_object(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any] | str,
    timeout_seconds: int,
    temperature: float = 0.0,
    urlopen_fn: Callable[..., Any] | None = None,
    network_error_prefix: str = "Mistral call failed",
    format_error_prefix: str = "Invalid Mistral response format",
    missing_json_error: str = "Mistral response did not contain a valid JSON object.",
) -> dict[str, object]:
    user_content = (
        user_payload
        if isinstance(user_payload, str)
        else json.dumps(user_payload, ensure_ascii=True)
    )
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }

    req = request.Request(
        url=MISTRAL_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    sender = urlopen_fn or request.urlopen
    try:
        with sender(req, timeout=timeout_seconds) as resp:
            raw_response = resp.read().decode("utf-8")
    except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"{network_error_prefix}: {exc}") from exc

    try:
        parsed_response = json.loads(raw_response)
        message_content = parsed_response["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{format_error_prefix}: {exc}") from exc

    raw_content = _extract_message_text(message_content)
    model_output = _extract_json_object(raw_content)
    if model_output is None:
        raise RuntimeError(missing_json_error)
    return model_output

