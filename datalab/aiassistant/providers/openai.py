# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
OpenAI (and OpenAI-compatible) chat-completion provider.

This implementation uses :mod:`urllib` from the standard library to avoid
adding any third-party HTTP dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    ToolCall,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API provider.

    Compatible with any OpenAI-style endpoint exposing
    ``POST /v1/chat/completions`` (e.g. Azure OpenAI, local proxies).
    """

    name = "openai"
    api_key_env_var = "OPENAI_API_KEY"

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantMessage:
        url = (self.base_url or _DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pylint: disable=broad-except
                detail = str(exc)
            raise LLMProviderError(f"OpenAI HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"OpenAI network error: {exc.reason}") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        return _parse_openai_response(data)


def _message_to_openai(message: ChatMessage) -> dict[str, Any]:
    """Convert a :class:`ChatMessage` into an OpenAI-format dict."""
    out: dict[str, Any] = {"role": message.role}
    if message.role == "tool":
        # Tool messages must carry plain text content.
        content = message.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )
        out["content"] = content
        if message.tool_call_id is not None:
            out["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            out["name"] = message.name
        return out
    if message.content:
        # Pass list content (multimodal blocks) through unchanged.
        out["content"] = message.content
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in message.tool_calls
        ]
    if not out.get("content") and not out.get("tool_calls"):
        out["content"] = ""
    return out


def _parse_openai_response(data: dict[str, Any]) -> AssistantMessage:
    """Parse an OpenAI ``chat/completions`` response."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError(
            f"Unexpected OpenAI response: {json.dumps(data)[:500]}"
        ) from exc
    message = choice.get("message", {}) or {}
    finish_reason = choice.get("finish_reason")
    content = message.get("content") or ""
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        function = raw.get("function") or {}
        name = function.get("name") or ""
        arguments_str = function.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {"_raw": arguments_str}
        tool_calls.append(
            ToolCall(id=raw.get("id") or name, name=name, arguments=arguments)
        )
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw=data,
    )
