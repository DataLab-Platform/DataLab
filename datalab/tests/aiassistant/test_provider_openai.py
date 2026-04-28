# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the OpenAI provider (HTTP layer mocked)."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from datalab.aiassistant.providers import LLMProviderError, OpenAIProvider
from datalab.aiassistant.providers.base import ChatMessage, ToolCall


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _make_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")


def test_chat_text_response() -> None:
    """Plain text reply is parsed into AssistantMessage.content."""
    payload = {
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
        ]
    }
    with mock.patch(
        "datalab.aiassistant.providers.openai.urllib.request.urlopen",
        return_value=_FakeResponse(payload),
    ):
        msg = _make_provider().chat([ChatMessage(role="user", content="hello")])
    assert msg.content == "hi"
    assert msg.tool_calls == []
    assert msg.finish_reason == "stop"


def test_chat_with_tool_calls_parsing() -> None:
    """Tool calls are parsed and JSON arguments decoded."""
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_objects",
                                "arguments": '{"panel": "signal"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    with mock.patch(
        "datalab.aiassistant.providers.openai.urllib.request.urlopen",
        return_value=_FakeResponse(payload),
    ):
        msg = _make_provider().chat([ChatMessage(role="user", content="?")])
    assert len(msg.tool_calls) == 1
    call: ToolCall = msg.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "list_objects"
    assert call.arguments == {"panel": "signal"}


def test_chat_request_payload() -> None:
    """The HTTP request body contains model, messages and tools."""
    captured: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        )

    tools = [{"name": "ping", "description": "pong", "parameters": {"type": "object"}}]
    with mock.patch(
        "datalab.aiassistant.providers.openai.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        _make_provider().chat([ChatMessage(role="user", content="hi")], tools=tools)

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["tools"][0]["function"]["name"] == "ping"


def test_http_error_raises_provider_error() -> None:
    """HTTP errors are wrapped into LLMProviderError."""
    import urllib.error

    err = urllib.error.HTTPError(
        url="x", code=401, msg="unauthorized", hdrs=None, fp=None
    )
    with mock.patch(
        "datalab.aiassistant.providers.openai.urllib.request.urlopen",
        side_effect=err,
    ):
        with pytest.raises(LLMProviderError):
            _make_provider().chat([ChatMessage(role="user", content="x")])
