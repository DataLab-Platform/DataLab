# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the MockProvider."""

from __future__ import annotations

from datalab.aiassistant.providers import MockProvider
from datalab.aiassistant.providers.base import ChatMessage


def _send(provider: MockProvider, text: str):
    return provider.chat([ChatMessage(role="user", content=text)])


def test_greeting_returns_text_only() -> None:
    msg = _send(MockProvider(), "Hello there")
    assert msg.content
    assert msg.tool_calls == []


def test_list_keyword_triggers_list_objects() -> None:
    msg = _send(MockProvider(), "please list objects in the panel")
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "list_objects"


def test_signal_keyword_triggers_create_signal() -> None:
    msg = _send(MockProvider(), "create a sine signal")
    assert len(msg.tool_calls) == 1
    call = msg.tool_calls[0]
    assert call.name == "create_synthetic_signal"
    assert call.arguments.get("kind") == "sin"


def test_macro_keyword_triggers_macro_tool() -> None:
    msg = _send(MockProvider(), "write a macro for me")
    assert msg.tool_calls and msg.tool_calls[0].name == "create_and_run_macro"
    assert "RemoteProxy" in msg.tool_calls[0].arguments["code"]


def test_fft_keyword_triggers_two_step_plan() -> None:
    """The FFT plan emits create + apply on consecutive chat() calls."""
    provider = MockProvider()
    first = _send(provider, "compute fft of a signal")
    assert first.tool_calls[0].name == "create_synthetic_signal"
    second = provider.chat([])
    assert second.tool_calls and second.tool_calls[0].name == "apply_operation"
    third = provider.chat([])
    assert third.tool_calls == []
    assert third.content


def test_unknown_input_returns_help_text() -> None:
    msg = _send(MockProvider(), "completely unrelated request")
    assert "Mock provider" in msg.content
    assert msg.tool_calls == []
