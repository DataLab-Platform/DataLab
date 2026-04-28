# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AIController (LLM provider mocked)."""

from __future__ import annotations

from unittest import mock

from datalab.aiassistant.controller import AIController
from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    ToolCall,
)
from datalab.aiassistant.tools.registry import Tool, ToolRegistry


class _ScriptedProvider(LLMProvider):
    """Provider returning a queued list of AssistantMessage objects."""

    name = "scripted"

    def __init__(self, replies: list[AssistantMessage]) -> None:
        super().__init__(api_key="x", model="x")
        self._replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, tools=None):  # noqa: ARG002
        self.calls.append(list(messages))
        return self._replies.pop(0)


def _registry_with(handler) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="do_thing",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            readonly=False,
        )
    )
    reg.register(
        Tool(
            name="inspect",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda *a, **kw: {"answer": 42},  # noqa: ARG005
            readonly=True,
        )
    )
    return reg


def test_text_only_response_returns_immediately() -> None:
    provider = _ScriptedProvider([AssistantMessage(content="hello")])
    ctrl = AIController(
        provider=provider,
        registry=ToolRegistry(),
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
    )
    result = ctrl.send("hi")
    assert result.assistant_message == "hello"
    assert result.tool_executions == []


def test_tool_call_loop_with_confirmation() -> None:
    """A tool call is confirmed, executed, and the loop continues until text."""
    handler = mock.MagicMock(return_value={"done": True})
    registry = _registry_with(handler)
    replies = [
        AssistantMessage(
            tool_calls=[ToolCall(id="c1", name="do_thing", arguments={"k": 1})]
        ),
        AssistantMessage(content="all done"),
    ]
    provider = _ScriptedProvider(replies)
    confirm = mock.MagicMock(return_value=True)
    ctrl = AIController(
        provider=provider,
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=confirm,
    )
    result = ctrl.send("please do")
    assert result.assistant_message == "all done"
    assert len(result.tool_executions) == 1
    name, args, tool_result = result.tool_executions[0]
    assert name == "do_thing"
    assert args == {"k": 1}
    assert tool_result.ok is True
    confirm.assert_called_once()
    handler.assert_called_once()


def test_readonly_tool_skips_confirmation() -> None:
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    replies = [
        AssistantMessage(tool_calls=[ToolCall(id="c1", name="inspect", arguments={})]),
        AssistantMessage(content="ok"),
    ]
    confirm = mock.MagicMock(return_value=True)
    ctrl = AIController(
        provider=_ScriptedProvider(replies),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=confirm,
        auto_approve_readonly=True,
    )
    result = ctrl.send("inspect please")
    assert result.assistant_message == "ok"
    confirm.assert_not_called()


def test_user_cancels_tool_call() -> None:
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    replies = [
        AssistantMessage(
            content="I will run it",
            tool_calls=[ToolCall(id="c1", name="do_thing", arguments={})],
        ),
    ]
    ctrl = AIController(
        provider=_ScriptedProvider(replies),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: False,
    )
    result = ctrl.send("please")
    assert result.cancelled is True
    assert result.tool_executions == []


def test_max_iterations_safety_cap() -> None:
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    # Always reply with another tool call -> would loop forever without the cap
    replies = [
        AssistantMessage(
            tool_calls=[ToolCall(id=f"c{i}", name="do_thing", arguments={})]
        )
        for i in range(10)
    ]
    ctrl = AIController(
        provider=_ScriptedProvider(replies),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
        max_iterations=3,
    )
    result = ctrl.send("loop")
    assert "maximum number of tool-call iterations" in result.assistant_message
    assert len(result.tool_executions) == 3
