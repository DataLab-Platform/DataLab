# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AIController (LLM provider mocked)."""

from __future__ import annotations

from unittest import mock

from datalab.aiassistant.controller import AIController, build_default_system_prompt
from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    TokenUsage,
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
    """A plain assistant text reply ends the loop without tool calls."""
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
    assert not result.tool_executions


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
    """Read-only tools are auto-approved and the confirm callback is skipped."""
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
    """Returning False from the confirm callback cancels the turn."""
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
    assert not result.tool_executions
    # OpenAI protocol: every assistant tool_call must have a matching tool
    # response in the history, even on cancellation. Otherwise the next
    # request to the API is rejected as malformed.
    tool_msgs = [m for m in ctrl.history if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1"]


def test_user_cancels_first_of_multiple_tool_calls() -> None:
    """All pending tool_calls in the cancelled turn get a tool response."""
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    replies = [
        AssistantMessage(
            content="batch",
            tool_calls=[
                ToolCall(id="c1", name="do_thing", arguments={}),
                ToolCall(id="c2", name="do_thing", arguments={}),
                ToolCall(id="c3", name="do_thing", arguments={}),
            ],
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
    tool_msgs = [m for m in ctrl.history if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2", "c3"]


def test_max_iterations_safety_cap() -> None:
    """The controller stops after max_iterations to avoid infinite tool loops."""
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


class _RaisingProvider(LLMProvider):
    """Provider that succeeds for the 1st call then raises on subsequent ones."""

    name = "raising"

    def __init__(self, first_reply: AssistantMessage, error: Exception) -> None:
        super().__init__(api_key="x", model="x")
        self._first = first_reply
        self._error = error
        self._calls = 0

    def chat(self, messages, tools=None):  # noqa: ARG002
        self._calls += 1
        if self._calls == 1:
            return self._first
        raise self._error


def test_send_rolls_back_history_on_provider_failure() -> None:
    """An exception mid-turn must not leave dangling tool_calls in history.

    Regression test: the OpenAI protocol requires every assistant.tool_calls
    to be followed by matching tool responses. If ``send()`` raises after
    appending tool_calls but before all tool responses are written, the
    next ``send()`` would build an invalid request payload and the API
    would respond with a 400 error. The controller must roll back to the
    pre-call snapshot on any unhandled exception.
    """
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    # Iter 1 returns a tool call (gets processed cleanly), iter 2 raises.
    first = AssistantMessage(
        tool_calls=[ToolCall(id="c1", name="do_thing", arguments={})]
    )
    ctrl = AIController(
        provider=_RaisingProvider(first, RuntimeError("network down")),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
    )
    history_before = list(ctrl.history)

    import pytest

    with pytest.raises(RuntimeError, match="network down"):
        ctrl.send("hello")

    assert ctrl.history == history_before, (
        "history must be rolled back to its pre-send snapshot when the turn "
        "fails, otherwise the next send() will produce a malformed request"
    )


def test_default_system_prompt_omits_macro_when_tool_disabled() -> None:
    """The system prompt must not advertise tools that aren't in the registry.

    Otherwise the LLM is pushed to call ``create_and_run_macro`` even though
    the schema doesn't expose it, producing wasteful KeyError loops.
    """
    enabled = build_default_system_prompt(
        {"list_objects", "apply_operation", "create_and_run_macro"}
    )
    disabled = build_default_system_prompt({"list_objects", "apply_operation"})
    assert "create_and_run_macro" in enabled
    assert "create_and_run_macro" not in disabled
    assert "macro creation is disabled" in disabled


def test_default_system_prompt_unrestricted_when_no_filter() -> None:
    """Backwards compatibility: no filter means all tools are advertised."""
    prompt = build_default_system_prompt()
    assert "create_and_run_macro" in prompt


def test_cumulative_usage_accumulates_across_iterations() -> None:
    """Token usage from each provider response is summed into the cumulative
    counter and forwarded to ``usage_callback``."""
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005
    replies = [
        AssistantMessage(
            tool_calls=[ToolCall(id="c1", name="do_thing", arguments={})],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        ),
        AssistantMessage(
            content="done",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        ),
    ]
    callbacks: list[tuple[TokenUsage, TokenUsage]] = []
    ctrl = AIController(
        provider=_ScriptedProvider(replies),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
        usage_callback=lambda turn, cum: callbacks.append((turn, cum)),
    )
    result = ctrl.send("go")
    assert result.assistant_message == "done"
    assert ctrl.get_usage().prompt_tokens == 30
    assert ctrl.get_usage().completion_tokens == 8
    assert ctrl.get_usage().total_tokens == 38
    # Two callback invocations (one per provider response).
    assert len(callbacks) == 2
    # Per-turn usage on TurnResult sums the same.
    assert result.turn_usage is not None
    assert result.turn_usage.total_tokens == 38


def test_load_messages_seeds_initial_usage() -> None:
    """``load_messages`` restores the cumulative counter for resumed conversations."""
    ctrl = AIController(
        provider=_ScriptedProvider([]),
        registry=ToolRegistry(),
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
    )
    seed = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    ctrl.load_messages([], initial_usage=seed)
    assert ctrl.get_usage().total_tokens == 150
    # Reset clears it.
    ctrl.reset()
    assert ctrl.get_usage().total_tokens is None


def test_abort_returns_aborted_result_and_rolls_back() -> None:
    """Calling ``abort()`` mid-loop returns an aborted TurnResult and the
    history snapshot is restored so the next ``send()`` is well-formed."""
    registry = _registry_with(lambda *a, **k: None)  # noqa: ARG005

    class _AbortingProvider(LLMProvider):
        name = "aborting"

        def __init__(self, ctrl_ref: list[AIController]) -> None:
            super().__init__(api_key="x", model="x")
            self._ctrl_ref = ctrl_ref

        def chat(self, messages, tools=None):  # noqa: ARG002
            # Trigger abort during the call so the next iteration sees it.
            self._ctrl_ref[0].abort()
            return AssistantMessage(
                tool_calls=[ToolCall(id="c1", name="do_thing", arguments={})]
            )

    ctrl_ref: list[AIController] = []
    ctrl = AIController(
        provider=_AbortingProvider(ctrl_ref),
        registry=registry,
        proxy=mock.MagicMock(),
        mainwindow=mock.MagicMock(),
        confirm_callback=lambda *a: True,
    )
    ctrl_ref.append(ctrl)
    history_before = list(ctrl.history)
    result = ctrl.send("go")
    assert result.aborted is True
    assert result.assistant_message == ""
    assert ctrl.history == history_before
    assert ctrl.is_running is False
