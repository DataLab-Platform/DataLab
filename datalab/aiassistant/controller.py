# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Conversation controller for the AI assistant.

The :class:`AIController` orchestrates the dialogue:

1. Append the user message to the history.
2. Send the history + tool schemas to the LLM provider.
3. If the LLM emits tool calls, ask for confirmation, execute, append
   results, loop back.
4. Otherwise, return the assistant text.

This module is GUI-agnostic: tool confirmation is delegated to a callback
``confirm_callback(tool, arguments) -> bool``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
)
from datalab.aiassistant.tools.registry import Tool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from datalab.control.proxy import LocalProxy
    from datalab.gui.main import DLMainWindow


_BASE_SYSTEM_PROMPT = """\
You are an AI assistant integrated in DataLab, a scientific data processing
application for 1D signals and 2D images. You help the user by:

- Inspecting the current workspace (use 'list_objects', 'get_object_info',
  'get_current_panel', 'list_available_operations').
- Creating synthetic signals/images and loading files.
- Applying registered processing operations through 'apply_operation'.
- Writing complete Python macros and running them through
  'create_and_run_macro' for complex workflows.
- Visually inspecting plots through 'capture_view' (returns the current
  signal/image plot as an image you can analyse) when you need to assess
  the shape of a signal or the appearance of an image before/after
  processing.
- Looking up the public API of the proxy and data objects via
  'get_api_help' BEFORE writing a macro that uses unfamiliar methods.

Guidelines:

- Always discover available operations via 'list_available_operations' before
  calling 'apply_operation' on an unknown name.
- Prefer atomic 'apply_operation' calls over macros when possible.
- When writing a macro, ONLY use the public API documented in the cheat
  sheet below or returned by 'get_api_help'. Never invent attributes or
  methods on 'proxy', 'SignalObj' or 'ImageObj'.
- 'create_and_run_macro' returns the macro console output (stdout + stderr,
  including Python tracebacks) and exit code. If exit_code != 0, READ the
  output, fix the macro, and call the tool again.
- Be concise. Confirm completion in one sentence after the last tool call.
- Never invent operation names or parameter fields.
"""


_MACRO_FEWSHOT = """\

# ----- Canonical macro examples (copy these patterns) -----

## Example 1 — create a signal and apply an FFT
```python
import numpy as np
from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()
x = np.linspace(0.0, 1.0, 1024)
y = np.sin(2 * np.pi * 50.0 * x)
proxy.add_signal("sine 50Hz", x, y)
proxy.calc("fft")
print("done")
```

## Example 2 — iterate on existing signals, normalise each
```python
import sigima.params
from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()
proxy.set_current_panel("signal")
for uuid in proxy.get_object_uuids("signal"):
    proxy.select_objects([uuid])
    proxy.calc("normalize", sigima.params.NormalizeParam.create(method="minmax"))
```

## Example 3 — read data out of an object
```python
from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()
obj = proxy.get_object()        # current selected signal/image
# SignalObj exposes: obj.x, obj.y, obj.title, obj.xunit, obj.yunit
# ImageObj  exposes: obj.data, obj.title, obj.x0, obj.y0, obj.dx, obj.dy
print(obj.title, getattr(obj, "y", getattr(obj, "data", None)).shape)
```
"""


def _summarise_signature(member) -> str:
    """Return a short ``name(sig)  # first docstring line`` summary."""
    try:
        sig = str(inspect.signature(member))
    except (TypeError, ValueError):
        sig = "(...)"
    doc = (inspect.getdoc(member) or "").strip().splitlines()
    first = doc[0] if doc else ""
    if len(first) > 90:
        first = first[:87] + "..."
    return f"{sig}  # {first}".rstrip("  # ")


def _build_proxy_cheatsheet() -> str:
    """Return a one-line-per-method cheat sheet of the public proxy API."""
    # pylint: disable-next=import-outside-toplevel
    from datalab.control.baseproxy import BaseProxy  # noqa: WPS433

    lines = [
        "# RemoteProxy / LocalProxy public API "
        "(import: `from datalab.control.proxy import RemoteProxy`)"
    ]
    for name, member in sorted(inspect.getmembers(BaseProxy, inspect.isfunction)):
        if name.startswith("_"):
            continue
        lines.append(f"- proxy.{name}{_summarise_signature(member)}")
    return "\n".join(lines)


def _build_objects_cheatsheet() -> str:
    """Return a short cheat sheet for SignalObj / ImageObj."""
    return (
        "# SignalObj (1D)\n"
        "- attributes: x, y, dx, dy, title, xunit, yunit, xlabel, ylabel, "
        "metadata, roi, uuid\n"
        "- properties: xydata -> (x, y)\n"
        "- methods: copy(), set_xydata(x, y, dx=None, dy=None), "
        "get_data(roi_index=None) -> (x, y)\n"
        "\n"
        "# ImageObj (2D)\n"
        "- attributes: data, title, x0, y0, dx, dy, xunit, yunit, zunit, "
        "metadata, roi, uuid\n"
        "- methods: copy(), set_data_type(dtype), get_data(roi_index=None)\n"
        "\n"
        "# Parameters: import via `import sigima.params` then "
        "`sigima.params.<Name>Param.create(...)`\n"
        "  e.g. NormalizeParam, MovingAverageParam, FFTParam, GaussianParam.\n"
    )


def build_default_system_prompt() -> str:
    """Return the default system prompt with auto-generated API cheat sheet."""
    return (
        _BASE_SYSTEM_PROMPT
        + "\n"
        + _build_proxy_cheatsheet()
        + "\n\n"
        + _build_objects_cheatsheet()
        + _MACRO_FEWSHOT
    )


# Kept for backwards compatibility (and so that callers can still override it).
DEFAULT_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT


@dataclass
class TurnResult:
    """Result of a single conversation turn.

    Args:
        assistant_message: Final assistant text (after all tool calls).
        tool_executions: Tool calls made during the turn.
        cancelled: True if the user cancelled at a confirmation prompt.
    """

    assistant_message: str
    tool_executions: list[tuple[str, dict, ToolResult]] = field(default_factory=list)
    cancelled: bool = False


ConfirmCallback = Callable[[Tool, dict], bool]
ExecuteCallback = Callable[[str, dict], ToolResult]


class AIController:
    """Orchestrate the dialogue between user, LLM and DataLab.

    Args:
        provider: LLM provider.
        registry: Tool registry exposed to the LLM.
        proxy: DataLab local proxy.
        mainwindow: DataLab main window.
        confirm_callback: Callback invoked before executing a tool;
         must return True to proceed, False to cancel the whole turn.
        system_prompt: Override the default system prompt.
        max_iterations: Safety cap on tool-call iterations per user prompt.
        auto_approve_readonly: Skip confirmation for read-only tools.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        proxy: LocalProxy,
        mainwindow: DLMainWindow,
        confirm_callback: ConfirmCallback,
        system_prompt: str | None = None,
        max_iterations: int = 8,
        auto_approve_readonly: bool = True,
        execute_callback: ExecuteCallback | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.proxy = proxy
        self.mainwindow = mainwindow
        self.confirm_callback = confirm_callback
        self.execute_callback = execute_callback
        self.system_prompt = system_prompt or build_default_system_prompt()
        self.max_iterations = int(max_iterations)
        self.auto_approve_readonly = bool(auto_approve_readonly)
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]

    def reset(self) -> None:
        """Clear the conversation history (keep the system prompt)."""
        self.history = [ChatMessage(role="system", content=self.system_prompt)]

    def load_messages(self, messages: list[ChatMessage]) -> None:
        """Replace the current history with ``messages``.

        The system prompt is always reset to the controller's current
        ``system_prompt`` (which is auto-generated and may have changed
        since the conversation was persisted). Any system messages from
        ``messages`` are dropped.
        """
        self.history = [ChatMessage(role="system", content=self.system_prompt)]
        for msg in messages:
            if msg.role == "system":
                continue
            self.history.append(msg)

    def get_messages(self) -> list[ChatMessage]:
        """Return the conversation messages, excluding the system prompt."""
        return [msg for msg in self.history if msg.role != "system"]

    def send(self, user_message: str) -> TurnResult:
        """Send a user message and run the tool-call loop."""
        self.history.append(ChatMessage(role="user", content=user_message))
        executions: list[tuple[str, dict, ToolResult]] = []
        tools_schema = self.registry.list_schemas()

        for _iteration in range(self.max_iterations):
            response: AssistantMessage = self.provider.chat(
                self.history, tools=tools_schema
            )
            self.history.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
            )
            if not response.tool_calls:
                return TurnResult(
                    assistant_message=response.content, tool_executions=executions
                )
            for call in response.tool_calls:
                try:
                    tool = self.registry.get(call.name)
                except KeyError as exc:
                    result = ToolResult(ok=False, error=str(exc))
                else:
                    needs_confirm = not (tool.readonly and self.auto_approve_readonly)
                    if needs_confirm and not self.confirm_callback(
                        tool, call.arguments
                    ):
                        return TurnResult(
                            assistant_message=response.content,
                            tool_executions=executions,
                            cancelled=True,
                        )
                    if self.execute_callback is None:
                        result = self.registry.call(
                            call.name,
                            call.arguments,
                            self.proxy,
                            self.mainwindow,
                        )
                    else:
                        result = self.execute_callback(call.name, call.arguments)
                executions.append((call.name, dict(call.arguments), result))
                self.history.append(
                    ChatMessage(
                        role="tool",
                        content=result.to_message_content(),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
                # Inject any follow-up messages attached by the tool (e.g. a
                # multimodal user message carrying a screenshot).
                for extra in result.followup_messages:
                    self.history.append(extra)
        return TurnResult(
            assistant_message=(
                "Stopped after reaching the maximum number of tool-call "
                f"iterations ({self.max_iterations})."
            ),
            tool_executions=executions,
        )
