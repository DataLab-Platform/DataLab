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


DEFAULT_SYSTEM_PROMPT = """\
You are an AI assistant integrated in DataLab, a scientific data processing
application for 1D signals and 2D images. You help the user by:

- Inspecting the current workspace (use 'list_objects', 'get_object_info',
  'get_current_panel', 'list_available_operations').
- Creating synthetic signals/images and loading files.
- Applying registered processing operations through 'apply_operation'.
- Writing complete Python macros and running them through
  'create_and_run_macro' for complex workflows.

Guidelines:

- Always discover available operations via 'list_available_operations' before
  calling 'apply_operation' on an unknown name.
- Prefer atomic 'apply_operation' calls over macros when possible.
- Macros may import 'numpy as np', 'scipy.signal as sps' and use
  'from datalab.control.proxy import RemoteProxy; proxy = RemoteProxy()'.
- Be concise. Confirm completion in one sentence after the last tool call.
- Never invent operation names or parameter fields.
"""


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
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_iterations = int(max_iterations)
        self.auto_approve_readonly = bool(auto_approve_readonly)
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]

    def reset(self) -> None:
        """Clear the conversation history (keep the system prompt)."""
        self.history = [ChatMessage(role="system", content=self.system_prompt)]

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
        return TurnResult(
            assistant_message=(
                "Stopped after reaching the maximum number of tool-call "
                f"iterations ({self.max_iterations})."
            ),
            tool_executions=executions,
        )
