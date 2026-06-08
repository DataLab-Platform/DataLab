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
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    TokenUsage,
    sum_usage,
)
from datalab.aiassistant.tools.registry import Tool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from datalab.control.proxy import LocalProxy
    from datalab.gui.main import DLMainWindow


_BASE_SYSTEM_PROMPT = """\
You are an AI assistant integrated in DataLab, a scientific data processing
application for 1D signals and 2D images. You help the user by:

- Inspecting the current workspace (use 'list_objects', 'get_object_info',
  'get_current_panel', 'list_available_operations', 'list_plugin_actions').
- Creating synthetic signals/images and loading files.
- Applying registered processing operations through 'apply_operation'.
- Invoking plugin-provided features through 'trigger_plugin_action' (use
  'list_plugin_actions' first to discover what each installed plugin
  exposes; this is how third-party plugins like ASNR are reached).
{macro_capability}\
- Visually inspecting plots through 'capture_view' (returns the current
  signal/image plot as an image you can analyse) when you need to assess
  the shape of a signal or the appearance of an image before/after
  processing.
{macro_api_help}\

Guidelines:

- Always discover available operations via 'list_available_operations' before
  calling 'apply_operation' on an unknown name.
- When the user mentions a plugin (or asks for a feature that is not in
  'list_available_operations'), call 'list_plugin_actions' first and then
  'trigger_plugin_action' with the matching action title or menu path.
{macro_guidelines}\
- Be concise. Confirm completion in one sentence after the last tool call.
- Never invent operation names or parameter fields.
- NEVER embed images in your prose. Do not emit Markdown image tags
  ('![alt](url)'), 'data:image/...' URIs, base64 blobs, HTML '<img>'
  tags, or any other inline image syntax — neither real nor invented.
  When a tool returns an image (e.g. 'capture_view'), the UI already
  displays it; just describe it in plain text. Same rule for binary
  payloads in general: never paste base64 strings back to the user.
"""


_MACRO_CAPABILITY = (
    "- Writing complete Python macros and running them through\n"
    "  'create_and_run_macro' for complex workflows.\n"
)

_MACRO_API_HELP = (
    "- Looking up the public API of the proxy and data objects via\n"
    "  'get_api_help' BEFORE writing a macro that uses unfamiliar methods.\n"
)

_MACRO_GUIDELINES = (
    "- Prefer atomic 'apply_operation' calls over macros when possible.\n"
    "- When writing a macro, ONLY use the public API documented in the cheat\n"
    "  sheet below or returned by 'get_api_help'. Never invent attributes or\n"
    "  methods on 'proxy', 'SignalObj' or 'ImageObj'.\n"
    "- 'create_and_run_macro' runs the script TRANSIENTLY — it is NOT added\n"
    "  to the Macro panel by default. The chat UI exposes a 'Save to Macros'\n"
    "  link so the user can opt in to persisting it. Always pick a short,\n"
    "  descriptive 'title' (it is shown in that link and used as the saved\n"
    "  macro's name).\n"
    "- 'create_and_run_macro' returns the macro console output (stdout + stderr,\n"
    "  including Python tracebacks) and exit code. If exit_code != 0, READ the\n"
    "  output, fix the macro, and call the tool again.\n"
)

_MACRO_DISABLED_NOTE = (
    "\nNote: macro creation is disabled by the user. Do not attempt to write "
    "or run Python macros; use 'apply_operation' or 'trigger_plugin_action' "
    "instead.\n"
)


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
    if not first:
        return sig
    return f"{sig}  # {first}"


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


def build_default_system_prompt(
    available_tool_names: set[str] | None = None,
) -> str:
    """Return the default system prompt with auto-generated API cheat sheet.

    Args:
        available_tool_names: Names of the tools actually exposed to the LLM.
         When provided, sections that mention a missing tool are omitted so
         the model is not pushed to call tools it does not have access to.
         When ``None`` (default), assume all built-in tools are available
         (backwards compatibility).
    """
    macro_enabled = available_tool_names is None or (
        "create_and_run_macro" in available_tool_names
    )
    base = _BASE_SYSTEM_PROMPT.format(
        macro_capability=_MACRO_CAPABILITY if macro_enabled else "",
        macro_api_help=_MACRO_API_HELP if macro_enabled else "",
        macro_guidelines=_MACRO_GUIDELINES if macro_enabled else "",
    )
    if not macro_enabled:
        base += _MACRO_DISABLED_NOTE
    return (
        base
        + "\n"
        + _build_proxy_cheatsheet()
        + "\n\n"
        + _build_objects_cheatsheet()
        + (_MACRO_FEWSHOT if macro_enabled else "")
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
        aborted: True if :meth:`AIController.abort` interrupted the loop.
        turn_usage: Cumulative token usage for this single turn (sum of
         every provider round-trip in the loop). ``None`` when no
         round-trip reported usage.
    """

    assistant_message: str
    tool_executions: list[tuple[str, dict, ToolResult]] = field(default_factory=list)
    cancelled: bool = False
    aborted: bool = False
    turn_usage: TokenUsage | None = None


class AIAbortError(RuntimeError):
    """Raised inside the controller loop when :meth:`AIController.abort` is
    called. Mirrors :class:`AbortError` in DataLab-Web."""


ConfirmCallback = Callable[[Tool, dict], bool]
ExecuteCallback = Callable[[str, dict], ToolResult]
UsageCallback = Callable[[TokenUsage, TokenUsage], None]


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
        max_history_messages: Maximum number of non-system messages sent to
         the provider on each request. ``0`` (the default) means unlimited.
         Useful to stay within a local model's context window (llama.cpp /
         LM Studio / Ollama all return HTTP 400 when the prompt exceeds
         ``n_ctx``).
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
        usage_callback: UsageCallback | None = None,
        max_history_messages: int = 0,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.proxy = proxy
        self.mainwindow = mainwindow
        self.confirm_callback = confirm_callback
        self.execute_callback = execute_callback
        self.usage_callback = usage_callback
        self.system_prompt = system_prompt or build_default_system_prompt()
        self.max_iterations = int(max_iterations)
        self.auto_approve_readonly = bool(auto_approve_readonly)
        self.max_history_messages = max(0, int(max_history_messages))
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        # Cumulative token usage across the whole conversation. Reset on
        # :meth:`reset` and :meth:`load_messages`.
        self._cumulative_usage: TokenUsage = TokenUsage()
        # Set while :meth:`send` is in flight to support :meth:`abort`.
        self._abort_event = threading.Event()
        self._running = False

    @classmethod
    def with_default_prompt(
        cls,
        provider: LLMProvider,
        registry: ToolRegistry,
        proxy: LocalProxy,
        mainwindow: DLMainWindow,
        confirm_callback: ConfirmCallback,
        max_iterations: int = 8,
        auto_approve_readonly: bool = True,
        execute_callback: ExecuteCallback | None = None,
        usage_callback: UsageCallback | None = None,
        max_history_messages: int = 0,
    ) -> AIController:
        """Build a controller whose system prompt matches ``registry``."""
        tool_names = {schema["name"] for schema in registry.list_schemas()}
        return cls(
            provider=provider,
            registry=registry,
            proxy=proxy,
            mainwindow=mainwindow,
            confirm_callback=confirm_callback,
            system_prompt=build_default_system_prompt(tool_names),
            max_iterations=max_iterations,
            auto_approve_readonly=auto_approve_readonly,
            execute_callback=execute_callback,
            usage_callback=usage_callback,
            max_history_messages=max_history_messages,
        )

    def reset(self) -> None:
        """Clear the conversation history (keep the system prompt)."""
        self.history = [ChatMessage(role="system", content=self.system_prompt)]
        self._cumulative_usage = TokenUsage()

    def load_messages(
        self,
        messages: list[ChatMessage],
        initial_usage: TokenUsage | None = None,
    ) -> None:
        """Replace the current history with ``messages``.

        The system prompt is always reset to the controller's current
        ``system_prompt`` (which is auto-generated and may have changed
        since the conversation was persisted). Any system messages from
        ``messages`` are dropped.

        Args:
            messages: Messages to restore (system messages are dropped).
            initial_usage: Cumulative token usage to seed the running
             counter with — typically the value persisted alongside the
             conversation. ``None`` resets it to zero.
        """
        self.history = [ChatMessage(role="system", content=self.system_prompt)]
        for msg in messages:
            if msg.role == "system":
                continue
            self.history.append(msg)
        self._cumulative_usage = (
            TokenUsage(
                prompt_tokens=initial_usage.prompt_tokens,
                completion_tokens=initial_usage.completion_tokens,
                total_tokens=initial_usage.total_tokens,
            )
            if initial_usage is not None
            else TokenUsage()
        )

    def get_messages(self) -> list[ChatMessage]:
        """Return the conversation messages, excluding the system prompt."""
        return [msg for msg in self.history if msg.role != "system"]

    def get_usage(self) -> TokenUsage:
        """Snapshot of the cumulative token usage across the conversation."""
        return TokenUsage(
            prompt_tokens=self._cumulative_usage.prompt_tokens,
            completion_tokens=self._cumulative_usage.completion_tokens,
            total_tokens=self._cumulative_usage.total_tokens,
        )

    def abort(self) -> None:
        """Request cancellation of the in-flight :meth:`send` call.

        Idempotent — a no-op when the controller is idle. The abort takes
        effect at the next safe point in the loop (between provider calls
        / between tool calls). The currently-running provider HTTP request
        is **not** interrupted; the abort is observed once it returns.
        """
        self._abort_event.set()

    @property
    def is_running(self) -> bool:
        """True while a :meth:`send` call is in flight."""
        return self._running

    def send(self, user_message: str) -> TurnResult:
        """Send a user message and run the tool-call loop.

        On any unhandled exception during the turn (network failure, GUI
        bridge error, etc.), the conversation history is rolled back to its
        state before this call. This guarantees that the OpenAI invariant
        "every assistant.tool_calls must be followed by matching tool
        responses" is never violated in the persisted history — a corrupt
        partial turn would otherwise poison every subsequent ``send()``.
        The user message is preserved in the GUI input history regardless.

        When :meth:`abort` is called while the loop is in flight, the call
        returns a :class:`TurnResult` with ``aborted=True`` (the partial
        transcript is rolled back so the persisted history stays valid).
        """
        snapshot_len = len(self.history)
        self._abort_event.clear()
        self._running = True
        try:
            return self._send_inner(user_message)
        except AIAbortError:
            del self.history[snapshot_len:]
            return TurnResult(
                assistant_message="",
                aborted=True,
            )
        except BaseException:
            del self.history[snapshot_len:]
            raise
        finally:
            self._running = False

    def _check_abort(self) -> None:
        if self._abort_event.is_set():
            raise AIAbortError()

    def _messages_for_provider(self) -> list[ChatMessage]:
        """Return the (possibly truncated) message window sent to the provider.

        When :attr:`max_history_messages` is positive, only the latest N
        non-system messages are kept. Leading messages are then trimmed so
        the window always starts on a ``user`` message — this avoids
        leaving an orphan ``tool`` reply or an assistant ``tool_calls``
        whose responses were dropped, which most providers reject.

        The current (in-flight) turn is always preserved: if the cap is
        smaller than the messages produced so far in this turn, the
        window falls back to ``[latest user message ... end]``.
        """
        if self.max_history_messages <= 0:
            return list(self.history)
        non_system = [m for m in self.history if m.role != "system"]
        window = non_system[-self.max_history_messages :]
        while window and window[0].role != "user":
            window.pop(0)
        if not window:
            for idx in range(len(non_system) - 1, -1, -1):
                if non_system[idx].role == "user":
                    window = non_system[idx:]
                    break
        return [ChatMessage(role="system", content=self.system_prompt), *window]

    def _send_inner(self, user_message: str) -> TurnResult:
        self.history.append(ChatMessage(role="user", content=user_message))
        executions: list[tuple[str, dict, ToolResult]] = []
        tools_schema = self.registry.list_schemas()
        turn_usage: TokenUsage = TokenUsage()

        for _iteration in range(self.max_iterations):
            self._check_abort()
            response: AssistantMessage = self.provider.chat(
                self._messages_for_provider(), tools=tools_schema
            )
            self.history.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
            )
            if response.usage is not None:
                self._cumulative_usage = sum_usage(
                    self._cumulative_usage, response.usage
                )
                turn_usage = sum_usage(turn_usage, response.usage)
                if self.usage_callback is not None:
                    try:
                        self.usage_callback(response.usage, self.get_usage())
                    # pylint: disable-next=broad-exception-caught
                    except BaseException:  # noqa: BLE001
                        # The callback is a UI hook — never let it break
                        # the conversation loop.
                        pass
            self._check_abort()
            if not response.tool_calls:
                return TurnResult(
                    assistant_message=response.content,
                    tool_executions=executions,
                    turn_usage=turn_usage if response.usage is not None else None,
                )
            for call_index, call in enumerate(response.tool_calls):
                self._check_abort()
                try:
                    tool = self.registry.get(call.name)
                except KeyError as exc:
                    result = ToolResult(ok=False, error=str(exc))
                else:
                    needs_confirm = not (tool.readonly and self.auto_approve_readonly)
                    if needs_confirm and not self.confirm_callback(
                        tool, call.arguments
                    ):
                        # User cancelled. The OpenAI protocol requires every
                        # 'tool_calls' entry to be followed by a matching
                        # 'tool' response message; otherwise the next request
                        # is rejected as malformed. Synthesize a cancellation
                        # response for THIS call and every remaining call in
                        # the same turn before returning.
                        cancel_result = ToolResult(ok=False, error="Cancelled by user.")
                        for pending in response.tool_calls[call_index:]:
                            self.history.append(
                                ChatMessage(
                                    role="tool",
                                    content=cancel_result.to_message_content(),
                                    tool_call_id=pending.id,
                                    name=pending.name,
                                )
                            )
                        return TurnResult(
                            assistant_message=response.content,
                            tool_executions=executions,
                            cancelled=True,
                            turn_usage=turn_usage,
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
            turn_usage=turn_usage,
        )
