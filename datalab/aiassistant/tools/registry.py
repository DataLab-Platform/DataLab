# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Tool registry: declarative definition of LLM-callable tools.

Each tool exposes a :class:`Tool` dataclass containing:

- a name and a description (sent to the LLM),
- a JSON schema (sent to the LLM, OpenAI function-calling format),
- a handler ``(proxy, mainwindow, **kwargs) -> dict``,
- a ``readonly`` flag (read-only inspection tools may be auto-approved).

The registry validates the arguments against the JSON schema before calling
the handler, for every client (AI Assistant chat, MCP server).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from datalab.aiassistant.tools.validation import validate_arguments

if TYPE_CHECKING:
    from datalab.aiassistant.providers.base import ChatMessage
    from datalab.control.proxy import LocalProxy
    from datalab.gui.main import DLMainWindow


ToolHandler = Callable[..., Any]

#: Kinds of tool errors, reported to the clients with the error message
ERROR_INVALID = "invalid"  # Arguments not matching the schema of the tool
ERROR_REFUSED = "refused"  # Call refused by the user or by the exposure policy
ERROR_FAILED = "failed"  # Error raised while running the tool
ERROR_CANCELLED = "cancelled"  # Call cancelled by the client
ERROR_UNKNOWN_TOOL = "unknown_tool"  # No tool registered under this name

_NON_INTERACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "datalab_tools_non_interactive", default=False
)


@contextlib.contextmanager
def non_interactive() -> Iterator[None]:
    """Run tools without any dialog box (e.g. for the MCP server).

    In this context, the tool handlers must neither open a parameter dialog nor
    ask the user to select an object: missing information raises an error.
    """
    token = _NON_INTERACTIVE.set(True)
    try:
        yield
    finally:
        _NON_INTERACTIVE.reset(token)


def is_non_interactive() -> bool:
    """Return True if tools run in the :func:`non_interactive` context."""
    return _NON_INTERACTIVE.get()


@dataclass
class Tool:
    """Definition of an LLM-callable tool.

    Args:
        name: Tool name (must be a valid Python identifier).
        description: Human-readable description (sent to the LLM).
        parameters: JSON schema of the tool parameters (OpenAI format).
        handler: Callable invoked with ``(proxy, mainwindow, **arguments)``
         and returning a JSON-serializable value.
        readonly: If True, the tool is considered safe (read-only) and may
         be auto-approved depending on user preferences.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    readonly: bool = False

    def needs_confirmation(self, auto_approve_readonly: bool) -> bool:
        """Return True if the user must confirm a call of this tool.

        Args:
            auto_approve_readonly: True if read-only tools are approved without
             confirmation.
        """
        return not (self.readonly and auto_approve_readonly)

    def to_schema(self) -> dict[str, Any]:
        """Return the JSON schema sent to the LLM (OpenAI format)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolResult:
    """Result of a tool execution.

    Args:
        ok: True if execution succeeded.
        data: Tool result payload (JSON-serializable).
        error: Error message if execution failed.
        followup_messages: Extra messages (e.g. a user message carrying an
         image) to append to the conversation history right after the tool
         result message. Useful for multimodal payloads that cannot live in
         a ``role:"tool"`` message.
        error_kind: Kind of error (one of the ``ERROR_*`` constants), if
         execution failed.
        images: Images produced by the tool, as ``(mime_type, base64_data)``
         tuples (e.g. a view capture), for the clients supporting them.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    followup_messages: list[ChatMessage] = field(default_factory=list)
    error_kind: str | None = None
    images: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ok and self.error_kind is None:
            self.error_kind = ERROR_FAILED

    def to_message_content(self) -> str:
        """Return the JSON content sent back to the LLM."""
        if self.ok:
            try:
                return json.dumps({"ok": True, "result": self.data}, default=str)
            except (TypeError, ValueError):
                return json.dumps({"ok": True, "result": str(self.data)})
        return json.dumps({"ok": False, "error": self.error or "unknown error"})


@dataclass
class ToolRegistry:
    """Registry of LLM-callable tools."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Register ``tool``. Raises ``ValueError`` if name is already used."""
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Raises:
            KeyError: if no tool is registered under ``name``.
        """
        try:
            return self.tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.tools)) or "<none>"
            raise KeyError(
                f"Unknown tool {name!r}. Available tools: {available}."
            ) from exc

    def list_schemas(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas (OpenAI format)."""
        return [tool.to_schema() for tool in self.tools.values()]

    def without(self, names: Iterable[str]) -> ToolRegistry:
        """Return a new registry without the tools named in ``names``.

        Args:
            names: Names of the tools to leave out. Unknown names are ignored.
        """
        excluded = set(names)
        return ToolRegistry(
            {name: tool for name, tool in self.tools.items() if name not in excluded}
        )

    def check_arguments(self, name: str, arguments: Any) -> ToolResult | None:
        """Check that ``name`` is a registered tool and ``arguments`` are valid.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            A failed result describing the problem, or None if the call is valid.
        """
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(ok=False, error=str(exc), error_kind=ERROR_UNKNOWN_TOOL)
        errors = validate_arguments(tool.parameters, arguments)
        if errors:
            return ToolResult(
                ok=False,
                error=f"Invalid arguments for tool {name!r}: " + "; ".join(errors),
                error_kind=ERROR_INVALID,
            )
        return None

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        proxy: LocalProxy,
        mainwindow: DLMainWindow,
    ) -> ToolResult:
        """Invoke the tool registered under ``name``.

        Args:
            name: Tool name.
            arguments: Tool arguments, validated against the JSON schema.
            proxy: DataLab local proxy.
            mainwindow: DataLab main window.

        Returns:
            Tool execution result.
        """
        failure = self.check_arguments(name, arguments)
        if failure is not None:
            return failure
        tool = self.get(name)
        try:
            data = tool.handler(proxy, mainwindow, **arguments)
        except Exception as exc:  # pylint: disable=broad-except
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        if isinstance(data, ToolResult):
            return data
        return ToolResult(ok=True, data=data)
