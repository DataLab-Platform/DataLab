# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Tool registry: declarative definition of LLM-callable tools.

Each tool exposes a :class:`Tool` dataclass containing:

- a name and a description (sent to the LLM),
- a JSON schema (sent to the LLM, OpenAI function-calling format),
- a handler ``(proxy, mainwindow, **kwargs) -> dict``,
- a ``readonly`` flag (read-only inspection tools may be auto-approved).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from datalab.control.proxy import LocalProxy
    from datalab.gui.main import DLMainWindow


ToolHandler = Callable[..., Any]


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
    """

    ok: bool
    data: Any = None
    error: str | None = None

    def to_message_content(self) -> str:
        """Return the JSON content sent back to the LLM."""
        import json

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
            arguments: Tool arguments (must match the JSON schema).
            proxy: DataLab local proxy.
            mainwindow: DataLab main window.

        Returns:
            Tool execution result.
        """
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(ok=False, error=str(exc))
        try:
            data = tool.handler(proxy, mainwindow, **arguments)
        except Exception as exc:  # pylint: disable=broad-except
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(ok=True, data=data)
