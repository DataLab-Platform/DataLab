# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Execution of the MCP tool calls on the Qt main thread.

The MCP server runs in a thread of its own. Each tool call is run on the Qt main
thread, one at a time, in the same steps as a call of the AI Assistant panel:
argument validation, confirmation by the user when the tool modifies the
workspace (unless the calls of external assistants are approved automatically),
then execution through the shared tool registry, without any other
dialog box.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import TYPE_CHECKING, Any

from datalab.aiassistant.tools.builtin import freeze_operation_targets
from datalab.aiassistant.tools.registry import (
    ERROR_CANCELLED,
    ERROR_INVALID,
    ERROR_REFUSED,
    ToolResult,
    non_interactive,
)
from datalab.aiassistant.widgets.toolconfirmdialog import ToolConfirmDialog
from datalab.config import Conf
from datalab.control.proxy import LocalProxy
from datalab.webapi.adapter import MainThreadExecutor

if TYPE_CHECKING:
    from datalab.aiassistant.tools.registry import ToolRegistry
    from datalab.gui.main import DLMainWindow

#: Maximum duration of a tool call, including its confirmation (seconds)
DEFAULT_CALL_TIMEOUT = 300.0


def _cancelled_result(message: str) -> ToolResult:
    return ToolResult(ok=False, error=message, error_kind=ERROR_CANCELLED)


class McpToolExecutor:
    """Run the tool calls of the MCP server on the Qt main thread.

    Args:
        mainwindow: DataLab main window.
        registry: Registry of the published tools.
        timeout: Maximum duration of a call (seconds).
    """

    def __init__(
        self,
        mainwindow: DLMainWindow,
        registry: ToolRegistry,
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> None:
        self._mainwindow = mainwindow
        self._registry = registry
        self._proxy = LocalProxy(mainwindow)
        self._executor = MainThreadExecutor()
        self._timeout = timeout
        self._busy = False
        self._closed = False
        self._lock: asyncio.Lock | None = None
        self._pending: set[concurrent.futures.Future] = set()
        self._pending_lock = threading.Lock()

    @property
    def registry(self) -> ToolRegistry:
        """Registry of the published tools."""
        return self._registry

    async def call(
        self, name: str, arguments: dict[str, Any], client: str | None = None
    ) -> ToolResult:
        """Run a tool call on the Qt main thread and return its result.

        Calls are run one at a time, in their order of arrival.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            client: Name of the MCP client, shown in the confirmation dialog.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        cancel = threading.Event()
        async with self._lock:
            if self._closed:
                return _cancelled_result("The DataLab MCP server is stopping.")
            future = self._executor.submit(
                lambda: self._run(name, arguments, client, cancel)
            )
            with self._pending_lock:
                self._pending.add(future)
            try:
                return await asyncio.wait_for(
                    asyncio.wrap_future(future), self._timeout
                )
            except asyncio.TimeoutError:
                cancel.set()
                return _cancelled_result(
                    f"The call of {name!r} did not complete within "
                    f"{self._timeout:g} s (pending confirmation or computation)."
                )
            except asyncio.CancelledError:
                cancel.set()
                future.cancel()
                raise
            finally:
                with self._pending_lock:
                    self._pending.discard(future)

    def close(self) -> None:
        """Cancel the calls which have not started yet and refuse new calls."""
        self._closed = True
        with self._pending_lock:
            pending = list(self._pending)
        for future in pending:
            # A call already running on the main thread completes, but its result
            # is no longer awaited
            future.cancel()

    def _run(
        self,
        name: str,
        arguments: dict[str, Any],
        client: str | None,
        cancel: threading.Event,
    ) -> ToolResult:
        """Run a tool call (on the Qt main thread)."""
        if cancel.is_set():
            return _cancelled_result("The call was cancelled by the client.")
        if self._busy:
            # A confirmation dialog runs a nested event loop: refuse to start
            # another call meanwhile
            return ToolResult(
                ok=False,
                error="DataLab is already running a tool call.",
                error_kind=ERROR_REFUSED,
            )
        failure = self._registry.check_arguments(name, arguments)
        if failure is not None:
            return failure
        tool = self._registry.get(name)
        self._busy = True
        try:
            if name == "apply_operation":
                try:
                    arguments = freeze_operation_targets(self._mainwindow, arguments)
                except ValueError as exc:
                    return ToolResult(
                        ok=False, error=str(exc), error_kind=ERROR_INVALID
                    )
            # Read-only tools never need confirmation for external assistants
            if not Conf.mcp_auto_approve.get(False) and tool.needs_confirmation(True):
                source = client or "MCP"
                if not ToolConfirmDialog.confirm(
                    tool, arguments, self._mainwindow, source
                ):
                    return ToolResult(
                        ok=False,
                        error="The user refused to run this action.",
                        error_kind=ERROR_REFUSED,
                    )
            if cancel.is_set():
                return _cancelled_result("The call was cancelled by the client.")
            with self._mainwindow.remote_request_context(), non_interactive():
                return self._registry.call(
                    name, arguments, self._proxy, self._mainwindow
                )
        finally:
            self._busy = False
