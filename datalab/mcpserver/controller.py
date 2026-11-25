# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Lifecycle of the MCP server: start, stop and token management.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Signal

from datalab.config import Conf
from datalab.mcpserver import MCP_AVAILABLE, auth, mcp_unavailable_message

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.mcpserver.executor import McpToolExecutor

HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 18090
#: Maximum duration of the server start (seconds)
START_TIMEOUT = 10.0


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Return the first free port of the loopback interface from ``start_port``.

    Raises:
        RuntimeError: No free port in the range.
    """
    for port in range(start_port, min(start_port + max_attempts, 65536)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((HOST, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found in the range {start_port}-{start_port + max_attempts}"
    )


class McpServerController(QObject):
    """Controller of the MCP server of a DataLab main window.

    Signals:
        server_started: Emitted when the server starts (URL).
        server_stopped: Emitted when the server stops.
    """

    server_started = Signal(str)
    server_stopped = Signal()

    def __init__(self, mainwindow: DLMainWindow) -> None:
        super().__init__()
        self._mainwindow = mainwindow
        self._thread: threading.Thread | None = None
        self._uvicorn_server = None
        self._executor: McpToolExecutor | None = None
        self._url: str | None = None
        self._token: str | None = None
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """True if the server is running."""
        return self._url is not None

    @property
    def url(self) -> str | None:
        """URL of the MCP endpoint, or None if the server is not running."""
        return self._url

    @property
    def token(self) -> str | None:
        """Token of the server, or None if the server is not running."""
        return self._token

    def _get_token(self) -> str:
        # Called by the server thread: a stopped server accepts no token
        return self._token or secrets_placeholder()

    def start(self, port: int | None = None) -> str:
        """Start the server and return the URL of its MCP endpoint.

        Args:
            port: Port to listen on. Defaults to the ``DATALAB_MCP_PORT``
             environment variable, else the first free port from the preferred
             port of the settings.

        Raises:
            RuntimeError: The server is already running, the MCP SDK is not
             installed, or the server could not start.
        """
        if self.is_running:
            raise RuntimeError("The MCP server is already running.")
        if not MCP_AVAILABLE:
            raise RuntimeError(mcp_unavailable_message())
        # pylint: disable=import-outside-toplevel
        import uvicorn

        from datalab.mcpserver.app import MCP_PATH, build_app
        from datalab.mcpserver.executor import McpToolExecutor
        from datalab.mcpserver.policy import build_mcp_registry

        if port is None:
            env_port = os.environ.get("DATALAB_MCP_PORT")
            if env_port:
                port = int(env_port)
            else:
                port = find_available_port(Conf.mcp_port.get(DEFAULT_MCP_PORT))
        token = auth.get_token(Conf.mcp_persistent_token.get(False))
        executor = McpToolExecutor(self._mainwindow, build_mcp_registry())
        config = uvicorn.Config(
            build_app(executor, port, self._get_token),
            host=HOST,
            port=port,
            lifespan="on",
            log_level="warning",
            access_log=False,
            log_config=None,
            timeout_graceful_shutdown=2,
        )
        server = uvicorn.Server(config)
        errors: list[str] = []
        thread = threading.Thread(
            target=self._run_server,
            args=(server, errors),
            name="DataLab-MCP",
            daemon=True,
        )
        self._token = token
        thread.start()
        deadline = time.monotonic() + START_TIMEOUT
        while not server.started and thread.is_alive():
            if time.monotonic() > deadline:
                server.should_exit = True
                break
            time.sleep(0.02)
        if not server.started:
            thread.join(timeout=5.0)
            self._token = None
            detail = errors[0] if errors else f"port {port} may be in use"
            raise RuntimeError(f"The MCP server could not start ({detail}).")
        self._uvicorn_server = server
        self._thread = thread
        self._executor = executor
        self._url = f"http://{HOST}:{port}{MCP_PATH}"
        self.server_started.emit(self._url)
        return self._url

    @staticmethod
    def _run_server(server, errors: list[str]) -> None:
        """Run the Uvicorn server (in the server thread)."""
        try:
            server.run()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            # Uvicorn exits (SystemExit) when it cannot bind its port
            errors.append(f"{type(exc).__name__}: {exc}")

    def stop(self) -> None:
        """Stop the server, cancelling the calls that have not started yet."""
        if not self.is_running or self._stopping:
            return
        self._stopping = True
        try:
            if self._executor is not None:
                self._executor.close()
            if self._uvicorn_server is not None:
                self._uvicorn_server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5.0)
        finally:
            self._thread = None
            self._uvicorn_server = None
            self._executor = None
            self._url = None
            self._token = None
            self._stopping = False
        self.server_stopped.emit()

    def renew_token(self) -> str:
        """Replace the token: clients using the former token are refused.

        Returns:
            The new token.
        """
        token = auth.renew_token(Conf.mcp_persistent_token.get(False))
        if self.is_running:
            self._token = token
        return token


def secrets_placeholder() -> str:
    """Return a random value matching no token (server stopped)."""
    return auth.generate_token()
