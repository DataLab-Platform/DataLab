# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Controller
==================

Manages the lifecycle of the Web API server embedded in DataLab.

The controller is responsible for:

- Starting and stopping the Uvicorn server
- Token generation and management
- Port selection
- Thread-safe server lifecycle

Usage
-----

::

    from datalab.webapi import get_webapi_controller

    controller = get_webapi_controller()
    controller.set_main_window(main_window)
    controller.start()

    # Later...
    controller.stop()
"""

from __future__ import annotations

import os
import socket
import threading
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Signal

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


class WebApiController(QObject):
    """Controller for the DataLab Web API server.

    This class manages the lifecycle of an embedded Uvicorn server running
    FastAPI routes for the Web API.

    Signals:
        server_started: Emitted when server starts (url, token)
        server_stopped: Emitted when server stops
        server_error: Emitted on server error (message)
    """

    # Qt signals for status updates
    server_started = Signal(str, str)  # url, token
    server_stopped = Signal()
    server_error = Signal(str)

    def __init__(self) -> None:
        """Initialize the controller."""
        super().__init__()
        self._main_window: DLMainWindow | None = None
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server = None
        self._adapter = None
        self._token: str | None = None
        self._url: str | None = None
        self._running = False
        self._lock = threading.Lock()

    def set_main_window(self, main_window: DLMainWindow) -> None:
        """Set the DataLab main window reference.

        Args:
            main_window: The DataLab main window.
        """
        self._main_window = main_window

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        with self._lock:
            return self._running

    @property
    def url(self) -> str | None:
        """Get the server URL."""
        return self._url

    @property
    def token(self) -> str | None:
        """Get the authentication token."""
        return self._token

    def _find_available_port(
        self, start_port: int = 8080, max_attempts: int = 100
    ) -> int:
        """Find an available port.

        Args:
            start_port: Port to start searching from.
            max_attempts: Maximum number of ports to try.

        Returns:
            Available port number.

        Raises:
            RuntimeError: If no available port found.
        """
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(
            f"No available port found in range {start_port}-{start_port + max_attempts}"
        )

    def start(
        self,
        host: str | None = None,
        port: int | None = None,
        token: str | None = None,
    ) -> tuple[str, str]:
        """Start the Web API server.

        Args:
            host: Host to bind to. Defaults to DATALAB_WEBAPI_HOST or "127.0.0.1".
            port: Port to bind to. Defaults to DATALAB_WEBAPI_PORT or auto-detect.
            token: Authentication token. Defaults to DATALAB_WEBAPI_TOKEN or generated.

        Returns:
            Tuple of (url, token).

        Raises:
            RuntimeError: If server already running or main window not set.
        """
        if self._main_window is None:
            raise RuntimeError("Main window not set. Call set_main_window() first.")

        with self._lock:
            if self._running:
                raise RuntimeError("Server already running")
            self._running = True

        try:
            # Import here to allow graceful failure if deps not installed

            # pylint: disable=import-outside-toplevel
            import uvicorn
            from fastapi import FastAPI

            from datalab.webapi.adapter import WorkspaceAdapter
            from datalab.webapi.routes import (
                generate_auth_token,
                router,
                set_adapter,
                set_auth_token,
                set_server_url,
            )

            # Resolve configuration
            host = host or os.environ.get("DATALAB_WEBAPI_HOST", "127.0.0.1")
            if port is None:
                env_port = os.environ.get("DATALAB_WEBAPI_PORT")
                port = int(env_port) if env_port else self._find_available_port()

            token = (
                token or os.environ.get("DATALAB_WEBAPI_TOKEN") or generate_auth_token()
            )

            # Create adapter
            self._adapter = WorkspaceAdapter(self._main_window)

            # Configure routes
            set_adapter(self._adapter)
            set_auth_token(token)

            self._url = f"http://{host}:{port}"
            set_server_url(self._url)
            self._token = token

            # Create FastAPI app
            app = FastAPI(
                title="DataLab Web API",
                description="HTTP/JSON API for DataLab workspace access",
                version="1.0.0",
            )
            app.include_router(router)

            # Configure Uvicorn
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                log_config=None,  # Disable Uvicorn's logging config
            )
            self._uvicorn_server = uvicorn.Server(config)

            # Start server in thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="DataLab-WebAPI",
                daemon=True,
            )
            self._server_thread.start()

            # Emit signal
            self.server_started.emit(self._url, self._token)

            return self._url, self._token

        except Exception as e:
            with self._lock:
                self._running = False
            self.server_error.emit(str(e))
            raise

    def _run_server(self) -> None:
        """Run the Uvicorn server (called in thread)."""
        try:
            self._uvicorn_server.run()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.server_error.emit(str(e))
        finally:
            with self._lock:
                self._running = False
            self.server_stopped.emit()

    def stop(self) -> None:
        """Stop the Web API server."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True

        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)
            self._server_thread = None

        self._uvicorn_server = None
        self._adapter = None
        self._url = None
        self._token = None

        self.server_stopped.emit()

    def get_connection_info(self) -> dict:
        """Get connection information for clients.

        Returns:
            Dictionary with url, token, and running status.
        """
        return {
            "running": self.is_running,
            "url": self._url,
            "token": self._token,
        }
