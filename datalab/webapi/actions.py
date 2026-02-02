# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API GUI Actions
===================

GUI actions for controlling the DataLab Web API server.

This module provides menu actions and status display for the Web API feature.
It integrates with the DataLab main window to provide UI controls for:

- Starting/stopping the Web API server
- Viewing connection information (URL, token)
- Copying connection info to clipboard
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtWidgets as QW

from datalab.config import APP_NAME, _

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


class WebApiActions:
    """Manager for Web API GUI actions.

    This class creates and manages the menu actions for the Web API feature.
    It handles the server lifecycle through the WebApiController.

    Attributes:
        main_window: Reference to the DataLab main window.
    """

    def __init__(self, main_window: DLMainWindow) -> None:
        """Initialize Web API actions.

        Args:
            main_window: The DataLab main window.
        """
        self._main_window = main_window
        self._controller = None
        self._menu: QW.QMenu | None = None
        self._start_action: QW.QAction | None = None
        self._stop_action: QW.QAction | None = None
        self._copy_action: QW.QAction | None = None
        self._status_action: QW.QAction | None = None

        self._init_controller()
        self._create_actions()

    def _init_controller(self) -> None:
        """Initialize the Web API controller if available."""
        try:
            # pylint: disable=import-outside-toplevel
            from datalab.webapi import WEBAPI_AVAILABLE, get_webapi_controller

            if WEBAPI_AVAILABLE:
                self._controller = get_webapi_controller()
                self._controller.set_main_window(self._main_window)
                self._controller.server_started.connect(self._on_server_started)
                self._controller.server_stopped.connect(self._on_server_stopped)
                self._controller.server_error.connect(self._on_server_error)
        except ImportError:
            self._controller = None

    def _create_actions(self) -> None:
        """Create menu actions."""
        available = self._controller is not None

        # Start action
        self._start_action = create_action(
            self._main_window,
            _("Start Web API Server"),
            icon=get_icon("start_webapi_server.svg"),
            triggered=self._start_server,
            tip=_("Start the HTTP/JSON Web API server for external access"),
        )
        self._start_action.setEnabled(available)

        # Stop action
        self._stop_action = create_action(
            self._main_window,
            _("Stop Web API Server"),
            icon=get_icon("stop_webapi_server.svg"),
            triggered=self._stop_server,
        )
        self._stop_action.setEnabled(False)

        # Copy connection info action
        self._copy_action = create_action(
            self._main_window,
            _("Copy Connection Info"),
            icon=get_icon("copy_connection_info.svg"),
            triggered=self._copy_connection_info,
            tip=_("Copy URL and token to clipboard"),
        )
        self._copy_action.setEnabled(False)

        # Status indicator (not clickable)
        self._status_action = create_action(
            self._main_window,
            _("Status: Not running"),
        )
        self._status_action.setEnabled(False)

        if not available:
            self._status_action.setText(
                _("Web API unavailable (install datalab-platform[webapi])")
            )

    def create_menu(self, parent_menu: QW.QMenu) -> QW.QMenu:
        """Create the Web API submenu.

        Args:
            parent_menu: Parent menu to add submenu to.

        Returns:
            The created submenu.
        """
        self._menu = parent_menu.addMenu(_("Web API"))
        add_actions(
            self._menu,
            [
                self._start_action,
                self._stop_action,
                None,
                self._copy_action,
                None,
                self._status_action,
            ],
        )
        return self._menu

    def _start_server(self) -> None:
        """Start the Web API server."""
        if self._controller is None:
            return

        try:
            url, token = self._controller.start()
            self._show_connection_dialog(url, token)
        except Exception as e:  # pylint: disable=broad-exception-caught
            QW.QMessageBox.critical(
                self._main_window,
                APP_NAME,
                _("Failed to start Web API server:") + f"\n{e}",
            )

    def _stop_server(self) -> None:
        """Stop the Web API server."""
        if self._controller is None:
            return

        self._controller.stop()

    def _copy_connection_info(self) -> None:
        """Copy connection info to clipboard."""
        if self._controller is None or not self._controller.is_running:
            return

        info = self._controller.get_connection_info()
        text = f"URL: {info['url']}\nToken: {info['token']}"

        clipboard = QW.QApplication.clipboard()
        clipboard.setText(text)

        # Show brief notification in status bar
        self._main_window.statusBar().showMessage(
            _("Connection info copied to clipboard"), 3000
        )

    def _show_connection_dialog(self, url: str, token: str) -> None:
        """Show dialog with connection information.

        Args:
            url: Server URL.
            token: Authentication token.
        """
        dialog = QW.QDialog(self._main_window)
        dialog.setWindowTitle(_("Web API Server Started"))
        dialog.setMinimumWidth(450)

        layout = QW.QVBoxLayout(dialog)

        # Info label
        info_label = QW.QLabel(
            _(
                "The Web API server is now running. "
                "Use the following credentials to connect:"
            )
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # URL field
        url_layout = QW.QHBoxLayout()
        url_layout.addWidget(QW.QLabel(_("URL:")))
        url_edit = QW.QLineEdit(url)
        url_edit.setReadOnly(True)
        url_layout.addWidget(url_edit)
        layout.addLayout(url_layout)

        # Token field
        token_layout = QW.QHBoxLayout()
        token_layout.addWidget(QW.QLabel(_("Token:")))
        token_edit = QW.QLineEdit(token)
        token_edit.setReadOnly(True)
        token_layout.addWidget(token_edit)
        layout.addLayout(token_layout)

        # Environment variable hint
        hint_label = QW.QLabel(
            _("Tip: Set these environment variables in your notebook:\n")
            + f"DATALAB_WORKSPACE_URL={url}\n"
            + f"DATALAB_WORKSPACE_TOKEN={token}"
        )
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint_label)

        # Buttons
        button_box = QW.QDialogButtonBox()

        copy_btn = button_box.addButton(
            _("Copy to Clipboard"), QW.QDialogButtonBox.ActionRole
        )
        copy_btn.clicked.connect(self._copy_connection_info)

        close_btn = button_box.addButton(QW.QDialogButtonBox.Close)
        close_btn.clicked.connect(dialog.accept)

        layout.addWidget(button_box)

        dialog.exec_()

    def _on_server_started(self, url: str, _token: str) -> None:
        """Handle server started signal."""
        self._start_action.setEnabled(False)
        self._stop_action.setEnabled(True)
        self._copy_action.setEnabled(True)
        self._status_action.setText(_("Status: Running at {}").format(url))
        # Update status bar widget
        if self._main_window.webapistatus is not None:
            # Extract port from URL
            try:
                # pylint: disable=import-outside-toplevel
                from urllib.parse import urlparse

                parsed = urlparse(url)
                port = parsed.port
            except Exception:  # pylint: disable=broad-exception-caught
                port = None
            self._main_window.webapistatus.set_status(url, port)

    def _on_server_stopped(self) -> None:
        """Handle server stopped signal."""
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)
        self._copy_action.setEnabled(False)
        self._status_action.setText(_("Status: Not running"))
        # Update status bar widget
        if self._main_window.webapistatus is not None:
            self._main_window.webapistatus.set_status(None, None)

    def _on_server_error(self, message: str) -> None:
        """Handle server error signal."""
        QW.QMessageBox.warning(
            self._main_window,
            APP_NAME,
            _("Web API server error:") + f"\n{message}",
        )
        self._on_server_stopped()

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self._controller is not None and self._controller.is_running:
            self._controller.stop()

    def show_connection_info(self) -> None:
        """Show connection info dialog or copy to clipboard.

        This is called when the status widget is clicked while server is running.
        """
        if self._controller is None or not self._controller.is_running:
            return

        # Show the connection info dialog
        info = self._controller.get_connection_info()
        self._show_connection_dialog(info["url"], info["token"])

    def start_server_from_status_widget(self) -> None:
        """Start server from status widget click.

        This is called when the status widget is clicked while server is not running.
        Shows a confirmation dialog before starting the server.
        """
        # Show confirmation dialog
        answer = QW.QMessageBox.question(
            self._main_window,
            _("Start Web API Server"),
            _(
                "Do you want to start the Web API server?\n\n"
                "This will allow external applications to connect to DataLab "
                "and control it remotely via HTTP/JSON."
            ),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )

        if answer == QW.QMessageBox.Yes:
            self._start_server()
