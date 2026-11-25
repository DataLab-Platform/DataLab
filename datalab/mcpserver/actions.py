# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
GUI of the MCP server: menu action and connection dialog.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import create_action
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import APP_NAME, Conf, _
from datalab.mcpserver import MCP_AVAILABLE, mcp_unavailable_message
from datalab.mcpserver.controller import McpServerController

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

#: Identifier of the input prompting for the token in the VS Code configuration
VSCODE_TOKEN_INPUT = "datalab-mcp-token"


def vscode_configuration(url: str) -> str:
    """Return the VS Code ``mcp.json`` configuration connecting to ``url``.

    The token is not written in the configuration: VS Code prompts for it once
    and stores it securely.
    """
    config = {
        "servers": {
            "datalab": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer ${{input:{VSCODE_TOKEN_INPUT}}}"},
            }
        },
        "inputs": [
            {
                "type": "promptString",
                "id": VSCODE_TOKEN_INPUT,
                "description": "DataLab MCP token",
                "password": True,
            }
        ],
    }
    return json.dumps(config, indent=2)


class McpConnectionDialog(QW.QDialog):
    """Dialog controlling the MCP server and showing how to connect to it."""

    def __init__(self, controller: McpServerController, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle(_("Connect an external assistant"))
        self.setWindowIcon(get_icon("ai-assistant.svg"))
        self.setMinimumWidth(560)
        layout = QW.QVBoxLayout(self)

        intro = QW.QLabel(
            _(
                "The MCP server lets an external assistant (e.g. VS Code Copilot) "
                "use the tools of the AI Assistant on this DataLab session. It "
                "listens on this computer only and needs the token below. Unless "
                "you approve its calls automatically, you confirm every action "
                "modifying the workspace, as in the AI Assistant panel."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        status_layout = QW.QHBoxLayout()
        self._status_label = QW.QLabel(self)
        status_layout.addWidget(self._status_label, 1)
        self._start_stop_button = QW.QPushButton(self)
        self._start_stop_button.clicked.connect(self._toggle_server)
        status_layout.addWidget(self._start_stop_button)
        layout.addLayout(status_layout)

        form = QW.QFormLayout()
        self._url_edit = QW.QLineEdit(self)
        self._url_edit.setReadOnly(True)
        form.addRow(_("URL:"), self._with_button(self._url_edit, self._copy_url))
        self._token_edit = QW.QLineEdit(self)
        self._token_edit.setReadOnly(True)
        self._token_edit.setEchoMode(QW.QLineEdit.Password)
        token_widget = self._with_button(self._token_edit, self._copy_token)
        self._renew_button = QW.QPushButton(_("Renew"), self)
        self._renew_button.setToolTip(
            _("Replace the token: the assistants using the former token are refused")
        )
        self._renew_button.clicked.connect(self._renew_token)
        token_widget.layout().addWidget(self._renew_button)
        form.addRow(_("Token:"), token_widget)
        self._show_token_box = QW.QCheckBox(_("Show token"), self)
        self._show_token_box.toggled.connect(
            lambda state: self._token_edit.setEchoMode(
                QW.QLineEdit.Normal if state else QW.QLineEdit.Password
            )
        )
        form.addRow("", self._show_token_box)
        layout.addLayout(form)

        layout.addWidget(
            QW.QLabel(
                _("VS Code configuration (.vscode/mcp.json), prompting for the token:"),
                self,
            )
        )
        self._config_edit = QW.QPlainTextEdit(self)
        self._config_edit.setReadOnly(True)
        self._config_edit.setMinimumHeight(220)
        layout.addWidget(self._config_edit)

        self._token_hint = QW.QLabel(self)
        self._token_hint.setWordWrap(True)
        layout.addWidget(self._token_hint)

        self._auto_approve_box = QW.QCheckBox(
            _("Approve the calls automatically (no confirmation)"), self
        )
        self._auto_approve_box.setToolTip(
            _(
                "Run the tool calls of external assistants without confirmation, "
                "including those modifying the workspace"
            )
        )
        self._auto_approve_box.toggled.connect(Conf.mcp_auto_approve.set)
        layout.addWidget(self._auto_approve_box)

        button_box = QW.QDialogButtonBox(self)
        self._copy_config_button = button_box.addButton(
            _("Copy configuration"), QW.QDialogButtonBox.ActionRole
        )
        self._copy_config_button.clicked.connect(self._copy_configuration)
        close_button = button_box.addButton(QW.QDialogButtonBox.Close)
        close_button.clicked.connect(self.accept)
        layout.addWidget(button_box)

        controller.server_started.connect(self.update_state)
        controller.server_stopped.connect(self.update_state)
        self.update_state()

    def _with_button(self, edit: QW.QLineEdit, slot) -> QW.QWidget:
        """Return a widget with ``edit`` and a copy button calling ``slot``."""
        widget = QW.QWidget(self)
        layout = QW.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QW.QToolButton(widget)
        button.setIcon(get_icon("copy_connection_info.svg"))
        button.setToolTip(_("Copy to clipboard"))
        button.clicked.connect(slot)
        layout.addWidget(button)
        return widget

    def update_state(self, *_args) -> None:
        """Update the dialog from the state of the server."""
        running = self._controller.is_running
        if running:
            self._status_label.setText(_("Status: running"))
            self._start_stop_button.setText(_("Stop server"))
        else:
            self._status_label.setText(_("Status: not running"))
            self._start_stop_button.setText(_("Start server"))
        self._url_edit.setText(self._controller.url or "")
        self._token_edit.setText(self._controller.token or "")
        for widget in (self._url_edit, self._token_edit, self._renew_button):
            widget.setEnabled(running)
        self._copy_config_button.setEnabled(running)
        self._config_edit.setPlainText(
            vscode_configuration(self._controller.url) if running else ""
        )
        if Conf.mcp_persistent_token.get(False):
            hint = _(
                "The token is persistent: it stays the same when DataLab restarts "
                "(see the settings)."
            )
        else:
            hint = _(
                "The token changes at each start of the server (see the settings "
                "to keep it). VS Code keeps the token entered: to enter the new "
                "one, hover over the token input in mcp.json and choose Edit."
            )
        self._token_hint.setText(hint)
        # The option may have been changed in the settings meanwhile
        with QC.QSignalBlocker(self._auto_approve_box):
            self._auto_approve_box.setChecked(bool(Conf.mcp_auto_approve.get(False)))

    def _toggle_server(self) -> None:
        """Start or stop the server."""
        if self._controller.is_running:
            self._controller.stop()
            return
        try:
            self._controller.start()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            QW.QMessageBox.critical(self, APP_NAME, str(exc))

    def _copy_url(self) -> None:
        QW.QApplication.clipboard().setText(self._controller.url or "")

    def _copy_token(self) -> None:
        QW.QApplication.clipboard().setText(self._controller.token or "")

    def _copy_configuration(self) -> None:
        QW.QApplication.clipboard().setText(self._config_edit.toPlainText())

    def _renew_token(self) -> None:
        """Renew the token after confirmation."""
        answer = QW.QMessageBox.question(
            self,
            APP_NAME,
            _(
                "The connected assistants will be refused until they use the new "
                "token. Renew the token?"
            ),
        )
        if answer == QW.QMessageBox.Yes:
            self._controller.renew_token()
            self.update_state()


class McpActions:
    """Menu action and dialog of the MCP server of a DataLab main window."""

    def __init__(self, main_window: DLMainWindow) -> None:
        self._main_window = main_window
        self.controller = McpServerController(main_window)
        self._dialog: McpConnectionDialog | None = None
        self.action = create_action(
            main_window,
            _("Connect an external assistant (MCP)..."),
            icon=get_icon("ai-assistant.svg"),
            triggered=self.show_dialog,
            tip=_(
                "Let an external assistant (e.g. VS Code Copilot) use the tools "
                "of the AI Assistant on this session"
            ),
        )
        if not MCP_AVAILABLE:
            self.action.setEnabled(False)
            self.action.setToolTip(mcp_unavailable_message(_))

    def show_dialog(self) -> None:
        """Show the connection dialog."""
        if self._dialog is None:
            self._dialog = McpConnectionDialog(self.controller, self._main_window)
        else:
            self._dialog.update_state()
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def cleanup(self) -> None:
        """Stop the server and close the dialog (main window closing)."""
        self.controller.stop()
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None
