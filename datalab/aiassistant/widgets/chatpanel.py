# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
AI Assistant chat dock panel.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.aiassistant.controller import AIController
from datalab.aiassistant.providers import get_provider
from datalab.aiassistant.tools.builtin import build_default_registry
from datalab.aiassistant.widgets.toolconfirmdialog import ToolConfirmDialog
from datalab.aiassistant.worker import AIWorker
from datalab.config import Conf, _
from datalab.control.proxy import LocalProxy

if TYPE_CHECKING:
    from datalab.aiassistant.controller import TurnResult
    from datalab.aiassistant.tools.registry import Tool
    from datalab.gui.main import DLMainWindow


class AIAssistantPanel(QW.QWidget, DockableWidgetMixin):
    """AI Assistant chat dock.

    Args:
        mainwindow: DataLab main window.
        parent: Parent widget.
    """

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("AI Assistant")

    SIG_OBJECT_MODIFIED = QC.Signal()

    def __init__(
        self,
        mainwindow: DLMainWindow,
        parent: QW.QWidget | None = None,
    ) -> None:
        QW.QWidget.__init__(self, parent)
        DockableWidgetMixin.__init__(self)
        self.mainwindow = mainwindow
        self._proxy = LocalProxy(mainwindow)
        self._controller: AIController | None = None
        self._worker: AIWorker | None = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QW.QHBoxLayout()
        self.new_button = QW.QPushButton(_("New conversation"))
        self.new_button.clicked.connect(self._on_new_conversation)
        self.settings_button = QW.QPushButton(_("Settings…"))
        self.settings_button.clicked.connect(self._on_open_settings)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.settings_button)
        toolbar.addStretch(1)
        self.status_label = QW.QLabel(_("Idle"))
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        # History view
        self.history_view = QW.QTextBrowser()
        self.history_view.setOpenExternalLinks(True)
        layout.addWidget(self.history_view, 1)

        # Input
        self.input_edit = QW.QPlainTextEdit()
        self.input_edit.setPlaceholderText(
            _("Ask the assistant…  (Ctrl+Enter to send)")
        )
        self.input_edit.installEventFilter(self)
        layout.addWidget(self.input_edit)

        send_row = QW.QHBoxLayout()
        send_row.addStretch(1)
        self.send_button = QW.QPushButton(_("Send"))
        self.send_button.clicked.connect(self._on_send)
        send_row.addWidget(self.send_button)
        layout.addLayout(send_row)

        self._append_system(
            _(
                "AI Assistant ready. Configure your provider through the "
                "Settings… button before sending the first message."
            )
        )

    def eventFilter(  # noqa: N802 - Qt API
        self, obj: QC.QObject, event: QC.QEvent
    ) -> bool:
        if obj is self.input_edit and event.type() == QC.QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if (
                key in (QC.Qt.Key_Return, QC.Qt.Key_Enter)
                and mods & QC.Qt.ControlModifier
            ):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # ----------------------------------------------------------- helpers

    def _append_html(self, html_text: str) -> None:
        self.history_view.append(html_text)
        cursor = self.history_view.textCursor()
        cursor.movePosition(QG.QTextCursor.End)
        self.history_view.setTextCursor(cursor)

    def _append_user(self, text: str) -> None:
        self._append_html(
            f"<div style='margin-top:6px;'><b>{_('You')}:</b><br>"
            f"{html.escape(text).replace(chr(10), '<br>')}</div>"
        )

    def _append_assistant(self, text: str) -> None:
        if not text:
            return
        self._append_html(
            f"<div style='margin-top:6px;color:#1a5fb4;'>"
            f"<b>{_('Assistant')}:</b><br>"
            f"{html.escape(text).replace(chr(10), '<br>')}</div>"
        )

    def _append_tool(self, name: str, ok: bool, summary: str) -> None:
        color = "#26a269" if ok else "#c01c28"
        symbol = "✓" if ok else "✗"
        self._append_html(
            f"<div style='margin-top:4px;color:{color};font-family:monospace;'>"
            f"{symbol} {html.escape(name)}: {html.escape(summary)}</div>"
        )

    def _append_system(self, text: str) -> None:
        self._append_html(
            f"<div style='margin-top:4px;color:#666;font-style:italic;'>"
            f"{html.escape(text)}</div>"
        )

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.input_edit.setEnabled(not busy)
        self.status_label.setText(_("Thinking…") if busy else _("Idle"))

    # --------------------------------------------------------- controller

    def _build_controller(self) -> AIController | None:
        from datalab.aiassistant.providers import PROVIDERS  # noqa: WPS433

        provider_name = Conf.ai.provider.get("openai")
        if provider_name not in PROVIDERS:
            provider_name = "openai"
        api_key = Conf.ai.api_key.get("")
        model = Conf.ai.model.get("gpt-4o-mini")
        base_url = Conf.ai.base_url.get("") or None
        temperature = float(Conf.ai.temperature.get(0.2))
        timeout = float(Conf.ai.timeout.get(60.0))
        max_iterations = int(Conf.ai.max_iterations.get(8))
        auto_approve = bool(Conf.ai.auto_approve_readonly.get(True))

        if not api_key and provider_name != "mock":
            QW.QMessageBox.warning(
                self,
                _("AI Assistant"),
                _(
                    "No API key configured.\n\n"
                    "Open the Settings… dialog to enter your "
                    "provider credentials, or select the 'mock' "
                    "provider for offline testing."
                ),
            )
            return None

        try:
            provider_cls = get_provider(provider_name)
        except KeyError as exc:
            QW.QMessageBox.critical(self, _("AI Assistant"), str(exc))
            return None
        provider = provider_cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )
        registry = build_default_registry()
        return AIController(
            provider=provider,
            registry=registry,
            proxy=self._proxy,
            mainwindow=self.mainwindow,
            confirm_callback=self._confirm_tool,
            max_iterations=max_iterations,
            auto_approve_readonly=auto_approve,
        )

    def _confirm_tool(self, tool: Tool, arguments: dict) -> bool:
        return ToolConfirmDialog.confirm(tool, arguments, self)

    # -------------------------------------------------------- slots

    def _on_new_conversation(self) -> None:
        if self._controller is not None:
            self._controller.reset()
        self.history_view.clear()
        self._append_system(_("Conversation reset."))

    def _on_open_settings(self) -> None:
        from datalab.aiassistant.widgets.settingsdialog import (  # noqa: WPS433
            AISettingsDialog,
        )

        if AISettingsDialog.edit(self):
            self._controller = None  # rebuild on next send

    def _on_send(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if self._controller is None:
            self._controller = self._build_controller()
            if self._controller is None:
                return
        self.input_edit.clear()
        self._append_user(text)
        self._set_busy(True)
        self._worker = AIWorker(self._controller, text, self)
        self._worker.finished_turn.connect(self._on_turn_finished)
        self._worker.failed.connect(self._on_turn_failed)
        self._worker.start()

    def _on_turn_finished(self, result: TurnResult) -> None:
        for name, _args, tool_result in result.tool_executions:
            if tool_result.ok:
                summary = (
                    str(tool_result.data)[:200]
                    if tool_result.data is not None
                    else "ok"
                )
            else:
                summary = tool_result.error or "error"
            self._append_tool(name, tool_result.ok, summary)
        if result.cancelled:
            self._append_system(_("Tool execution cancelled by user."))
        self._append_assistant(result.assistant_message)
        self._set_busy(False)
        self._worker = None

    def _on_turn_failed(self, error: str) -> None:
        self._append_system(_("Error: %s") % error)
        self._set_busy(False)
        self._worker = None
