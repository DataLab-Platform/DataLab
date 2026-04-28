# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
AI Assistant chat dock panel.
"""

from __future__ import annotations

import html
import os.path as osp
from typing import TYPE_CHECKING

from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.aiassistant.controller import AIController
from datalab.aiassistant.conversation import (
    Conversation,
    ConversationStore,
    derive_title,
)
from datalab.aiassistant.inputhistory import InputHistory
from datalab.aiassistant.providers import get_provider
from datalab.aiassistant.tools.builtin import build_default_registry
from datalab.aiassistant.widgets.toolconfirmdialog import ToolConfirmDialog
from datalab.aiassistant.worker import AIWorker
from datalab.config import Conf, _
from datalab.control.proxy import LocalProxy

if TYPE_CHECKING:
    from datalab.aiassistant.controller import TurnResult
    from datalab.aiassistant.providers.base import ChatMessage
    from datalab.aiassistant.tools.registry import Tool
    from datalab.gui.main import DLMainWindow


_AI_CONFIG_SUBDIR = "aiassistant"
_CONVERSATIONS_SUBDIR = "conversations"
_INPUT_HISTORY_FILENAME = "input_history.txt"
_INPUT_HISTORY_MAX = 500
_CONVERSATIONS_MAX = 200


class _GuiBridge(QC.QObject):
    """Marshal callables from worker threads onto the GUI thread.

    The bridge lives in the GUI thread. Background threads call
    :meth:`call_in_gui` which posts the callable to the GUI thread via a
    queued signal and waits on a semaphore until the result (or exception)
    is available.
    """

    _request = QC.Signal(object, object)  # callable, holder

    def __init__(self, parent: QC.QObject | None = None) -> None:
        super().__init__(parent)
        self._request.connect(self._on_request, QC.Qt.QueuedConnection)

    def _on_request(self, func, holder: dict) -> None:
        try:
            holder["result"] = func()
        # pylint: disable-next=broad-exception-caught
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            holder["error"] = exc
        finally:
            holder["sem"].release()

    def call_in_gui(self, func):
        """Run ``func()`` on the GUI thread and return its result."""
        if QC.QThread.currentThread() is self.thread():
            return func()
        holder: dict = {"sem": QC.QSemaphore(0)}
        self._request.emit(func, holder)
        holder["sem"].acquire()
        if "error" in holder:
            raise holder["error"]
        return holder.get("result")


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
        self._bridge = _GuiBridge(self)
        base_dir = Conf.get_path(_AI_CONFIG_SUBDIR)
        self._conv_store = ConversationStore(
            osp.join(base_dir, _CONVERSATIONS_SUBDIR),
            max_conversations=_CONVERSATIONS_MAX,
        )
        self._input_history = InputHistory(
            osp.join(base_dir, _INPUT_HISTORY_FILENAME),
            max_size=_INPUT_HISTORY_MAX,
        )
        self._current_conversation: Conversation | None = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QW.QHBoxLayout()
        self.new_button = QW.QPushButton(_("New conversation"))
        self.new_button.clicked.connect(self._on_new_conversation)
        self.history_button = QW.QPushButton(_("History…"))
        self.history_button.setToolTip(_("Browse, load or delete past conversations."))
        self.history_button.clicked.connect(self._on_open_history)
        self.settings_button = QW.QPushButton(_("Settings…"))
        self.settings_button.clicked.connect(self._on_open_settings)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.history_button)
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
            _(
                "Ask the assistant…  (Ctrl+Enter to send, "
                "Ctrl+Up/Down to navigate previous prompts)"
            )
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

    def eventFilter(  # noqa: N802 - Qt API  pylint: disable=invalid-name
        self, obj: QC.QObject, event: QC.QEvent
    ) -> bool:
        """Handle Ctrl+Enter to send and Ctrl+Up/Down to browse history."""
        if obj is self.input_edit and event.type() == QC.QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if (
                key in (QC.Qt.Key_Return, QC.Qt.Key_Enter)
                and mods & QC.Qt.ControlModifier
            ):
                self._on_send()
                return True
            if mods & QC.Qt.ControlModifier and key == QC.Qt.Key_Up:
                self._navigate_input_history(forward=False)
                return True
            if mods & QC.Qt.ControlModifier and key == QC.Qt.Key_Down:
                self._navigate_input_history(forward=True)
                return True
            # Any other typing invalidates the navigation cursor so that the
            # next Ctrl+Up call captures the freshly edited draft.
            if key not in (
                QC.Qt.Key_Control,
                QC.Qt.Key_Shift,
                QC.Qt.Key_Alt,
                QC.Qt.Key_Meta,
                QC.Qt.Key_Up,
                QC.Qt.Key_Down,
                QC.Qt.Key_Left,
                QC.Qt.Key_Right,
                QC.Qt.Key_Home,
                QC.Qt.Key_End,
                QC.Qt.Key_PageUp,
                QC.Qt.Key_PageDown,
            ):
                self._input_history.reset_navigation()
        return super().eventFilter(obj, event)

    def _navigate_input_history(self, *, forward: bool) -> None:
        current = self.input_edit.toPlainText()
        text = (
            self._input_history.next(current)
            if forward
            else self._input_history.previous(current)
        )
        if text is None:
            return
        self.input_edit.blockSignals(True)
        self.input_edit.setPlainText(text)
        self.input_edit.blockSignals(False)
        cursor = self.input_edit.textCursor()
        cursor.movePosition(QG.QTextCursor.End)
        self.input_edit.setTextCursor(cursor)

    # ----------------------------------------------------------- helpers

    def _append_html(self, html_text: str) -> None:
        self.history_view.append(html_text)
        cursor = self.history_view.textCursor()
        cursor.movePosition(QG.QTextCursor.End)
        self.history_view.setTextCursor(cursor)

    def _append_user(self, text: str) -> None:
        you = _("You")
        self._append_html(
            f"<div style='margin-top:6px;'><b>{you}:</b><br>"
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
        self.new_button.setEnabled(not busy)
        self.history_button.setEnabled(not busy)
        self.status_label.setText(_("Thinking…") if busy else _("Idle"))

    # --------------------------------------------------------- controller

    def _build_controller(self) -> AIController | None:
        # pylint: disable-next=import-outside-toplevel
        from datalab.aiassistant.providers import PROVIDERS  # noqa: WPS433

        provider_name = Conf.ai.provider.get("openai")
        if provider_name not in PROVIDERS:
            provider_name = "openai"
        configured_key = Conf.ai.api_key.get("")
        model = Conf.ai.model.get("gpt-4o-mini")
        base_url = Conf.ai.base_url.get("") or None
        temperature = float(Conf.ai.temperature.get(0.2))
        timeout = float(Conf.ai.timeout.get(60.0))
        max_iterations = int(Conf.ai.max_iterations.get(8))
        auto_approve = bool(Conf.ai.auto_approve_readonly.get(True))

        try:
            provider_cls = get_provider(provider_name)
        except KeyError as exc:
            QW.QMessageBox.critical(self, _("AI Assistant"), str(exc))
            return None

        # Resolve the API key: explicit configuration wins, then fall back
        # to the provider-specific environment variable (e.g. OPENAI_API_KEY).
        api_key = provider_cls.resolve_api_key(configured_key)

        if not api_key and provider_name != "mock":
            env_var = provider_cls.api_key_env_var
            env_hint = (
                _(
                    "\n\nTip: you can also set the {var} environment "
                    "variable to avoid storing the key in the "
                    "configuration file."
                ).format(var=env_var)
                if env_var
                else ""
            )
            QW.QMessageBox.warning(
                self,
                _("AI Assistant"),
                _(
                    "No API key configured.\n\n"
                    "Open the Settings… dialog to enter your "
                    "provider credentials, or select the 'mock' "
                    "provider for offline testing."
                )
                + env_hint,
            )
            return None

        provider = provider_cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )
        registry = build_default_registry()

        def confirm_in_gui(tool: Tool, arguments: dict) -> bool:
            return self._bridge.call_in_gui(
                lambda: ToolConfirmDialog.confirm(tool, arguments, self)
            )

        def execute_in_gui(name: str, arguments: dict):
            return self._bridge.call_in_gui(
                lambda: registry.call(name, arguments, self._proxy, self.mainwindow)
            )

        return AIController(
            provider=provider,
            registry=registry,
            proxy=self._proxy,
            mainwindow=self.mainwindow,
            confirm_callback=confirm_in_gui,
            max_iterations=max_iterations,
            auto_approve_readonly=auto_approve,
            execute_callback=execute_in_gui,
        )

    def _confirm_tool(self, tool: Tool, arguments: dict) -> bool:
        return ToolConfirmDialog.confirm(tool, arguments, self)

    # -------------------------------------------------------- slots

    def _on_new_conversation(self) -> None:
        if self._controller is not None:
            self._controller.reset()
        self._current_conversation = None
        self.history_view.clear()
        self._append_system(_("Conversation reset."))

    def _on_open_history(self) -> None:
        # pylint: disable-next=import-outside-toplevel
        from datalab.aiassistant.widgets.conversationsdialog import (  # noqa: WPS433
            ConversationsDialog,
        )

        dialog = ConversationsDialog(self._conv_store, self)
        if dialog.exec_() and dialog.selected_id is not None:
            self._load_conversation(dialog.selected_id)

    def _on_open_settings(self) -> None:
        # pylint: disable-next=import-outside-toplevel
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
        self._input_history.add(text)
        if self._current_conversation is None:
            self._current_conversation = Conversation.new()
            self._current_conversation.title = derive_title(text)
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
        self._persist_current_conversation()
        self._set_busy(False)
        self._worker = None

    def _on_turn_failed(self, error: str) -> None:
        self._append_system(_("Error: %s") % error)
        self._persist_current_conversation()
        self._set_busy(False)
        self._worker = None

    # ----------------------------------------------------- conversations

    def _persist_current_conversation(self) -> None:
        if self._controller is None or self._current_conversation is None:
            return
        self._current_conversation.messages = self._controller.get_messages()
        try:
            self._conv_store.save(self._current_conversation)
        except OSError as exc:
            self._append_system(_("Failed to save conversation: %s") % exc)

    def _load_conversation(self, conv_id: str) -> None:
        try:
            conversation = self._conv_store.load(conv_id)
        except (OSError, ValueError) as exc:
            QW.QMessageBox.critical(
                self,
                _("AI Assistant"),
                _("Failed to load conversation: %s") % exc,
            )
            return
        if self._controller is None:
            self._controller = self._build_controller()
            if self._controller is None:
                return
        self._controller.load_messages(conversation.messages)
        self._current_conversation = conversation
        self.history_view.clear()
        self._render_messages(conversation.messages)
        self._append_system(
            _("Loaded conversation: %s") % (conversation.title or conversation.id)
        )

    def _render_messages(self, messages: list[ChatMessage]) -> None:
        """Re-render persisted messages in the chat view."""
        for msg in messages:
            if msg.role == "user":
                self._append_user(self._content_to_text(msg.content))
            elif msg.role == "assistant":
                if msg.content:
                    self._append_assistant(self._content_to_text(msg.content))
                for call in msg.tool_calls:
                    self._append_system(
                        _("Tool call: %(name)s(%(args)s)")
                        % {"name": call.name, "args": call.arguments}
                    )
            elif msg.role == "tool":
                content = self._content_to_text(msg.content)
                summary = content[:200] if content else ""
                self._append_tool(msg.name or "tool", True, summary)

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(str(part["text"]))
                    elif part.get("type") == "image_url":
                        parts.append("[image]")
            return "\n".join(parts)
        return str(content)
