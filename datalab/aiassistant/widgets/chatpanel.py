# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
AI Assistant chat dock panel.
"""

from __future__ import annotations

import html
import json
import os.path as osp
import textwrap
import uuid
from typing import TYPE_CHECKING, Any

from guidata.configtools import get_icon
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
from datalab.aiassistant.providers.base import TokenUsage
from datalab.aiassistant.tools.builtin import build_default_registry
from datalab.aiassistant.widgets.markdown import (
    is_python_code_block,
    markdown_to_html,
)
from datalab.aiassistant.widgets.toolconfirmdialog import ToolConfirmDialog
from datalab.aiassistant.worker import AIWorker
from datalab.config import Conf, _, get_config_path
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
_MACRO_TOOL_NAME = "create_and_run_macro"
_DEFAULT_MACRO_TITLE = "AI Assistant macro"
_SAVE_MACRO_LINK = "save-macro"
_SHOW_MACRO_PANEL_LINK = "show-macro-panel"


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
        base_dir = get_config_path(_AI_CONFIG_SUBDIR)
        self._conv_store = ConversationStore(
            osp.join(base_dir, _CONVERSATIONS_SUBDIR),
            max_conversations=_CONVERSATIONS_MAX,
        )
        self._input_history = InputHistory(
            osp.join(base_dir, _INPUT_HISTORY_FILENAME),
            max_size=_INPUT_HISTORY_MAX,
        )
        self._current_conversation: Conversation | None = None
        # Macros proposed by the LLM, either through 'create_and_run_macro'
        # or as Python code blocks in its replies. They are NOT added to the
        # Macro panel by default; the user can save them on demand by
        # clicking the 'Save to Macros' link surfaced next to them. Keyed by
        # short uuid.
        self._pending_macros: dict[str, tuple[str, str]] = {}
        # Last user request, used to name the macros saved from the code
        # blocks of the reply.
        self._last_user_text: str | None = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QW.QHBoxLayout()
        self.new_button = QW.QPushButton(
            get_icon("libre-gui-add.svg"), _("New conversation")
        )
        self.new_button.clicked.connect(self._on_new_conversation)
        self.history_button = QW.QPushButton(get_icon("history.svg"), _("History…"))
        self.history_button.setToolTip(_("Browse, load or delete past conversations."))
        self.history_button.clicked.connect(self._on_open_history)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.history_button)
        toolbar.addStretch(1)
        self.usage_label = QW.QLabel("")
        self.usage_label.setStyleSheet(
            "color:#666;font-family:monospace;padding:0 6px;"
        )
        self.usage_label.setVisible(False)
        toolbar.addWidget(self.usage_label)
        self.status_label = QW.QLabel(_("Idle"))
        toolbar.addWidget(self.status_label)
        layout.addLayout(toolbar)

        # History view
        self.history_view = QW.QTextBrowser()
        # We handle anchor clicks ourselves to support our internal
        # ``datalab:`` scheme (e.g. ``datalab:save-macro/<id>``) used to
        # offer a "Save to Macros" link after a transient macro run.
        # External http(s) links are forwarded to QDesktopServices.
        self.history_view.setOpenLinks(False)
        self.history_view.setOpenExternalLinks(False)
        self.history_view.anchorClicked.connect(self._on_anchor_clicked)
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
        self.stop_button = QW.QPushButton(get_icon("tour/stop.svg"), _("Stop"))
        self.stop_button.setToolTip(_("Cancel the in-flight request (Esc)."))
        self.stop_button.setStyleSheet(
            "QPushButton { color: white; background-color: #c01c28; }"
            "QPushButton:hover { background-color: #a51d2d; }"
        )
        self.stop_button.clicked.connect(self._on_stop)
        self.stop_button.setVisible(False)
        send_row.addWidget(self.stop_button)
        self.send_button = QW.QPushButton(get_icon("apply.svg"), _("Send"))
        self.send_button.clicked.connect(self._on_send)
        send_row.addWidget(self.send_button)
        layout.addLayout(send_row)

        self._append_system(
            _(
                "AI Assistant ready. Configure your provider in the "
                "Edit > Preferences dialog (AI Assistant tab) before "
                "sending the first message."
            )
        )

    def eventFilter(  # noqa: N802 - Qt API  pylint: disable=invalid-name
        self, obj: QC.QObject, event: QC.QEvent
    ) -> bool:
        """Handle Ctrl+Enter to send and Ctrl+Up/Down to browse history."""
        if obj is self.input_edit and event.type() == QC.QEvent.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if key == QC.Qt.Key_Escape and self._worker is not None:
                self._on_stop()
                return True
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
        footer = (
            self._code_block_save_link if Conf.ai_expose_macro_tool.get(True) else None
        )
        self._append_html(
            f"<div style='margin-top:6px;color:#1a5fb4;'>"
            f"<b>{_('Assistant')}:</b>"
            f"{markdown_to_html(text, code_block_footer=footer)}</div>"
        )

    def _code_block_save_link(self, lang: str, code: str) -> str:
        """Return a "Save to Macros" link for a Python code block of a reply.

        The macro is named after the user request the reply answers.
        """
        if not is_python_code_block(lang, code):
            return ""
        title = (
            derive_title(self._last_user_text)
            if self._last_user_text and self._last_user_text.strip()
            else _DEFAULT_MACRO_TITLE
        )
        return (
            "<p style='margin:2px 0 4px 0;font-size:90%;'>"
            f"{self._save_macro_link_html(title, textwrap.dedent(code))}</p>"
        )

    def _append_tool(self, name: str, ok: bool, summary: str) -> None:
        color = "#26a269" if ok else "#c01c28"
        symbol = "✓" if ok else "✗"
        self._append_html(
            f"<div style='margin-top:4px;color:{color};font-family:monospace;'>"
            f"{symbol} {html.escape(name)}: {html.escape(summary)}</div>"
        )

    def _append_tool_result(
        self, name: str, ok: bool, data: Any, error: str | None
    ) -> None:
        """Append the outcome of a tool call (live or reloaded).

        For a macro proposed through 'create_and_run_macro', a "Save to
        Macros" link is added so the user can opt in to persisting it in the
        Macro panel (mirrors the DataLab-Web behaviour).
        """
        if ok:
            summary = str(data)[:200] if data is not None else "ok"
        else:
            summary = error or "error"
        self._append_tool(name, ok, summary)
        if name == _MACRO_TOOL_NAME and isinstance(data, dict):
            macro_payload = data.get("_macro")
            if isinstance(macro_payload, dict):
                title = str(macro_payload.get("title") or _DEFAULT_MACRO_TITLE)
                code = str(macro_payload.get("code") or "")
                if code:
                    self._append_save_macro_link(title, code)

    def _append_system(self, text: str) -> None:
        self._append_html(
            f"<div style='margin-top:4px;color:#666;font-style:italic;'>"
            f"{html.escape(text)}</div>"
        )

    def _save_macro_link_html(self, title: str, code: str) -> str:
        """Return an inline "Save to Macros" link for a proposed macro.

        The link uses the internal ``datalab:save-macro/<id>`` scheme,
        intercepted by :meth:`_on_anchor_clicked`. The macro source is
        stashed in ``self._pending_macros`` keyed by ``id``.
        """
        macro_id = uuid.uuid4().hex[:12]
        self._pending_macros[macro_id] = (title, code)
        url = f"datalab:{_SAVE_MACRO_LINK}/{macro_id}"
        save_label = _("Save to Macros")
        return (
            f"💾 <a href='{html.escape(url, quote=True)}'>"
            f"{html.escape(save_label)}</a> — "
            f"<i>{html.escape(title)}</i>"
        )

    def _append_save_macro_link(self, title: str, code: str) -> None:
        """Append an inline "Save to Macros" link for a proposed macro."""
        self._append_html(
            "<div style='margin-top:2px;margin-left:18px;font-size:90%;'>"
            f"{self._save_macro_link_html(title, code)}</div>"
        )

    def _on_anchor_clicked(self, url: QC.QUrl) -> None:
        """Dispatch internal ``datalab:`` links; forward others to the OS."""
        if url.scheme() == "datalab":
            path = url.path().lstrip("/")
            prefix = f"{_SAVE_MACRO_LINK}/"
            if url.host() == _SAVE_MACRO_LINK or path.startswith(prefix):
                # QUrl.path() for ``datalab:save-macro/<id>`` puts the
                # ``save-macro`` segment in ``host`` on some platforms
                # and in the leading path segment on others — handle both.
                macro_id = path[len(prefix) :] if path.startswith(prefix) else path
                self._save_macro(macro_id)
                return
            if _SHOW_MACRO_PANEL_LINK in (url.host(), path):
                self.mainwindow.set_current_panel("macro")
                return
        # External link — open in default browser (mirrors the previous
        # ``setOpenExternalLinks(True)`` behaviour).
        QG.QDesktopServices.openUrl(url)

    def _save_macro(self, macro_id: str) -> None:
        """Persist a pending macro to the Macro panel."""
        entry = self._pending_macros.pop(macro_id, None)
        if entry is None:
            self._append_system(
                _("This macro is no longer available (already saved or cleared).")
            )
            return
        title, code = entry
        try:
            self.mainwindow.macropanel.add_macro_with_code(title, code, source="ai")
        # pylint: disable-next=broad-exception-caught
        except BaseException as exc:  # noqa: BLE001
            self._append_system(_("Failed to save macro: %s") % exc)
            # Re-stash so the user can retry.
            self._pending_macros[macro_id] = entry
            return
        # The Macro panel is tabbed with this one: offer to show it rather
        # than switching automatically, which would hide the conversation.
        saved = html.escape(_("Saved macro '%s' to the Macro panel.") % title)
        show = html.escape(_("Show in Macro panel"))
        self._append_html(
            f"<div style='margin-top:4px;color:#666;font-style:italic;'>"
            f"{saved} <a href='datalab:{_SHOW_MACRO_PANEL_LINK}'>{show}</a></div>"
        )

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setVisible(not busy)
        self.send_button.setEnabled(not busy)
        self.stop_button.setVisible(busy)
        self.stop_button.setEnabled(busy)
        self.input_edit.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.history_button.setEnabled(not busy)
        self.status_label.setText(_("Thinking…") if busy else _("Idle"))

    # ----------------------------------------------------- token usage

    @staticmethod
    def _format_token_count(value: int | None) -> str:
        if value is None:
            return "—"
        if value < 1000:
            return str(value)
        if value < 1_000_000:
            return f"{value / 1000:.1f}k".replace(".0k", "k")
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")

    @classmethod
    def _format_usage_badge(
        cls, turn: TokenUsage | None, cumulative: TokenUsage | None
    ) -> str:
        parts: list[str] = []
        if turn is not None and turn.prompt_tokens:
            parts.append(f"ctx {cls._format_token_count(turn.prompt_tokens)}")
        if cumulative is not None and (
            cumulative.prompt_tokens
            or cumulative.completion_tokens
            or cumulative.total_tokens
        ):
            if (
                cumulative.prompt_tokens is not None
                or cumulative.completion_tokens is not None
            ):
                parts.append(
                    f"↑{cls._format_token_count(cumulative.prompt_tokens)} "
                    f"↓{cls._format_token_count(cumulative.completion_tokens)}"
                )
            else:
                parts.append(f"Σ{cls._format_token_count(cumulative.total_tokens)}")
        return " · ".join(parts)

    @classmethod
    def _format_usage_tooltip(
        cls, turn: TokenUsage | None, cumulative: TokenUsage | None
    ) -> str:
        lines: list[str] = []
        if turn is not None and turn.prompt_tokens:
            lines.append(_("Last request context size: %d tokens") % turn.prompt_tokens)
            lines.append("")
        if cumulative is not None:
            lines.append(_("Cumulative token usage"))
            if cumulative.prompt_tokens is not None:
                lines.append(_("Prompt: %d") % cumulative.prompt_tokens)
            if cumulative.completion_tokens is not None:
                lines.append(_("Completion: %d") % cumulative.completion_tokens)
            if cumulative.total_tokens is not None:
                lines.append(_("Total: %d") % cumulative.total_tokens)
        return "\n".join(lines).strip()

    def _update_usage_badge(
        self, turn: TokenUsage | None, cumulative: TokenUsage | None
    ) -> None:
        text = self._format_usage_badge(turn, cumulative)
        self.usage_label.setText(text)
        self.usage_label.setToolTip(self._format_usage_tooltip(turn, cumulative))
        self.usage_label.setVisible(bool(text))

    def _on_usage(self, turn: TokenUsage, cumulative: TokenUsage) -> None:
        """Called from a worker thread whenever a provider reports usage."""
        # Marshal to the GUI thread before touching widgets.
        self._bridge.call_in_gui(lambda: self._update_usage_badge(turn, cumulative))

    def _on_stop(self) -> None:
        if self._controller is not None and self._controller.is_running:
            self._controller.abort()
            self.status_label.setText(_("Stopping…"))
            self.stop_button.setEnabled(False)

    # --------------------------------------------------------- controller

    def _build_controller(self) -> AIController | None:
        # pylint: disable-next=import-outside-toplevel
        from datalab.aiassistant.providers import PROVIDERS  # noqa: WPS433

        provider_name = Conf.ai_provider.get("openai")
        if provider_name not in PROVIDERS:
            provider_name = "openai"
        configured_key = Conf.ai_api_key.get("")
        model = Conf.ai_model.get("gpt-4o-mini")
        base_url = Conf.ai_base_url.get("") or None
        temperature = float(Conf.ai_temperature.get(0.2))
        timeout = float(Conf.ai_timeout.get(60.0))
        max_iterations = int(Conf.ai_max_iterations.get(8))
        max_history_messages = int(Conf.ai_max_history_messages.get(0))
        auto_approve = bool(Conf.ai_auto_approve_readonly.get(True))

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
                    "Open the Edit > Preferences dialog (AI Assistant tab) "
                    "to enter your provider credentials, or select the "
                    "'mock' provider for offline testing."
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
        registry = build_default_registry(
            expose_macro_tool=bool(Conf.ai_expose_macro_tool.get(True))
        )

        def confirm_in_gui(tool: Tool, arguments: dict) -> bool:
            return self._bridge.call_in_gui(
                lambda: ToolConfirmDialog.confirm(tool, arguments, self)
            )

        def execute_in_gui(name: str, arguments: dict):
            return self._bridge.call_in_gui(
                lambda: registry.call(name, arguments, self._proxy, self.mainwindow)
            )

        return AIController.with_default_prompt(
            provider=provider,
            registry=registry,
            proxy=self._proxy,
            mainwindow=self.mainwindow,
            confirm_callback=confirm_in_gui,
            max_iterations=max_iterations,
            auto_approve_readonly=auto_approve,
            execute_callback=execute_in_gui,
            usage_callback=self._on_usage,
            max_history_messages=max_history_messages,
        )

    def _confirm_tool(self, tool: Tool, arguments: dict) -> bool:
        return ToolConfirmDialog.confirm(tool, arguments, self)

    # -------------------------------------------------------- slots

    def _on_new_conversation(self) -> None:
        if self._controller is not None:
            self._controller.reset()
        self._current_conversation = None
        self._last_user_text = None
        self.history_view.clear()
        self._update_usage_badge(None, None)
        self._append_system(_("Conversation reset."))

    def _on_open_history(self) -> None:
        # pylint: disable-next=import-outside-toplevel
        from datalab.aiassistant.widgets.conversationsdialog import (  # noqa: WPS433
            ConversationsDialog,
        )

        dialog = ConversationsDialog(self._conv_store, self)
        if dialog.exec_() and dialog.selected_id is not None:
            self._load_conversation(dialog.selected_id)

    def invalidate_controller(self) -> None:
        """Drop the cached controller so the next prompt rebuilds it.

        Called whenever AI assistant settings are modified through
        DataLab's main Settings dialog (Edit > Preferences).
        """
        self._controller = None

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
        self._last_user_text = text
        self._append_user(text)
        self._set_busy(True)
        self._worker = AIWorker(self._controller, text, self)
        self._worker.finished_turn.connect(self._on_turn_finished)
        self._worker.failed.connect(self._on_turn_failed)
        self._worker.start()

    def _on_turn_finished(self, result: TurnResult) -> None:
        for name, _args, tool_result in result.tool_executions:
            self._append_tool_result(
                name, tool_result.ok, tool_result.data, tool_result.error
            )
        if result.cancelled:
            self._append_system(_("Tool execution cancelled by user."))
        if result.aborted:
            self._append_system(_("Stopped by user."))
        else:
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
        self._current_conversation.usage = self._controller.get_usage()
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
        self._controller.load_messages(
            conversation.messages, initial_usage=conversation.usage
        )
        self._current_conversation = conversation
        self.history_view.clear()
        self._update_usage_badge(None, conversation.usage)
        self._render_messages(conversation.messages)
        self._append_system(
            _("Loaded conversation: %s") % (conversation.title or conversation.id)
        )

    def _render_messages(self, messages: list[ChatMessage]) -> None:
        """Re-render persisted messages in the chat view."""
        self._last_user_text = None
        for msg in messages:
            if msg.role == "user":
                text = self._content_to_text(msg.content)
                self._last_user_text = text
                self._append_user(text)
            elif msg.role == "assistant":
                if msg.content:
                    self._append_assistant(self._content_to_text(msg.content))
                for call in msg.tool_calls:
                    self._append_system(
                        _("Tool call: %(name)s(%(args)s)")
                        % {"name": call.name, "args": call.arguments}
                    )
            elif msg.role == "tool":
                ok, data, error = self._parse_tool_content(
                    self._content_to_text(msg.content)
                )
                self._append_tool_result(msg.name or "tool", ok, data, error)

    @staticmethod
    def _parse_tool_content(content: str) -> tuple[bool, Any, str | None]:
        """Decode a persisted tool message into ``(ok, data, error)``.

        Tool messages hold the JSON produced by
        :meth:`ToolResult.to_message_content`; anything else is shown as is.
        """
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("ok"), bool):
            if payload["ok"]:
                return True, payload.get("result"), None
            return False, None, str(payload.get("error") or "error")
        return True, content or None, None

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
