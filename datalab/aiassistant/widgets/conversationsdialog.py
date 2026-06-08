# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dialog allowing the user to browse, load and delete persisted AI assistant
conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.aiassistant.markdown_export import (
    conversation_to_markdown,
    sanitize_filename,
)
from datalab.config import _

if TYPE_CHECKING:
    from datalab.aiassistant.conversation import ConversationInfo, ConversationStore


class ConversationsDialog(QW.QDialog):
    """Browse and manage persisted AI assistant conversations.

    Args:
        store: Conversation store to browse.
        parent: Parent widget.
    """

    def __init__(
        self,
        store: ConversationStore,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._selected_id: str | None = None
        self._infos: list[ConversationInfo] = []
        self.setWindowTitle(_("AI Assistant — Conversation history"))
        self.resize(640, 420)

        layout = QW.QVBoxLayout(self)

        info = QW.QLabel(
            _(
                "Double-click a conversation to load it. Loading a "
                "conversation replaces the current one."
            )
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QW.QListWidget(self)
        self._list.itemDoubleClicked.connect(self._on_load)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        # F2 to rename — matches DataLab-Web's keyboard shortcut.
        rename_shortcut = QG.QShortcut(QG.QKeySequence(QC.Qt.Key_F2), self._list)
        rename_shortcut.activated.connect(self._on_rename)
        layout.addWidget(self._list, 1)

        button_row = QW.QHBoxLayout()
        self._load_button = QW.QPushButton(get_icon("restore.svg"), _("Load"))
        self._load_button.clicked.connect(self._on_load)
        self._delete_button = QW.QPushButton(get_icon("edit/delete.svg"), _("Delete"))
        self._delete_button.clicked.connect(self._on_delete)
        self._rename_button = QW.QPushButton(
            get_icon("libre-gui-pencil.svg"), _("Rename")
        )
        self._rename_button.setToolTip(_("Rename the selected conversation (F2)."))
        self._rename_button.clicked.connect(self._on_rename)
        self._export_button = QW.QPushButton(get_icon("export.svg"), _("Export…"))
        self._export_button.setToolTip(
            _("Save the selected conversation as a Markdown file.")
        )
        self._export_button.clicked.connect(self._on_export)
        self._refresh_button = QW.QPushButton(
            get_icon("refresh-manual.svg"), _("Refresh")
        )
        self._refresh_button.clicked.connect(self._refresh)
        button_row.addWidget(self._load_button)
        button_row.addWidget(self._delete_button)
        button_row.addWidget(self._rename_button)
        button_row.addWidget(self._export_button)
        button_row.addWidget(self._refresh_button)
        button_row.addStretch(1)
        close_button = QW.QPushButton(get_icon("libre-gui-close.svg"), _("Close"))
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self._refresh()

    # --------------------------------------------------------- helpers

    def _refresh(self) -> None:
        self._list.clear()
        self._infos = list(self._store.list())
        for info in self._infos:
            label = self._format_info(info)
            item = QW.QListWidgetItem(label)
            item.setData(QC.Qt.UserRole, info.id)
            tooltip = _(
                "Created: {created}\nUpdated: {updated}\nMessages: {count}"
            ).format(
                created=info.created_at or _("unknown"),
                updated=info.updated_at or _("unknown"),
                count=info.message_count,
            )
            item.setToolTip(tooltip)
            self._list.addItem(item)
        self._on_selection_changed()

    @staticmethod
    def _format_info(info: ConversationInfo) -> str:
        title = info.title or _("(untitled)")
        when = info.updated_at or info.created_at or ""
        return f"[{when}]  {title}  ({info.message_count})"

    def _current_id(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(QC.Qt.UserRole)

    def _on_selection_changed(self) -> None:
        has = self._current_id() is not None
        self._load_button.setEnabled(has)
        self._delete_button.setEnabled(has)
        self._rename_button.setEnabled(has)
        self._export_button.setEnabled(has)

    def _on_load(self) -> None:
        conv_id = self._current_id()
        if conv_id is None:
            return
        self._selected_id = conv_id
        self.accept()

    def _on_delete(self) -> None:
        conv_id = self._current_id()
        if conv_id is None:
            return
        confirm = QW.QMessageBox.question(
            self,
            _("Delete conversation"),
            _("Delete the selected conversation? This cannot be undone."),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if confirm == QW.QMessageBox.Yes:
            self._store.delete(conv_id)
            self._refresh()

    def _current_info(self) -> ConversationInfo | None:
        conv_id = self._current_id()
        if conv_id is None:
            return None
        for info in self._infos:
            if info.id == conv_id:
                return info
        return None

    def _on_rename(self) -> None:
        info = self._current_info()
        if info is None:
            return
        new_title, ok = QW.QInputDialog.getText(
            self,
            _("Rename conversation"),
            _("New title:"),
            QW.QLineEdit.Normal,
            info.title,
        )
        if not ok:
            return
        self._store.rename(info.id, new_title.strip())
        self._refresh()
        # Re-select the renamed item.
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.data(QC.Qt.UserRole) == info.id:
                self._list.setCurrentItem(item)
                break

    def _on_export(self) -> None:
        info = self._current_info()
        if info is None:
            return
        try:
            conv = self._store.load(info.id)
        except (OSError, ValueError) as exc:
            QW.QMessageBox.critical(
                self,
                _("Export failed"),
                _("Could not load conversation: %s") % exc,
            )
            return
        date_part = (conv.updated_at or conv.created_at or "")[:10]
        prefix = f"{date_part}-" if date_part else ""
        default_name = f"{prefix}{sanitize_filename(conv.title or 'conversation')}.md"
        path, _selected = QW.QFileDialog.getSaveFileName(
            self,
            _("Export conversation"),
            default_name,
            _("Markdown files (*.md);;All files (*)"),
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(conversation_to_markdown(conv))
        except OSError as exc:
            QW.QMessageBox.critical(
                self,
                _("Export failed"),
                _("Could not write file: %s") % exc,
            )

    # ----------------------------------------------------------- API

    @property
    def selected_id(self) -> str | None:
        """Identifier of the conversation chosen by the user (or ``None``)."""
        return self._selected_id
