# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dialog allowing the user to browse, load and delete persisted AI assistant
conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

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
        layout.addWidget(self._list, 1)

        button_row = QW.QHBoxLayout()
        self._load_button = QW.QPushButton(_("Load"))
        self._load_button.clicked.connect(self._on_load)
        self._delete_button = QW.QPushButton(_("Delete"))
        self._delete_button.clicked.connect(self._on_delete)
        self._refresh_button = QW.QPushButton(_("Refresh"))
        self._refresh_button.clicked.connect(self._refresh)
        button_row.addWidget(self._load_button)
        button_row.addWidget(self._delete_button)
        button_row.addWidget(self._refresh_button)
        button_row.addStretch(1)
        close_button = QW.QPushButton(_("Close"))
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self._refresh()

    # --------------------------------------------------------- helpers

    def _refresh(self) -> None:
        self._list.clear()
        for info in self._store.list():
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

    # ----------------------------------------------------------- API

    @property
    def selected_id(self) -> str | None:
        """Identifier of the conversation chosen by the user (or ``None``)."""
        return self._selected_id
