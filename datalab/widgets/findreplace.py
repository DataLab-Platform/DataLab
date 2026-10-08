# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Lightweight Find/Replace widget for :class:`guidata.widgets.codeeditor.CodeEditor`.

The widget is meant to be inserted at the bottom of the editor's container.
It exposes the standard Ctrl+F / Ctrl+H / F3 / Shift+F3 / Escape shortcuts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _

if TYPE_CHECKING:
    from guidata.widgets.codeeditor import CodeEditor


class FindReplaceBar(QW.QWidget):
    """Compact find/replace bar bound to a (possibly dynamic) code editor.

    Args:
        editor_provider: Callable returning the current code editor, or a
         :class:`CodeEditor` instance to use a fixed target.
        shortcut_parent: Widget on which Ctrl+F / Ctrl+H / F3 / Shift+F3
         shortcuts are registered. Defaults to the bar itself.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        editor_provider: CodeEditor | Callable[[], CodeEditor | None],
        shortcut_parent: QW.QWidget | None = None,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if callable(editor_provider):
            self._provider = editor_provider
        else:
            fixed = editor_provider
            self._provider = lambda: fixed

        self.find_edit = QW.QLineEdit(self)
        self.find_edit.setPlaceholderText(_("Find"))
        self.find_edit.returnPressed.connect(self.find_next)
        self.find_edit.textChanged.connect(self._on_text_changed)

        self.replace_edit = QW.QLineEdit(self)
        self.replace_edit.setPlaceholderText(_("Replace with"))
        self.replace_edit.returnPressed.connect(self.replace_one)

        self.case_cb = QW.QCheckBox(_("Aa"), self)
        self.case_cb.setToolTip(_("Match case"))
        self.whole_cb = QW.QCheckBox(_("W"), self)
        self.whole_cb.setToolTip(_("Whole words only"))

        next_btn = QW.QPushButton(_("Next"), self)
        next_btn.clicked.connect(self.find_next)
        prev_btn = QW.QPushButton(_("Prev"), self)
        prev_btn.clicked.connect(self.find_previous)
        replace_btn = QW.QPushButton(_("Replace"), self)
        replace_btn.clicked.connect(self.replace_one)
        replace_all_btn = QW.QPushButton(_("Replace all"), self)
        replace_all_btn.clicked.connect(self.replace_all)
        close_btn = QW.QToolButton(self)
        close_btn.setText("✕")
        close_btn.setAutoRaise(True)
        close_btn.clicked.connect(self.hide_and_focus_editor)

        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(QW.QLabel(_("Find:"), self))
        layout.addWidget(self.find_edit, 1)
        layout.addWidget(prev_btn)
        layout.addWidget(next_btn)
        layout.addWidget(self.case_cb)
        layout.addWidget(self.whole_cb)
        layout.addWidget(QW.QLabel(_("Replace:"), self))
        layout.addWidget(self.replace_edit, 1)
        layout.addWidget(replace_btn)
        layout.addWidget(replace_all_btn)
        layout.addWidget(close_btn)

        self.hide()

        # Register shortcuts on the chosen widget (typically the panel).
        target = shortcut_parent if shortcut_parent is not None else self
        QW.QShortcut(QG.QKeySequence("Ctrl+F"), target, self.show_find)
        QW.QShortcut(QG.QKeySequence("Ctrl+H"), target, self.show_replace)
        QW.QShortcut(QG.QKeySequence("F3"), target, self.find_next)
        QW.QShortcut(QG.QKeySequence("Shift+F3"), target, self.find_previous)
        QW.QShortcut(QG.QKeySequence("Escape"), self, self.hide_and_focus_editor)

    # ------------------------------------------------------------------
    # Editor access
    # ------------------------------------------------------------------

    def _editor(self):  # type: ignore[override]
        return self._provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_find(self) -> None:
        """Show the bar pre-filled with the editor selection (if any)."""
        editor = self._editor()
        if editor is not None:
            sel = editor.textCursor().selectedText()
            if sel and "\u2029" not in sel:
                self.find_edit.setText(sel)
        self.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def show_replace(self) -> None:
        """Show the bar focused on the replacement field."""
        self.show_find()
        self.replace_edit.setFocus()

    def hide_and_focus_editor(self) -> None:
        """Hide the bar and return focus to the editor."""
        self.hide()
        editor = self._editor()
        if editor is not None:
            editor.setFocus()

    def find_next(self) -> bool:
        """Search forward from the current cursor position."""
        return self._search(False)

    def find_previous(self) -> bool:
        """Search backward from the current cursor position."""
        return self._search(True)

    def replace_one(self) -> None:
        """Replace the current match (if any) and move to the next one."""
        editor = self._editor()
        if editor is None:
            return
        cur = editor.textCursor()
        needle = self.find_edit.text()
        if needle and cur.selectedText() == needle:
            cur.insertText(self.replace_edit.text())
        self.find_next()

    def replace_all(self) -> int:
        """Replace every match in the document.

        Returns:
            Number of replacements performed.
        """
        needle = self.find_edit.text()
        if not needle:
            return 0
        editor = self._editor()
        if editor is None:
            return 0
        replacement = self.replace_edit.text()
        doc = editor.document()
        count = 0
        cursor = QG.QTextCursor(doc)
        cursor.beginEditBlock()
        try:
            flags = self._flags(False)
            cursor.movePosition(QG.QTextCursor.Start)
            found = doc.find(needle, cursor, flags)
            while not found.isNull():
                found.insertText(replacement)
                count += 1
                found = doc.find(needle, found, flags)
        finally:
            cursor.endEditBlock()
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flags(self, backward: bool) -> QG.QTextDocument.FindFlags:
        flags = QG.QTextDocument.FindFlags()
        if backward:
            flags |= QG.QTextDocument.FindBackward
        if self.case_cb.isChecked():
            flags |= QG.QTextDocument.FindCaseSensitively
        if self.whole_cb.isChecked():
            flags |= QG.QTextDocument.FindWholeWords
        return flags

    def _search(self, backward: bool) -> bool:
        needle = self.find_edit.text()
        if not needle:
            return False
        editor = self._editor()
        if editor is None:
            return False
        flags = self._flags(backward)
        found_cursor = editor.document().find(needle, editor.textCursor(), flags)
        if found_cursor.isNull():
            # Wrap-around search
            wrap = QG.QTextCursor(editor.document())
            if backward:
                wrap.movePosition(QG.QTextCursor.End)
            found_cursor = editor.document().find(needle, wrap, flags)
        if found_cursor.isNull():
            self.find_edit.setStyleSheet("background-color: #ffd6d6;")
            return False
        self.find_edit.setStyleSheet("")
        editor.setTextCursor(found_cursor)
        return True

    def _on_text_changed(self, _text: str) -> None:
        self.find_edit.setStyleSheet("")
