# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Python autocompletion helper for :class:`guidata.widgets.codeeditor.CodeEditor`.

Uses Jedi (mandatory dependency, declared in ``pyproject.toml``). The
buffer-parsed fallback (symbol scan + standard Python keywords/builtins) is
kept as a runtime safety net in case Jedi raises unexpectedly on malformed
source.

Trigger: Ctrl+Space, or 2+ identifier characters typed in a row.
"""

from __future__ import annotations

import builtins
import keyword
import re
from typing import TYPE_CHECKING

import jedi
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

if TYPE_CHECKING:
    from guidata.widgets.codeeditor import CodeEditor

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+(.+)$", re.MULTILINE)

_DEBOUNCE_MS = 200
_MIN_CHARS = 2


class PythonCompleter(QC.QObject):
    """Attach Python autocompletion to a :class:`CodeEditor` instance."""

    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

        self._completer = QW.QCompleter([], editor)
        self._completer.setWidget(editor)
        self._completer.setCompletionMode(QW.QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(QC.Qt.CaseSensitive)
        self._completer.activated.connect(self._insert_completion)

        self._timer = QC.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DEBOUNCE_MS)
        self._timer.timeout.connect(self._maybe_show)

        editor.textChanged.connect(self._timer.start)

        QW.QShortcut(QG.QKeySequence("Ctrl+Space"), editor, self.trigger)

        # Forward key events to handle popup-aware Tab/Return/Escape via
        # a Qt event filter on the editor.
        editor.installEventFilter(self)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):  # pylint: disable=invalid-name
        """Reimplement Qt method to forward key events
        to the completer when its popup is visible."""
        if event.type() == QC.QEvent.KeyPress and self._completer.popup().isVisible():
            if event.key() in (
                QC.Qt.Key_Enter,
                QC.Qt.Key_Return,
                QC.Qt.Key_Tab,
            ):
                # Let the completer consume the event.
                event.ignore()
                return True
            if event.key() == QC.Qt.Key_Escape:
                self._completer.popup().hide()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger(self) -> None:
        """Force the completion popup to show now."""
        self._timer.stop()
        self._show(force=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _word_under_cursor(self) -> str:
        cur = self._editor.textCursor()
        cur.select(QG.QTextCursor.WordUnderCursor)
        return cur.selectedText()

    def _maybe_show(self) -> None:
        self._show(force=False)

    def _show(self, force: bool) -> None:
        prefix = self._word_under_cursor()
        if not force and len(prefix) < _MIN_CHARS:
            self._completer.popup().hide()
            return
        suggestions = self._compute_suggestions(prefix)
        if not suggestions:
            self._completer.popup().hide()
            return
        model = QC.QStringListModel(suggestions, self._completer)
        self._completer.setModel(model)
        self._completer.setCompletionPrefix(prefix)
        rect = self._editor.cursorRect()
        popup = self._completer.popup()
        rect.setWidth(
            popup.sizeHintForColumn(0) + popup.verticalScrollBar().sizeHint().width()
        )
        self._completer.complete(rect)

    def _compute_suggestions(self, prefix: str) -> list[str]:
        jedi_words = self._jedi_suggestions()
        if jedi_words:
            return self._filter(jedi_words, prefix)
        return self._filter(self._fallback_words(), prefix)

    def _jedi_suggestions(self) -> list[str]:
        try:
            cur = self._editor.textCursor()
            source = self._editor.toPlainText()
            line = cur.blockNumber() + 1
            col = cur.columnNumber()
            script = jedi.Script(source)
            return [c.name for c in script.complete(line, col)]
        except Exception:  # pylint: disable=broad-except
            return []

    def _fallback_words(self) -> list[str]:
        words = set(keyword.kwlist)
        words.update(dir(builtins))
        text = self._editor.toPlainText()
        words.update(_DEF_RE.findall(text))
        words.update(_ASSIGN_RE.findall(text))
        for token in _IDENT_RE.findall(text):
            words.add(token)
        return sorted(words)

    @staticmethod
    def _filter(words: list[str], prefix: str) -> list[str]:
        if not prefix:
            return words
        return [w for w in words if w.startswith(prefix) and w != prefix]

    def _insert_completion(self, completion: str) -> None:
        cur = self._editor.textCursor()
        prefix = self._completer.completionPrefix()
        # Replace the prefix already typed.
        for _ in range(len(prefix)):
            cur.deletePreviousChar()
        cur.insertText(completion)
        self._editor.setTextCursor(cur)
