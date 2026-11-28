# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro code editor with Jedi completion and signature help.
"""

from __future__ import annotations

import builtins
import io
import keyword
import logging
import re
import tokenize

import jedi
from guidata.widgets.codeeditor import CodeEditor
from guidata.widgets.console.calltip import CallTipWidget
from guidata.widgets.console.mixins import BaseEditMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
_PREFIX_RE = re.compile(r"\w+$")

_DEBOUNCE_MS = 200
_MIN_CHARS = 2


def _is_dot_operator(source: str, line: int, column: int) -> bool:
    """Distinguish a member-access dot from one in a number, comment or string."""
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.end >= (line, column):
                return token.type == tokenize.OP and token.string == "."
    except (tokenize.TokenError, IndentationError):
        # An unfinished multiline string, for example, has no dot token.
        pass
    return False


class MacroCodeEditor(CodeEditor, BaseEditMixin):
    """Python editor with keyboard completion and the console's signature popup."""

    def __init__(self) -> None:
        super().__init__(language="python")
        BaseEditMixin.__init__(self)
        self.calltip_widget = CallTipWidget(self)
        self.calltip_widget.setParent(self, QC.Qt.ToolTip)
        self.completer = PythonCompleter(self)
        self.completer.signature_requested.connect(self._show_signature)
        self.completer.highlighted.connect(self.calltip_widget.hide)

    def _show_signature(self, line: int, column: int) -> None:
        self.calltip_widget.hide()
        try:
            signatures = jedi.Script(self.toPlainText()).get_signatures(line, column)
        except (jedi.InternalError, ValueError):
            logging.getLogger(__name__).exception("Jedi signature lookup failed")
            return
        if signatures:
            self.show_calltip(
                _("Parameters"), signatures[0].to_string(), signature=True
            )


class PythonCompleter(QW.QCompleter):
    """Attach Python completion to a code editor.

    All automatic prompts use the child timer: plugins such as Showcase block
    its signals during simulated typing. Explicit requests use :meth:`trigger`.
    """

    signature_requested = QC.Signal(int, int)

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self._editor = editor
        self._model = QC.QStringListModel(self)
        self.setModel(self._model)
        self.setWidget(editor)
        self.setCompletionMode(QW.QCompleter.PopupCompletion)
        self.setCaseSensitivity(QC.Qt.CaseSensitive)
        self.activated.connect(self._insert_completion)

        self._timer = QC.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_completions)
        editor.textChanged.connect(self._text_changed)
        shortcut = QW.QShortcut(QG.QKeySequence("Ctrl+Space"), editor)
        shortcut.setContext(QC.Qt.WidgetShortcut)
        shortcut.activated.connect(self.trigger)

    def trigger(self) -> None:
        """Show suggestions at the cursor without waiting for automatic completion."""
        self._show_completions(force=True)

    def eventFilter(self, obj: QC.QObject, event: QC.QEvent) -> bool:
        """Accept popup keys before QCompleter forwards them to the editor."""
        if isinstance(event, QG.QKeyEvent) and event.type() == QC.QEvent.KeyPress:
            popup = self.popup()
            if not popup.isVisible():
                return super().eventFilter(obj, event)
            if event.key() in (QC.Qt.Key_Tab, QC.Qt.Key_Return, QC.Qt.Key_Enter):
                index = popup.currentIndex()
                if index.isValid():
                    self._insert_completion(index.data())
                    event.accept()
                    return True
            if event.key() == QC.Qt.Key_Escape:
                popup.hide()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _before_cursor(self) -> str:
        cursor = self._editor.textCursor()
        cursor.movePosition(QG.QTextCursor.StartOfBlock, QG.QTextCursor.KeepAnchor)
        return cursor.selectedText()

    @staticmethod
    def _prefix(before: str) -> str:
        match = _PREFIX_RE.search(before)
        return match[0] if match else ""

    def _text_changed(self) -> None:
        before = self._before_cursor()
        popup = self.popup()
        if popup.isVisible():
            prefix = self._prefix(before)
            if prefix and prefix.startswith(self.completionPrefix()):
                self.setCompletionPrefix(prefix)
                if self.completionCount():
                    popup.setCurrentIndex(self.currentIndex())
                    return
            popup.hide()
        self._timer.start(0 if before.endswith((".", "(")) else _DEBOUNCE_MS)

    def _show_completions(self, force: bool = False) -> None:
        self._timer.stop()
        if self._editor.isReadOnly() or (not force and not self._editor.hasFocus()):
            return
        before = self._before_cursor()
        line = self._editor.textCursor().blockNumber() + 1
        if before.endswith("(") and not force:
            self.signature_requested.emit(line, len(before))
            return
        prefix = self._prefix(before)
        dot_column = len(before) - len(prefix)
        member = before[:dot_column].endswith(".")
        popup = self.popup()
        if not (force or member or len(prefix) >= _MIN_CHARS):
            popup.hide()
            return
        source = self._editor.toPlainText()
        if member and not _is_dot_operator(source, line, dot_column):
            popup.hide()
            return
        try:
            words = [c.name for c in jedi.Script(source).complete(line, len(before))]
        except (jedi.InternalError, ValueError):
            logging.getLogger(__name__).exception("Jedi completion failed")
            words = []
        if not words and not member:
            words = self._fallback_words()
        suggestions = self._filter(words, prefix)
        if not suggestions:
            popup.hide()
            return
        self._model.setStringList(suggestions)
        self.setCompletionPrefix(prefix)
        rect = self._editor.cursorRect()
        rect.setWidth(
            popup.sizeHintForColumn(0) + popup.verticalScrollBar().sizeHint().width()
        )
        self.complete(rect)
        popup.setCurrentIndex(self.currentIndex())

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
        self.popup().hide()
        cur = self._editor.textCursor()
        if self.completionPrefix():
            # At the boundary before ')' or ',', WordUnderCursor selects punctuation.
            cur.movePosition(QG.QTextCursor.PreviousCharacter)
            cur.select(QG.QTextCursor.WordUnderCursor)
        cur.insertText(completion)
        self._editor.setTextCursor(cur)
        self._timer.stop()
