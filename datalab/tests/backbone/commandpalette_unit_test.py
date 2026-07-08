# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Command palette unit tests for DataLab

Testing the command palette helpers (fuzzy matcher, command collection)
and the dialog construction.
"""

# guitest: show

from __future__ import annotations

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from sigima.tests.data import create_paracetamol_signal

from datalab.gui.commandpalette import (
    COMMAND_PATH_SEPARATOR,
    CommandPaletteDialog,
    CommandSearchField,
    collect_commands,
    fuzzy_score,
)
from datalab.tests import datalab_test_app_context


def test_fuzzy_score():
    """Test the fuzzy subsequence matcher."""
    # Empty query matches anything with a neutral score
    assert fuzzy_score("", "anything") == 0
    # A plain substring (single contiguous run) matches, even mid-word
    assert fuzzy_score("fft", "processing › fourier analysis › fft") is not None
    assert fuzzy_score("rota", "rotate") is not None
    assert fuzzy_score("bration", "calibration") is not None
    # Acronym-style matches where every run starts at a word boundary match
    assert fuzzy_score("fan", "fourier analysis") is not None
    # A missing character or an over-long query does not match
    assert fuzzy_score("xyz", "fourier analysis") is None
    assert fuzzy_score("abcdef", "abc") is None
    # Scattered mid-word noise is rejected: "rota" must not match paths that
    # merely contain r, o, t, a as scattered mid-word runs
    assert fuzzy_score("rota", "edit › annotations › import annotations") is None
    assert fuzzy_score("rota", "analysis › horizontal projection") is None
    assert fuzzy_score("abc", "a1b2c3 def") is None
    # A contiguous match scores higher than a boundary-run one
    contiguous = fuzzy_score("abc", "abc def")
    boundary = fuzzy_score("abc", "a b c")
    assert contiguous is not None and boundary is not None
    assert contiguous > boundary


def test_collect_commands():
    """Test command collection and dialog filtering on the main window."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("signal")
        win.signalpanel.add_object(create_paracetamol_signal(500))
        panel = win.signalpanel

        commands = collect_commands(win, panel)
        assert commands, "Command palette should list commands"
        # Every command is identified by a separator-joined menu path
        assert all(COMMAND_PATH_SEPARATOR in path for _action, path in commands)

        # The dialog builds and filters live
        dialog = CommandPaletteDialog(win, commands)
        try:
            assert dialog.listwidget.count() == len(commands)
            dialog.search.setText("fft")
            assert dialog.listwidget.count() > 0
            assert dialog.listwidget.count() < len(commands)
        finally:
            dialog.close()


def test_command_search_field():
    """Test the menu-bar search-field launcher."""
    with datalab_test_app_context(console=False) as win:
        opened = []
        field = CommandSearchField(win, lambda: opened.append(True), "Ctrl+Shift+P")
        try:
            # The field advertises both the action and its shortcut
            assert field.placeholder_label.text()
            assert field.shortcut_label is not None
            assert "Ctrl+Shift+P" in field.shortcut_label.text()
            # Its size hint fits the full content (icon + text + shortcut)
            assert field.sizeHint().width() > field.placeholder_label.width()
            # A click opens the palette
            field.mousePressEvent(None)
            assert opened == [True]
            # Enter opens it too; plain typing does not
            event = QG.QKeyEvent(QC.QEvent.KeyPress, QC.Qt.Key_Return, QC.Qt.NoModifier)
            field.keyPressEvent(event)
            assert opened == [True, True]
        finally:
            field.deleteLater()


if __name__ == "__main__":
    test_fuzzy_score()
    test_collect_commands()
    test_command_search_field()
