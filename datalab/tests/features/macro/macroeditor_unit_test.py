# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro Panel unit tests
----------------------

The objective of this test is to check all the functionalities of the Macro Panel
widget, by calling all its methods and checking the results.

Some methods are not tested here, as they are tested in remote control tests
(see datalab/tests/features/control/remoteclient_app.py):
- `run_macro`
- `stop_macro`

All other methods should be tested here.
"""

# guitest: show

from __future__ import annotations

import os
import os.path as osp
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
from guidata import qapplication
from guidata.io import HDF5Reader, HDF5Writer
from guidata.qthelpers import qt_app_context
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.QtTest import QTest

from datalab.aiassistant.tools.builtin import _wait_macro_finished
from datalab.config import _
from datalab.env import execenv
from datalab.gui.macroeditor import Macro
from datalab.gui.macros_templates import list_templates
from datalab.gui.panel.macro import MacroPanel
from datalab.tests import datalab_test_app_context, helpers
from datalab.utils import macrorecovery
from datalab.widgets.codecompleter import MacroCodeEditor, PythonCompleter


@pytest.fixture(autouse=True)
def isolate_macro_cache(tmp_path, monkeypatch):
    """Keep the editor tests out of the user's recovery and recent caches."""
    monkeypatch.setattr(
        macrorecovery.Configuration,
        "get_path",
        staticmethod(lambda n: str(tmp_path / n)),
    )


@pytest.fixture
def editor():
    """A focused macro editor without a running macro or recovery file."""
    app = qapplication()
    widget = MacroCodeEditor()
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    app.processEvents()
    yield widget
    widget.close()


@pytest.mark.parametrize("key", [QC.Qt.Key_Tab, QC.Qt.Key_Return, QC.Qt.Key_Enter])
def test_macro_completion_keys(editor: MacroCodeEditor, key: QC.Qt.Key):
    """Accept the highlighted row, without inserting a tab or a newline."""
    editor.setPlainText("print(str.st)")
    editor.moveCursor(QG.QTextCursor.End)
    editor.moveCursor(QG.QTextCursor.PreviousCharacter)
    QTest.keyClick(editor, QC.Qt.Key_Space, QC.Qt.ControlModifier)
    popup = QW.QApplication.activePopupWidget()
    assert isinstance(popup, QW.QAbstractItemView)
    assert popup.currentIndex().data() == "startswith"
    if key != QC.Qt.Key_Tab:
        QTest.keyClick(popup, QC.Qt.Key_Down)
        assert popup.currentIndex().data() == "strip"
    selected = popup.currentIndex().data()
    QTest.keyClick(popup, key)
    assert editor.toPlainText() == f"print(str.{selected})"
    assert not popup.isVisible()
    editor.moveCursor(QG.QTextCursor.End)
    QTest.keyClick(editor, QC.Qt.Key_Return)
    QTest.keyClick(editor, QC.Qt.Key_Tab)
    assert editor.toPlainText() == f"print(str.{selected})\n\t"


def test_macro_member_completion(editor: MacroCodeEditor):
    """A dot offers the inferred object's members, not unrelated global names."""
    editor.setPlainText('sample = ""\nsample')
    editor.moveCursor(QG.QTextCursor.End)
    QTest.keyClicks(editor, ".")
    QW.QApplication.processEvents()
    popup = QW.QApplication.activePopupWidget()
    assert isinstance(popup, QW.QAbstractItemView)
    QTest.keyClicks(popup, "st")
    assert popup.currentIndex().data() == "startswith"
    QTest.keyClick(popup, QC.Qt.Key_Tab)
    assert editor.toPlainText() == 'sample = ""\nsample.startswith'

    for text in ("3", "# sample", "unknown_object"):
        editor.setPlainText(text)
        editor.moveCursor(QG.QTextCursor.End)
        QTest.keyClicks(editor, ".")
        QW.QApplication.processEvents()
        assert not popup.isVisible()
        assert editor.toPlainText() == text + "."


def test_macro_signature_help(editor: MacroCodeEditor):
    """An opening parenthesis shows the local function's signature, not a menu."""
    editor.setPlainText(
        "def calibrate(gain, offset=0): pass\ndef calibrate_raw(): pass\ncalibrate"
    )
    editor.moveCursor(QG.QTextCursor.End)
    QTest.keyClick(editor, QC.Qt.Key_Space, QC.Qt.ControlModifier)
    popup = QW.QApplication.activePopupWidget()
    assert popup is not None
    QTest.keyClicks(popup, "(")
    QW.QApplication.processEvents()
    tip = editor.calltip_widget
    assert tip.isVisible() and QW.QApplication.activePopupWidget() is None
    doc = QG.QTextDocument()
    doc.setHtml(tip.text())
    assert "calibrate(gain, offset=0)" in doc.toPlainText().replace("\xa0", " ")
    QTest.keyClicks(editor, "2)")
    assert not tip.isVisible()
    assert editor.toPlainText().endswith("calibrate(2)")


def test_macro_completion_suspended(editor: MacroCodeEditor):
    """Showcase suspends all automatic prompts by blocking the helper's timers."""
    (completer,) = editor.findChildren(PythonCompleter)
    (timer,) = completer.findChildren(QC.QTimer)
    with QC.QSignalBlocker(timer):
        for text in ("pri", "str.", "len("):
            editor.clear()
            editor.insertPlainText(text)
            QTest.qWait(timer.interval() + 50)
            assert QW.QApplication.activePopupWidget() is None
            assert not editor.calltip_widget.isVisible()
        timer.stop()
    editor.clear()
    QTest.keyClicks(editor, "str.")
    QW.QApplication.processEvents()
    assert completer.popup().isVisible()
    QTest.keyClick(completer.popup(), QC.Qt.Key_Escape)
    editor.clear()
    QTest.keyClicks(editor, "len(")
    QW.QApplication.processEvents()
    assert editor.calltip_widget.isVisible()


def test_python_completer_on_plain_editor():
    """Plugins can still attach the helper to a guidata editor and trigger it."""
    app = qapplication()
    widget = CodeEditor(language="python")
    helper = PythonCompleter(widget)
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    app.processEvents()
    widget.insertPlainText("str.st")
    helper.trigger()
    popup = helper.popup()
    assert popup.isVisible() and popup.currentIndex().data() == "startswith"
    QTest.keyClick(popup, QC.Qt.Key_Tab)
    assert widget.toPlainText() == "str.startswith"
    widget.close()


def get_macro_example_path() -> str:
    """Return macro example path"""
    path = helpers.get_temporary_directory()
    contents = """
# Simple DataLab macro example

import numpy as np

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.calc("fft")

print("All done! :)")
"""
    filename = osp.join(path, "macro_example.py")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(contents)
    return filename


def test_macro_editor():
    """Test dep viewer window"""
    with qt_app_context(exec_loop=True):
        widget = MacroPanel(None)
        widget.resize(800, 600)
        widget.show()

        # Create a new macro
        new_macro = widget.add_macro()
        assert new_macro is widget.get_macro()

        # Check out the macro title, serializable name, number, ...
        execenv.print("Macro title:", new_macro.title)
        execenv.print("Serializable name:", widget.get_serializable_name(new_macro))
        nb1 = widget.get_number_from_macro(new_macro)
        nb2 = widget.get_number_from_title(new_macro.title)
        assert nb1 == nb2 and nb1 == 1
        execenv.print("Macro number:", nb1)
        titles = widget.get_macro_titles()
        assert titles[0] == new_macro.title and len(titles) == 1
        new_title = "New title"
        widget.rename_macro(1, new_title)
        assert widget.get_macro_titles()[0] == new_title

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            fname = osp.join(tmpdir, "macro.py")
            widget.export_macro_to_file(1, fname)
            widget.import_macro_from_file(fname)
            imported_macro: Macro = widget.get_macro(2)
            assert imported_macro.title == new_macro.title
            assert imported_macro.get_code() == new_macro.get_code()
            widget.rename_macro(1, "Other title")
            widget.remove_macro(1)
            assert len(widget.get_macro_titles()) == 1
            assert widget.get_macro_titles()[0] == imported_macro.title

        # Remove all macros
        widget.remove_all_objects()
        assert len(widget.get_macro_titles()) == 0

        # Load a macro from file
        macro_path = get_macro_example_path()
        widget.import_macro_from_file(macro_path)
        assert len(widget.get_macro_titles()) == 1
        assert widget.get_macro_titles()[0] == osp.basename(macro_path)


def test_macro_uid_and_new_actions():
    """Test new Macro panel features: UID, duplicate, blank, templates."""
    with qt_app_context(exec_loop=True):
        widget = MacroPanel(None)
        widget.resize(800, 600)
        widget.show()

        # --- UID stability ---
        m1 = widget.add_macro()
        m2 = widget.add_macro()
        assert m1.uid and m2.uid
        assert m1.uid != m2.uid, "Two macros must have distinct UIDs"

        # UID is preserved across serialize/deserialize round-trips.
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            h5_path = osp.join(tmpdir, "macro_uid.h5")
            with HDF5Writer(h5_path) as writer:
                with writer.group("macro"):
                    m1.serialize(writer)
            restored = Macro(widget.console)
            with HDF5Reader(h5_path) as reader:
                with reader.group("macro"):
                    restored.deserialize(reader)
            assert restored.uid == m1.uid

            # Macros without ``uid`` (workspaces before 1.4, DataLab-Web) are read
            # and the reader stays usable for the following groups.
            legacy_path = osp.join(tmpdir, "macro_no_uid.h5")
            with HDF5Writer(legacy_path) as writer:
                with writer.group("macro"):
                    with writer.group("title"):
                        writer.write("Legacy")
                    with writer.group("contents"):
                        writer.write("print('legacy')")
                with writer.group("after"):
                    writer.write("ok")
            legacy = Macro(widget.console)
            with HDF5Reader(legacy_path) as reader:
                with reader.group("macro"):
                    legacy.deserialize(reader)
                with reader.group("after"):
                    assert reader.read_any() == "ok"
            assert legacy.title == "Legacy" and legacy.uid

        # --- duplicate_macro ---
        widget.remove_all_objects()
        original = widget.add_macro()
        original_code = original.get_code()
        widget.duplicate_macro()
        assert len(widget.get_macro_titles()) == 2
        copy = widget.get_macro(2)
        assert copy is not original
        assert copy.uid != original.uid
        assert copy.get_code() == original_code
        assert copy.title == _("%s (copy)") % original.title

        # --- add_blank_macro ---
        widget.remove_all_objects()
        blank = widget.add_blank_macro()
        assert blank.get_code() == ""

        # --- add_macro_from_template ---
        widget.remove_all_objects()
        templates = list_templates()
        assert templates, "Expected bundled macro templates to be available"
        tmpl_macro = widget.add_macro_from_template(templates[0])
        # ``set_code`` may normalize whitespace (e.g. add trailing spaces on
        # blank lines via the highlighter); compare the textual content
        # line-by-line after stripping trailing whitespace.
        actual = "\n".join(line.rstrip() for line in tmpl_macro.get_code().splitlines())
        expected = "\n".join(line.rstrip() for line in templates[0].code.splitlines())
        assert actual == expected
        assert tmpl_macro.title == templates[0].title

        widget.remove_all_objects()


def test_macro_unicode_encoding():
    """Test that macros can print Unicode characters without encoding errors.

    This test verifies the fix for the UnicodeEncodeError that occurred on Windows
    systems with locales like cp1252 when macros printed Unicode characters.

    The test creates and runs a macro that prints various Unicode characters,
    simulating the scenario where RemoteProxy connection messages (which contain
    arrows →) would cause encoding errors on Windows with cp1252 locale.

    Without the UTF-8 encoding fix in Macro.run(), this test would fail with:
    UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
    """
    with helpers.WorkdirRestoringTempDir():
        with datalab_test_app_context(console=False) as win:
            win.set_current_panel("macro")

            # Create a macro that prints various Unicode characters
            macro = win.macropanel.add_macro()
            macro.title = "Unicode Test Macro"

            # This test verifies that Unicode characters can be printed successfully.
            # The macro prints Unicode characters without any encoding manipulation.
            # With the UTF-8 fix in Macro.run(), these print statements work correctly.
            # Without the fix, on systems with cp1252 locale, these would fail.
            #
            # Note: We cannot reliably simulate cp1252 locale in the test because:
            # 1. Modern Python often defaults to UTF-8
            # 2. If we manually reconfigure to cp1252 in the macro, it overrides
            #    any fix done before the macro code runs
            # 3. The PYTHONIOENCODING env var might be set system-wide
            #
            # This test serves as a regression test - it will catch if the fix
            # is removed, but only on systems that actually default to cp1252.
            unicode_code = """
import sys

# Print encoding info for debugging
print(f"stdout encoding: {sys.stdout.encoding}")
print(f"stderr encoding: {sys.stderr.encoding}")

# Print various Unicode characters that are not in cp1252
# On systems with cp1252 default locale, without the UTF-8 fix,
# these would cause UnicodeEncodeError
print("Testing Unicode output:")
print("  → Arrow character (U+2192)")
print("  ✓ Check mark (U+2713)")
print("  • Bullet point (U+2022)")
print("  … Ellipsis (U+2026)")
print("  Emoji: 🎉 🚀 ⚡")

# Simulate RemoteProxy connection message format
print("Setting XML-RPC port... [input:None] →[execenv.xmlrpcport:None] OK")

print("All Unicode tests passed! ✓")
"""
            macro.set_code(unicode_code)

            # Run the macro and wait for completion
            execenv.print("Running Unicode test macro...")
            win.macropanel.run_macro()

            # Wait for macro to complete (with timeout)
            # We need to process Qt events for the QProcess signals to be delivered
            max_wait = 10  # seconds
            elapsed = 0
            while macro.is_running() and elapsed < max_wait:
                QW.QApplication.processEvents()
                time.sleep(0.1)
                elapsed += 0.1

            # Verify the macro completed (not still running)
            # If there was an encoding error, the process would have crashed
            assert not macro.is_running(), (
                "Macro did not complete within timeout - "
                "likely failed with encoding error"
            )

            # Check the exit code - should be 0 for success
            # With the UTF-8 fix, the macro completes successfully (exit code 0)
            # Without the fix, it crashes with UnicodeEncodeError (exit code 1)
            exit_code = macro.get_exit_code()
            assert exit_code == 0, (
                f"Macro exited with error code {exit_code} - "
                f"likely UnicodeEncodeError when trying to print Unicode characters"
            )

            execenv.print("✓ Unicode test macro completed successfully!")
            execenv.print(
                "Note: This test verifies Unicode support works. On systems with "
                "UTF-8 as default encoding, it may pass even without the fix. "
                "The fix is critical for Windows systems with cp1252 locale."
            )

            # Clean up
            win.macropanel.remove_all_objects()


def _wait_macro(macro: Macro, timeout_ms: int = 10000) -> None:
    """Wait for all runs, including an immediate restart from FINISHED."""
    timer = QC.QElapsedTimer()
    timer.start()
    while macro.is_running() and timer.elapsed() < timeout_ms:
        QTest.qWait(10)
    assert not macro.is_running(), "macro did not finish before the deadline"


def test_macro_start_failure():
    """Test that a macro which cannot start reports why (issue #587).

    Without the fix, the error message box was given the macro (a QObject) as
    parent, raising a TypeError which hid the cause of the failure.
    """
    missing = osp.join(helpers.get_temporary_directory(), "missing_python.exe")
    messages = []

    def critical(parent, title, text):
        """Record the message box instead of showing it"""
        assert isinstance(parent, QW.QWidget), f"Invalid parent: {parent!r}"
        messages.append(text)

    with qt_app_context():
        widget = MacroPanel(None)
        macro = widget.add_macro()
        states = []
        macro.STARTED.connect(lambda: states.append("started"))
        macro.FINISHED.connect(lambda: states.append("finished"))
        created_scripts: list[str | None] = []
        orig_start = QC.QProcess.start

        def spy_start(self_proc, program, arguments, *rest, **kwargs):
            created_scripts.append(macro._temp_script_path)
            return orig_start(self_proc, program, arguments, *rest, **kwargs)

        with (
            mock.patch.object(sys, "executable", missing),
            mock.patch.object(QW.QMessageBox, "critical", side_effect=critical),
            mock.patch.object(QC.QProcess, "start", spy_start),
        ):
            widget.run_macro()
            timer = QC.QElapsedTimer()
            timer.start()
            while "finished" not in states and timer.elapsed() < 10000:
                QW.QApplication.processEvents()
                time.sleep(0.05)
        assert states and states[-1] == "finished"
        assert not macro.is_running()
        assert macro.get_exit_code() is None
        assert len(created_scripts) == 1 and created_scripts[0] is not None
        assert not osp.exists(created_scripts[0])
        assert len(messages) == 1 and missing in messages[0]
        error_msg = _("# ==> Unable to run '%s' macro") % macro.title
        assert error_msg in widget.console.toPlainText()

        # The AI Assistant does not wait for a macro which failed to start
        timer.restart()
        _output, timed_out = _wait_macro_finished(macro, widget.console, 0, 5.0)
        assert not timed_out and timer.elapsed() < 1000
        widget.remove_all_objects()


def test_macro_temp_script_long_code_and_traceback():
    """Test running long macros via a temporary file with accurate line numbers.

    Regression test for issue #588: macro code must not be passed on the command
    line, traceback line numbers must match the editor, and the temporary file
    must be removed after execution (both on success and on error).
    """
    with qt_app_context():
        widget = MacroPanel(None)
        macro = widget.add_macro()

        # 1. Long macro (>40,000 chars, exceeding Windows' 32,767 char limit)
        # with an encoding cookie and special shell/quoting characters.
        padding = ("# " + "x" * 78 + "\n") * 550
        special = r"""quote' "double" \backslash\ %percent% $dollar$ → ✓"""
        long_code = (
            "\ufeff# -*- coding: latin-1 -*-\n"
            "from __future__ import annotations\n"
            f"{padding}"
            "import pickle, sys\n"
            "class Example: pass\n"
            "assert globals() is vars(sys.modules['__main__'])\n"
            "assert '__file__' not in globals()\n"
            "assert type(pickle.loads(pickle.dumps(Example()))) is Example\n"
            f"MARKER = {special!r}\n"
            "print('LONG_OK:', MARKER.encode().hex(), __name__)\n"
        )
        assert len(long_code) > 40000
        macro.set_code(long_code)

        recorded_args: list[list[str]] = []
        orig_start = QC.QProcess.start

        def spy_start(self_proc, program, arguments, *rest, **kwargs):
            recorded_args.append(list(arguments))
            return orig_start(self_proc, program, arguments, *rest, **kwargs)

        with mock.patch.object(QC.QProcess, "start", spy_start):
            widget.run_macro()
            script_path = macro._temp_script_path
            _wait_macro(macro)

        assert not macro.is_running()
        assert macro.get_exit_code() == 0
        assert len(recorded_args) == 1 and len(" ".join(recorded_args[0])) < 4096
        assert (
            f"LONG_OK: {special.encode().hex()} __main__"
            in widget.console.toPlainText()
        )
        assert script_path is not None and not osp.exists(script_path)

        # 2. Traceback line numbers must match the editor (line 3 -> "line 3"),
        # and the temporary script file must still be removed after an error.
        macro.set_code("a = 1\nb = 2\nraise ValueError('boom_line_3')\n")
        before = len(widget.console.toPlainText())
        widget.run_macro()
        err_script_path = macro._temp_script_path
        _wait_macro(macro)

        assert not macro.is_running()
        assert macro.get_exit_code() == 1
        err_output = widget.console.toPlainText()[before:]
        assert f'File "<macro:{macro.title}>", line 3,' in err_output
        assert "raise ValueError('boom_line_3')" in err_output
        assert "ValueError: boom_line_3" in err_output
        assert err_script_path is not None and not osp.exists(err_script_path)

        widget.remove_all_objects()


@pytest.mark.parametrize("operation", ["create", "write", "close"])
def test_macro_preparation_error(tmp_path, monkeypatch, operation):
    """Report preparation errors and remove even a partially written file."""
    create_file = tempfile.NamedTemporaryFile

    @contextmanager
    def failing_file(*args, **kwargs):
        error = OSError(f"test: {operation} failed")
        if operation == "create":
            raise error
        with create_file(*args, dir=tmp_path, **kwargs) as stream:
            if operation == "write":
                stream.write = mock.Mock(side_effect=error)
            yield stream
        if operation == "close":
            raise error

    with qt_app_context():
        panel = MacroPanel(None)
        macro = panel.add_macro()
        states = []
        macro.STARTED.connect(lambda: states.append("started"))
        macro.FINISHED.connect(lambda: states.append("finished"))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_file)
        with mock.patch.object(QW.QMessageBox, "critical"):
            macro.run()
        assert states == ["finished"]
        assert macro.get_exit_code() is None and not macro.is_running()
        assert f"test: {operation} failed" in panel.console.toPlainText()
        assert not list(tmp_path.glob("datalab_macro_*.py"))
        panel.remove_all_objects()


def test_removing_running_macro_stops_and_cleans():
    """Removing a running tab stops its child and removes its script."""
    with qt_app_context():
        panel = MacroPanel(None)
        macro = panel.add_macro()
        macro.set_code("import time; time.sleep(20)")
        macro.run()
        path = Path(macro._temp_script_path)
        try:
            panel.remove_macro(1)
            assert not macro.is_running()
            assert not path.exists()
        finally:
            if macro.is_running():
                macro.kill()
                _wait_macro(macro)


def test_macro_independent_runs_and_restart():
    """A refused duplicate must not disturb another tab or a FINISHED restart."""
    with qt_app_context():
        panel = MacroPanel(None)
        first = panel.add_macro_with_code("one", "import time; time.sleep(30)")
        second = panel.add_macro_with_code("two", "raise SystemExit(7)")
        exits = []
        first_finished = []
        first.FINISHED.connect(lambda: first_finished.append(True))

        def restart():
            exits.append(second.get_exit_code())
            if len(exits) == 1:
                second.set_code("print('second run completed')")
                panel.run_macro(2)

        second.FINISHED.connect(restart)
        panel.run_macro(1)
        try:
            path = Path(first._temp_script_path)
            panel.run_macro(1)
            panel.run_macro(2)
            _wait_macro(second)
            assert exits == [7, 0]
            assert "second run completed" in panel.console.toPlainText()
            assert first.is_running() and not first_finished
            assert path == Path(first._temp_script_path) and path.is_file()
        finally:
            panel.remove_all_objects()
        assert first_finished == [True] and not path.exists()


def test_hiding_macro_panel_is_not_shutdown():
    """Only closing the application stops the hidden tab and preserves recovery."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.macropanel
        macro = panel.add_macro_with_code("hidden", "import time; time.sleep(30)")
        panel.run_macro()
        path = Path(macro._temp_script_path)
        win.docks[panel].hide()
        assert macro.is_running() and path.exists()
    assert not macro.is_running() and not path.exists()
    assert macro.uid in macrorecovery.load_pending()


@pytest.mark.parametrize("permanent_lock", [False, True])
def test_macro_syntax_error_and_cleanup_retry(monkeypatch, permanent_lock):
    """Retry locks and report persistent locks without hiding the syntax error."""
    with qt_app_context():
        panel = MacroPanel(None)
        macro = panel.add_macro_with_code("syntax", "a = 1\nb = 2\nif :")
        macro.run()
        path = macro._temp_script_path
        remove = os.remove
        attempts = []

        def locked_once(filename):
            if filename == path:
                attempts.append(filename)
                if permanent_lock or len(attempts) == 1:
                    raise PermissionError("locked")
            return remove(filename)

        monkeypatch.setattr(os, "remove", locked_once)
        _wait_macro(macro)
        assert attempts == [path, path] and osp.exists(path) == permanent_lock
        output = panel.console.toPlainText()
        if permanent_lock:
            remove(path)
            assert path in output and "locked" in output
        assert macro.get_exit_code() == 1
        assert 'File "<macro:syntax>", line 3' in output
        assert "if :" in output and "SyntaxError:" in output
        panel.remove_all_objects()


if __name__ == "__main__":
    test_macro_editor()
    test_macro_uid_and_new_actions()
    test_macro_unicode_encoding()
    test_macro_start_failure()
    test_macro_temp_script_long_code_and_traceback()
