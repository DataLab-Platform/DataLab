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

import os.path as osp
import time

from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.gui.macroeditor import Macro
from datalab.gui.panel.macro import MacroPanel
from datalab.tests import datalab_test_app_context, helpers


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


if __name__ == "__main__":
    test_macro_editor()
    test_macro_unicode_encoding()
