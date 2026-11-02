# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Window title bars on Windows

With the dark theme on Windows, Windows draws the title bar of windows with its own
color scheme: since guidata 3.17.2, every window is given a title bar following the
color theme, including the DataLab windows which do not fix it themselves (e.g. the
conversation history of the AI Assistant), the dialogs created by static methods, and
the floating panels.

Native Windows platform only (``--show-windows``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np
import pytest
from guidata import qthelpers as guidata_qth
from plotpy import config as plotpy_config
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import create_signal

from datalab.aiassistant.conversation import ConversationStore
from datalab.aiassistant.widgets.conversationsdialog import ConversationsDialog
from datalab.config import Conf
from datalab.gui.commandpalette import CommandPaletteDialog
from datalab.gui.main import DLMainWindow
from datalab.gui.objectview import GetObjectsDialog
from datalab.gui.panel.base import NonModalInfoDialog
from datalab.gui.panel.macro import _RecentMacrosDialog
from datalab.tests import datalab_test_app_context
from datalab.widgets.capsule import CapsuleDialog
from datalab.widgets.instconfviewer import InstallConfigViewerWindow
from datalab.widgets.textimport import TextImportWizard

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows only")


def is_dark_title_bar(widget: QW.QWidget) -> bool:
    """Return True if the title bar of a window is dark (Windows API)"""
    import ctypes  # pylint: disable=import-outside-toplevel
    from ctypes import wintypes  # pylint: disable=import-outside-toplevel

    value = wintypes.BOOL()
    hwnd = wintypes.HWND(int(widget.winId()))
    for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (Windows 10 20H1+)
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
        )
        if result == 0:
            return bool(value.value)
    raise OSError("DwmGetWindowAttribute failed")


def process_events(duration: float = 0.3) -> None:
    """Process pending events (the title bars may be given again after a color mode
    change, once Qt has processed it, e.g. with Qt 6)"""
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        QW.QApplication.processEvents()
        time.sleep(0.01)


def set_color_mode(win: DLMainWindow, mode: str) -> None:
    """Set the color mode, as from the settings dialog"""
    Conf.color_mode.set(mode)
    win._update_color_mode()  # pylint: disable=protected-access
    process_events()


def static_dialog_title_bar(
    dialog_type: type[QW.QDialog], open_dialog: Callable[[], object]
) -> bool:
    """Open a dialog created by a static method, and return True if its title bar
    is dark"""
    app = QW.QApplication.instance()
    # With Qt 5, event loops return at once after `quit` was called without a running
    # event loop (e.g. by the unattended mode of previous tests): run it again
    QC.QTimer.singleShot(0, app.quit)
    app.exec()
    result: dict[str, bool] = {}

    def close_dialog() -> None:
        for widget in QW.QApplication.topLevelWidgets():
            if isinstance(widget, dialog_type) and widget.isVisible():
                result["dark"] = is_dark_title_bar(widget)
                widget.done(0)

    QC.QTimer.singleShot(0, close_dialog)
    open_dialog()
    return result["dark"]


def test_title_bars_follow_color_mode(monkeypatch, tmp_path) -> None:
    """The title bar of DataLab windows follows the color theme"""
    # Switching the color mode changes these globals: restore them at teardown
    monkeypatch.setenv(guidata_qth.ENV_COLOR_MODE, guidata_qth.get_color_mode())
    monkeypatch.setattr(guidata_qth, "CURRENT_THEME", guidata_qth.CURRENT_THEME)
    initial_mode = Conf.color_mode.get()
    initial_guidata_mode = guidata_qth.get_color_mode()
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        if QG.QGuiApplication.platformName() != "windows":
            pytest.skip("Native Windows platform only (use --show-windows)")
        dialogs: dict[str, QW.QWidget] = {}
        try:
            set_color_mode(win, "dark")
            win.signalpanel.add_object(
                create_signal("Signal", np.arange(10.0), np.arange(10.0))
            )
            dialogs = {
                "conversation history": ConversationsDialog(
                    ConversationStore(str(tmp_path)), win
                ),
                "workspace capsule": CapsuleDialog(win),
                "command palette": CommandPaletteDialog(win, []),
                "installation configuration": InstallConfigViewerWindow(win),
                "object selection": GetObjectsDialog(win, win.signalpanel, "Title"),
                "recent macros": _RecentMacrosDialog(win.macropanel),
                "pattern help": NonModalInfoDialog(win, "Pattern help", "Text"),
                "text import wizard": TextImportWizard("signal", win),
            }
            for dialog in dialogs.values():
                dialog.show()
            process_events()
            windows = {"main window": win, **dialogs}
            for name, window in windows.items():
                assert is_dark_title_bar(window), name

            # Dialogs created by static methods (e.g. renaming a conversation)
            assert static_dialog_title_bar(
                QW.QInputDialog, lambda: QW.QInputDialog.getText(win, "Rename", "Name")
            )
            assert static_dialog_title_bar(
                QW.QMessageBox, lambda: QW.QMessageBox.warning(win, "Title", "Text")
            )

            # Floating panels (new native windows)
            for widget, dock in win.docks.items():
                if widget is win.applicationspanel:
                    continue
                visible = dock.isVisible()
                dock.show()
                dock.setFloating(True)
                process_events()
                assert is_dark_title_bar(dock), dock.windowTitle()
                dock.setFloating(False)
                dock.setVisible(visible)
            # Applications panel: floating, closed, then shown again
            dock = win.docks[win.applicationspanel]
            for _index in range(2):
                win.show_applications()
                process_events()
                assert dock.isFloating() and is_dark_title_bar(dock)
                dock.close()
                process_events()
            win.show_applications()
            process_events()
            windows["Applications panel"] = dock

            set_color_mode(win, "light")
            for name, window in windows.items():
                assert not is_dark_title_bar(window), name
            set_color_mode(win, "dark")
            for name, window in windows.items():
                assert is_dark_title_bar(window), name
        finally:
            for dialog in dialogs.values():
                dialog.close()
                dialog.deleteLater()
            Conf.color_mode.set(initial_mode)
            # The style depends on the color mode: following tests must keep it
            plotpy_config.set_plotpy_color_mode(initial_guidata_mode)
