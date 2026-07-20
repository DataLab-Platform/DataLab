# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""DataLab Qt utilities."""

from __future__ import annotations

import functools
import os
import os.path as osp
import shutil
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager

from guidata.configtools import get_icon
from guidata.qthelpers import grab_save_window as guidata_grab_save_window
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigimax.utils.qthelpers import (
    CallbackWorker,
    add_corner_menu,
    block_signals,
    bring_to_front,
    configure_menu_about_to_show,
    create_progress_bar,
    is_running_tests,
    qt_handle_error_message,
    qt_long_callback,
    qt_try_loadsave_file,
    resize_widget_to_parent,
    save_restore_stds,
    sigimax_app_context,
    try_or_log_error,
)

from datalab.config import SHOTPATH, Conf

__all__ = [
    "CallbackWorker",
    "add_corner_menu",
    "block_signals",
    "bring_to_front",
    "configure_menu_about_to_show",
    "create_menu_button",
    "create_progress_bar",
    "datalab_app_context",
    "grab_save_window",
    "is_running_tests",
    "open_local_path",
    "qt_handle_error_message",
    "qt_long_callback",
    "qt_try_except",
    "qt_try_loadsave_file",
    "resize_widget_to_parent",
    "save_restore_stds",
    "show_in_folder",
    "try_or_log_error",
]


def open_local_path(path: str) -> bool:
    """Open a local path with the desktop handler."""
    return QG.QDesktopServices.openUrl(QC.QUrl.fromLocalFile(path))


def show_in_folder(path: str) -> bool:
    """Show a file in its containing folder, selecting it when supported."""
    filepath = osp.abspath(path)
    directory = osp.dirname(filepath)

    if sys.platform.startswith("win"):
        commands = [["explorer", f"/select,{osp.normpath(filepath)}"]]
    elif sys.platform == "darwin":
        commands = [["open", "-R", filepath]]
    else:
        commands = []
        if shutil.which("nautilus"):
            commands.append(["nautilus", "--select", filepath])
        if shutil.which("dolphin"):
            commands.append(["dolphin", "--select", filepath])
        if shutil.which("nemo"):
            commands.append(["nemo", filepath])
        if shutil.which("caja"):
            commands.append(["caja", "--select", filepath])

    for command in commands:
        try:
            subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            continue
    return open_local_path(directory)


@contextmanager
def datalab_app_context(
    exec_loop: bool = False, enable_logs: bool = True
) -> Generator[QW.QApplication, None, None]:
    """Create a DataLab application context using SigimaX infrastructure."""
    Conf.reload_from_ini()
    with sigimax_app_context(exec_loop=exec_loop, enable_logs=enable_logs) as qapp:
        yield qapp


def qt_try_except(message=None, context=None):
    """Decorate a DataLab Qt method with status and error handling."""

    def qt_try_except_decorator(func):
        @functools.wraps(func)
        def method_wrapper(*args, **kwargs):
            self = args[0]
            panel = getattr(self, "panel", self)
            if message is not None:
                panel.SIG_STATUS_MESSAGE.emit(message, 0)
                QW.QApplication.setOverrideCursor(QG.QCursor(QC.Qt.WaitCursor))
                panel.repaint()
            output = None
            try:
                output = func(*args, **kwargs)
            except Exception as msg:  # pylint: disable=broad-except
                if is_running_tests():
                    raise
                qt_handle_error_message(panel.parentWidget(), msg, context)
            finally:
                if message is not None:
                    panel.SIG_STATUS_MESSAGE.emit("", 0)
                    QW.QApplication.restoreOverrideCursor()
            return output

        return method_wrapper

    return qt_try_except_decorator


def grab_save_window(
    widget: QW.QWidget, name: str | None = None, add_timestamp: bool = True
) -> None:  # pragma: no cover
    """Save a screenshot using DataLab naming and localization conventions."""
    if name is None:
        name = widget.objectName()

    if name.endswith("_"):
        add_timestamp = True
    elif name[-1].isdigit() or name.startswith(("s_", "i_")):
        add_timestamp = False

    lang_env = (os.environ.get("LANG") or "en").lower()
    lang = "fr" if lang_env.startswith("fr") else "en"
    name = f"{name}.{lang}"

    guidata_grab_save_window(
        widget=widget, name=name, save_dir=SHOTPATH, add_timestamp=add_timestamp
    )


def create_menu_button(
    parent: QW.QWidget | None = None, menu: QW.QMenu | None = None
) -> QW.QPushButton:
    """Create a DataLab menu button."""
    button = QW.QPushButton(get_icon("libre-gui-menu.svg"), "", parent)
    button.setFlat(True)
    if menu is not None:
        button.setMenu(menu)
    return button
