# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab launcher module
"""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import APP_NAME, Conf, _
from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.utils.instancecheck import ApplicationInstanceRegistry
from datalab.utils.qthelpers import datalab_app_context

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj


def create(
    splash: bool = True,
    console: bool | None = None,
    objects: list[ImageObj | SignalObj] | None = None,
    h5files: list[str] | None = None,
    size: tuple[int, int] | None = None,
) -> DLMainWindow:
    """Create DataLab application and return mainwindow instance

    Args:
        splash: if True, show splash screen
        console: if True, show console
        objects: list of objects to add to the mainwindow
        h5files: list of h5files to open
        size: mainwindow size (width, height)

    Returns:
        Main window instance
    """
    if splash:
        # Showing splash screen
        pixmap = QG.QPixmap(get_image_file_path("DataLab-Splash.png"))
        splashscreen = QW.QSplashScreen(pixmap, QC.Qt.WindowStaysOnTopHint)
        splashscreen.show()
    window = DLMainWindow(console=console)
    if size is not None:
        width, height = size
        window.resize(width, height)
    if splash:
        splashscreen.finish(window)
    if Conf.main.window_maximized.get(None):
        window.showMaximized()
    else:
        window.showNormal()
    if h5files is not None:
        window.open_h5_files(h5files, import_all=True)
    if objects is not None:
        for obj in objects:
            window.add_object(obj)
    if execenv.h5browser_file is not None:
        window.import_h5_file(execenv.h5browser_file)
    return window


def run(
    console: bool | None = None,
    objects: list[ImageObj | SignalObj] | None = None,
    h5files: list[str] | None = None,
    size: tuple[int, int] | None = None,
) -> None:
    """Run the DataLab application

    Note: this function is an entry point in `setup.py` and therefore
    may not be moved without modifying the package setup script.

    Args:
        console: if True, show console
        objects: list of objects to add to the mainwindow
        h5files: list of h5files to open
        size: mainwindow size (width, height)
    """
    if execenv.h5files:
        h5files = ([] if h5files is None else h5files) + execenv.h5files

    with datalab_app_context(exec_loop=True):
        # --- Instance detection -------------------------------------------
        # In unattended mode (tests), skip the lock check entirely so that
        # tests are never blocked by a running DataLab instance or a stale
        # lock file.
        if not execenv.unattended:
            registry = ApplicationInstanceRegistry()
            running_pid = registry.is_another_instance_running()
            if running_pid is not None:
                answer = QW.QMessageBox.warning(
                    None,
                    APP_NAME,
                    _(
                        "Another instance of DataLab (PID %d) appears to be "
                        "running.\n"
                        "Running multiple instances simultaneously may cause "
                        "side effects: preferences being overwritten, remote "
                        "control discovery issues, etc.\n\n"
                        "Do you want to continue anyway?"
                    )
                    % running_pid,
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                    QW.QMessageBox.No,
                )
                if answer == QW.QMessageBox.No:
                    QC.QTimer.singleShot(0, QW.QApplication.instance().quit)
                    return 0
            registry.create_lock_file(force=running_pid is not None)
            atexit.register(registry.remove_lock_file)
        # ------------------------------------------------------------------
        window = create(
            splash=True,
            console=console,
            objects=objects,
            h5files=h5files,
            size=size,
        )
        QW.QApplication.processEvents()
        window.execute_post_show_actions()


if __name__ == "__main__":
    run()
