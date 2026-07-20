# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab launcher module
"""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigimax.app import create as sigimax_create

from datalab.config import APP_NAME, _
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
    window = sigimax_create(
        window_class=DLMainWindow,
        splash=splash,
        console=console,
        h5files=h5files,
        size=size,
    )
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
        # DataLab-specific protection: concurrent instances share XML-RPC
        # connection settings and may overwrite each other's network config.
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
                    return
            registry.create_lock_file(force=running_pid is not None)
            atexit.register(registry.remove_lock_file)

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
