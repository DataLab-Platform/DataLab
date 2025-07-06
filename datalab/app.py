# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab launcher module
"""

from __future__ import annotations

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.main import CDLMainWindow
from datalab.utils.qthelpers import datalab_app_context


def create(
    splash: bool = True,
    console: bool | None = None,
    objects=None,
    h5files=None,
    size=None,
) -> CDLMainWindow:
    """Create DataLab application and return mainwindow instance

    Args:
        splash: if True, show splash screen
        console: if True, show console
        objects: list of objects to add to the mainwindow
        h5files: list of h5files to open
        size: mainwindow size (width, height)
    """
    if splash:
        # Showing splash screen
        pixmap = QG.QPixmap(get_image_file_path("DataLab-Splash.png"))
        splashscreen = QW.QSplashScreen(pixmap, QC.Qt.WindowStaysOnTopHint)
        splashscreen.show()
    window = CDLMainWindow(console=console)
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


def run(console=None, objects=None, h5files=None, size=None):
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
        window = create(
            splash=True, console=console, objects=objects, h5files=h5files, size=size
        )
        QW.QApplication.processEvents()
        window.execute_post_show_actions()


if __name__ == "__main__":
    run()
