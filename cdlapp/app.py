# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab launcher module
"""

from __future__ import annotations

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdlapp.config import Conf
from cdlapp.core.gui.main import CDLMainWindow
from cdlapp.env import execenv
from cdlapp.utils.qthelpers import cdl_app_context


def create(
    splash: bool = True,
    console: bool | None = None,
    objects=None,
    h5files=None,
    size=None,
) -> CDLMainWindow:
    """Create DataLab application and return mainwindow instance"""
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
    may not be moved without modifying the package setup script."""
    if execenv.h5files:
        h5files = ([] if h5files is None else h5files) + execenv.h5files

    with cdl_app_context(exec_loop=True):
        window = create(
            splash=True, console=console, objects=objects, h5files=h5files, size=size
        )
        QW.QApplication.processEvents()
        window.check_stable_release()
        window.check_for_previous_crash()


if __name__ == "__main__":
    run()
