# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT launcher module
"""

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from codraft.config import Conf
from codraft.core.gui.main import CodraFTMainWindow
from codraft.utils.qthelpers import qt_app_context


def create(splash=True, console=None, objects=None, h5files=None, size=None):
    """Create CodraFT application and return mainwindow instance"""
    if splash:
        # Showing splash screen
        pixmap = QG.QPixmap(get_image_file_path("codraft_titleicon.png"))
        splashscreen = QW.QSplashScreen(pixmap, QC.Qt.WindowStaysOnTopHint)
        splashscreen.show()
    window = CodraFTMainWindow(console=console)
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
    return window


def run(console=None, objects=None, h5files=None, size=None):
    """Run the CodraFT application

    Note: this function is an entry point in `setup.py` and therefore
    may not be moved without modifying the package setup script."""

    with qt_app_context(exec_loop=True):
        window = create(
            splash=True, console=console, objects=objects, h5files=h5files, size=size
        )
        window.check_dependencies()


if __name__ == "__main__":
    run()
