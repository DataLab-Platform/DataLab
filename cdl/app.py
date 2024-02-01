# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab launcher module
"""

from __future__ import annotations

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import Conf
from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.utils.qthelpers import cdl_app_context


def create(
    splash: bool = True,
    console: bool | None = None,
    objects=None,
    h5files=None,
    size=None,
    tour: bool | None = None,
) -> CDLMainWindow:
    """Create DataLab application and return mainwindow instance

    Args:
        splash: if True, show splash screen
        console: if True, show console
        objects: list of objects to add to the mainwindow
        h5files: list of h5files to open
        size: mainwindow size (width, height)
        tour: if True, show tour at startup, if False, don't show tour at startup,
         if None, use configuration setting (i.e. show tour at first startup only)
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
    if tour is None:
        tour = Conf.main.tour_enabled.get()
        Conf.main.tour_enabled.set(False)
    if tour:
        window.show_tour()
    return window


def run(console=None, objects=None, h5files=None, size=None, tour=None):
    """Run the DataLab application

    Note: this function is an entry point in `setup.py` and therefore
    may not be moved without modifying the package setup script.

    Args:
        console: if True, show console
        objects: list of objects to add to the mainwindow
        h5files: list of h5files to open
        size: mainwindow size (width, height)
        tour: if True, show tour at startup, if False, don't show tour at startup,
         if None, use configuration setting (i.e. show tour at first startup only)
    """
    if execenv.h5files:
        h5files = ([] if h5files is None else h5files) + execenv.h5files

    with cdl_app_context(exec_loop=True):
        window = create(
            splash=True,
            console=console,
            objects=objects,
            h5files=h5files,
            size=size,
            tour=tour,
        )
        QW.QApplication.processEvents()
        window.check_stable_release()
        window.check_for_previous_crash()


if __name__ == "__main__":
    run()
