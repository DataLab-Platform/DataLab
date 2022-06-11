# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT launcher module
"""

import argparse

from guidata.configtools import get_image_file_path
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from codraft.config import Conf
from codraft.core.gui.main import CodraFTMainWindow
from codraft.utils.qthelpers import qt_app_context


def create(
    splash: bool = True, console: bool = None, objects=None, h5files=None, size=None
) -> CodraFTMainWindow:
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


def parse_args():
    """Parse command line arguments and run CodraFT"""
    parser = argparse.ArgumentParser(description="Run CodraFT")
    parser.add_argument(
        "--h5",
        type=str,
        required=False,
        help="HDF5 file names (separated by ';') with dataset name (separated by ',')",
    )
    args, _unknown = parser.parse_known_args()
    return args


def run(console=None, objects=None, h5files=None, size=None):
    """Run the CodraFT application

    Note: this function is an entry point in `setup.py` and therefore
    may not be moved without modifying the package setup script."""

    args = parse_args()
    if args.h5:
        h5files = ([] if h5files is None else h5files) + args.h5.split(";")

    with qt_app_context(exec_loop=True):
        window = create(
            splash=True, console=console, objects=objects, h5files=h5files, size=size
        )
        window.check_dependencies()
        window.check_for_previous_crash()


if __name__ == "__main__":
    run()
