# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT unit tests
"""

from contextlib import contextmanager

from guidata.guitest import run_testlauncher
from qtpy import QtCore as QC

import codraft.config  # Loading icons
from codraft import env
from codraft.core.gui.main import CodraFTMainWindow
from codraft.utils import qthelpers as qth
from codraft.utils import tests

# TODO: [P2] Documentation: add more screenshots from tests
# TODO: [P3] Create subpackages "app" & "unit" + add support for subpackages in
# test launcher


@contextmanager
def codraft_app_context(size=None, maximized=False, save=False, console=None):
    """Context manager handling CodraFT mainwindow creation and Qt event loop"""
    if size is None:
        size = 950, 450

    with qth.qt_app_context(exec_loop=True):
        try:
            win = CodraFTMainWindow(console=console)
            if maximized:
                win.showMaximized()
            else:
                width, height = size
                win.resize(width, height)
                win.showNormal()
            win.show()
            win.setObjectName(tests.get_default_test_name())  # screenshot name
            yield win
        finally:
            if save:
                try:
                    win.save_to_h5_file(tests.get_output_data_path("h5"))
                except PermissionError:
                    pass


def take_plotwidget_screenshot(panel, name):
    """Eventually takes plotwidget screenshot (only in screenshot mode)"""
    if env.execenv.screenshot:
        qth.grab_save_window(panel.itmlist.plotwidget, f"{panel.PREFIX}_{name}")


def run():
    """Run CodraFT test launcher"""
    run_testlauncher(codraft)


if __name__ == "__main__":
    run()
