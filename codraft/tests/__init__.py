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
from codraft.core.gui.main import CodraFTMainWindow
from codraft.utils.qthelpers import QtTestEnv, grab_save_window, qt_app_context
from codraft.utils.tests import get_default_test_name, get_output_data_path

# TODO: [P2] Documentation: add more screenshots from tests
# TODO: [P3] Create subpackages "app" & "unit" + add support for subpackages in
# test launcher


@contextmanager
def codraft_app_context(size=None, maximized=False, save=False, console=None):
    """Context manager handling CodraFT mainwindow creation and Qt event loop"""
    if size is None:
        size = 950, 450

    with qt_app_context(exec_loop=True):
        try:
            win = CodraFTMainWindow(console=console)
            if maximized:
                win.showMaximized()
            else:
                width, height = size
                win.resize(width, height)
                win.showNormal()
            win.show()
            win.setObjectName(get_default_test_name())  # screenshot name
            yield win
        finally:
            if save:
                win.save_to_h5_file(get_output_data_path("h5"))


def take_plotwidget_screenshot(panel, name):
    """Eventually takes plotwidget screenshot (only in screenshot mode)"""
    if QtTestEnv().screenshot:
        grab_save_window(panel.itmlist.plotwidget, f"{panel.PREFIX}_{name}")


def run():
    """Run CodraFT test launcher"""
    run_testlauncher(codraft)


if __name__ == "__main__":
    run()
