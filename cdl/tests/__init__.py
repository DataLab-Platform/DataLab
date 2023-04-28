# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
DataLab unit tests
"""

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

from contextlib import contextmanager
from typing import Generator

from guidata.guitest import run_testlauncher

import cdl.config  # Loading icons
from cdl import env
from cdl.core.gui.main import CDLMainWindow
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.utils import qthelpers as qth
from cdl.utils import tests

# TODO: [P2] Documentation: add more screenshots from tests
# TODO: [P3] Create subpackages "app" & "unit" + add support for subpackages in
# test launcher


@contextmanager
def cdl_app_context(
    size=None, maximized=False, save=False, console=None
) -> Generator[CDLMainWindow, None, None]:
    """Context manager handling DataLab mainwindow creation and Qt event loop"""
    if size is None:
        size = 950, 600

    with qth.qt_app_context(exec_loop=True):
        try:
            win = CDLMainWindow(console=console)
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


def take_plotwidget_screenshot(panel: SignalPanel | ImagePanel, name: str):
    """Eventually takes plotwidget screenshot (only in screenshot mode)"""
    if env.execenv.screenshot:
        prefix = panel.PARAMCLASS.PREFIX
        qth.grab_save_window(panel.plothandler.plotwidget, f"{prefix}_{name}")


def run():
    """Run DataLab test launcher"""
    run_testlauncher(cdl)


if __name__ == "__main__":
    run()
