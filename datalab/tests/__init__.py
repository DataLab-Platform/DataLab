# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Tests (:mod:`datalab.tests`)
------------------------

The DataLab test suite is based on the `pytest <https://pytest.org>`_ framework.

The test suite modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`datalab.tests.backbone`: backbone tests
- :mod:`datalab.tests.features`: feature tests (unit tests and application tests)
- :mod:`datalab.tests.scenarios`: high-level scenarios tests

.. seealso::

    :ref:`validation` for more information about DataLab's testing strategy.
"""

from __future__ import annotations

import os
import os.path as osp
from contextlib import contextmanager
from typing import Generator

import pytest
from guidata.guitest import run_testlauncher
from sigima.tests import helpers

import datalab.config  # Loading icons
from datalab.config import MOD_NAME
from datalab.env import execenv
from datalab.gui.main import CDLMainWindow
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalPanel
from datalab.utils import qthelpers as qth

# Add test data files and folders pointed by `CDL_DATA` environment variable:
helpers.add_test_path_from_env("CDL_DATA")

# Add test data files and folders for the DataLab module:
helpers.add_test_module_path(MOD_NAME, osp.join("data", "tests"))


@contextmanager
def cdltest_app_context(
    size: tuple[int, int] = None,
    maximized: bool = False,
    save: bool = False,
    console: bool | None = None,
    exec_loop: bool = True,
) -> Generator[CDLMainWindow, None, None]:
    """Context manager handling DataLab mainwindow creation and Qt event loop
    with optional HDF5 file save and other options for testing purposes

    Args:
        size: mainwindow size (default: (950, 600))
        maximized: whether to maximize mainwindow (default: False)
        save: whether to save HDF5 file (default: False)
        console: whether to show console (default: None)
        exec_loop: whether to execute Qt event loop (default: True)
    """
    if size is None:
        size = 950, 600
    with qth.datalab_app_context(exec_loop=exec_loop):
        try:
            win = CDLMainWindow(console=console)
            if maximized:
                win.showMaximized()
            else:
                width, height = size
                win.resize(width, height)
                win.showNormal()
            win.show()
            win.setObjectName(helpers.get_default_test_name())  # screenshot name
            yield win
        finally:
            if save:
                path = helpers.get_output_data_path("h5")
                try:
                    os.remove(path)
                    win.save_to_h5_file(path)
                except (FileNotFoundError, PermissionError):
                    pass
            if not exec_loop:
                # Closing main window properly
                win.set_modified(False)
                win.close()


@contextmanager
def skip_if_opencv_missing() -> Generator[None, None, None]:
    """Skip test if OpenCV is not available"""
    try:
        yield
    except ImportError as exc:
        if "cv2" in str(exc).lower():
            pytest.skip("OpenCV not available, skipping test")
        raise exc


def take_plotwidget_screenshot(panel: SignalPanel | ImagePanel, name: str) -> None:
    """Eventually takes plotwidget screenshot (only in screenshot mode)"""
    if execenv.screenshot:
        prefix = panel.PARAMCLASS.PREFIX
        qth.grab_save_window(panel.plothandler.plotwidget, f"{prefix}_{name}")


def run() -> None:
    """Run DataLab test launcher"""
    run_testlauncher(datalab)


if __name__ == "__main__":
    run()
