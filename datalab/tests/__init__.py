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
import sys
import time
from contextlib import contextmanager
from typing import Generator

import psutil
import pytest
from guidata.guitest import run_testlauncher
from sigima.tests import helpers

import datalab.config  # Loading icons
from datalab.config import MOD_NAME, SHOTPATH
from datalab.control.proxy import RemoteProxy, proxy_context
from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalPanel
from datalab.utils import qthelpers as qth

# Add test data files and folders pointed by `DATALAB_DATA` environment variable:
helpers.add_test_path_from_env("DATALAB_DATA")

# Add test data files and folders for the DataLab module:
helpers.add_test_module_path(MOD_NAME, osp.join("data", "tests"))

# Set default screenshot path for tests
execenv.screenshot_path = SHOTPATH


@contextmanager
def datalab_test_app_context(
    size: tuple[int, int] = None,
    maximized: bool = False,
    save: bool = False,
    console: bool | None = None,
    exec_loop: bool = True,
) -> Generator[DLMainWindow, None, None]:
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
        size = 1200, 700
    with qth.datalab_app_context(exec_loop=exec_loop):
        win: DLMainWindow | None = None
        try:
            win = DLMainWindow(console=console)
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
            has_exception_occurred = sys.exc_info()[0] is not None
            if not exec_loop or has_exception_occurred and win is not None:
                # Closing main window properly
                win.set_modified(False)
                win.close()


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive

    Args:
        Process ID to check

    Returns:
        True if the process is alive, False otherwise
    """
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()


def run_datalab_in_background(wait_until_ready: bool = True) -> None:
    """Run DataLab application as a service.

    This function starts the DataLab application in a separate process, ensuring that
    it runs independently of the current script. It sets the necessary environment
    variables to prevent the application from quitting automatically (since the script
    is executed in a non-interactive mode - the so-called "unattended" mode) and to
    avoid port conflicts. After starting the application, it waits for a short period
    to allow the application to initialize and then checks if the process is alive.

    The main use case for this function is in testing scenarios where the DataLab
    application needs to be running in the background while a client connects to it
    and performs various operations.

    Args:
        wait_until_ready: If True, waits until the DataLab application is ready to
         accept connections (default: True). Uses RemoteProxy's built-in retry logic
         with extended timeout to handle DataLab startup time.

    Raises:
        RuntimeError: If the DataLab application fails to start
    """
    env = os.environ.copy()
    env[execenv.DO_NOT_QUIT_ENV] = "1"
    if execenv.XMLRPCPORT_ENV in env:
        # May happen when executing other tests before
        env.pop(execenv.XMLRPCPORT_ENV)

    proc = helpers.exec_script(
        "-m", args=["datalab.app"], wait=False, env=env, verbose=False
    )
    # If the process fails to start, it will raise the `AssertionError` exception
    # with the message "Unable to start DataLab application".
    # In that case, it might be useful to set `wait=True` and `verbose=True` in the
    # `exec_script` call above, so that the script waits for the DataLab application
    # to start and prints the output to the console. This way, you can see any
    # error messages or logs that might help you understand why the application failed
    # to start.
    # If the script is executed within a pytest session, add the `-s` option to pytest.

    # Give the process a moment to actually start
    time.sleep(1)
    if not is_pid_alive(proc.pid):
        raise RuntimeError("DataLab process terminated immediately after start")

    if wait_until_ready:
        # Use RemoteProxy's built-in retry mechanism with extended timeout
        # DataLab startup: Python imports, Qt init, GUI creation, XML-RPC server
        try:
            proxy = RemoteProxy(autoconnect=False)
            proxy.connect(timeout=30.0)  # 30 seconds max for DataLab to be ready
            proxy.disconnect()
        except ConnectionRefusedError as exc:
            if is_pid_alive(proc.pid):
                proc.kill()
            raise RuntimeError(
                "Failed to connect to DataLab application. "
                "Process may have started but XML-RPC server is not responding."
            ) from exc


def close_datalab_background() -> None:
    """Close DataLab application running as a service.

    This function connects to the DataLab application running in the background
    (started with `run_datalab_in_background`) and sends a command to close it.
    It uses the `RemoteProxy` class to establish the connection and send the
    close command.

    Raises:
        ConnectionRefusedError: If unable to connect to the DataLab application.
    """
    proxy = RemoteProxy(autoconnect=False)
    proxy.connect(timeout=5.0)  # 5 seconds max to connect
    proxy.close_application()
    proxy.disconnect()


@contextmanager
def datalab_in_background_context() -> Generator[RemoteProxy, None, None]:
    """Context manager for DataLab instance with proxy connection"""
    run_datalab_in_background()
    with proxy_context("remote") as proxy:
        try:
            yield proxy
        except Exception as exc:  # pylint: disable=broad-except
            proxy.close_application()
            raise exc
        # Cleanup
        proxy.close_application()


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
