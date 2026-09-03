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

import multiprocessing
import os
import os.path as osp
import socket
import sys
import time
from contextlib import contextmanager
from typing import Generator

import psutil
import pytest
from guidata.guitest import run_testlauncher
from sigima.tests import helpers

import datalab.config  # Loading icons
from datalab.config import MOD_NAME, SHOTPATH, ensure_initialized
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

_BACKGROUND_PROCESS: multiprocessing.Process | None = None


@contextmanager
def datalab_test_app_context(
    size: tuple[int, int] = None,
    maximized: bool = False,
    save: bool = False,
    console: bool | None = None,
    exec_loop: bool = True,
    history: bool = False,
) -> Generator[DLMainWindow, None, None]:
    """Context manager handling DataLab mainwindow creation and Qt event loop
    with optional HDF5 file save and other options for testing purposes

    Args:
        size: mainwindow size (default: (950, 600))
        maximized: whether to maximize mainwindow (default: False)
        save: whether to save HDF5 file (default: False)
        console: whether to show console (default: None)
        exec_loop: whether to execute Qt event loop (default: True)
        history: whether to enable and show history tracking (default: False)
    """
    if size is None:
        size = 1200, 700
    ensure_initialized(load_user_config=False)
    with qth.datalab_app_context(exec_loop=exec_loop):
        win: DLMainWindow | None = None
        try:
            win = DLMainWindow(console=console)
            if not history:
                win.historypanel.set_tracking_enabled(False)
                win.historypanel.setEnabled(False)
                win.docks[win.historypanel].hide()
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    global _BACKGROUND_PROCESS  # pylint: disable=global-statement
    if _BACKGROUND_PROCESS is not None:
        if _BACKGROUND_PROCESS.is_alive():
            raise RuntimeError("A background DataLab process is already running")
        _BACKGROUND_PROCESS.close()
        _BACKGROUND_PROCESS = None

    execenv.xmlrpcport = port
    from datalab import app  # pylint: disable=import-outside-toplevel

    proc = multiprocessing.Process(
        target=app.run,
        kwargs={
            "load_user_config": False,
            "option_overrides": {
                "rpc_server_enabled": True,
                "tour_enabled": False,
            },
            "xmlrpc_port": port,
        },
    )
    previous_do_not_quit = os.environ.get(execenv.DO_NOT_QUIT_ENV)
    os.environ[execenv.DO_NOT_QUIT_ENV] = "1"
    try:
        proc.start()
    finally:
        if previous_do_not_quit is None:
            os.environ.pop(execenv.DO_NOT_QUIT_ENV, None)
        else:
            os.environ[execenv.DO_NOT_QUIT_ENV] = previous_do_not_quit
    _BACKGROUND_PROCESS = proc
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
    if not proc.is_alive():
        raise RuntimeError("DataLab process terminated immediately after start")

    if wait_until_ready:
        # Use RemoteProxy's built-in retry mechanism with extended timeout
        # DataLab startup: Python imports, Qt init, GUI creation, XML-RPC server
        try:
            proxy = RemoteProxy(autoconnect=False)
            proxy.connect(port=str(port), timeout=30.0)
            proxy.disconnect()
        except ConnectionRefusedError as exc:
            close_datalab_background(request_close=False)
            raise RuntimeError(
                "Failed to connect to DataLab application. "
                "Process may have started but XML-RPC server is not responding."
            ) from exc


def close_datalab_background(request_close: bool = True) -> None:
    """Close DataLab application running as a service.

    This function connects to the DataLab application running in the background
    (started with `run_datalab_in_background`) and sends a command to close it.
    It uses the `RemoteProxy` class to establish the connection and send the
    close command.

    Args:
        request_close: If True, first request a graceful shutdown through XML-RPC.
    """
    global _BACKGROUND_PROCESS  # pylint: disable=global-statement
    process = _BACKGROUND_PROCESS
    if process is None:
        execenv.xmlrpcport = None
        return
    try:
        if request_close and process.is_alive():
            try:
                proxy = RemoteProxy(autoconnect=False)
                proxy.connect(timeout=5.0)
                proxy.close_application()
                proxy.disconnect()
            except (ConnectionRefusedError, OSError):
                pass
        process.join(10.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
    finally:
        if not process.is_alive():
            process.close()
        _BACKGROUND_PROCESS = None
        execenv.xmlrpcport = None


@contextmanager
def datalab_in_background_context() -> Generator[RemoteProxy, None, None]:
    """Context manager for DataLab instance with proxy connection"""
    run_datalab_in_background()
    try:
        with proxy_context("remote") as proxy:
            yield proxy
    finally:
        close_datalab_background()


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
