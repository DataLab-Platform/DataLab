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
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import psutil
import pytest
from guidata.guitest import run_testlauncher
from sigima.tests import helpers

import datalab.config  # Loading icons
from datalab.config import MOD_NAME, SHOTPATH, Conf, get_typed_config_filename
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


@dataclass
class _BackgroundTestState:
    """Mutable state shared by background DataLab test helpers."""

    process: subprocess.Popen | None = None
    typed_config_path: str | None = None
    typed_config_bytes: bytes | None = None
    typed_config_exists: bool = False
    parent_option_values: dict[str, object] | None = field(default=None)
    parent_xmlrpc_port: str | None = None


_BACKGROUND_STATE = _BackgroundTestState()


def _set_gui_test_options() -> dict[str, object]:
    """Enable GUI features required by tests without writing user config."""
    previous = {
        "plugins_enabled": Conf.plugins_enabled.get(sync_env=False),
        "plugins_enabled_list": Conf.plugins_enabled_list.get(None, sync_env=False),
    }
    Conf.plugins_enabled.set(True, sync_env=False)
    Conf.plugins_enabled_list.set(None, sync_env=False)
    return previous


def _restore_gui_test_options(previous: dict[str, object]) -> None:
    """Restore GUI options changed temporarily for a test context."""
    Conf.plugins_enabled.set(previous["plugins_enabled"], sync_env=False)
    Conf.plugins_enabled_list.set(previous["plugins_enabled_list"], sync_env=False)


def _wait_for_background_process(proc: subprocess.Popen | None) -> None:
    """Wait for a test DataLab process and terminate it if it lingers."""
    if proc is None:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _restore_background_typed_config() -> None:
    """Restore the typed config changed temporarily by a background test."""
    if _BACKGROUND_STATE.typed_config_path is None:
        return
    if _BACKGROUND_STATE.typed_config_exists:
        with open(_BACKGROUND_STATE.typed_config_path, "wb") as stream:
            stream.write(_BACKGROUND_STATE.typed_config_bytes or b"")
    elif osp.isfile(_BACKGROUND_STATE.typed_config_path):
        os.remove(_BACKGROUND_STATE.typed_config_path)
    _BACKGROUND_STATE.typed_config_bytes = None
    _BACKGROUND_STATE.typed_config_exists = False


def _prepare_background_typed_config() -> None:
    """Enable services in the shared typed file for a child test process."""
    # Lazy imports avoid importing the test helpers during DataLab config startup.
    # pylint: disable=import-outside-toplevel
    from sigimax.utils import conf as conf_module

    from datalab.config.config_persistence import (
        remove_persisted_option,
        save_options_to_ini,
    )

    _BACKGROUND_STATE.parent_option_values = {
        "rpc_server_enabled": Conf.rpc_server_enabled.get(sync_env=False),
        "plugins_enabled": Conf.plugins_enabled.get(sync_env=False),
        "plugins_enabled_list": Conf.plugins_enabled_list.get(None, sync_env=False),
    }
    persist_enabled = Conf._ini_persist_enabled  # pylint: disable=protected-access
    Conf.set_ini_persist_enabled(False)
    try:
        Conf.rpc_server_enabled.set(True, sync_env=False)
        Conf.plugins_enabled.set(True, sync_env=False)
        Conf.plugins_enabled_list.set(None, sync_env=False)
        save_options_to_ini(Conf, conf_module.CONF)
        remove_persisted_option(Conf, "rpc_server_port", conf_module.CONF)
    finally:
        Conf.set_ini_persist_enabled(persist_enabled)


def _restore_background_parent_options() -> None:
    """Restore parent option values after a background test."""
    if _BACKGROUND_STATE.parent_option_values is None:
        return
    Conf.rpc_server_enabled.set(
        _BACKGROUND_STATE.parent_option_values["rpc_server_enabled"], sync_env=False
    )
    Conf.plugins_enabled.set(
        _BACKGROUND_STATE.parent_option_values["plugins_enabled"], sync_env=False
    )
    Conf.plugins_enabled_list.set(
        _BACKGROUND_STATE.parent_option_values["plugins_enabled_list"], sync_env=False
    )
    _BACKGROUND_STATE.parent_option_values = None


def _restore_background_port_environment() -> None:
    """Restore the XML-RPC port environment variable after a test."""
    if _BACKGROUND_STATE.parent_xmlrpc_port is None:
        os.environ.pop(execenv.XMLRPCPORT_ENV, None)
    else:
        os.environ[execenv.XMLRPCPORT_ENV] = _BACKGROUND_STATE.parent_xmlrpc_port
    _BACKGROUND_STATE.parent_xmlrpc_port = None


def cleanup_datalab_background() -> None:
    """Clean up any background DataLab state left by a test."""
    proc = _BACKGROUND_STATE.process
    try:
        if proc is not None and is_pid_alive(proc.pid):
            proxy = RemoteProxy(autoconnect=False)
            proxy.connect(timeout=5.0)
            proxy.close_application()
            proxy.disconnect()
    except (ConnectionRefusedError, OSError):
        pass
    finally:
        _wait_for_background_process(proc)
        _restore_background_typed_config()
        _restore_background_parent_options()
        _restore_background_port_environment()
        _BACKGROUND_STATE.process = None


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
        previous_options = _set_gui_test_options()
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
            _restore_gui_test_options(previous_options)


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive

    Args:
        Process ID to check

    Returns:
        True if the process is alive, False otherwise
    """
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()


def run_datalab_in_background(wait_until_ready: bool = True) -> subprocess.Popen:
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
    env["DATALAB_CONFIG_BACKEND"] = "typed"
    env.pop("DATALAB_OPTIONS_JSON", None)
    if execenv.XMLRPCPORT_ENV in env:
        # May happen when executing other tests before
        env.pop(execenv.XMLRPCPORT_ENV)

    _BACKGROUND_STATE.typed_config_path = get_typed_config_filename()
    _BACKGROUND_STATE.typed_config_exists = osp.isfile(
        _BACKGROUND_STATE.typed_config_path
    )
    if _BACKGROUND_STATE.typed_config_exists:
        with open(_BACKGROUND_STATE.typed_config_path, "rb") as stream:
            _BACKGROUND_STATE.typed_config_bytes = stream.read()
    _prepare_background_typed_config()

    proc = helpers.exec_script(
        "-m", args=["datalab.app"], wait=False, env=env, verbose=False
    )
    _BACKGROUND_STATE.process = proc
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
            _BACKGROUND_STATE.parent_xmlrpc_port = os.environ.get(
                execenv.XMLRPCPORT_ENV
            )
            os.environ[execenv.XMLRPCPORT_ENV] = str(proxy.port)
            proxy.disconnect()
        except ConnectionRefusedError as exc:
            if is_pid_alive(proc.pid):
                proc.kill()
            _wait_for_background_process(proc)
            _restore_background_typed_config()
            _restore_background_parent_options()
            _restore_background_port_environment()
            _BACKGROUND_STATE.process = None
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
    try:
        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=5.0)  # 5 seconds max to connect
        proxy.close_application()
        proxy.disconnect()
    finally:
        _wait_for_background_process(_BACKGROUND_STATE.process)
        _restore_background_typed_config()
        _restore_background_parent_options()
        _restore_background_port_environment()
        _BACKGROUND_STATE.process = None


@contextmanager
def datalab_in_background_context() -> Generator[RemoteProxy, None, None]:
    """Context manager for DataLab instance with proxy connection"""
    run_datalab_in_background()
    try:
        with proxy_context("remote") as proxy:
            yield proxy
    finally:
        try:
            close_datalab_background()
        except ConnectionRefusedError:
            _wait_for_background_process(_BACKGROUND_STATE.process)
            _restore_background_typed_config()
            _restore_background_parent_options()
            _restore_background_port_environment()


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
