# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Temporary pytest plugin diagnosing Qt event loops that never quit (CI only).

Loaded with ``-p qt_hang_probe`` by the PyQt6 workflow of a diagnostic branch;
it is not part of the test suite. It records every unattended
``close_widgets_and_quit`` call (event loop level, top-level widgets,
exceptions, widgets still visible afterwards, which make Qt 6 ignore
``QApplication.quit()``), and stops the run with a stack dump when an event
loop outlives its close request or when a test blocks.
"""

from __future__ import annotations

import faulthandler
import gc
import multiprocessing
import os
import sys
import threading
import time
import traceback

import pytest

LOG_PATH = os.path.abspath(os.environ.get("QT_HANG_PROBE_LOG", "qt-hang-probe.log"))
#: Seconds after a close request before reporting a still running event loop
CLOSE_GRACE = 30
#: Seconds after a close request before stopping the run
CLOSE_LIMIT = 300
#: Seconds before reporting an event loop running without any close request
LOOP_LIMIT = 240
#: Seconds before stopping a blocked test (longest normal test: about 110 s)
TEST_LIMIT = 600
MODE_VARIABLES = ("GUIDATA_UNATTENDED", "GUIDATA_DELAY", "SIGIMAX_DO_NOT_QUIT")

_log = None
_lock = threading.Lock()
_test = "<session>"
_closes = 0


def _write(message: str) -> None:
    with _lock:
        _log.write(f"{time.strftime('%H:%M:%S')} [{_test}] {message}\n")
        _log.flush()


def _stop(reason: str, code: int) -> None:
    _write(f"STOP: {reason}")
    with _lock:
        faulthandler.dump_traceback(file=_log, all_threads=True)
        _log.flush()
    os._exit(code)  # pylint: disable=protected-access


def _qt():
    from qtpy import QtCore, QtWidgets  # pylint: disable=import-outside-toplevel

    return QtCore, QtWidgets


def _loop_level() -> int:
    return _qt()[0].QThread.currentThread().loopLevel()


def _blocks_quit(widget) -> bool:
    """Return True if ``QApplication`` (Qt 6) ignores a quit while it is shown."""
    QtCore = _qt()[0]
    kind = widget.windowType()
    return (
        widget.isVisible()
        and kind not in (QtCore.Qt.Desktop, QtCore.Qt.Popup)
        and (kind != QtCore.Qt.Dialog or widget.parentWidget() is None)
        and not widget.testAttribute(QtCore.Qt.WA_DontShowOnScreen)
    )


def _describe(widget) -> str:
    try:
        return (
            f"{type(widget).__module__}.{type(widget).__qualname__}"
            f" name={widget.objectName()!r} visible={widget.isVisible()}"
            f" type={widget.windowType()!r} parent={widget.parentWidget() is not None}"
            f" blocks_quit={_blocks_quit(widget)}"
        )
    except RuntimeError as exc:
        return f"{type(widget).__qualname__} <{exc}>"


def _snapshot(prefix: str) -> None:
    for widget in _qt()[1].QApplication.topLevelWidgets():
        _write(f"  {prefix}: {_describe(widget)}")


def _modes() -> str:
    return " ".join(f"{name}={os.environ.get(name)!r}" for name in MODE_VARIABLES)


def _check_close(count: int, test: str, elapsed: int) -> None:
    if count != _closes or test != _test or _loop_level() == 0:
        return  # The loop quit: this timer fires later, elsewhere
    _write(f"EVENT LOOP STILL RUNNING {elapsed} s after close #{count} ({_modes()})")
    _snapshot("now")
    if elapsed >= CLOSE_LIMIT:
        _stop("the event loop did not quit after its close request", 86)
    _qt()[0].QTimer.singleShot(60_000, lambda: _check_close(count, test, elapsed + 60))


def _check_loop(test: str) -> None:
    if test != _test or _loop_level() == 0:
        return
    _write(
        f"EVENT LOOP RUNNING {LOOP_LIMIT} s+ (close requests: {_closes}, {_modes()})"
    )
    _snapshot("now")


def _wrap(module) -> None:
    original = module.close_widgets_and_quit

    def close_widgets_and_quit(screenshot: bool = False) -> None:
        global _closes  # pylint: disable=global-statement
        _closes += 1
        count, test, level = _closes, _test, _loop_level()
        outside = " (outside any event loop)" if level == 0 else ""
        _write(
            f"close #{count} by {module.__name__}: loop level {level}{outside}"
            f" ({_modes()})"
        )
        _snapshot("before")
        try:
            original(screenshot=screenshot)
        except BaseException:
            _write("  EXCEPTION, quit() not called:\n" + traceback.format_exc())
            raise
        finally:
            try:
                for widget in _qt()[1].QApplication.topLevelWidgets():
                    if _blocks_quit(widget):
                        _write(f"  STILL VISIBLE, quit ignored: {_describe(widget)}")
            except RuntimeError:
                _write("  after: " + traceback.format_exc())
            if level > 0:
                _qt()[0].QTimer.singleShot(
                    CLOSE_GRACE * 1000,
                    lambda: _check_close(count, test, CLOSE_GRACE),
                )

    module.close_widgets_and_quit = close_widgets_and_quit


def pytest_configure(config) -> None:  # pylint: disable=unused-argument
    """Open the probe log."""
    global _log  # pylint: disable=global-statement
    _log = open(LOG_PATH, "w", encoding="utf-8")  # pylint: disable=consider-using-with


def pytest_sessionstart(session) -> None:  # pylint: disable=unused-argument
    """Wrap the unattended close helpers once the test environment is set up."""
    import guidata.qthelpers  # pylint: disable=import-outside-toplevel
    import sigimax.utils.qthelpers  # pylint: disable=import-outside-toplevel

    QtCore = _qt()[0]
    _write(
        f"Python {sys.version.split()[0]}, Qt {QtCore.qVersion()}, "
        f"start method {multiprocessing.get_start_method(allow_none=True)}, "
        f"GC thresholds {gc.get_threshold()}, {_modes()}"
    )
    for module in (guidata.qthelpers, sigimax.utils.qthelpers):
        _wrap(module)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):  # pylint: disable=unused-argument
    """Watch each test: a blocked test stops the run with a stack dump."""
    global _test  # pylint: disable=global-statement
    _test = item.nodeid
    stopper = threading.Timer(
        TEST_LIMIT, _stop, args=(f"test still running after {TEST_LIMIT} s", 87)
    )
    stopper.daemon = True
    stopper.start()
    if _qt()[1].QApplication.instance() is not None:
        test = item.nodeid
        _qt()[0].QTimer.singleShot(LOOP_LIMIT * 1000, lambda: _check_loop(test))
    try:
        return (yield)
    finally:
        stopper.cancel()
        _test = "<between tests>"


def pytest_unconfigure(config) -> None:  # pylint: disable=unused-argument
    """Close the probe log."""
    if _log is not None:
        _write(f"session end: {_closes} close requests")
        _log.close()
