# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Qt utilities
"""

from __future__ import annotations

import faulthandler
import functools
import inspect
import logging
import os
import os.path as osp
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import guidata
from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.io.common.converters import to_string

from datalab.config import (
    APP_NAME,
    DATETIME_FORMAT,
    SHOTPATH,
    Conf,
    _,
    get_old_log_fname,
)
from datalab.env import execenv


def close_widgets_and_quit(screenshot=False) -> None:
    """Close Qt top level widgets and quit Qt event loop"""
    for widget in QW.QApplication.instance().topLevelWidgets():
        try:
            wname = widget.objectName()
        except RuntimeError:
            # Object has been deleted
            continue
        if screenshot and wname and widget.isVisible():  # pragma: no cover
            grab_save_window(widget, wname.lower())
        assert widget.close()
    QW.QApplication.instance().quit()


QAPP_INSTANCE = None


def get_log_contents(fname: str) -> str | None:
    """Return True if file exists and something was logged in it"""
    if osp.exists(fname):
        with open(fname, "rb") as fdesc:
            return to_string(fdesc.read()).strip()
    return None


def initialize_log_file(fname: str) -> bool:
    """Eventually keep the previous log file
    Returns True if there was a previous log file"""
    contents = get_log_contents(fname)
    if contents:
        try:
            shutil.move(fname, get_old_log_fname(fname))
        except Exception:  # pylint: disable=broad-except
            pass
        return True
    return False


def remove_empty_log_file(fname: str) -> None:
    """Eventually remove empty log files"""
    if not get_log_contents(fname):
        try:
            os.remove(fname)
        except Exception:  # pylint: disable=broad-except
            pass


@contextmanager
def datalab_app_context(
    exec_loop=False, enable_logs=True
) -> Generator[QW.QApplication, None, None]:
    """DataLab Qt application context manager, handling Qt application creation
    and persistance, faulthandler/traceback logging features, screenshot mode
    and unattended mode.

    Args:
        exec_loop: whether to execute Qt event loop (default: False)
        enable_logs: whether to enable logs (default: True)
    """
    global QAPP_INSTANCE  # pylint: disable=global-statement
    if QAPP_INSTANCE is None:
        QAPP_INSTANCE = guidata.qapplication()

    # === Set application name and version ---------------------------------------------
    # pylint: disable=import-outside-toplevel
    import datalab

    QAPP_INSTANCE.setApplicationName(APP_NAME)
    QAPP_INSTANCE.setApplicationVersion(datalab.__version__)
    QAPP_INSTANCE.setOrganizationName(APP_NAME + " project")

    if enable_logs:
        # === Create a logger for standard exceptions ----------------------------------
        tb_log_fname = Conf.main.traceback_log_path.get()
        Conf.main.traceback_log_available.set(initialize_log_file(tb_log_fname))
        logger = logging.getLogger(__name__)
        fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
        logging.basicConfig(
            filename=tb_log_fname,
            filemode="w",
            level=logging.ERROR,
            format=fmt,
            datefmt=DATETIME_FORMAT,
        )

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            "Custom exception hook"
            logger.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
            )
            return sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = custom_excepthook

    # === Use faulthandler for other exceptions ------------------------------------
    fh_log_fname = Conf.main.faulthandler_log_path.get()
    Conf.main.faulthandler_log_available.set(initialize_log_file(fh_log_fname))

    with open(fh_log_fname, "w", encoding="utf-8") as fh_log_fn:
        if enable_logs and Conf.main.faulthandler_enabled.get(True):
            faulthandler.enable(file=fh_log_fn)
        exception_occured = False
        try:
            yield QAPP_INSTANCE
        except Exception:  # pylint: disable=broad-except
            exception_occured = True
        finally:
            if (
                execenv.unattended or execenv.screenshot
            ) and not execenv.do_not_quit:  # pragma: no cover
                if execenv.delay > 0:
                    mode = "Screenshot" if execenv.screenshot else "Unattended"
                    message = f"{mode} mode (delay: {execenv.delay}ms)"
                    msec = execenv.delay - 200
                    for widget in QW.QApplication.instance().topLevelWidgets():
                        if isinstance(widget, QW.QMainWindow):
                            widget.statusBar().showMessage(message, msec)
                QC.QTimer.singleShot(
                    execenv.delay,
                    lambda: close_widgets_and_quit(screenshot=execenv.screenshot),
                )
            if exec_loop and not exception_occured:
                QAPP_INSTANCE.exec()
        if exception_occured:
            raise  # pylint: disable=misplaced-bare-raise

    if enable_logs and Conf.main.faulthandler_enabled.get():
        faulthandler.disable()
    remove_empty_log_file(fh_log_fname)
    if enable_logs:
        logging.shutdown()
        remove_empty_log_file(tb_log_fname)


def is_running_tests() -> bool:
    """Check if code is running during test execution"""
    return "pytest" in sys.modules


@contextmanager
def try_or_log_error(context: str) -> Generator[None, None, None]:
    """Try to execute a function and log an error message if it fails"""
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        if is_running_tests():
            # If we are running tests, we want to raise the exception
            raise
        traceback.print_exc()
        logger = logging.getLogger(__name__)
        logger.error("Error in %s", context, exc_info=traceback.format_exc())
        Conf.main.traceback_log_available.set(True)
    finally:
        pass


@contextmanager
def create_progress_bar(
    parent: QW.QWidget, label: str, max_: int, show_after: int = 1000
) -> Generator[QW.QProgressDialog, None, None]:
    """Create modal progress bar

    Args:
        parent: Parent widget
        label: Progress dialog title
        max_: Maximum progress value
        show_after: Delay before showing the progress dialog (ms, default: 1000)
    """
    prog = QW.QProgressDialog(label, _("Cancel"), 0, max_, parent, QC.Qt.SplashScreen)
    prog.setWindowModality(QC.Qt.WindowModal)
    prog.setMinimumDuration(show_after)
    try:
        yield prog
    finally:
        prog.close()
        prog.deleteLater()


class CallbackWorker(QC.QThread):
    """Worker for executing long operations in a separate thread (this must not be
    confused with the :py:class:`datalab.gui.processor.base.Worker` class, which
    handles the execution of computations in a another process)

    Implements `CallbackWorkerProtocol` from `sigima.worker`, used for computations
    that support cancellation and progress reporting.

    Args:
        callback: The function to be executed in a separate thread, that takes
         optionnally 'worker' as argument (instance of this class), and any other
         argument passed with **kwargs
        kwargs: Callback keyword arguments
    """

    SIG_PROGRESS_UPDATE = QC.Signal(int)

    def __init__(self, callback: Callable, **kwargs) -> None:
        super().__init__()
        self.callback = callback
        if "worker" in inspect.signature(callback).parameters:
            kwargs["worker"] = self
        self.kwargs = kwargs
        self.result: Any | None = None
        self.__canceled = False
        self.__exc = None

    def run(self) -> None:
        """Start thread"""
        # Initialize progress bar: setting progress to 0.0 has the effect of
        # showing the progress dialog after the `minimumDuration` time has elapsed.
        # If we don't set the progress to 0.0, the progress dialog will be shown only
        # after the first call to `set_progress` method even if the `minimumDuration`
        # time has elapsed.
        self.set_progress(0.0)

        try:
            self.result = self.callback(**self.kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            self.__exc = exc

    def cancel(self) -> None:
        """Progress bar was canceled"""
        self.__canceled = True

    def was_canceled(self) -> bool:
        """Return whether the progress dialog was canceled by user"""
        return self.__canceled

    def set_progress(self, value: float) -> None:
        """Set progress bar value

        Args:
            value: float between 0.0 and 1.0
        """
        self.SIG_PROGRESS_UPDATE.emit(int(100 * value))

    def get_result(self) -> Any:
        """Return callback result"""
        if self.__exc is not None:
            raise self.__exc
        return self.result


def qt_long_callback(
    parent: QW.QWidget,
    label: str,
    worker: CallbackWorker,
    progress: bool,
    show_after: int = 500,
) -> Any:
    """Handle long callbacks: run in a separate thread while showing a busy bar

    Args:
        parent: Parent widget
        label: Progress dialog title
        worker: Callback worker handling the function execution in a separate thread
        progress: Whether the progress feature is handled or not. If True, a progress
         bar and a 'Cancel' button are shown on the progress dialog. The progress value
         is updated by the `worker.set_progress` method (which takes a float between
         0.0 and 1.0). Moreover, if `progress` is True, we wait for the callback
         function to return (it means that the callback function must implement a
         mechanism to return an intermediate result or `None` if the
         `worker.was_canceled` method returns True).
        show_after: Delay before showing the progress dialog (ms, default: 1000)

    Returns:
        Callback result
    """
    if progress:
        prog = QW.QProgressDialog(
            label, _("Cancel"), 0, 100, parent, QC.Qt.SplashScreen
        )
        prog.setMinimumDuration(show_after)
        worker.SIG_PROGRESS_UPDATE.connect(prog.setValue)
        prog.canceled.connect(worker.cancel)
    else:
        prog = QW.QProgressDialog(label, None, 0, 0, parent, QC.Qt.SplashScreen)
        prog.setMinimumDuration(0)
        prog.setCancelButton(None)
        prog.setRange(0, 0)
        prog.show()
    prog.setWindowModality(QC.Qt.WindowModal)

    worker.start()
    while worker.isRunning() and not worker.was_canceled():
        QW.QApplication.processEvents()
        time.sleep(0.005)
    if progress:
        worker.SIG_PROGRESS_UPDATE.disconnect(prog.setValue)
        worker.wait()
    try:
        result = worker.get_result()
    except Exception as exc:  # pylint: disable=broad-except
        prog.close()
        prog.deleteLater()
        raise exc
    prog.close()
    prog.deleteLater()
    return result


def qt_handle_error_message(widget: QW.QWidget, message: str, context: str = None):
    """Handles application (QWidget) error message"""
    traceback.print_exc()
    txt = str(message)
    msglines = txt.splitlines()
    firstline = _("Error:") if context is None else f"%s: {context}" % _("Context")
    msglines.insert(0, firstline)
    if len(msglines) > 10:
        msglines = msglines[:10] + ["..."]
    title = widget.window().objectName()
    QW.QMessageBox.critical(widget, title, os.linesep.join(msglines))


def qt_try_except(message=None, context=None):
    """Try...except Qt widget method decorator"""

    def qt_try_except_decorator(func):
        """Try...except Qt widget method decorator"""

        @functools.wraps(func)
        def method_wrapper(*args, **kwargs):
            """Decorator wrapper function"""
            self = args[0]  # extracting 'self' from method arguments
            #  If "self" is a BaseProcessor, then we need to get the panel instance
            panel = getattr(self, "panel", self)
            if message is not None:
                panel.SIG_STATUS_MESSAGE.emit(message)
                QW.QApplication.setOverrideCursor(QG.QCursor(QC.Qt.WaitCursor))
                panel.repaint()
            output = None
            try:
                output = func(*args, **kwargs)
            except Exception as msg:  # pylint: disable=broad-except
                if is_running_tests():
                    # If we are running tests, we want to raise the exception
                    raise
                qt_handle_error_message(panel.parent(), msg, context)
            finally:
                panel.SIG_STATUS_MESSAGE.emit("")
                QW.QApplication.restoreOverrideCursor()
            return output

        return method_wrapper

    return qt_try_except_decorator


@contextmanager
def qt_try_loadsave_file(
    parent: QW.QWidget, filename: str, operation: str
) -> Generator[str, None, None]:
    """Try and open file (operation: "load" or "save")"""
    if operation not in ("load", "save"):
        raise ValueError("operation argument must be 'load' or 'save'")
    try:
        yield filename
    except Exception as msg:  # pylint: disable=broad-except
        if is_running_tests():
            # If we are running tests, we want to raise the exception
            raise
        traceback.print_exc()
        url = osp.dirname(filename).replace("\\", "/")
        if operation == "load":
            text = _("The file %s could not be read:")
        else:
            text = _("The file %s could not be written:")
        in_folder = _("in this folder")
        message = text % (
            f"<span style='font-weight:bold;color:#555555;'>{osp.basename(filename)}"
            f"</span> (<a href='file:///{url}'>{in_folder}</a>)"
        )
        QW.QMessageBox.critical(parent, APP_NAME, f"{message}<br><br>{str(msg)}")
    finally:
        pass


def grab_save_window(
    widget: QW.QWidget, name: str | None = None
) -> None:  # pragma: no cover
    """Grab window screenshot and save it"""
    if name is None:
        name = widget.objectName()
    widget.activateWindow()
    widget.raise_()
    QW.QApplication.processEvents()
    pixmap = widget.grab()
    suffix = datetime.now().strftime("%Y-%m-%d-%H%M%S") if name.endswith("_") else ""
    pixmap.save(osp.join(SHOTPATH, f"{name}{suffix}.png"))


@contextmanager
def save_restore_stds() -> Generator[None, None, None]:
    """Save/restore standard I/O before/after doing some things
    (e.g. calling Qt open/save dialogs)"""
    saved_in, saved_out, saved_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdout = None
    try:
        yield
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved_in, saved_out, saved_err


@contextmanager
def block_signals(
    widget: QW.QWidget, enable: bool = True, children: bool = False
) -> Generator[None, None, None]:
    """Eventually block/unblock widget Qt signals before/after doing some things

    Args:
        widget: Widget to block/unblock signals
        enable: Whether to block/unblock signals (default: True). This is useful
         to avoid blocking signals when not needed without having to handle it by
         adding an `if` statement which would require to duplicate the code that is
         inside the `with` statement in the `else` branch.
        children: Whether to block/unblock signals for child widgets (default: False).

    Returns:
        Context manager
    """
    if enable:
        widget.blockSignals(True)
        if children:
            for child in widget.findChildren(QW.QWidget):
                child.blockSignals(True)
    try:
        yield
    finally:
        if enable:
            widget.blockSignals(False)
            if children:
                for child in widget.findChildren(QW.QWidget):
                    child.blockSignals(False)


def create_menu_button(
    parent: QW.QWidget | None = None, menu: QW.QMenu | None = None
) -> QW.QPushButton:
    """Create a menu button

    Args:
        parent (QWidget): Parent widget
        menu (QMenu): Menu to attach to the button

    Returns:
        QW.QPushButton: Menu button
    """
    button = QW.QPushButton(get_icon("libre-gui-menu.svg"), "", parent)
    button.setFlat(True)
    if menu is not None:
        button.setMenu(menu)
    return button


def bring_to_front(window: QW.QWidget) -> None:
    """Bring window to front

    Args:
        window: Window to bring to front
    """
    # Show window on top of others
    eflags = window.windowFlags()
    window.setWindowFlags(eflags | QC.Qt.WindowStaysOnTopHint)
    window.show()
    window.setWindowFlags(eflags)
    window.show()
    # If window is minimized, restore it
    if window.isMinimized():
        window.showNormal()


def configure_menu_about_to_show(menu: QW.QMenu, slot: Callable) -> None:
    """Configure menu about to show.
    This method is only used to connect the "aboutToShow" signal of menus,
    and more importantly to fix Issue #15 (Part 2) which is the fact that
    dynamic menus are not supported on MacOS unless an action is added to
    the menu before it is displayed.

    Args:
        menu: menu
        slot: slot
    """
    # On MacOS, add an empty action to the menu before connecting the
    # "aboutToShow" signal to the slot. This is required to fix Issue #15 (Part 2)
    if sys.platform == "darwin":
        menu.addAction(QW.QAction(menu))
    menu.aboutToShow.connect(slot)


def add_corner_menu(
    tabwidget: QW.QTabWidget, corner: QC.Qt.Corner | None = None
) -> QW.QMenu:
    """Add menu as corner widget to tab widget

    Args:
        tabwidget: Tab widget
        corner: Corner

    Returns:
        Menu
    """
    if corner is None:
        corner = QC.Qt.TopRightCorner
    menu = QW.QMenu(tabwidget)
    btn = QW.QToolButton(tabwidget)
    btn.setMenu(menu)
    btn.setPopupMode(QW.QToolButton.InstantPopup)
    btn.setIcon(get_icon("menu.svg"))
    btn.setToolTip(_("Open tab menu"))
    tabwidget.setCornerWidget(btn, corner)
    return menu
