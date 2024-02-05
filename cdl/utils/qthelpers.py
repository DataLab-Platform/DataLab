# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Qt utilities
"""

from __future__ import annotations

import faulthandler
import functools
import logging
import os
import os.path as osp
import shutil
import sys
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime

import guidata
from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import APP_NAME, DATETIME_FORMAT, SHOTPATH, Conf, _, get_old_log_fname
from cdl.env import execenv
from cdl.utils.strings import to_string


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
def cdl_app_context(
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
                    message = f"{mode} mode (delay: {execenv.delay}s)"
                    msec = execenv.delay * 1000 - 200
                    for widget in QW.QApplication.instance().topLevelWidgets():
                        if isinstance(widget, QW.QMainWindow):
                            widget.statusBar().showMessage(message, msec)
                QC.QTimer.singleShot(
                    execenv.delay * 1000,
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


@contextmanager
def try_or_log_error(context: str) -> Generator[None, None, None]:
    """Try to execute a function and log an error message if it fails"""
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        logger = logging.getLogger(__name__)
        logger.error("Error in %s", context, exc_info=traceback.format_exc())
        Conf.main.traceback_log_available.set(True)
    finally:
        pass


@contextmanager
def create_progress_bar(
    parent: QW.QWidget, label: str, max_: int
) -> Generator[QW.QProgressDialog, None, None]:
    """Create modal progress bar"""
    prog = QW.QProgressDialog(label, _("Cancel"), 0, max_, parent, QC.Qt.SplashScreen)
    prog.setWindowModality(QC.Qt.WindowModal)
    prog.show()
    try:
        yield prog
    finally:
        prog.close()
        prog.deleteLater()


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
    if operation == "load":
        text = _("%s could not be opened:")
    elif operation == "save":
        text = _("%s could not be written:")
    else:
        raise ValueError("operation argument must be 'load' or 'save'")
    try:
        yield filename
    except Exception as msg:  # pylint: disable=broad-except
        traceback.print_exc()
        message = (text % osp.basename(filename)) + "\n" + str(msg)
        QW.QMessageBox.critical(parent, APP_NAME, message)
    finally:
        pass


def grab_save_window(widget: QW.QWidget, name: str) -> None:  # pragma: no cover
    """Grab window screenshot and save it"""
    widget.activateWindow()
    widget.raise_()
    QW.QApplication.processEvents()
    pixmap = widget.grab()
    suffix = ""
    if not name[-1].isdigit() and not name.startswith(("s_", "i_")):
        suffix = "_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")
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
def block_signals(widget: QW.QWidget, enable: bool) -> Generator[None, None, None]:
    """Eventually block/unblock widget Qt signals before/after doing some things
    (enable: True if feature is enabled)"""
    if enable:
        widget.blockSignals(True)
    try:
        yield
    finally:
        if enable:
            widget.blockSignals(False)


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
