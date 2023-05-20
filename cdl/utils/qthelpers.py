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
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

import guidata
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import APP_NAME, DATETIME_FORMAT, SHOTPATH, Conf, _, get_old_log_fname
from cdl.env import execenv
from cdl.utils.misc import to_string
from cdl.widgets.errormessagebox import ErrorMessageBox


def close_widgets_and_quit(screenshot=False) -> None:
    """Close Qt top level widgets and quit Qt event loop"""
    for widget in QW.QApplication.instance().topLevelWidgets():
        wname = widget.objectName()
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
def qt_app_context(
    exec_loop=False, enable_logs=True
) -> Generator[QW.QApplication, None, None]:
    """Context manager handling Qt application creation and persistance"""
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
        try:
            yield QAPP_INSTANCE
        finally:
            if execenv.unattended:  # pragma: no cover
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
            if exec_loop:
                QAPP_INSTANCE.exec()

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


def close_dialog_and_quit(widget, screenshot=False):
    """Close QDialog and quit Qt event loop"""
    wname = widget.objectName()
    if screenshot and wname and widget.isVisible():  # pragma: no cover
        grab_save_window(widget, wname.lower())
    widget.done(QW.QDialog.Accepted)
    # QW.QApplication.instance().quit()


def exec_dialog(dlg: QW.QDialog) -> int:
    """Run QDialog Qt execution loop without blocking,
    depending on environment test mode"""
    if execenv.unattended:
        QC.QTimer.singleShot(
            execenv.delay * 1000,
            lambda: close_dialog_and_quit(dlg, screenshot=execenv.screenshot),
        )
    delete_later = not dlg.testAttribute(QC.Qt.WA_DeleteOnClose)
    result = dlg.exec()
    if delete_later:
        dlg.deleteLater()
    return result


def qt_wait(timeout, except_unattended=False) -> None:  # pragma: no cover
    """Freeze GUI during timeout (seconds) while processing Qt events"""
    if except_unattended and execenv.unattended:
        return
    start = time.time()
    while time.time() <= start + timeout:
        time.sleep(0.01)
        QW.QApplication.processEvents()


@contextmanager
def create_progress_bar(
    parent: QW.QWidget, label: str, max_: int
) -> Generator[QW.QProgressDialog, None, None]:
    """Create modal progress bar"""
    prog = QW.QProgressDialog(label, _("Cancel"), 0, max_, parent, QC.Qt.SplashScreen)
    prog.setWindowModality(QC.Qt.WindowModal)
    prog.show()
    QW.QApplication.processEvents()
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
def qt_try_context(parent: QW.QWidget, context: str = None, tip: str = None):
    """Try...except Qt widget context manager"""
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        dlg = ErrorMessageBox(parent, context, tip)
        exec_dialog(dlg)
    finally:
        pass


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
