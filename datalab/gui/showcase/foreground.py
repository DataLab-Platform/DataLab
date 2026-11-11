# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase window activation
==========================

The :mod:`datalab.gui.showcase.foreground` module brings the DataLab main window
to the foreground when a showcase starts, even when another application is
active.

On Windows, only the application owning the foreground window may activate a
window: a showcase started unattended (e.g. from a shortcut or a scheduled task,
while another application is active) would otherwise play behind the windows of
the other applications, with only its menus and dialogs showing up.
"""

from __future__ import annotations

import ctypes
import logging
import sys

from qtpy import QtWidgets as QW

from datalab.gui.showcase.engine import is_headless

logger = logging.getLogger("datalab.showcase")


def _activate_windows_window(hwnd: int) -> bool:
    """Make a window the foreground window, whatever the active application

    The input of the thread owning the foreground window is attached to the
    current thread for the time of the activation: Windows then lets the current
    thread activate its window, as if the user had clicked it.

    Args:
        hwnd: Native handle of the window

    Returns:
        True if the window is the foreground window
    """
    # pylint: disable=import-outside-toplevel
    from ctypes import wintypes

    # Private instances: the prototypes are not shared with other modules
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, wintypes.LPDWORD]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.AttachThreadInput.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    foreground = user32.GetForegroundWindow()
    if foreground == hwnd:
        return True
    current_thread = kernel32.GetCurrentThreadId()
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    )
    attached = bool(
        foreground_thread
        and foreground_thread != current_thread
        and user32.AttachThreadInput(foreground_thread, current_thread, True)
    )
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(foreground_thread, current_thread, False)
    return user32.GetForegroundWindow() == hwnd


def bring_to_front(win: QW.QWidget) -> None:
    """Bring a top-level window to the foreground, and activate it

    Args:
        win: Top-level window (shown)
    """
    win.raise_()
    win.activateWindow()
    if (
        sys.platform != "win32"
        or is_headless()
        or not win.isVisible()
        or win.isMinimized()
    ):
        return
    try:
        if not _activate_windows_window(int(win.winId())):
            logger.info("Showcase window could not be brought to the foreground")
    except OSError:
        logger.warning("Showcase window activation failed", exc_info=True)
