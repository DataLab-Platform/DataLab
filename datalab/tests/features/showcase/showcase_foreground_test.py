# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase window activation tests
"""

# guitest: show

from __future__ import annotations

import sys

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

from datalab.gui.showcase import foreground


@pytest.fixture(name="activations")
def fixture_activations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the native activations of windows instead of running them

    Returns:
        Native handles of the windows activated
    """
    handles: list[int] = []

    def record(hwnd: int) -> bool:
        handles.append(hwnd)
        return True

    monkeypatch.setattr(foreground, "_activate_windows_window", record)
    return handles


def test_window_brought_to_front_on_windows(
    monkeypatch: pytest.MonkeyPatch, activations: list[int]
) -> None:
    """On Windows, a shown window is activated whatever the active application"""
    monkeypatch.setattr(foreground, "is_headless", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")
    with qt_app_context():
        win = QW.QMainWindow()
        win.show()
        foreground.bring_to_front(win)
        assert activations == [int(win.winId())]
        # A minimized window (e.g. during the DataLab Web interlude) is left alone
        win.showMinimized()
        foreground.bring_to_front(win)
        assert len(activations) == 1
        win.close()


def test_window_not_activated_natively_without_screen(
    activations: list[int],
) -> None:
    """Without screen (offscreen platform), the native activation is skipped"""
    with qt_app_context():
        if not foreground.is_headless():
            pytest.skip("Needs a Qt platform without screen")
        win = QW.QMainWindow()
        win.show()
        foreground.bring_to_front(win)
        win.close()
    assert not activations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
