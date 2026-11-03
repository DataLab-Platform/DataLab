# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Demo application test

Plays the whole demo (**?** > **Demo**), without the delays between its steps.
"""

from __future__ import annotations

from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.gui import demo
from datalab.tests import datalab_test_app_context


def test_play_demo(monkeypatch) -> None:
    """Play the demo from the main window, as from the menu"""
    for name in ("DELAY1", "DELAY2", "DELAY3"):
        monkeypatch.setattr(demo, name, 0)
    messages: list[str] = []

    def information(_parent, _title, text, *_args) -> QW.QMessageBox.StandardButton:
        """Accept the start and end messages of the demo"""
        messages.append(text)
        return QW.QMessageBox.Ok

    monkeypatch.setattr(QW.QMessageBox, "information", staticmethod(information))
    # Restored after the test: the demo changes these settings when it ends
    monkeypatch.setenv(execenv.UNATTENDED_ENV, "1")
    monkeypatch.setenv(execenv.DELAY_ENV, "0")

    with datalab_test_app_context(console=False) as win:
        try:
            win.play_demo()
        finally:
            execenv.unattended = True  # Quits the event loop of the test
        assert len(messages) == 2
        assert not execenv.demo_mode
        assert len(win.signalpanel.objmodel) > 0
        assert len(win.imagepanel.objmodel) > 0
