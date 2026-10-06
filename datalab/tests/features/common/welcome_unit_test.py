# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Welcome page unit tests
"""

from __future__ import annotations

from qtpy import QtWidgets as QW

import datalab
from datalab.config import Conf, _
from datalab.gui import welcome
from datalab.gui.main import DLMainWindow
from datalab.tests import datalab_test_app_context


def test_release_notes_url(monkeypatch) -> None:
    """Release notes URL follows the documentation page naming"""
    monkeypatch.setattr(datalab, "__version__", "1.4.0rc1")
    url = welcome.get_release_notes_url()
    assert url == "https://datalab-platform.com/en/release_notes/release_1.04.html"


def is_in_front(dock: QW.QDockWidget) -> bool:
    """Return True if the dock is the selected tab of its group"""
    # Qt keeps background dock tabs visible: only their visible region is empty
    return not dock.visibleRegion().isEmpty()


def test_welcome_page(monkeypatch) -> None:
    """Welcome page dock, entries and startup preference"""
    calls: list[object] = []
    monkeypatch.setattr(DLMainWindow, "show_tour", lambda self: calls.append("tour"))
    monkeypatch.setattr(DLMainWindow, "play_demo", lambda self: calls.append("demo"))
    monkeypatch.setattr(welcome.webbrowser, "open", calls.append)

    def choose_image(menu: QW.QMenu, *_args) -> QW.QAction:
        return next(act for act in menu.actions() if act.text() == _("Image"))

    def create_gaussian(menu: QW.QMenu, *_args) -> None:
        sig_menu = next(
            act.menu() for act in menu.actions() if act.text() == _("Signal")
        )
        next(act for act in sig_menu.actions() if act.text() == _("Gaussian")).trigger()

    with datalab_test_app_context(console=False) as win:
        page = win.welcomepanel
        dock = win.docks[page]
        sigdock = win.docks[win.signalpanel]
        assert dock in win.tabifiedDockWidgets(sigdock)
        assert page not in win.panels

        win.show_welcome_page()
        QW.QApplication.processEvents()
        assert is_in_front(dock) and not is_in_front(sigdock)
        assert page.start_title.text() == _("Get started")

        # Panel-specific entries ask for the destination panel first
        monkeypatch.setattr(QW.QMenu, "exec", choose_image)
        for key, method in (
            ("open", "load_from_files"),
            ("import_text", "exec_import_wizard"),
        ):
            monkeypatch.setattr(
                win.imagepanel, method, lambda key=key: calls.append(key)
            )
            page.entries[key].SIG_CLICKED.emit()
            assert win.tabwidget.currentWidget() is win.imagepanel
        monkeypatch.setattr(
            win, "open_h5_files", lambda import_all=None: calls.append(import_all)
        )
        for key in ("browse_h5", "open_h5", "tour", "demo"):
            page.entries[key].SIG_CLICKED.emit()
        for key in ("documentation", "release_notes"):
            page.entries[key].SIG_CLICKED.emit()
        assert calls == [
            "open",
            "import_text",
            None,
            True,
            "tour",
            "demo",
            Conf.app_docurl.get(),
            welcome.get_release_notes_url(),
        ]

        page.entries["ai_assistant"].SIG_CLICKED.emit()
        QW.QApplication.processEvents()
        assert is_in_front(win.docks[win.aiassistantpanel])

        # Creating an object brings the signal view back to the front
        monkeypatch.setattr(QW.QMenu, "exec", create_gaussian)
        win.show_welcome_page()
        page.entries["create"].SIG_CLICKED.emit()
        QW.QApplication.processEvents()
        assert len(win.signalpanel) == 1
        assert is_in_front(sigdock) and not is_in_front(dock)
        win.show_welcome_page()
        QW.QApplication.processEvents()
        assert page.start_title.text() == _("Quick actions")

        initial = Conf.welcome_on_startup.get()
        try:
            page.startup_checkbox.setChecked(not initial)
            assert Conf.welcome_on_startup.get() is (not initial)
        finally:
            Conf.welcome_on_startup.set(initial)
