# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D3 (image analysis) tests

The scenario is played in check mode (no dialog), then at showcase speed: its
parameter dialogs are driven as a user would do.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Generator
from concurrent.futures import Future

import pytest
import sigima.params
from guidata import qthelpers
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.config import options as sigima_options

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.processor.base import run_with_env
from datalab.gui.processor.preview import PreviewExecutorCache
from datalab.gui.showcase import ShowcaseOptions, engine, play_showcase, player
from datalab.gui.showcase.scenarios import d3_image_analysis
from datalab.tests import datalab_test_app_context


class InlineExecutor:
    """Preview executor computing the previews at once, in the test process"""

    def submit(self, function, args) -> Future:
        """Compute a preview"""
        future = Future()
        future.set_result(run_with_env(function, args, sigima_options.get_env()))
        return future

    def close(self, wait: bool = False) -> None:
        """Close the executor"""


def exec_dialog_without_event_loop(dlg: QW.QDialog) -> int:
    """Execute a dialog as ``exec_dialog`` does, without a nested event loop

    In the test suite, once a test ran the event loop of the application, a nested
    event loop (e.g. ``QDialog.exec``) returns at once: the modal dialog is shown,
    then the events are processed until it is closed.
    """
    if execenv.unattended:
        QC.QTimer.singleShot(
            execenv.delay, lambda: qthelpers.close_dialog_and_quit(dlg)
        )
    dlg.setWindowModality(QC.Qt.ApplicationModal)
    dlg.show()
    end = time.monotonic() + 120.0
    while dlg.isVisible() and time.monotonic() < end:
        QW.QApplication.processEvents()
        time.sleep(0.005)
    if dlg.isVisible():
        dlg.reject()
    result = dlg.result()
    if not dlg.testAttribute(QC.Qt.WA_DeleteOnClose):
        dlg.deleteLater()
    return result


@pytest.fixture(name="modal_dialogs")
def fixture_modal_dialogs(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Execute the modal dialogs without nested event loops, and never move the
    mouse pointer"""
    original = qthelpers.exec_dialog
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if (
            name.partition(".")[0] in ("datalab", "guidata", "plotpy", "sigimax")
            and getattr(module, "exec_dialog", None) is original
        ):
            monkeypatch.setattr(module, "exec_dialog", exec_dialog_without_event_loop)
    monkeypatch.setattr(engine, "is_headless", lambda: True)
    monkeypatch.setattr(player, "is_headless", lambda: True)
    yield


def test_d3_check_mode_restores_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """In check mode, D3 creates its results and restores the global state"""
    with datalab_test_app_context(console=False) as win:
        defaults = win.imagepanel.processor.PARAM_DEFAULTS
        remembered = sigima.params.LineProfileParam.create(y=12.0)
        monkeypatch.setitem(defaults, "LineProfileParam", remembered)
        remembered_defaults = dict(defaults)
        show_result_dialog = Conf.show_result_dialog.get()
        results = play_showcase(win, ShowcaseOptions(["d3"], speed=0.0))
        assert [(r.scenario_id, r.status) for r in results] == [("d3", "done")], (
            results[0].message
        )
        assert defaults == remembered_defaults
        assert defaults["LineProfileParam"] is remembered
        assert Conf.show_result_dialog.get() == show_result_dialog
        assert win.tabwidget.currentWidget() is win.signalpanel
        assert len(win.imagepanel.objmodel) == 3
        # Results plot, radial profile and line profile
        assert len(win.signalpanel.objmodel) >= 3


@pytest.mark.usefixtures("modal_dialogs")
def test_d3_dialogs_are_driven(monkeypatch: pytest.MonkeyPatch) -> None:
    """At showcase speed, D3 drives its parameter dialogs and the live preview"""
    with datalab_test_app_context(console=False) as win:
        monkeypatch.setattr(
            win, "preview_executor_cache", PreviewExecutorCache(InlineExecutor)
        )
        delay = execenv.delay
        results = play_showcase(win, ShowcaseOptions(["d3"], speed=0.05))
        assert [(r.scenario_id, r.status) for r in results] == [("d3", "done")], (
            results[0].message
        )
        assert execenv.delay == delay
        assert len(win.imagepanel.objmodel) == 3
        # Results plot (peak coordinates), radial profile and line profile
        assert len(win.signalpanel.objmodel) == 3
        filtered = win.imagepanel.objmodel.get_all_objects()[-1]
        assert f"cut_off={d3_image_analysis.CUT_OFF:.3f}" in filtered.title
        assert not [
            widget
            for widget in QW.QApplication.topLevelWidgets()
            if widget is not win and widget.isVisible()
        ]
