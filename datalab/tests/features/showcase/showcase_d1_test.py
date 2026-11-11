# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D1 tests

The D1 scenario (full traceability) turns the History recording and edit modes on
and moves the History panel: it must restore them, and leave nothing behind once
the workspace is reset, even when played in a loop.
"""

from __future__ import annotations

import pytest

from datalab.gui.main import DLMainWindow
from datalab.gui.showcase import ShowcaseOptions, ShowcasePlayer, ShowcaseScenario
from datalab.gui.showcase.engine import ShowcaseContext
from datalab.gui.showcase.scenarios import d1_provenance
from datalab.tests import datalab_test_app_context


def get_history_state(win: DLMainWindow) -> tuple:
    """Return the state of the History panel changed by the D1 scenario"""
    hpanel = win.historypanel
    dock = win.docks[hpanel]
    return (
        hpanel.record_mode_enabled,
        hpanel.ui.actions["record"].isChecked(),
        hpanel.is_edit_mode(),
        hpanel.isEnabled(),
        dock.toggleViewAction().isChecked(),
        win.dockWidgetArea(dock),
        sorted(widget.objectName() for widget in win.tabifiedDockWidgets(dock)),
    )


@pytest.mark.parametrize(
    "history, record", [(True, True), (False, False)], ids=["recording", "hidden"]
)
def test_d1_loop_restores_state(history: bool, record: bool) -> None:
    """D1 plays twice in a loop, restores the History panel and leaves nothing"""
    plays: list[int] = []

    def stop_after_two_loops(ctx: ShowcaseContext) -> None:
        plays.append(len(ctx.win.historypanel))
        if len(plays) == 2:
            ctx.request_stop()

    scenarios = [
        d1_provenance.SCENARIO,
        ShowcaseScenario("stop", "Stop", stop_after_two_loops),
    ]
    with datalab_test_app_context(console=False, history=history) as win:
        hpanel = win.historypanel
        hpanel.ui.actions["record"].setChecked(record)
        state = get_history_state(win)
        options = ShowcaseOptions(loop=True, speed=0.0)
        results = ShowcasePlayer(win, scenarios, options).play()
        statuses = [(result.scenario_id, result.status) for result in results]
        assert statuses == [("d1", "done"), ("stop", "done")] * 2, results
        # Each loop starts from a clean workspace: the History is cleared
        assert plays == [0, 0]
        assert get_history_state(win) == state
        ShowcaseContext(win, 0.0).reset_workspace()
        assert len(hpanel) == 0 and not hpanel.history_sessions
        assert not win.has_objects()
