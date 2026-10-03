# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace provenance with previews and History replay (Desktop)."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import sigima.params
import sigima.proc.signal as sips
from qtpy import QtWidgets as QW

from datalab.gui.panel.history import recompute as hrec
from datalab.gui.processor.catcher import CompOut
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.processing_preview_unit_test import FakeExecutor
from datalab.tests.features.common.provenance_unit_test import (
    AMPLITUDE_Y,
    MAXIMUM_Y,
    add_signal,
    app,
    normalize,
    output_state,
)
from datalab.widgets import processingpreview


@pytest.fixture(autouse=True)
def drain_pending_qt_timers():
    """Prevent unattended close timers from affecting the next test."""
    yield
    if QW.QApplication.instance() is not None:
        for _index in range(3):
            QW.QApplication.processEvents()


def test_preview_cancelled_and_adopted(monkeypatch) -> None:
    """A cancelled preview records nothing; an adopted one records once."""
    with app() as win:
        panel = win.signalpanel
        source = add_signal(win)
        executor = FakeExecutor()
        win.preview_executor_cache.reset()
        win.preview_executor_cache._executor_factory = lambda: executor  # pylint: disable=protected-access
        execution = panel.processor.execution

        def preview_and(accept: bool):
            def dialog_handler(dialog):
                dialog.preview.enabled.setChecked(True)
                preview = sips.normalize(source, dialog.instance)
                executor.requests[-1][0].set_result(CompOut(result=preview))
                dialog.preview.controller.poll()
                if accept:
                    dialog.accept()
                    return 1
                dialog.reject()
                return 0

            return dialog_handler

        param = sigima.params.NormalizeParam.create(method="amplitude")
        monkeypatch.setattr(processingpreview, "exec_dialog", preview_and(False))
        panel.processor.compute_1_to_1(sips.normalize, param=param, edit=True)
        assert win.provenance.ledger.activities == ()
        assert len(panel.objmodel) == 1

        computations = execution.computations
        monkeypatch.setattr(processingpreview, "exec_dialog", preview_and(True))
        panel.processor.compute_1_to_1(sips.normalize, param=param, edit=True)
        assert execution.computations == computations
        (activity,) = win.provenance.ledger.activities
        assert activity["call"]["parameters"] == {"method": "amplitude"}
        result = panel.objmodel[panel.objmodel.get_object_ids()[-1]]
        assert np.array_equal(result.y, AMPLITUDE_Y)
        assert output_state(win, activity)["object_uuid"] == get_uuid(result)


def test_history_recompute_through_shared_path() -> None:
    """History replay records new activities on the final objects only."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        result = normalize(win, "maximum")
        action = history[len(history)]
        assert action.output_uuids == [get_uuid(result)]
        first = json.dumps(win.provenance.ledger.activities[0], sort_keys=True)
        execution = win.signalpanel.processor.execution
        computations = execution.computations

        assert hrec.recompute_action_in_place(history, action) is True
        assert execution.computations == computations + 1
        replay = win.provenance.ledger.activities[-1]
        assert replay["origin"] == "history_replay"
        assert output_state(win, replay)["object_uuid"] == get_uuid(result)

        action.kwargs["param"] = sigima.params.NormalizeParam.create(method="amplitude")
        assert hrec.recompute_action_in_place(history, action) is True
        edited = win.provenance.ledger.activities[-1]
        assert edited["call"]["parameters"] == {"method": "amplitude"}
        assert output_state(win, edited)["object_uuid"] == get_uuid(result)
        assert np.array_equal(win.signalpanel.objmodel[get_uuid(result)].y, AMPLITUDE_Y)
        assert len(win.provenance.ledger.activities) == 3
        assert json.dumps(win.provenance.ledger.activities[0], sort_keys=True) == first
        known = set(win.signalpanel.objmodel.get_object_ids())
        for state in win.provenance.ledger.states.values():
            assert state["object_uuid"] in known
        win.provenance.ledger.validate()


def test_failed_history_replay_records_nothing() -> None:
    """Discarded temporary outputs leave no activity behind."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        normalize(win, "maximum")
        action = history[len(history)]
        count = len(win.provenance.ledger.activities)
        original_execute = hrec.execute_compute_via_ui

        def failing_execute(panel_data, act, obj2_uuids) -> None:
            original_execute(panel_data, act, obj2_uuids)
            raise RuntimeError("simulated mid-batch failure")

        with patch.object(hrec, "execute_compute_via_ui", failing_execute):
            assert hrec.recompute_action_in_place(history, action) is False
        assert len(win.provenance.ledger.activities) == count
        history.runtime.execution.cascade_warnings.clear()
        assert np.array_equal(
            win.signalpanel.objmodel[action.output_uuids[0]].y, MAXIMUM_Y
        )
