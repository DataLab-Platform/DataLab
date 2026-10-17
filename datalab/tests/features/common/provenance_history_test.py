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


def link_of(win, obj) -> str:
    """Return the activity that produced the latest recorded state of *obj*."""
    return win.provenance.ledger.latest_state(get_uuid(obj))["produced_by"]


def test_actions_link_the_activities_of_their_execution() -> None:
    """Every compute action links exactly the activities of its execution."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        first = add_signal(win)
        second = add_signal(win, title="S1")
        panel.objview.select_objects([get_uuid(first)])
        result = normalize(win, "maximum")
        action = history[len(history)]
        assert action.provenance_activity_ids == [link_of(win, result)]
        panel.objview.select_objects([get_uuid(first), get_uuid(second)])
        panel.processor.compute_n_to_1(sips.average, edit=False)
        mean = panel.objmodel[panel.objmodel.get_object_ids()[-1]]
        action = history[len(history)]
        assert action.provenance_activity_ids == [link_of(win, mean)]
        panel.objview.select_objects([get_uuid(first)])
        params = [
            sigima.params.NormalizeParam.create(method="maximum"),
            sigima.params.NormalizeParam.create(method="amplitude"),
        ]
        panel.processor.compute_1_to_n(sips.normalize, params)
        action = history[len(history)]
        outputs = [panel.objmodel[uid] for uid in action.output_uuids]
        assert len(outputs) == 2
        assert action.provenance_activity_ids == [link_of(win, o) for o in outputs]
        command = win.provenance.ledger.activity(action.provenance_activity_ids[0])
        assert win.provenance.command_activity_ids(command["command_id"]) == (
            action.provenance_activity_ids
        )
        assert history.find_action_for_activity(link_of(win, outputs[1])) is action


def test_replay_edit_and_failure_keep_exact_links() -> None:
    """Replays relink on final UUIDs; edits and failures keep the last execution."""
    # pylint: disable=import-outside-toplevel
    from datalab.processingcall import recipe_differs_from_execution
    from datalab.widgets.provenance import ExecutionProvenanceDialog

    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        result = normalize(win, "maximum")
        action = history[len(history)]
        (first,) = action.provenance_activity_ids
        before = json.dumps(win.provenance.ledger.activity(first), sort_keys=True)
        assert recipe_differs_from_execution(win, action) is False
        assert hrec.recompute_action_in_place(history, action) is True
        (replay,) = action.provenance_activity_ids
        assert replay != first and replay == link_of(win, result)
        assert win.provenance.ledger.activity(replay)["origin"] == "history_replay"
        assert json.dumps(win.provenance.ledger.activity(first), sort_keys=True) == (
            before
        )
        # Edited, not executed: the recipe differs from the last execution.
        action.kwargs["param"] = sigima.params.NormalizeParam.create(method="amplitude")
        assert recipe_differs_from_execution(win, action) is True
        dialog = ExecutionProvenanceDialog(win, action)
        assert "edited" in dialog.note.text() and len(dialog.views) == 1
        assert dialog.views[0].description["activity"]["activity_id"] == replay
        # A failing replay keeps the link to the last successful execution.
        original_execute = hrec.execute_compute_via_ui

        def failing_execute(panel_data, act, obj2_uuids) -> None:
            original_execute(panel_data, act, obj2_uuids)
            raise RuntimeError("simulated mid-batch failure")

        with patch.object(hrec, "execute_compute_via_ui", failing_execute):
            assert hrec.recompute_action_in_place(history, action) is False
        history.runtime.execution.cascade_warnings.clear()
        assert action.provenance_activity_ids == [replay]
        assert hrec.recompute_action_in_place(history, action) is True
        (edited,) = action.provenance_activity_ids
        assert win.provenance.ledger.activity(edited)["call"]["parameters"] == {
            "method": "amplitude"
        }
        assert recipe_differs_from_execution(win, action) is False


def test_processing_edit_updates_the_linked_action() -> None:
    """Two calls of one function: Processing edits the action of its result."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        source = add_signal(win)
        first_result = normalize(win, "maximum")
        first_action = history[len(history)]
        win.signalpanel.objview.select_objects([get_uuid(source)])
        second_result = normalize(win, "maximum")
        second_action = history[len(history)]
        history.toggle_edit_mode(True)
        objprop = win.signalpanel.objprop
        report = objprop.apply_processing_parameters(
            second_result,
            interactive=False,
            param=sigima.params.NormalizeParam.create(method="amplitude"),
        )
        assert report.success, report.message
        assert second_action.kwargs["param"].method == "amplitude"
        assert first_action.kwargs["param"].method == "maximum"
        assert second_action.provenance_activity_ids == [link_of(win, second_result)]
        assert first_action.provenance_activity_ids == [link_of(win, first_result)]


def test_cascade_relinks_every_recomputed_action() -> None:
    """A cascade gives each recomputed action its new activities."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        first = normalize(win, "maximum")
        root = history[len(history)]
        win.signalpanel.objview.select_objects([get_uuid(first)])
        second = normalize(win, "amplitude")
        child = history[len(history)]
        old = (list(root.provenance_activity_ids), list(child.provenance_activity_ids))
        root.kwargs["param"] = sigima.params.NormalizeParam.create(method="amplitude")
        root.is_stale = True
        history.recompute_cascade(root)
        history.runtime.execution.cascade_warnings.clear()
        assert root.provenance_activity_ids == [link_of(win, first)]
        assert child.provenance_activity_ids == [link_of(win, second)]
        assert root.provenance_activity_ids != old[0]
        assert child.provenance_activity_ids != old[1]
        for state in win.provenance.ledger.states.values():
            assert win.find_object_by_uuid(state["object_uuid"]) is not None


def test_same_computation_by_menu_processing_and_history() -> None:
    """Menu, Processing and History run the same scientific call."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        result = normalize(win, "maximum")
        action = history[len(history)]
        menu = win.provenance.ledger.activity(link_of(win, result))
        menu_y = result.y.copy()
        report = win.signalpanel.processor.recompute_processing(
            result, interactive=False
        )
        assert report.success
        processing = win.provenance.ledger.activity(link_of(win, result))
        processing_y = result.y.copy()
        assert hrec.recompute_action_in_place(history, action) is True
        replay = win.provenance.ledger.activity(link_of(win, result))
        assert [a["origin"] for a in (menu, processing, replay)] == [
            "ordinary",
            "recompute_in_place",
            "history_replay",
        ]
        for activity in (processing, replay):
            for key in ("operation", "parameters", "inputs"):
                assert activity["call"][key] == menu["call"][key]
            assert activity["context"] == menu["context"]
        assert np.array_equal(menu_y, processing_y)
        assert np.array_equal(menu_y, result.y)
        assert np.array_equal(menu_y, MAXIMUM_Y)


def test_links_are_saved_and_never_copied(tmp_path) -> None:
    """Links reopen with the workspace; copies and imports carry none."""
    # pylint: disable=import-outside-toplevel
    from datalab.gui import historytools_ops as htools
    from datalab.tests.features.common.history_test_helpers import (
        select_tree_session,
    )
    from datalab.widgets.provenance import ExecutionProvenanceDialog

    path = str(tmp_path / "links.h5")
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        normalize(win, "maximum")
        links = list(history[len(history)].provenance_activity_ids)
        assert links
        win.save_h5_workspace(path)
    with app(history=True) as win:
        win.load_h5_workspace([path], reset_all=True)
        history = win.historypanel
        compute = [
            action
            for session in history.history_sessions
            for action in session.actions
            if action.kind == "compute"
        ]
        assert compute[-1].provenance_activity_ids == links
        assert history.find_action_for_activity(links[0]) is compute[-1]
        session = history.history_sessions[-1]
        select_tree_session(history, session)
        htools.duplicate_selected_entries(history)
        duplicate = history.history_sessions[-1]
        assert duplicate is not session
        assert all(not a.provenance_activity_ids for a in duplicate.actions)
        copied = [a for a in duplicate.actions if a.kind == "compute"][-1]
        dialog = ExecutionProvenanceDialog(win, copied)
        assert "No recorded execution" in dialog.note.text() and not dialog.views
        # A History without the matching ledger (e.g. another workspace)
        win.provenance.reset()
        dialog = ExecutionProvenanceDialog(win, compute[-1])
        assert "not available" in dialog.note.text() and not dialog.views


def test_verification_creates_no_action_nor_activity() -> None:
    """Verifying an execution changes neither the History nor the ledger."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        add_signal(win)
        normalize(win, "maximum")
        (activity_id,) = history[len(history)].provenance_activity_ids
        count, actions = len(win.provenance.ledger.activities), len(history)
        report = win.verify_provenance_activity(activity_id)
        assert report["verdict"] == "exact"
        assert len(win.provenance.ledger.activities) == count
        assert len(history) == actions


def _grouped(win, method: str) -> list:
    """Run a grouped command (two Gaussian filters, sigma 1 and 2) on S0."""
    panel = win.signalpanel
    s0 = add_signal(win)
    panel.objview.select_objects([get_uuid(s0)])
    before = set(panel.objmodel.get_object_ids())
    params = [
        sigima.params.GaussianParam.create(sigma=1.0),
        sigima.params.GaussianParam.create(sigma=2.0),
    ]
    if method == "compute_1_to_n":
        panel.processor.compute_1_to_n(sips.gaussian_filter, params=params, edit=False)
    else:
        panel.processor.compute_multiple_1_to_1(
            [sips.gaussian_filter, sips.gaussian_filter], params=params, edit=False
        )
    return [
        panel.objmodel[uid]
        for uid in panel.objmodel.get_object_ids()
        if uid not in before
    ]


def _sigma(win, obj) -> float:
    return win.provenance.describe_object(get_uuid(obj))["production"]["parameters"][
        "sigma"
    ]


@pytest.mark.parametrize("method", ["compute_1_to_n", "compute_multiple_1_to_1"])
def test_edited_output_of_a_grouped_command_is_replayed(method: str) -> None:
    """Editing one output of a grouped command updates that element only."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        first, second = _grouped(win, method)
        action = history[len(history)]
        history.toggle_edit_mode(True)
        report = win.signalpanel.objprop.apply_processing_parameters(
            first,
            interactive=False,
            param=sigima.params.GaussianParam.create(sigma=5.0),
        )
        assert report.success, report.message
        assert "param" not in action.kwargs
        assert [p.sigma for p in action.kwargs["params"]] == [5.0, 2.0]
        assert link_of(win, first) in action.provenance_activity_ids
        assert link_of(win, second) in action.provenance_activity_ids
        assert hrec.recompute_action_in_place(history, action) is True
        history.runtime.execution.cascade_warnings.clear()
        assert (_sigma(win, first), _sigma(win, second)) == (5.0, 2.0)


def test_processing_outside_edit_mode_keeps_the_recipe() -> None:
    """Outside History edit mode, the History action stays the recipe."""
    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        first, _second = _grouped(win, "compute_1_to_n")
        action = history[len(history)]
        count = len(history)
        report = win.signalpanel.objprop.apply_processing_parameters(
            first,
            interactive=False,
            param=sigima.params.GaussianParam.create(sigma=5.0),
        )
        assert report.success, report.message
        assert len(history) == count
        assert [p.sigma for p in action.kwargs["params"]] == [1.0, 2.0]
        assert _sigma(win, first) == 5.0  # Recorded by the provenance
        assert hrec.recompute_action_in_place(history, action) is True
        history.runtime.execution.cascade_warnings.clear()
        assert _sigma(win, first) == 1.0  # The recipe is replayed


def _analyses(win, obj) -> list[str]:
    return [
        activity["activity_id"]
        for activity in win.provenance.describe_object(get_uuid(obj))["analyses"]
    ]


def test_analysis_actions_link_their_executions() -> None:
    """Analyses from the menu and from the Analysis tab are linked."""
    # pylint: disable=import-outside-toplevel
    from datalab.widgets.provenance import ExecutionProvenanceDialog

    with app(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        s0 = add_signal(win)
        s1 = add_signal(win)
        panel.objview.select_objects([get_uuid(s0), get_uuid(s1)])
        panel.processor.compute_1_to_0(
            sips.fwhm, param=sigima.params.FWHMParam(), edit=False
        )
        action = history[len(history)]
        assert action.pattern == "1_to_0"
        # One activity per analysed object, linked from the menu run
        first0, first1 = _analyses(win, s0)[0], _analyses(win, s1)[0]
        assert action.provenance_activity_ids == [first0, first1]
        dialog = ExecutionProvenanceDialog(win, action)
        assert len(dialog.views) == 2 and dialog.note.isHidden()
        # A new analysis of s0 from the Analysis tab replaces its link only
        panel.objview.set_current_object(s0)
        QW.QApplication.processEvents()
        assert panel.objprop.apply_analysis_parameters(
            s0, interactive=False, param=sigima.params.FWHMParam()
        )
        (latest,) = [a for a in _analyses(win, s0) if a != first0]
        assert action.provenance_activity_ids == [latest, first1]
        # A History replay links the activities of the replay
        assert hrec.recompute_action_in_place(history, action) is True
        replayed = action.provenance_activity_ids
        assert len(replayed) == 2 and latest not in replayed and first1 not in replayed
