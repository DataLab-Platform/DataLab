# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace provenance capture, replay preparation and verification (Desktop)."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import sigima.objects
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import SignalObj, create_signal
from sigima.tests.data import create_sincos_image

from datalab import provenance as provenance_module
from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.processor.base import BaseProcessor
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context

X = np.array([0.0, 0.25, 0.5, 0.75])
Y = np.array([-2.0, 0.0, 1.0, 4.0])
# Exact analytical oracles (literals, never derived from Sigima)
MAXIMUM_Y = np.array([-0.5, 0.0, 0.25, 1.0])
AMPLITUDE_Y = np.array([0.0, 1 / 3, 0.5, 1.0])
SHIFTED_Y = np.array([-1.0, 1.0, 2.0, 5.0])
SHIFTED_MAXIMUM_Y = np.array([-1 / 5, 1 / 5, 2 / 5, 1.0])


@pytest.fixture(autouse=True)
def isolated_param_defaults(monkeypatch):
    """``PARAM_DEFAULTS`` is a class attribute shared by the whole process."""
    monkeypatch.setattr(BaseProcessor, "PARAM_DEFAULTS", {})


@contextlib.contextmanager
def app(history: bool = False, isolation: bool = False):
    """Return a DataLab main window for one test."""
    try:
        with qt_app_context(), Conf.process_isolation_enabled.context(isolation):
            with datalab_test_app_context(
                console=False, history=history, exec_loop=False
            ) as win:
                yield win
    finally:
        # Unattended close timers scheduled on exit would close the next window.
        for _index in range(3):
            QW.QApplication.processEvents()


def add_signal(win, y=Y, title: str = "S0") -> SignalObj:
    """Add a signal and select it."""
    obj = create_signal(title, X.copy(), np.array(y, dtype=float), units=("s", ""))
    win.signalpanel.add_object(obj)
    win.signalpanel.objview.select_objects([get_uuid(obj)])
    return obj


def run(win, func, param=None, **kwargs) -> list[SignalObj]:
    """Run a 1-to-1 processing on the selection and return the new objects."""
    panel = win.signalpanel
    before = set(panel.objmodel.get_object_ids())
    panel.processor.compute_1_to_1(func, param=param, edit=False, **kwargs)
    return [
        panel.objmodel[uid]
        for uid in panel.objmodel.get_object_ids()
        if uid not in before
    ]


def normalize(win, method: str) -> SignalObj:
    """Normalize the selection with an explicit method."""
    param = sigima.params.NormalizeParam.create(method=method)
    (result,) = run(win, sips.normalize, param)
    return result


def select(win, obj) -> None:
    """Select one signal."""
    win.signalpanel.objview.select_objects([get_uuid(obj)])


def output_state(win, activity) -> dict:
    """Return the result state of an activity."""
    return win.provenance.ledger.states[activity["outputs"][0]["state_id"]]


def input_state(win, activity) -> dict:
    """Return the source state of an activity."""
    state_id = activity["call"]["inputs"][0]["binding"]["state_id"]
    return win.provenance.ledger.states[state_id]


def test_capture_with_record_off_and_on() -> None:
    """Capture does not depend on History; Record on creates no duplicate."""
    for history in (False, True):
        with app(history=history) as win:
            if history:
                win.historypanel.toggle_record_mode(True)
            source = add_signal(win)
            result = normalize(win, "maximum")
            assert np.array_equal(result.y, MAXIMUM_Y)
            (activity,) = win.provenance.ledger.activities
            assert activity["call"]["operation"] == {
                "id": "sigima.signal.normalize",
                "contract_version": 1,
            }
            assert activity["call"]["parameters"] == {"method": "maximum"}
            assert activity["origin"] == "ordinary"
            assert activity["edition"] == "desktop"
            assert input_state(win, activity)["object_uuid"] == get_uuid(source)
            assert output_state(win, activity)["object_uuid"] == get_uuid(result)
            win.provenance.ledger.validate()
            if history:
                assert len(win.historypanel) >= 1


def test_two_sources_share_one_command() -> None:
    """Each source/result pair is one activity; one command groups them."""
    with app() as win:
        first = add_signal(win, title="A")
        second = add_signal(win, Y * 2, title="B")
        win.signalpanel.objview.select_objects([get_uuid(first), get_uuid(second)])
        results = run(win, sips.normalize, sigima.params.NormalizeParam.create())
        activities = win.provenance.ledger.activities
        assert len(activities) == 2
        assert activities[0]["command_id"] == activities[1]["command_id"]
        for activity, source, result in zip(activities, (first, second), results):
            assert input_state(win, activity)["object_uuid"] == get_uuid(source)
            assert output_state(win, activity)["object_uuid"] == get_uuid(result)


def test_error_and_cancellation_record_nothing() -> None:
    """A failing or cancelled computation records no activity, no object."""

    def failing(src: SignalObj, p: sigima.params.NormalizeParam) -> SignalObj:
        raise ValueError("simulated failure")

    with app() as win, execenv.context(catcher_test=True):
        add_signal(win)
        count = len(win.signalpanel.objmodel)
        assert run(win, failing, sigima.params.NormalizeParam.create()) == []
        execution = win.signalpanel.processor.execution
        execution._exec_func = lambda *args: None  # pylint: disable=protected-access
        assert run(win, sips.normalize, sigima.params.NormalizeParam.create()) == []
        assert len(win.signalpanel.objmodel) == count
        assert win.provenance.ledger.activities == ()


def test_omitted_parameter_records_effective_values(monkeypatch) -> None:
    """Defaults and remembered values are recorded as actually used."""
    with app() as win:
        processor = win.signalpanel.processor
        add_signal(win)
        before = set(win.signalpanel.objmodel.get_object_ids())
        processor.compute_1_to_1(
            sips.normalize, paramclass=sigima.params.NormalizeParam, edit=False
        )
        (activity,) = win.provenance.ledger.activities
        assert activity["call"]["parameters"] == {"method": "maximum"}
        new = [u for u in win.signalpanel.objmodel.get_object_ids() if u not in before]
        assert np.array_equal(win.signalpanel.objmodel[new[0]].y, MAXIMUM_Y)

        remembered = sigima.params.NormalizeParam.create(method="amplitude")
        monkeypatch.setitem(processor.PARAM_DEFAULTS, "NormalizeParam", remembered)
        select(win, win.signalpanel.objmodel.get_object_from_number(1))
        before = set(win.signalpanel.objmodel.get_object_ids())
        processor.compute_1_to_1(
            sips.normalize, paramclass=sigima.params.NormalizeParam, edit=False
        )
        activity = win.provenance.ledger.activities[-1]
        assert activity["call"]["parameters"] == {"method": "amplitude"}
        new = [u for u in win.signalpanel.objmodel.get_object_ids() if u not in before]
        assert np.array_equal(win.signalpanel.objmodel[new[0]].y, AMPLITUDE_Y)
        for act in win.provenance.ledger.activities:
            report = win.verify_provenance_activity(act["activity_id"])
            assert report["verdict"] == "exact"


def test_processing_tab_recompute_in_place() -> None:
    """Recomputing from the Processing tab keeps the UUID, adds a new state."""
    with app() as win:
        add_signal(win)
        result = normalize(win, "maximum")
        first_state = output_state(win, win.provenance.ledger.activities[0])
        report = win.signalpanel.processor.recompute_processing(
            result,
            param=sigima.params.NormalizeParam.create(method="amplitude"),
            interactive=False,
        )
        assert report.success
        assert np.array_equal(result.y, AMPLITUDE_Y)
        activity = win.provenance.ledger.activities[-1]
        assert activity["origin"] == "recompute_in_place"
        assert activity["call"]["parameters"] == {"method": "amplitude"}
        state = output_state(win, activity)
        assert state["object_uuid"] == get_uuid(result)
        assert state["state_id"] != first_state["state_id"]


def test_verification_computes_a_separate_candidate() -> None:
    """Verification really computes and never changes the workspace."""
    with app() as win:
        source = add_signal(win)
        result = normalize(win, "maximum")
        (activity,) = win.provenance.ledger.activities
        ledger_before = win.provenance.ledger.to_json()
        objects_before = win.signalpanel.objmodel.get_object_ids()
        execution = win.signalpanel.processor.execution
        computations = execution.computations
        report = win.verify_provenance_activity(activity["activity_id"])
        assert execution.computations == computations + 1
        assert report["verdict"] == "exact"
        assert report["restoration"] == "replayable"
        assert report["eligibility"] == "ready"
        assert report["reference"]["status"] == "available"
        assert report["verification_activity"]["outputs"][0]["locator"] is None
        assert win.provenance.ledger.to_json() == ledger_before
        assert win.signalpanel.objmodel.get_object_ids() == objects_before
        assert np.array_equal(source.y, Y) and np.array_equal(result.y, MAXIMUM_Y)


def test_verification_detects_divergence(monkeypatch) -> None:
    """A deliberately altered candidate gives ``different`` with its errors."""
    with app() as win:
        add_signal(win)
        normalize(win, "maximum")
        (activity,) = win.provenance.ledger.activities
        execution = win.signalpanel.processor.execution
        original = execution.execute_candidate

        def altered(func, source, param):
            candidate = original(func, source, param)
            candidate.y[2] += 0.5
            return candidate

        monkeypatch.setattr(execution, "execute_candidate", altered)
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "different"
        assert report["observed"]["mismatches"] == 1
        assert report["observed"]["max_abs_error"] == 0.5


def test_verification_refusals() -> None:
    """Changed, deleted or ROI-bearing sources are refused before computation."""
    with app() as win:
        source = add_signal(win)
        normalize(win, "maximum")
        (activity,) = win.provenance.ledger.activities
        act_id = activity["activity_id"]
        execution = win.signalpanel.processor.execution
        computations = execution.computations

        source.roi = sigima.objects.create_signal_roi([0.0, 0.5])
        report = win.verify_provenance_activity(act_id)
        assert report["eligibility"] == "unsupported_context"
        source.roi = None

        source.set_xydata(X, Y + 1.0)
        report = win.verify_provenance_activity(act_id)
        assert report["eligibility"] == "input_changed"
        assert report["verdict"] == "not_verified"

        select(win, source)
        win.signalpanel.remove_object(force=True)
        report = win.verify_provenance_activity(act_id)
        assert report["eligibility"] == "missing_input"
        assert execution.computations == computations


def test_chain_scenario() -> None:
    """Opaque step, fan-out, linked states and replays in the middle of a chain."""
    with app() as win:
        s0 = add_signal(win)
        (s1,) = run(
            win, sips.addition_constant, sigima.params.ConstantParam.create(value=1.0)
        )
        assert np.array_equal(s1.y, SHIFTED_Y)
        select(win, s1)
        s2 = normalize(win, "maximum")
        assert np.array_equal(s2.y, SHIFTED_MAXIMUM_Y)
        select(win, s2)
        s3 = normalize(win, "amplitude")
        select(win, s0)
        s4 = normalize(win, "amplitude")
        assert np.array_equal(s4.y, AMPLITUDE_Y)

        ledger = win.provenance.ledger
        ledger.validate()
        a1, a2, a3, a4 = ledger.activities
        assert a1["call"]["operation"] is None
        assert a1["call"]["parameters"] == {"value": 1.0}
        assert a1["implementation"]["python_name"].endswith("addition_constant")
        assert all(a["call"]["operation"] for a in (a2, a3, a4))
        s0_state = a1["call"]["inputs"][0]["binding"]["state_id"]
        assert a4["call"]["inputs"][0]["binding"]["state_id"] == s0_state
        assert input_state(win, a2)["state_id"] == output_state(win, a1)["state_id"]
        assert input_state(win, a3)["state_id"] == output_state(win, a2)["state_id"]
        assert output_state(win, a3)["object_uuid"] == get_uuid(s3)

        opaque = win.verify_provenance_activity(a1["activity_id"])
        assert (opaque["restoration"], opaque["eligibility"]) == (
            "opaque",
            "unsupported_operation",
        )
        for activity in (a4, a2):
            report = win.verify_provenance_activity(activity["activity_id"])
            assert report["verdict"] == "exact"

        select(win, s1)
        win.signalpanel.remove_object(force=True)
        assert win.verify_provenance_activity(a2["activity_id"])["eligibility"] == (
            "missing_input"
        )
        assert win.verify_provenance_activity(a4["activity_id"])["verdict"] == "exact"


def test_capture_failure_keeps_result(monkeypatch) -> None:
    """A failing capture keeps the processing result and is counted."""

    def broken(obj):
        raise RuntimeError("simulated capture failure")

    with app() as win:
        add_signal(win)
        monkeypatch.setattr(provenance_module, "signal_state_facts", broken)
        result = normalize(win, "maximum")
        assert np.array_equal(result.y, MAXIMUM_Y)
        assert win.provenance.ledger.activities == ()
        assert win.provenance.capture_failures == 1


def test_scope_of_capture() -> None:
    """Unqualified signal operations are opaque; images are not captured."""
    with app() as win:
        add_signal(win)
        (smoothed,) = run(
            win, sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
        )
        (activity,) = win.provenance.ledger.activities
        assert activity["call"]["operation"] is None
        assert activity["call"]["parameters"] == {"sigma": 1.0}
        assert output_state(win, activity)["object_uuid"] == get_uuid(smoothed)

        ipanel = win.imagepanel
        ipanel.add_object(create_sincos_image())
        ipanel.objview.select_objects([1])
        ipanel.processor.compute_1_to_1(sipi.normalize, edit=False)
        assert len(win.provenance.ledger.activities) == 1


def test_multiple_and_1_to_n_executions_are_captured() -> None:
    """Executions of the shared 1-to-1 step are captured under one command."""
    with app() as win:
        add_signal(win)
        params = [
            sigima.params.NormalizeParam.create(method="maximum"),
            sigima.params.NormalizeParam.create(method="amplitude"),
        ]
        win.signalpanel.processor.compute_1_to_n(sips.normalize, params)
        activities = win.provenance.ledger.activities
        assert [a["call"]["parameters"]["method"] for a in activities] == [
            "maximum",
            "amplitude",
        ]
        assert activities[0]["command_id"] == activities[1]["command_id"]


def test_capture_with_process_isolation() -> None:
    """Capture and verification work when computations run in a worker."""
    with app(isolation=True) as win:
        add_signal(win)
        result = normalize(win, "maximum")
        assert np.array_equal(result.y, MAXIMUM_Y)
        (activity,) = win.provenance.ledger.activities
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "exact"


def test_reset_starts_a_new_ledger() -> None:
    """Resetting the workspace resets the ledger."""
    with app() as win:
        add_signal(win)
        normalize(win, "maximum")
        workspace_id = win.provenance.ledger.workspace_id
        win.reset_all()
        assert win.provenance.ledger.activities == ()
        assert win.provenance.ledger.workspace_id != workspace_id
