# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal difference: X alignment on the source grid, capture and verification."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest
import sigima.proc.signal as sips
from sigima.objects import SignalObj, create_signal

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import helpers

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    app,
    isolated_param_defaults,
)

HDF5_DIR = osp.join(osp.dirname(osp.dirname(__file__)), "hdf5")


def add(win, x, y, title: str, units=("s", "V")) -> SignalObj:
    """Add a signal to the signal panel."""
    obj = create_signal(
        title, np.array(x, dtype=float), np.array(y, dtype=float), units=units
    )
    win.signalpanel.add_object(obj)
    return obj


def difference(win, source: SignalObj, operand: SignalObj) -> list[SignalObj]:
    """Compute ``source - operand`` (single operand mode); return new objects."""
    panel = win.signalpanel
    panel.objview.select_objects([get_uuid(source)])
    before = set(panel.objmodel.get_object_ids())
    panel.processor.compute_2_to_1(operand, "operand", sips.difference, edit=False)
    return [
        panel.objmodel[uid]
        for uid in panel.objmodel.get_object_ids()
        if uid not in before
    ]


def test_same_size_and_ends_are_not_the_same_grid() -> None:
    """[0, 1, 2] and [0, 0.5, 2] differ: no index-by-index subtraction."""
    with app() as win:
        source = add(win, [0, 1, 2], [0, 1, 2], "source")
        operand = add(win, [0, 0.5, 2], [0, 0.5, 2], "operand")
        (result,) = difference(win, source, operand)
        assert np.array_equal(result.x, [0.0, 1.0, 2.0])
        assert np.array_equal(result.y, [0.0, 0.0, 0.0])


# Asymmetric case where both orders are valid (exact analytical oracles).
A_X, A_Y = [0, 1, 2, 3], [0, 1, 4, 9]
B_X, B_Y = [0, 1.5, 3], [0, 3, 6]
A_MINUS_B = np.array([0.0, -1.0, 0.0, 3.0])
B_MINUS_A = np.array([0.0, 0.5, -3.0])
RULE = {
    "rule": "sigima.signal.x_alignment.source_grid_linear",
    "version": 1,
    "interpolated": True,
}


def roles(win, activity) -> list[tuple[str, str]]:
    """Return the ``(role, object uuid)`` inputs of an activity."""
    states = win.provenance.ledger.states
    return [
        (item["role"], states[item["binding"]["state_id"]]["object_uuid"])
        for item in activity["call"]["inputs"]
    ]


def build_differences(win) -> tuple[SignalObj, ...]:
    """Compute A - B and B - A; return A, B and both results."""
    a = add(win, A_X, A_Y, "A")
    b = add(win, B_X, B_Y, "B")
    (a_b,) = difference(win, a, b)
    (b_a,) = difference(win, b, a)
    return a, b, a_b, b_a


def check_differences(win, edition: str = "desktop") -> None:
    """Check the two recorded differences of a workspace and replay them."""
    ledger = win.provenance.ledger
    ledger.validate()
    act_ab, act_ba = ledger.activities
    objects = {}
    for activity in (act_ab, act_ba):
        assert activity["edition"] == edition
        assert activity["call"]["operation"] == {
            "id": "sigima.signal.difference",
            "contract_version": 1,
        }
        assert activity["call"]["parameters"] == {}
        assert activity["context"]["x_alignment"] == RULE
        for role, uid in roles(win, activity):
            objects.setdefault(role, []).append(uid)
    assert objects["source"] == objects["operand"][::-1]
    for activity, expected in ((act_ab, A_MINUS_B), (act_ba, B_MINUS_A)):
        state = ledger.states[activity["outputs"][0]["state_id"]]
        result = win.find_object_by_uuid(state["object_uuid"])
        assert np.array_equal(result.y, expected)
    computations = win.signalpanel.processor.execution.computations
    for activity in (act_ab, act_ba):
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "exact", report
        assert report["context"]["x_alignment"] == RULE
    assert win.signalpanel.processor.execution.computations == computations + 2


def test_capture_and_verification() -> None:
    """Roles, original inputs and the rule are recorded; replays are exact."""
    with app() as win:
        a, b, a_b, b_a = build_differences(win)
        assert np.array_equal(a_b.x, A_X) and np.array_equal(a_b.y, A_MINUS_B)
        assert np.array_equal(b_a.x, B_X) and np.array_equal(b_a.y, B_MINUS_A)
        for obj, x, y in ((a, A_X, A_Y), (b, B_X, B_Y)):
            assert np.array_equal(obj.x, x) and np.array_equal(obj.y, y)
        act_ab, act_ba = win.provenance.ledger.activities
        assert roles(win, act_ab) == [("source", get_uuid(a)), ("operand", get_uuid(b))]
        assert roles(win, act_ba) == [("source", get_uuid(b)), ("operand", get_uuid(a))]
        check_differences(win)


def test_desktop_round_trip() -> None:
    """Saved and reopened in a fresh window, both differences replay exactly."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "differences.h5")
        with app() as win:
            build_differences(win)
            win.save_h5_workspace(path)
            saved = win.provenance.ledger.to_dict()
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            assert win.provenance.ledger.to_dict() == saved
            check_differences(win)


@pytest.mark.parametrize("edition", ["desktop", "web"])
def test_reference_files(edition: str) -> None:
    """Reference files of both editions reopen and replay exactly on Desktop."""
    path = osp.join(HDF5_DIR, f"provenance_{edition}_difference.h5")
    with app() as win:
        win.load_h5_workspace([path], reset_all=True)
        check_differences(win, edition)


def test_swapped_roles_give_a_different_result() -> None:
    """Replaying with the roles swapped computes the other difference."""
    with app() as win:
        build_differences(win)
        act_ab = win.provenance.ledger.activities[0]
        inputs = act_ab["call"]["inputs"]
        inputs[0]["binding"], inputs[1]["binding"] = (
            inputs[1]["binding"],
            inputs[0]["binding"],
        )
        report = win.verify_provenance_activity(act_ab["activity_id"])
        assert report["eligibility"] == "ready"
        assert report["verdict"] == "different"


def test_verification_refusals() -> None:
    """Missing or changed operands and unknown rules are refused, uncomputed."""
    with app() as win:
        _a, b, _a_b, _b_a = build_differences(win)
        act_ab = win.provenance.ledger.activities[0]
        act_id = act_ab["activity_id"]
        execution = win.signalpanel.processor.execution
        computations = execution.computations

        act_ab["context"]["x_alignment"] = dict(RULE, version=2)
        report = win.verify_provenance_activity(act_id)
        assert report["eligibility"] == "unsupported_context"
        act_ab["context"]["x_alignment"] = dict(RULE, rule="unknown.rule")
        assert win.verify_provenance_activity(act_id)["eligibility"] == (
            "unsupported_context"
        )
        act_ab["context"]["x_alignment"] = dict(RULE)

        b.set_xydata(np.array(B_X, dtype=float), np.array([0.0, 3.0, 7.0]))
        report = win.verify_provenance_activity(act_id)
        assert report["eligibility"] == "input_changed"
        assert report["verdict"] == "not_verified"

        win.signalpanel.objview.select_objects([get_uuid(b)])
        win.signalpanel.remove_object(force=True)
        assert win.verify_provenance_activity(act_id)["eligibility"] == (
            "missing_input"
        )
        assert execution.computations == computations


def test_refused_alignment_creates_nothing() -> None:
    """An uncovered or unit-mismatched operand gives no result, no activity."""
    with app() as win, execenv.context(catcher_test=True):
        source = add(win, [0, 1, 2], [0, 1, 2], "source")
        short = add(win, [0.5, 1, 2], [0, 1, 2], "short")
        millivolts = add(win, [0, 1, 2], [0, 1, 2], "mV", units=("s", "mV"))
        count = len(win.signalpanel.objmodel)
        assert difference(win, source, short) == []
        assert difference(win, source, millivolts) == []
        assert len(win.signalpanel.objmodel) == count
        assert win.provenance.ledger.activities == ()


def test_other_two_input_operations_are_opaque() -> None:
    """Other 2-to-1 operations keep their behaviour and are captured opaque."""
    with app() as win:
        a = add(win, [0, 1, 2], [2, 4, 6], "A")
        b = add(win, [0, 1, 2], [1, 2, 3], "B")
        win.signalpanel.objview.select_objects([get_uuid(a)])
        win.signalpanel.processor.compute_2_to_1(b, "B", sips.division, edit=False)
        (activity,) = win.provenance.ledger.activities
        assert activity["call"]["operation"] is None
        assert roles(win, activity) == [
            ("source", get_uuid(a)),
            ("operand", get_uuid(b)),
        ]
        assert activity["context"]["x_alignment"] is None
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["restoration"] == "opaque"
