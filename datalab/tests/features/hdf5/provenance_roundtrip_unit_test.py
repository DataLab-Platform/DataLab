# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace provenance saved in HDF5 files, reopened in a fresh context and replayed.

The chain workspace (S0 -> S1 opaque -> S2 -> S3, S0 -> S4) is checked for files
written by Desktop and for the reference file written by DataLab-Web.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp
import shutil

import h5py
import numpy as np
import pytest
import sigima.params
import sigima.proc.signal as sips
from datalab_capsule.hdf5 import ProvenanceFormatError
from sigima.objects import SignalObj, create_signal

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import helpers
from datalab.tests.features.common.provenance_unit_test import app

HERE = osp.dirname(__file__)
DESKTOP_FIXTURE = osp.join(HERE, "provenance_desktop_chain.h5")
WEB_FIXTURE = osp.join(HERE, "provenance_web_chain.h5")

X = np.array([0.0, 0.25, 0.5, 0.75])
Y = np.array([-2.0, 0.0, 1.0, 4.0])
# Exact analytical oracles (literals, never derived from Sigima)
S1_Y = np.array([-1.0, 1.0, 2.0, 5.0])
S2_Y = np.array([-1 / 5, 1 / 5, 2 / 5, 1.0])
S4_Y = np.array([0.0, 1 / 3, 0.5, 1.0])
NORMALIZE = {"id": "sigima.signal.normalize", "contract_version": 1}


def _run(win, func, param) -> SignalObj:
    panel = win.signalpanel
    before = set(panel.objmodel.get_object_ids())
    panel.processor.compute_1_to_1(func, param=param, edit=False)
    (uid,) = [u for u in panel.objmodel.get_object_ids() if u not in before]
    return panel.objmodel[uid]


def _select(win, obj) -> None:
    win.signalpanel.objview.select_objects([get_uuid(obj)])


def build_chain(win) -> dict[str, SignalObj]:
    """Run the chain scenario in *win* and return its objects."""
    s0 = create_signal("S0", X.copy(), Y.copy(), units=("s", ""))
    win.signalpanel.add_object(s0)
    _select(win, s0)
    s1 = _run(
        win, sips.addition_constant, sigima.params.ConstantParam.create(value=1.0)
    )
    _select(win, s1)
    s2 = _run(
        win, sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
    )
    _select(win, s2)
    s3 = _run(
        win, sips.normalize, sigima.params.NormalizeParam.create(method="amplitude")
    )
    _select(win, s0)
    s4 = _run(
        win, sips.normalize, sigima.params.NormalizeParam.create(method="amplitude")
    )
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4}


def _output_object(win, activity):
    state = win.provenance.ledger.states[activity["outputs"][0]["state_id"]]
    return win.find_object_by_uuid(state["object_uuid"])


def check_chain(win, edition: str) -> None:
    """Check a reopened chain workspace: facts, data and real replays."""
    ledger = win.provenance.ledger
    ledger.validate()
    a1, a2, a3, a4 = ledger.activities
    assert {a["edition"] for a in ledger.activities} == {edition}
    assert a1["call"]["operation"] is None
    assert a1["call"]["parameters"] == {"value": 1.0}
    assert [a["call"]["operation"] for a in (a2, a3, a4)] == [NORMALIZE] * 3
    assert [a["call"]["parameters"]["method"] for a in (a2, a3, a4)] == [
        "maximum",
        "amplitude",
        "amplitude",
    ]
    s0_state = a1["call"]["inputs"][0]["binding"]["state_id"]
    assert a4["call"]["inputs"][0]["binding"]["state_id"] == s0_state
    for prev, nxt in ((a1, a2), (a2, a3)):
        assert (
            nxt["call"]["inputs"][0]["binding"]["state_id"]
            == prev["outputs"][0]["state_id"]
        )
    assert win.provenance.state_status == {}
    for activity, expected in ((a1, S1_Y), (a2, S2_Y), (a4, S4_Y)):
        obj = _output_object(win, activity)
        assert obj.y.dtype == np.float64 and np.array_equal(obj.y, expected)
        assert np.array_equal(obj.x, X)
    references = {
        a["activity_id"]: _output_object(win, a).y.copy() for a in ledger.activities
    }
    opaque = win.verify_provenance_activity(a1["activity_id"])
    assert (opaque["restoration"], opaque["eligibility"]) == (
        "opaque",
        "unsupported_operation",
    )
    computations = win.signalpanel.processor.execution.computations
    for activity in (a4, a2):
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "exact"
        assert report["restoration"] == "replayable"
        if edition != "desktop":
            assert report["environment"]["match"] == "different"
            assert "edition" in report["environment"]["differences"]
    assert win.signalpanel.processor.execution.computations == computations + 2
    for activity in ledger.activities:
        assert np.array_equal(
            _output_object(win, activity).y, references[activity["activity_id"]]
        )


def _save_chain(path: str) -> dict:
    with app() as win:
        build_chain(win)
        win.save_h5_workspace(path)
        return win.provenance.ledger.to_dict()


def _open(win, path: str) -> None:
    win.load_h5_workspace([path], reset_all=True)


def test_desktop_round_trip() -> None:
    """Desktop -> fresh Desktop: same ledger, exact data, real replays."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        saved = _save_chain(path)
        with app() as win:
            _open(win, path)
            assert win.provenance.file_status == "loaded"
            assert win.provenance.ledger.to_dict() == saved
            assert len(win.provenance.ledger.activities) == 4
            check_chain(win, "desktop")


@pytest.mark.parametrize(
    "path, edition", [(DESKTOP_FIXTURE, "desktop"), (WEB_FIXTURE, "web")]
)
def test_reference_files(path: str, edition: str) -> None:
    """Reference files of both editions reopen and replay on Desktop."""
    with app() as win:
        _open(win, path)
        check_chain(win, edition)


def test_altered_and_missing_objects() -> None:
    """Altered data and a deleted source are refused before any computation."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        with app() as win:
            objs = build_chain(win)
            _select(win, objs["S1"])
            win.signalpanel.remove_object(force=True)
            win.save_h5_workspace(path)
        altered = osp.join(tmpdir, "altered.h5")
        shutil.copy(path, altered)
        with app() as win:
            _open(win, path)
            a1, a2, _a3, a4 = win.provenance.ledger.activities
            s1_state = a1["outputs"][0]["state_id"]
            assert win.provenance.state_status == {s1_state: "unavailable"}
            report = win.verify_provenance_activity(a2["activity_id"])
            assert report["eligibility"] == "missing_input"
            assert win.verify_provenance_activity(a4["activity_id"])["verdict"] == (
                "exact"
            )
            s0_uuid = win.provenance.ledger.states[
                a4["call"]["inputs"][0]["binding"]["state_id"]
            ]["object_uuid"]
        with h5py.File(altered, "r+") as h5file:
            for name, group in h5file["DataLab_Sig"].items():
                for objname in group:
                    meta_uuid = group[objname]["metadata"].attrs.get("__uuid")
                    if isinstance(meta_uuid, bytes):
                        meta_uuid = meta_uuid.decode("utf-8")
                    if meta_uuid == s0_uuid:
                        h5file[f"DataLab_Sig/{name}/{objname}/xydata"][1, 0] = -3.0
        with app() as win:
            _open(win, altered)
            a4 = win.provenance.ledger.activities[3]
            s0_state = a4["call"]["inputs"][0]["binding"]["state_id"]
            assert win.provenance.state_status[s0_state] == "altered"
            computations = win.signalpanel.processor.execution.computations
            report = win.verify_provenance_activity(a4["activity_id"])
            assert report["eligibility"] == "input_changed"
            assert win.signalpanel.processor.execution.computations == computations


def test_invalid_block_is_refused_before_replacement() -> None:
    """An invalid block raises before the current workspace is replaced."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        with h5py.File(path, "r+") as h5file:
            del h5file["DataLab_Provenance/ledger_json"]
            h5file["DataLab_Provenance/ledger_json"] = "{not json"
        with app() as win:
            current = create_signal("current", X.copy(), Y.copy())
            win.signalpanel.add_object(current)
            with pytest.raises(ProvenanceFormatError):
                _open(win, path)
            assert win.signalpanel.objmodel.get_object_ids() == [get_uuid(current)]


def test_file_without_block_and_append() -> None:
    """An old file reports provenance as absent; appending never merges ledgers."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        old = osp.join(tmpdir, "old.h5")
        with app() as win:
            win.signalpanel.add_object(create_signal("S", X.copy(), Y.copy()))
            win.save_h5_workspace(old)
        with h5py.File(old, "r+") as h5file:
            del h5file["DataLab_Provenance"]
        with app() as win:
            _open(win, old)
            assert win.provenance.file_status == "absent"
            assert win.provenance.ledger.activities == ()
            build_chain(win)
            activities = win.provenance.ledger.activities
            win.load_h5_workspace([DESKTOP_FIXTURE], reset_all=False)
            assert win.provenance.ledger.activities == activities
            assert win.provenance.notices


def main() -> None:
    """Regenerate the Desktop reference file (run from the repository root)."""
    if osp.exists(DESKTOP_FIXTURE):
        raise SystemExit(f"Remove {DESKTOP_FIXTURE} first")
    execenv.unattended = True
    _save_chain(DESKTOP_FIXTURE)
    print(f"Written {DESKTOP_FIXTURE}")


if __name__ == "__main__":
    main()
