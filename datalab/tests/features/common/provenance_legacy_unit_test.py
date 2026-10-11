# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Legacy workspaces: processing records saved without provenance (Desktop).

The reference file was saved by the released DataLab 1.3.0 (Sigima 1.2.0) with
``scripts/provenance_legacy_fixture.py``: S0 -> normalize -> S1 -> moving
average -> S2; average of S0 and T (T then deleted) -> S3; statistics on S1;
image I with a ROI -> Gaussian filter -> J; centroid on I.
"""

from __future__ import annotations

import os.path as osp

import sigima.params
import sigima.proc.signal as sips

import datalab
from datalab.objectmodel import get_uuid
from datalab.tests import helpers

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    app,
    isolated_param_defaults,
)

HDF5_DIR = osp.join(osp.dirname(osp.dirname(__file__)), "hdf5")
LEGACY = osp.join(HDF5_DIR, "provenance_legacy_1.3.0.h5")
VERSION_ISSUES = [] if datalab.__version__ == "1.3.0" else ["version_mismatch"]
EXPECTED = [
    # (holder title prefix, record key, operation, input titles, issues)
    ("normalize", "__processing_parameters", "normalize", ["S0"], []),
    ("normalize", "__analysis_parameters", "stats", None, ["result_not_identified"]),
    ("moving_average", "__processing_parameters", "moving_average", None, []),
    ("\u00b5(s001", "__processing_parameters", "average", None, ["unavailable_input"]),
    ("I", "__analysis_parameters", "centroid", ["I"], ["result_not_identified"]),
    ("gaussian_filter", "__processing_parameters", "gaussian_filter", ["I"], []),
]


def _titles(win, uuids: list[str]) -> list[str | None]:
    objs = [win.find_object_by_uuid(uid) for uid in uuids]
    return [None if obj is None else obj.title for obj in objs]


def check_legacy(win) -> None:
    """Check the activities imported from the reference legacy workspace."""
    provenance = win.provenance
    provenance.ledger.validate()
    summary = provenance.legacy_summary()
    assert summary["activities"] == len(EXPECTED)
    for item, (title, key, operation, inputs, issues) in zip(
        summary["items"], EXPECTED
    ):
        holder = win.find_object_by_uuid(item["object_uuid"])
        assert holder.title.startswith(title), (holder.title, title)
        assert (item["key"], item["operation"]) == (key, operation)
        assert item["issues"] == issues + VERSION_ISSUES
        if inputs is not None:
            assert _titles(win, item["inputs"]) == inputs
    average = summary["items"][3]
    assert _titles(win, average["inputs"]) == ["S0", None]
    for activity in provenance.ledger.activities:
        assert activity["origin"] == "legacy_metadata"
        assert activity["call"]["operation"] is None
        assert activity["call"]["parameters"] is None
        assert activity["environment_id"] is None and activity["edition"] is None
    normalize = provenance.ledger.activities[0]
    assert normalize["legacy"]["parameters"] == {
        "class": "sigima.proc.base.NormalizeParam",
        "values": {"method": "MAXIMUM"},
    }
    # Each processed object's current state is the output of its record.
    for activity in provenance.ledger.activities:
        for output in activity["outputs"]:
            state = provenance.ledger.states[output["state_id"]]
            obj = win.find_object_by_uuid(state["object_uuid"])
            assert provenance.observe(obj) == state["state_id"]


def test_legacy_workspace_is_imported() -> None:
    """Records become legacy activities with explicit unknowns and issues."""
    with app() as win:
        win.load_h5_workspace([LEGACY], reset_all=True)
        assert win.provenance.file_status == "absent"
        check_legacy(win)
        assert any("legacy" in notice for notice in win.provenance.notices)


def test_legacy_activities_are_never_replayed() -> None:
    """Verification refuses a legacy activity before any computation."""
    with app() as win:
        win.load_h5_workspace([LEGACY], reset_all=True)
        processor = win.signalpanel.processor
        computations = processor.execution.computations
        for activity in win.provenance.ledger.activities:
            report = win.verify_provenance_activity(activity["activity_id"])
            assert report["restoration"] == "opaque"
            assert report["eligibility"] == "unsupported_operation"
            assert report["environment"]["match"] == "unknown"
            assert report["verdict"] == "not_verified"
        assert processor.execution.computations == computations


def test_new_processing_continues_the_legacy_chain() -> None:
    """A new activity's input is the legacy output state of its source."""
    with app() as win:
        win.load_h5_workspace([LEGACY], reset_all=True)
        s2 = next(o for o in win.signalpanel.objmodel if o.title.startswith("moving"))
        legacy_state = win.provenance.ledger.latest_state(get_uuid(s2))["state_id"]
        win.signalpanel.objview.select_objects([get_uuid(s2)])
        win.signalpanel.processor.compute_1_to_1(
            sips.normalize, sigima.params.NormalizeParam.create(), edit=False
        )
        activity = win.provenance.ledger.activities[-1]
        assert activity["origin"] == "ordinary"
        assert activity["call"]["inputs"][0]["binding"]["state_id"] == legacy_state
        win.provenance.ledger.validate()


def test_round_trip_keeps_legacy_activities() -> None:
    """Saved with its ledger, the workspace reopens without a second import."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "imported.h5")
        with app() as win:
            win.load_h5_workspace([LEGACY], reset_all=True)
            win.save_h5_workspace(path)
            saved = win.provenance.ledger.to_dict()
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            assert win.provenance.file_status == "loaded"
            assert win.provenance.ledger.to_dict() == saved
            assert win.provenance.state_status == {}
            check_legacy(win)


def test_appending_a_legacy_workspace_imports_nothing() -> None:
    """Appending never merges provenance, legacy records included."""
    with app() as win:
        win.load_h5_workspace([LEGACY], reset_all=False)
        assert win.provenance.ledger.activities == ()
        assert win.provenance.legacy_summary()["activities"] == 0
