# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Consultation of the provenance of a result (Desktop).

The History tab of the object properties shows the recorded production of the
selected object, its inputs and their ancestors, and lets the user check the data
and verify the computation; none of these operations changes the workspace or
the ledger. The scenario of reference: S0 -> Gaussian filter (sigma = 2) -> S1
-> normalization -> S2, then S1 recomputed with sigma = 5.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.objectmodel import get_short_id, get_uuid
from datalab.tests import helpers
from datalab.tests.features.common.provenance_processing_unit_test import (
    add_signal,
    compute,
)

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    Y,
    app,
    isolated_param_defaults,
)


def build_chain(win) -> dict:
    """Build the sigma scenario and return its objects."""
    s0 = add_signal(win, Y, "S0")
    panel = win.signalpanel
    (s1,) = compute(
        panel,
        [s0],
        "compute_1_to_1",
        sips.gaussian_filter,
        param=sigima.params.GaussianParam.create(sigma=2.0),
        edit=False,
    )
    (s2,) = compute(
        panel,
        [s1],
        "compute_1_to_1",
        sips.normalize,
        param=sigima.params.NormalizeParam.create(method="maximum"),
        edit=False,
    )
    report = panel.processor.recompute_processing(
        s1, param=sigima.params.GaussianParam.create(sigma=5.0), interactive=False
    )
    assert report.success, report.message
    return {"s0": s0, "s1": s1, "s2": s2}


def snapshot(win) -> tuple:
    """Return what consultation, check and verification must never change."""
    objects = [
        (get_uuid(obj), obj.title, obj.xydata.copy())
        for obj in win.signalpanel.objmodel
    ]
    return win.provenance.ledger.to_json(), objects, len(win.historypanel)


def same(before: tuple, after: tuple) -> bool:
    """Return True if two snapshots are equal."""
    ledger_a, objects_a, history_a = before
    ledger_b, objects_b, history_b = after
    return (
        ledger_a == ledger_b
        and history_a == history_b
        and len(objects_a) == len(objects_b)
        and all(
            ua == ub and ta == tb and np.array_equal(da, db)
            for (ua, ta, da), (ub, tb, db) in zip(objects_a, objects_b)
        )
    )


def view_of(win, obj):
    """Select *obj* and return the provenance view of its panel."""
    panel = win.signalpanel
    panel.objview.set_current_object(obj)
    QW.QApplication.processEvents()
    objprop = panel.objprop
    assert objprop.history_stack.currentWidget() is objprop.provenance_widget
    return objprop.provenance_widget


def check_sigma_scenario(win, objs: dict) -> None:
    """S2 describes the sigma = 2 state actually used, whatever S1 became."""
    before = snapshot(win)
    view = view_of(win, objs["s2"])
    description = view.description
    assert description["anchor"]["content"] == "unchecked"
    assert "Data to be checked" in view.summary.text()
    production = description["production"]
    assert production["operation"]["id"] == "sigima.signal.normalize"
    (source,) = production["inputs"]
    assert source["object_uuid"] == get_uuid(objs["s1"])
    ancestor = win.provenance.describe_state(source["state_id"])
    assert ancestor["production"]["parameters"]["sigma"] == 2.0
    assert ancestor["anchor"]["latest"] is False
    # The input item is expanded on demand to the sigma = 2 production.
    item = view.tree.findItems("Inputs", QC.Qt.MatchRecursive, 0)[0].child(0)
    item.setExpanded(True)
    texts = [item.child(i).text(0) for i in range(item.childCount())]
    assert any("gaussian" in text.lower() for text in texts), texts
    view.check_data()
    checked = view.description["production"]
    assert view.description["anchor"]["content"] == "matches"
    assert checked["inputs"][0]["data"] == "changed"
    report = view.verify()
    assert report["eligibility"] == "input_changed"
    assert not view.report.isHidden()
    assert same(before, snapshot(win))


def test_sigma_scenario_is_readable() -> None:
    """Consultation, check and verification describe sigma = 2, change nothing."""
    with app(history=True) as win:
        objs = build_chain(win)
        check_sigma_scenario(win, objs)


def test_sigma_scenario_after_reopening_and_deletion() -> None:
    """The same explanation after reopening; a deleted input is never replaced."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "sigma.h5")
        with app() as win:
            uuids = {name: get_uuid(obj) for name, obj in build_chain(win).items()}
            win.save_h5_workspace(path)
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            objs = {name: win.find_object_by_uuid(uid) for name, uid in uuids.items()}
            check_sigma_scenario(win, objs)
            panel = win.signalpanel
            panel.objview.set_current_object(objs["s1"])
            panel.remove_object(force=True)
            view = view_of(win, objs["s2"])
            view.check_data()
            (source,) = view.description["production"]["inputs"]
            assert source["object_present"] is False
            assert source["data"] == "missing"
            ancestor = win.provenance.describe_state(source["state_id"])
            assert ancestor["production"]["parameters"]["sigma"] == 2.0


def test_modified_output_and_presentation_changes() -> None:
    """A changed result is not attributed; a new title is not a change."""
    with app() as win:
        objs = build_chain(win)
        s2 = objs["s2"]
        s2.title = "Renamed"
        view = view_of(win, s2)
        view.check_data()
        assert view.description["anchor"]["content"] == "matches"
        s2.y = s2.y + 1.0
        win.signalpanel.SIG_OBJECT_MODIFIED.emit()
        QW.QApplication.processEvents()
        assert view.description["anchor"]["content"] == "unchecked"
        view.check_data()
        assert view.description["anchor"]["content"] == "differs"
        assert "not attributed" in view.summary.text()
        assert view.description["production"]["outputs"][0]["status"] == "altered"


def test_stale_results_are_never_shown() -> None:
    """A result arriving after the selection changed is dropped."""
    with app() as win:
        objs = build_chain(win)
        view = view_of(win, objs["s2"])
        provenance = win.provenance
        original = provenance.describe_object

        def describe_then_select(object_uuid, check_data=False):
            result = original(object_uuid, check_data)
            if check_data:
                view.set_object(objs["s1"])
            return result

        provenance.describe_object = describe_then_select
        try:
            view.check_data()
        finally:
            del provenance.describe_object
        assert view.description["object"]["object_uuid"] == get_uuid(objs["s1"])
        assert view.description["anchor"]["content"] == "unchecked"

        view = view_of(win, objs["s2"])
        verify = win.verify_provenance_activity

        def verify_then_select(activity_id):
            report = verify(activity_id)
            view.set_object(objs["s0"])
            return report

        win.verify_provenance_activity = verify_then_select
        try:
            assert view.verify() is None
        finally:
            del win.verify_provenance_activity
        assert view.report.isHidden()


def test_object_without_recorded_production_keeps_its_history() -> None:
    """Objects without recorded production keep the metadata history."""
    with app() as win:
        s0 = add_signal(win, Y, "S0")
        panel = win.signalpanel
        panel.objview.set_current_object(s0)
        objprop = panel.objprop
        assert objprop.history_stack.currentWidget() is objprop.processing_history
        assert not objprop.provenance_widget.has_content()


def test_history_session_deletion_keeps_the_provenance() -> None:
    """Deleting the History session leaves the provenance consultable."""
    with app(history=True) as win:
        win.historypanel.toggle_record_mode(True)
        objs = build_chain(win)
        win.historypanel.toggle_record_mode(False)
        win.historypanel.remove_all_objects()
        view = view_of(win, objs["s2"])
        assert view.description["production"]["operation"]["id"] == (
            "sigima.signal.normalize"
        )
        s3 = compute(
            win.signalpanel,
            [objs["s2"]],
            "compute_1_to_1",
            sips.normalize,
            param=sigima.params.NormalizeParam.create(method="amplitude"),
            edit=False,
        )[0]
        assert win.provenance.describe_object(get_uuid(s3))["production"] is not None


def test_view_survives_metadata_deletion() -> None:
    """Object numbers are briefly absent while metadata are deleted."""
    with app() as win:
        objs = build_chain(win)
        view = view_of(win, objs["s2"])
        win.signalpanel.delete_metadata(keep_roi=True)
        for _index in range(3):
            QW.QApplication.processEvents()
        assert view.description["object"]["object_uuid"] == get_uuid(objs["s2"])
        (results,) = view.tree.findItems("Results", QC.Qt.MatchRecursive, 0)
        assert get_short_id(objs["s2"]) in results.child(0).text(0)
