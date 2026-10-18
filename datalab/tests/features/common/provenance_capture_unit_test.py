# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Provenance capture of images, analyses, n-to-1 operations, ROI and uncertainty."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp
import sys

import h5py
import numpy as np
import pytest
import sigima.config
import sigima.objects
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from sigima.objects import create_image, create_signal

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import helpers

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    X,
    Y,
    app,
    input_state,
    isolated_param_defaults,
    output_state,
)


def add_signal(win, y=Y, title: str = "S"):
    """Add a signal (X in s) and return it."""
    obj = create_signal(title, X.copy(), np.array(y, dtype=float), units=("s", ""))
    win.signalpanel.add_object(obj)
    return obj


def add_image(win, title: str = "I"):
    """Add a 3x4 image and return it."""
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    obj = create_image(title, data, units=("mm", "mm", "counts"))
    win.imagepanel.add_object(obj)
    return obj


def states_of(win, activity) -> list[dict]:
    """Return the input states of an activity, in role order."""
    states = win.provenance.ledger.states
    return [states[i["binding"]["state_id"]] for i in activity["call"]["inputs"]]


def test_image_processing_is_captured() -> None:
    """An image 1-to-1 records image states: shape, units, image fingerprint."""
    with app() as win:
        image = add_image(win)
        win.imagepanel.objview.select_objects([get_uuid(image)])
        win.imagepanel.processor.compute_1_to_1(
            sipi.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
        )
        (activity,) = win.provenance.ledger.activities
        source = input_state(win, activity)
        assert source["kind"] == "image" and source["shape"] == [3, 4]
        assert source["units"] == {"x": "mm", "y": "mm", "z": "counts"}
        assert source["fingerprint"]["scheme"] == "datalab-image-v1"
        assert output_state(win, activity)["kind"] == "image"
        assert activity["call"]["parameters"]["sigma"] == 1.0
        win.provenance.ledger.validate()


def test_n_to_1_records_every_source_in_order() -> None:
    """Average of three signals: three ``sources`` inputs, in selection order."""
    with app() as win:
        signals = [add_signal(win, Y * k, f"S{k}") for k in (1, 2, 3)]
        win.signalpanel.objview.select_objects([get_uuid(s) for s in signals])
        win.signalpanel.processor.compute_n_to_1(sips.average, edit=False)
        (activity,) = win.provenance.ledger.activities
        assert [i["role"] for i in activity["call"]["inputs"]] == ["sources"] * 3
        assert [s["object_uuid"] for s in states_of(win, activity)] == [
            get_uuid(s) for s in signals
        ]
        result = win.find_object_by_uuid(output_state(win, activity)["object_uuid"])
        assert np.array_equal(result.y, Y * 2)
        assert activity["limits"] == []


def test_list_alignment_is_recorded() -> None:
    """Inputs aligned before an n-to-1 call are recorded as originals, with the
    alignment policy declared by the operation and its outcome."""
    with app() as win:
        first = add_signal(win, title="first")
        second = create_signal(
            "second", np.array([0.0, 0.5, 0.6, 0.75]), Y.copy(), units=("s", "")
        )
        win.signalpanel.add_object(second)
        win.signalpanel.objview.select_objects([get_uuid(first), get_uuid(second)])
        win.signalpanel.processor.compute_n_to_1(sips.average, edit=False)
        (activity,) = win.provenance.ledger.activities
        assert activity["limits"] == []
        assert activity["context"]["x_alignment"] == {
            "rule": "sigima.signal.x_alignment.smallest_grid_linear_hold",
            "version": 1,
            "interpolated": [1],
            "target": 0,
        }
        recorded = states_of(win, activity)[1]
        assert recorded["object_uuid"] == get_uuid(second)
        assert recorded["state_id"] == win.provenance.observe(second)


def test_analyses_record_artifacts() -> None:
    """Signal and image analyses record their result as an artifact output."""
    with app() as win:
        signal = add_signal(win)
        win.signalpanel.objview.select_objects([get_uuid(signal)])
        win.signalpanel.processor.run_feature("stats")
        image = add_image(win)
        win.imagepanel.objview.select_objects([get_uuid(image)])
        win.imagepanel.processor.run_feature("centroid")
        stats, centroid = win.provenance.ledger.activities
        for activity, obj, kind in (
            (stats, signal, "table"),
            (centroid, image, "geometry"),
        ):
            (output,) = activity["outputs"]
            artifact = output["artifact"]
            assert artifact["kind"] == kind
            assert artifact["object_uuid"] == get_uuid(obj)
            assert artifact["key"] in obj.metadata
            assert states_of(win, activity)[0]["object_uuid"] == get_uuid(obj)
        win.provenance.ledger.validate()


def test_recomputed_analyses_are_recorded_as_such() -> None:
    """Analyses refreshed on demand are recorded as recomputations in place."""
    with app() as win:
        signal = add_signal(win)
        win.signalpanel.objview.select_objects([get_uuid(signal)])
        win.signalpanel.processor.run_feature("stats")
        win.signalpanel.recompute_selected()
        first, second = win.provenance.ledger.activities
        assert (first["origin"], second["origin"]) == ("ordinary", "recompute_in_place")
        win.provenance.ledger.validate()


def test_analyses_are_verified() -> None:
    """Analyses are recomputed and compared; a normal absence of result too."""
    with app() as win:
        signal = add_signal(win)
        win.signalpanel.objview.select_objects([get_uuid(signal)])
        win.signalpanel.processor.run_feature("stats")
        (stats,) = win.provenance.ledger.activities
        report = win.verify_provenance_activity(stats["activity_id"])
        assert report["verdict"] == "exact"
        (output,) = report["outputs"]
        assert output["recorded"] == "artifact"
        assert output["observed"]["structure_match"] is True
        # A ramp has no local maximum: the detection normally finds nothing.
        image = create_image("ramp", np.add.outer(np.arange(16.0), np.arange(16.0)))
        win.imagepanel.add_object(image)
        win.imagepanel.objview.select_objects([get_uuid(image)])
        win.imagepanel.processor.compute_1_to_0(
            sipi.peak_detection, sigima.params.Peak2DDetectionParam(), edit=False
        )
        empty = win.provenance.ledger.activities[-1]
        assert empty["call"]["operation"]["id"] == "sigima.image.peak_detection"
        assert empty["outputs"] == []
        report = win.verify_provenance_activity(empty["activity_id"])
        assert report["verdict"] == "exact"
        assert report["outputs"][0]["recorded"] == "none"
        win.provenance.ledger.validate()


def test_candidate_errors_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A computation error during a verification is reported, not raised."""
    with app() as win:
        signal = add_signal(win)
        win.signalpanel.objview.select_objects([get_uuid(signal)])
        win.signalpanel.processor.compute_1_to_1(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
        )
        (activity,) = win.provenance.ledger.activities

        def failing(*_args, **_kwargs):
            raise RuntimeError("computation failed")

        execution = win.signalpanel.processor.execution
        monkeypatch.setattr(execution, "_exec_func", failing)
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "not_verified"
        assert "computation failed" in report["reason"]


def run_families(win) -> None:
    """Run one operation of each family on fresh objects (none is opaque)."""
    spanel, ipanel = win.signalpanel, win.imagepanel
    signals = [add_signal(win, Y * k, f"S{k}") for k in (1, 2, 3)]
    uuids = [get_uuid(s) for s in signals]
    spanel.objview.select_objects(uuids)
    spanel.processor.compute_n_to_1(sips.average, edit=False)
    spanel.objview.select_objects(uuids)
    spanel.processor.compute_n_to_1(
        sips.signals_to_image, sigima.params.SignalsToImageParam(), edit=False
    )
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.compute_1_to_1(
        sips.fft, sigima.params.FFTParam.create(shift=True), edit=False
    )
    rois = [
        sigima.objects.ROI1DParam.create(xmin=0.0, xmax=0.5),
        sigima.objects.ROI1DParam.create(xmin=0.25, xmax=0.75),
    ]
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.compute_1_to_1(sips.extract_rois, rois, edit=False)
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.compute_1_to_n(sips.extract_roi, rois, edit=False)
    spanel.objview.select_objects(uuids[:1])
    noise = sigima.objects.NormalDistributionParam.create(seed=7, sigma=0.5)
    spanel.processor.compute_1_to_1(sips.add_gaussian_noise, noise, edit=False)
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.run_feature("stats")
    image = add_image(win)
    ipanel.objview.select_objects([get_uuid(image)])
    profile = sigima.params.LineProfileParam.create(direction="horizontal", row=1)
    ipanel.processor.compute_1_to_1(sipi.line_profile, profile, edit=False)
    ipanel.objview.select_objects([get_uuid(image)])
    ipanel.processor.run_feature("centroid")
    # Registered wrapped (analysis results carried in metadata are transformed)
    ipanel.objview.select_objects([get_uuid(image)])
    ipanel.processor.run_feature("fliph")
    kernel = add_image(win, "K")
    ipanel.objview.select_objects([get_uuid(image)])
    with sigima.config.options.auto_normalize_kernel.context(True):
        ipanel.processor.compute_2_to_1(kernel, "K", sipi.convolution, edit=False)


FAMILIES = [
    "sigima.signal.average",
    "sigima.signal.signals_to_image",
    "sigima.signal.fft",
    "sigima.signal.extract_rois",
    "sigima.signal.extract_roi",
    "sigima.signal.extract_roi",
    "sigima.signal.add_gaussian_noise",
    "sigima.signal.stats",
    "sigima.image.line_profile",
    "sigima.image.centroid",
    "sigima.image.fliph",
    "sigima.image.convolution",
]


def check_families(win) -> None:
    """Every family is identified and replayed exactly, every output compared."""
    activities = win.provenance.ledger.activities
    assert [a["call"]["operation"]["id"] for a in activities] == FAMILIES
    roi_a, roi_b = activities[4:6]
    assert roi_a["command_id"] is not None
    assert roi_a["command_id"] == roi_b["command_id"]
    assert activities[2]["call"]["parameters"] == {"shift": True}
    assert activities[-1]["context"]["options"] == {"auto_normalize_kernel": True}
    for activity in activities:
        assert activity["limits"] == [], activity
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "exact", report
        assert [o["recorded"] for o in report["outputs"]] == [
            "artifact" if "artifact" in o else "state" for o in activity["outputs"]
        ]


def test_operation_families_replay() -> None:
    """Each family is recorded, saved, reopened and replayed with its outputs."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "families.h5")
        with app() as win:
            run_families(win)
            # The recorded option is applied, whatever the current setting.
            with sigima.config.options.auto_normalize_kernel.context(False):
                check_families(win)
            _reopen(win, path)
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            check_families(win)


def test_metadata_read_by_evaluate_fit() -> None:
    """Fit parameters read by ``evaluate_fit`` are recorded and checked."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "fit.h5")
        with app() as win:
            spanel = win.signalpanel
            data = add_signal(win, X * 2.0 + 1.0, "line")
            spanel.objview.select_objects([get_uuid(data)])
            spanel.processor.compute_1_to_1(sips.linear_fit, edit=False)
            fit = win.find_object_by_uuid(
                output_state(win, win.provenance.ledger.activities[-1])["object_uuid"]
            )
            assert "fit_params" in fit.metadata
            grid = create_signal("grid", np.linspace(0.0, 2.0, 9), np.zeros(9))
            spanel.add_object(grid)
            spanel.objview.select_objects([get_uuid(fit)])
            spanel.processor.compute_2_to_1(
                grid,
                "grid",
                sips.evaluate_fit,
                edit=False,
                skip_xarray_compat=True,
                pre_execute_hook=spanel.processor.prepare_fit_evaluation,
            )
            activity = win.provenance.ledger.activities[-1]
            assert activity["call"]["operation"]["id"] == "sigima.signal.evaluate_fit"
            (read,) = activity["context"]["metadata"]
            assert (read["role"], read["key"]) == ("source", "fit_params")
            activity_id = activity["activity_id"]
            assert win.verify_provenance_activity(activity_id)["verdict"] == "exact"
            win.save_h5_workspace(path)
            original = dict(fit.metadata["fit_params"])
            fit.metadata["fit_params"] = {**original, "a": original["a"] + 1.0}
            report = win.verify_provenance_activity(activity_id)
            assert report["eligibility"] == "input_changed"
            assert "fit_params" in report["reason"]
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            report = win.verify_provenance_activity(activity_id)
            assert report["verdict"] == "exact", report


def test_roi_and_uncertainty_are_recorded() -> None:
    """ROI definitions and uncertainty rows are part of the recorded states."""
    with app() as win:
        signal = add_signal(win)
        signal.dy = np.full(4, 0.1)
        signal.roi = sigima.objects.create_signal_roi([0.0, 0.5])
        win.signalpanel.objview.select_objects([get_uuid(signal)])
        win.signalpanel.processor.compute_1_to_1(
            sips.normalize, sigima.params.NormalizeParam.create(), edit=False
        )
        (activity,) = win.provenance.ledger.activities
        source = input_state(win, activity)
        assert source["rows"] == ["x", "y", "dy"] and source["limits"] == []
        assert source["fingerprint"] is not None
        assert source["roi"]["definition"]["single_rois"][0]["coords"] == [0.0, 0.5]
        # ROI and uncertainty no longer restrict the qualified normalisation.
        assert activity["call"]["operation"] == {
            "id": "sigima.signal.normalize",
            "contract_version": 1,
        }
        report = win.verify_provenance_activity(activity["activity_id"])
        assert (report["eligibility"], report["verdict"]) == ("ready", "exact")
        image = add_image(win)
        image.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 2, 2])
        state = win.provenance.ledger.states[win.provenance.observe(image)]
        assert state["roi"]["definition"]["single_rois"][0]["coords"] == [0, 0, 2, 2]


def test_round_trip_with_images_and_analyses() -> None:
    """Images, ROI, uncertainty and artifacts survive a save and a reopening."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "capture.h5")
        with app() as win:
            image = add_image(win)
            image.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 2, 2])
            win.imagepanel.objview.select_objects([get_uuid(image)])
            win.imagepanel.processor.run_feature("centroid")
            signal = add_signal(win)
            signal.dy = np.full(4, 0.1)
            win.signalpanel.objview.select_objects([get_uuid(signal)])
            win.signalpanel.processor.compute_1_to_1(
                sips.gaussian_filter,
                sigima.params.GaussianParam.create(sigma=1.0),
                edit=False,
            )
            win.save_h5_workspace(path)
            saved = win.provenance.ledger.to_dict()
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            assert win.provenance.ledger.to_dict() == saved
            assert win.provenance.state_status == {}
            win.provenance.ledger.validate()


def _node_of(h5file, object_uuid: str):
    """Return the HDF5 group of the object with *object_uuid*."""
    for panel in ("DataLab_Sig", "DataLab_Ima"):
        for group in h5file.get(panel, {}).values():
            for obj in group.values():
                if obj["metadata"].attrs["__uuid"] == object_uuid:
                    return obj
    raise KeyError(object_uuid)


def _reopen(win, path: str, edit=None):
    """Save *win*'s workspace, optionally edit the file, and return its path."""
    win.save_h5_workspace(path)
    if edit is not None:
        with h5py.File(path, "r+") as h5file:
            edit(h5file)
    return path


def _filter_image_with_roi(win, coords=(0, 0, 2, 2)):
    """Filter an image holding a ROI; return it and the recorded activity."""
    image = add_image(win)
    image.roi = sigima.objects.create_image_roi("rectangle", list(coords))
    win.imagepanel.objview.select_objects([get_uuid(image)])
    win.imagepanel.processor.compute_1_to_1(
        sipi.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
    )
    return image, win.provenance.ledger.activities[-1]


def test_roi_only_change_is_a_new_state() -> None:
    """Same pixels, different ROI: a new state is recorded, not reused."""
    with app() as win:
        image, first = _filter_image_with_roi(win)
        image.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 3, 2])
        win.imagepanel.objview.select_objects([get_uuid(image)])
        win.imagepanel.processor.compute_1_to_1(
            sipi.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
        )
        before = input_state(win, first)
        after = input_state(win, win.provenance.ledger.activities[-1])
        assert before["state_id"] != after["state_id"]
        assert before["fingerprint"] == after["fingerprint"]
        assert before["roi"]["digest"] != after["roi"]["digest"]


FILE_EDITS = {
    "roi": lambda n: n["metadata/_roi_/single_rois/__seq0/coords"].__setitem__(2, 3.0),
    "roi_removed": lambda n: n["metadata"].__delitem__("_roi_"),
    "x0": lambda n: n.attrs.__setitem__("x0", 0.5),
    "dy": lambda n: n.attrs.__setitem__("dy", 2.0),
    "zunit": lambda n: n.attrs.__setitem__("zunit", "V"),
}


@pytest.mark.parametrize("name", list(FILE_EDITS))
def test_roi_calibration_or_unit_change_in_file_is_altered(name: str) -> None:
    """Unchanged pixels with another ROI, calibration or unit are not intact."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        intact, edited = osp.join(tmpdir, "intact.h5"), osp.join(tmpdir, "edited.h5")
        with app() as win:
            image, activity = _filter_image_with_roi(win)
            uid = get_uuid(image)
            source = activity["call"]["inputs"][0]["binding"]["state_id"]
            _reopen(win, intact)
            _reopen(win, edited, lambda f: FILE_EDITS[name](_node_of(f, uid)))
        with app() as win:
            win.load_h5_workspace([intact], reset_all=True)
            assert win.provenance.state_status == {}
            assert win.provenance.observe(win.find_object_by_uuid(uid)) == source
        with app() as win:
            win.load_h5_workspace([edited], reset_all=True)
            assert win.provenance.state_status[source] == "altered"
            # The reopened object is not taken for the recorded state either.
            assert win.provenance.observe(win.find_object_by_uuid(uid)) != source


def test_replaced_analysis_result_is_not_attributed() -> None:
    """Two analyses replacing the same result: the first is reported replaced."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path, edited = osp.join(tmpdir, "w.h5"), osp.join(tmpdir, "edited.h5")
        with app() as win:
            signal = add_signal(win)
            uid = get_uuid(signal)
            win.signalpanel.objview.select_objects([uid])
            win.signalpanel.processor.run_feature("stats")
            signal.set_xydata(X, Y * 2.0)
            win.signalpanel.processor.run_feature("stats")
            first, second = win.provenance.ledger.activities
            artifacts = [a["outputs"][0]["artifact"] for a in (first, second)]
            assert artifacts[0]["key"] == artifacts[1]["key"]
            assert artifacts[0]["digest"] != artifacts[1]["digest"]
            expected = [["replaced"], ["available"]]
            statuses = [
                [
                    s["status"]
                    for s in win.provenance.artifact_statuses(a["activity_id"])
                ]
                for a in (first, second)
            ]
            assert statuses == expected
            _reopen(win, path)
            key = artifacts[0]["key"]
            _reopen(
                win, edited, lambda f: _node_of(f, uid)["metadata"].__delitem__(key)
            )
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            statuses = [
                [
                    s["status"]
                    for s in win.provenance.artifact_statuses(a["activity_id"])
                ]
                for a in win.provenance.ledger.activities
            ]
            assert statuses == expected
        with app() as win:
            win.load_h5_workspace([edited], reset_all=True)
            for activity in win.provenance.ledger.activities:
                (status,) = win.provenance.artifact_statuses(activity["activity_id"])
                assert status["status"] == "missing"


# -- Cross-edition reference files ----------------------------------------
#
# The same scenario is written by each edition: an image with a rectangular ROI
# (x0=0, y0=0, dx=2, dy=2) analysed (centroid) then filtered (Gaussian, sigma=1),
# and a signal analysed (statistics). Desktop writes ``provenance_desktop_
# capture.h5`` (run this module as a script); DataLab-Web writes ``web_capture.h5``
# from Pyodide (Playwright, ``DLW_WRITE_PROVENANCE_FIXTURE=1``).

HDF5_DIR = osp.join(osp.dirname(osp.dirname(__file__)), "hdf5")
CAPTURE_SHAPE = [
    ("image", None, "geometry"),
    ("image", "image", None),
    ("signal", None, "table"),
]


def build_capture(win) -> None:
    """Run the cross-edition capture scenario."""
    image = add_image(win)
    image.roi = sigima.objects.create_image_roi("rectangle", [0, 0, 2, 2])
    win.imagepanel.objview.select_objects([get_uuid(image)])
    win.imagepanel.processor.run_feature("centroid")
    win.imagepanel.processor.compute_1_to_1(
        sipi.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0), edit=False
    )
    signal = add_signal(win)
    win.signalpanel.objview.select_objects([get_uuid(signal)])
    win.signalpanel.processor.run_feature("stats")


def check_capture(win, edition: str) -> None:
    """Check a reopened capture workspace written by *edition*."""
    provenance = win.provenance
    ledger = provenance.ledger
    ledger.validate()
    assert provenance.state_status == {}
    shape = []
    for activity in ledger.activities:
        assert activity["edition"] == edition
        (item,) = activity["call"]["inputs"]
        (output,) = activity["outputs"]
        source = ledger.states[item["binding"]["state_id"]]
        result = ledger.states[output["state_id"]] if "state_id" in output else None
        artifact = output.get("artifact")
        shape.append(
            (source["kind"], result and result["kind"], artifact and artifact["kind"])
        )
        if source["kind"] == "image":
            (roi,) = source["roi"]["definition"]["single_rois"]
            assert roi["coords"] == [0, 0, 2, 2]
        if artifact is not None:
            (status,) = provenance.artifact_statuses(activity["activity_id"])
            assert status["status"] == "available", status
        # Replayed here, whichever edition recorded it, under the declared rule;
        # activities recorded without identity (older files) are never replayed.
        report = win.verify_provenance_activity(activity["activity_id"])
        if activity["call"]["operation"] is None:
            assert report["restoration"] == "opaque"
        else:
            assert report["verdict"] in ("exact", "within_tolerance"), report
    assert shape == CAPTURE_SHAPE
    centroid, gaussian, _stats = ledger.activities
    assert centroid["call"]["inputs"] == gaussian["call"]["inputs"]
    # Every reopened object is recognised as its latest recorded state, ROI
    # included, by this edition's own reading of the other edition's file.
    for object_uuid in ledger.object_uuids():
        obj = win.find_object_by_uuid(object_uuid)
        assert provenance.observe(obj) == ledger.latest_state(object_uuid)["state_id"]


def test_capture_round_trip() -> None:
    """The capture scenario reopens intact in a fresh Desktop window."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "capture.h5")
        with app() as win:
            build_capture(win)
            win.save_h5_workspace(path)
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            check_capture(win, "desktop")


@pytest.mark.parametrize("edition", ["desktop", "web"])
def test_capture_reference_files(edition: str) -> None:
    """Reference capture files of both editions reopen intact on Desktop."""
    path = osp.join(HDF5_DIR, f"provenance_{edition}_capture.h5")
    with app() as win:
        win.load_h5_workspace([path], reset_all=True)
        check_capture(win, edition)


# -- Cross-edition families reference files -----------------------------------
#
# One operation of each family, run identically by each edition: Desktop writes
# ``provenance_desktop_families.h5`` (run this module as a script); DataLab-Web
# writes ``web_families.h5`` from Pyodide (Playwright,
# ``DLW_WRITE_PROVENANCE_FIXTURE=1``). Each edition replays the other's file:
# exact, or within the tolerance declared by the qualification of the operation
# (different numerical libraries).

FAMILIES_REFERENCE = [
    "sigima.signal.average",
    "sigima.signal.signals_to_image",
    "sigima.signal.fft",
    "sigima.signal.add_gaussian_noise",
    "sigima.signal.moving_average",
    "sigima.signal.stats",
    "sigima.image.line_profile",
    "sigima.image.centroid",
    "sigima.image.convolution",
    "sigima.image.peak_detection",
]


def build_families_reference(win) -> None:
    """Run the cross-edition families scenario."""
    spanel, ipanel = win.signalpanel, win.imagepanel
    signals = [add_signal(win, Y * k, f"S{k}") for k in (1, 2, 3)]
    uuids = [get_uuid(s) for s in signals]
    spanel.objview.select_objects(uuids)
    spanel.processor.compute_n_to_1(sips.average, edit=False)
    spanel.objview.select_objects(uuids)
    spanel.processor.compute_n_to_1(
        sips.signals_to_image, sigima.params.SignalsToImageParam(), edit=False
    )
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.compute_1_to_1(
        sips.fft, sigima.params.FFTParam.create(shift=True), edit=False
    )
    spanel.objview.select_objects(uuids[:1])
    noise = sigima.objects.NormalDistributionParam.create(seed=7, sigma=0.5)
    spanel.processor.compute_1_to_1(sips.add_gaussian_noise, noise, edit=False)
    single = create_signal(
        "F", X.astype(np.float32), Y.astype(np.float32), units=("s", "")
    )
    spanel.add_object(single)
    spanel.objview.select_objects([get_uuid(single)])
    spanel.processor.compute_1_to_1(
        sips.moving_average,
        sigima.params.MovingAverageParam.create(n=3),
        edit=False,
    )
    spanel.objview.select_objects(uuids[:1])
    spanel.processor.run_feature("stats")
    image = add_image(win)
    ipanel.objview.select_objects([get_uuid(image)])
    profile = sigima.params.LineProfileParam.create(direction="horizontal", row=1)
    ipanel.processor.compute_1_to_1(sipi.line_profile, profile, edit=False)
    ipanel.objview.select_objects([get_uuid(image)])
    ipanel.processor.run_feature("centroid")
    kernel = add_image(win, "K")
    ipanel.objview.select_objects([get_uuid(image)])
    with sigima.config.options.auto_normalize_kernel.context(False):
        ipanel.processor.compute_2_to_1(kernel, "K", sipi.convolution, edit=False)
    ramp = create_image("ramp", np.add.outer(np.arange(16.0), np.arange(16.0)))
    ipanel.add_object(ramp)
    ipanel.objview.select_objects([get_uuid(ramp)])
    ipanel.processor.compute_1_to_0(
        sipi.peak_detection, sigima.params.Peak2DDetectionParam(), edit=False
    )


def check_families_reference(win, edition: str) -> list[dict]:
    """Replay a reopened families file written by *edition*; return the reports."""
    ledger = win.provenance.ledger
    ledger.validate()
    assert win.provenance.state_status == {}
    activities = ledger.activities
    assert [a["edition"] for a in activities] == [edition] * len(FAMILIES_REFERENCE)
    assert [a["call"]["operation"]["id"] for a in activities] == FAMILIES_REFERENCE
    assert activities[-1]["outputs"] == []  # nothing detected on the ramp
    single = ledger.states[activities[4]["call"]["inputs"][0]["binding"]["state_id"]]
    assert single["fingerprint"]["scheme"] == "datalab-signal-v2"  # float32 rows
    reports = []
    for activity in activities:
        report = win.verify_provenance_activity(activity["activity_id"])
        accepted = ("exact",) if edition == "desktop" else ("exact", "within_tolerance")
        assert report["verdict"] in accepted, report
        reports.append(report)
    return reports


def test_families_reference_round_trip() -> None:
    """The families scenario reopens and replays exactly in a fresh window."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "families.h5")
        with app() as win:
            build_families_reference(win)
            win.save_h5_workspace(path)
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            check_families_reference(win, "desktop")


@pytest.mark.parametrize("edition", ["desktop", "web"])
def test_families_reference_files(edition: str) -> None:
    """Families files of both editions are replayed here under the declared rule."""
    path = osp.join(HDF5_DIR, f"provenance_{edition}_families.h5")
    with app() as win:
        win.load_h5_workspace([path], reset_all=True)
        check_families_reference(win, edition)


def write_reference_file(name: str = "families") -> None:
    """Write a Desktop reference file (unattended).

    Args:
        name: ``families`` or ``capture``. The capture file in the repository was
         recorded by an earlier version (activities without identity) and is
         kept as such, to check that older files stay readable.
    """
    build = {"capture": build_capture, "families": build_families_reference}[name]
    path = osp.join(HDF5_DIR, f"provenance_desktop_{name}.h5")
    execenv.unattended = True
    with app() as win:
        build(win)
        win.save_h5_workspace(path)
    print(f"Written: {path}")


if __name__ == "__main__":
    write_reference_file(*sys.argv[1:])
