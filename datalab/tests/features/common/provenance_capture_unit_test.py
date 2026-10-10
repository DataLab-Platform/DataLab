# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Provenance capture of images, analyses, n-to-1 operations, ROI and uncertainty."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp

import numpy as np
import sigima.objects
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from sigima.objects import create_image, create_signal

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


def test_legacy_interpolation_is_flagged() -> None:
    """Inputs interpolated before an n-to-1 call are flagged; originals recorded."""
    with app() as win:
        first = add_signal(win, title="first")
        second = create_signal(
            "second", np.array([0.0, 0.5, 0.6, 0.75]), Y.copy(), units=("s", "")
        )
        win.signalpanel.add_object(second)
        win.signalpanel.objview.select_objects([get_uuid(first), get_uuid(second)])
        win.signalpanel.processor.compute_n_to_1(sips.average, edit=False)
        (activity,) = win.provenance.ledger.activities
        assert activity["limits"] == ["x_interpolated"]
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
        # Qualified normalisation refuses ROI and uncertainty: recorded as opaque.
        assert activity["call"]["operation"] is None
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
