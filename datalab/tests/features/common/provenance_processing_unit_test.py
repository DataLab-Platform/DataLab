# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Processing restores and applies again every recorded call (Desktop).

After saving and reopening a workspace, the Processing tab restores the call
that produced a result from the provenance ledger: 1-to-1, 2-to-1 and n-to-1
processing, conversions between signals and images, with or without
parameters, every input in order. Applying it again keeps the identity of the
result and records a new activity describing all its inputs.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import contextlib
import dataclasses
import os.path as osp

import numpy as np
import pytest
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj, create_image, create_signal

from datalab.gui.processor.catcher import CompOut
from datalab.objectmodel import get_uuid
from datalab.processingcall import resolve_processing_call
from datalab.tests import helpers
from datalab.tests.features.common.provenance_difference_unit_test import (
    A_MINUS_B,
    RULE,
    build_differences,
)

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    AMPLITUDE_Y,
    MAXIMUM_Y,
    X,
    Y,
    app,
    isolated_param_defaults,
)


def add_signal(win, y, title: str) -> SignalObj:
    """Add a signal to the signal panel."""
    obj = create_signal(title, X.copy(), np.array(y, dtype=float), units=("s", ""))
    win.signalpanel.add_object(obj)
    return obj


def add_image(win, data, title: str) -> ImageObj:
    """Add an image to the image panel."""
    obj = create_image(title, np.array(data, dtype=float))
    win.imagepanel.add_object(obj)
    return obj


def compute(panel, objs, method: str, *args, **kwargs) -> list:
    """Select *objs* in *panel*, run ``processor.<method>``; return new objects."""
    win = panel.mainwindow
    panel.objview.select_objects([get_uuid(obj) for obj in objs])
    before = {
        p.PANEL_STR_ID: set(p.objmodel.get_object_ids())
        for p in (win.signalpanel, win.imagepanel)
    }
    getattr(panel.processor, method)(*args, **kwargs)
    return [
        p.objmodel[uid]
        for p in (win.signalpanel, win.imagepanel)
        for uid in p.objmodel.get_object_ids()
        if uid not in before[p.PANEL_STR_ID]
    ]


@contextlib.contextmanager
def reopened(build):
    """Build a workspace, save it, then yield ``(window, uuids)`` after reopening.

    Args:
        build: ``window -> dict of named objects`` building the workspace.
    """
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "workspace.h5")
        with app() as win:
            uuids = {name: get_uuid(obj) for name, obj in build(win).items()}
            win.save_h5_workspace(path)
        with app() as win:
            win.load_h5_workspace([path], reset_all=True)
            yield win, uuids


def inputs_of(win, activity) -> list[tuple[str, str]]:
    """Return the ``(role, object uuid)`` inputs of an activity."""
    states = win.provenance.ledger.states
    return [
        (item["role"], states[item["binding"]["state_id"]]["object_uuid"])
        for item in activity["call"]["inputs"]
    ]


def output_uuid(win, activity) -> str:
    """Return the object UUID of the result of an activity."""
    return win.provenance.ledger.states[activity["outputs"][0]["state_id"]][
        "object_uuid"
    ]


def reapply_button(panel) -> QW.QPushButton | None:
    """Return the Reapply button of the Processing tab, if any."""
    return panel.objprop.processing_scroll.findChild(
        QW.QPushButton, "reapply_processing"
    )


def test_filter_is_restored_and_applied_again() -> None:
    """1-to-1 with parameters: restored from the ledger, applied in place."""

    def build(win):
        s0 = add_signal(win, Y, "S0")
        param = sigima.params.GaussianParam.create(sigma=2.0)
        (s1,) = compute(
            win.signalpanel,
            [s0],
            "compute_1_to_1",
            sips.gaussian_filter,
            param=param,
            edit=False,
        )
        return {"s0": s0, "s1": s1}

    with reopened(build) as (win, uuids):
        s1 = win.find_object_by_uuid(uuids["s1"])
        call = resolve_processing_call(win, s1)
        assert (call.origin, call.pattern, call.roles) == (
            "ledger",
            "1-to-1",
            ["source"],
        )
        assert call.param.sigma == 2.0 and call.can_reapply
        objprop = win.signalpanel.objprop
        assert objprop.setup_processing_tab(s1)
        assert objprop.processing_param_editor.dataset.sigma == 2.0
        count = len(win.provenance.ledger.activities)
        report = objprop.apply_processing_parameters(
            s1,
            interactive=False,
            param=sigima.params.GaussianParam.create(sigma=5.0),
        )
        assert report.success, report.message
        activities = win.provenance.ledger.activities
        assert len(activities) == count + 1
        activity = activities[-1]
        assert activity["origin"] == "recompute_in_place"
        assert activity["call"]["parameters"]["sigma"] == 5.0
        assert inputs_of(win, activity) == [("source", uuids["s0"])]
        assert output_uuid(win, activity) == uuids["s1"]
        assert win.find_object_by_uuid(uuids["s1"]) is s1
        win.provenance.ledger.validate()


def test_difference_keeps_source_and_operand() -> None:
    """2-to-1: source and operand in order, recorded context kept."""

    def build(win):
        a, b, a_b, b_a = build_differences(win)
        return {"a": a, "b": b, "a_b": a_b, "b_a": b_a}

    with reopened(build) as (win, uuids):
        for name, roles in (
            ("a_b", [("source", uuids["a"]), ("operand", uuids["b"])]),
            ("b_a", [("source", uuids["b"]), ("operand", uuids["a"])]),
        ):
            call = resolve_processing_call(win, win.find_object_by_uuid(uuids[name]))
            assert call.pattern == "2-to-1" and call.param is None
            assert [(i.role, i.object_uuid) for i in call.inputs] == roles
        a_b = win.find_object_by_uuid(uuids["a_b"])
        a_b.y[:] = 0.0
        report = win.signalpanel.processor.recompute_processing(a_b, interactive=False)
        assert report.success, report.message
        activity = win.provenance.ledger.activities[-1]
        assert activity["origin"] == "recompute_in_place"
        assert inputs_of(win, activity) == [
            ("source", uuids["a"]),
            ("operand", uuids["b"]),
        ]
        assert activity["context"]["x_alignment"] == RULE
        assert output_uuid(win, activity) == uuids["a_b"]
        assert np.array_equal(a_b.y, A_MINUS_B)
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["verdict"] == "exact", report


@pytest.mark.parametrize("kind", ["signal", "image"])
def test_average_without_parameters_has_a_reapply_action(kind: str) -> None:
    """n-to-1 without parameters: every source restored, explicit Reapply."""

    def build(win):
        if kind == "signal":
            objs = [add_signal(win, Y * (i + 1), f"S{i}") for i in range(3)]
            panel, func = win.signalpanel, sips.average
        else:
            base = np.arange(12.0).reshape(3, 4)
            objs = [add_image(win, base * (i + 1), f"I{i}") for i in range(3)]
            panel, func = win.imagepanel, sipi.average
        (mean,) = compute(panel, objs, "compute_n_to_1", func, edit=False)
        return {"mean": mean, **{f"src{i}": obj for i, obj in enumerate(objs)}}

    with reopened(build) as (win, uuids):
        panel = win.signalpanel if kind == "signal" else win.imagepanel
        mean = win.find_object_by_uuid(uuids["mean"])
        sources = [uuids[f"src{i}"] for i in range(3)]
        call = resolve_processing_call(win, mean)
        assert call.pattern == "n-to-1" and call.param is None
        assert [(i.role, i.object_uuid) for i in call.inputs] == [
            ("sources", uid) for uid in sources
        ]
        panel.objview.set_current_object(mean)
        assert panel.objprop.processing_param_editor is None
        button = reapply_button(panel)
        assert button is not None and button.isEnabled()
        expected = mean.data.copy() if kind == "image" else mean.y.copy()
        if kind == "image":
            mean.data[:] = 0.0
        else:
            mean.y[:] = 0.0
        count = len(win.provenance.ledger.activities)
        button.click()
        activities = win.provenance.ledger.activities
        assert len(activities) == count + 1
        activity = activities[-1]
        assert activity["origin"] == "recompute_in_place"
        assert inputs_of(win, activity) == [("sources", uid) for uid in sources]
        assert output_uuid(win, activity) == uuids["mean"]
        current = win.find_object_by_uuid(uuids["mean"])
        values = current.data if kind == "image" else current.y
        assert np.array_equal(values, expected)
        win.provenance.ledger.validate()


def test_two_input_aggregation_is_not_a_difference() -> None:
    """n-to-1 with two inputs keeps two aggregation roles."""

    def build(win):
        a, b = add_signal(win, Y, "A"), add_signal(win, Y * 3, "B")
        (mean,) = compute(
            win.signalpanel, [a, b], "compute_n_to_1", sips.average, edit=False
        )
        return {"a": a, "b": b, "mean": mean}

    with reopened(build) as (win, uuids):
        mean = win.find_object_by_uuid(uuids["mean"])
        call = resolve_processing_call(win, mean)
        assert call.pattern == "n-to-1"
        assert call.roles == ["sources", "sources"]
        report = win.signalpanel.processor.recompute_processing(mean, interactive=False)
        assert report.success, report.message
        activity = win.provenance.ledger.activities[-1]
        assert inputs_of(win, activity) == [
            ("sources", uuids["a"]),
            ("sources", uuids["b"]),
        ]


def test_signals_to_image_keeps_its_parameters_and_processor() -> None:
    """Conversion signals -> image: the signal processor computes the image."""

    def build(win):
        signals = [add_signal(win, Y + i, f"S{i}") for i in range(3)]
        param = sigima.params.SignalsToImageParam.create(
            orientation="columns", normalize=False
        )
        (image,) = compute(
            win.signalpanel,
            signals,
            "compute_n_to_1",
            sips.signals_to_image,
            param=param,
            edit=False,
        )
        return {"image": image, **{f"s{i}": obj for i, obj in enumerate(signals)}}

    with reopened(build) as (win, uuids):
        image = win.find_object_by_uuid(uuids["image"])
        call = resolve_processing_call(win, image)
        assert call.processor is win.signalpanel.processor
        assert call.feature.function is sips.signals_to_image
        assert call.param.orientation == "columns"
        ipanel = win.imagepanel
        assert ipanel.objprop.setup_processing_tab(image)
        assert ipanel.objprop.processing_param_editor is not None
        shape = image.data.shape
        param = sigima.params.SignalsToImageParam.create(
            orientation="rows", normalize=False
        )
        report = ipanel.objprop.apply_processing_parameters(
            image, interactive=False, param=param
        )
        assert report.success, report.message
        assert image.data.shape == shape[::-1]
        assert win.find_object_by_uuid(uuids["image"]) is image
        activity = win.provenance.ledger.activities[-1]
        assert activity["call"]["parameters"]["orientation"] == "rows"
        assert inputs_of(win, activity) == [
            ("sources", uuids[f"s{i}"]) for i in range(3)
        ]
        assert output_uuid(win, activity) == uuids["image"]


def test_image_to_signal_conversion() -> None:
    """Conversion image -> signal: the image processor computes the signal."""

    def build(win):
        image = add_image(win, np.outer(np.hanning(32), np.hanning(32)), "I")
        param = sigima.params.RadialProfileParam.create(center="user", xc=16, yc=16)
        (profile,) = compute(
            win.imagepanel,
            [image],
            "compute_1_to_1",
            sipi.radial_profile,
            param=param,
            edit=False,
        )
        return {"image": image, "profile": profile}

    with reopened(build) as (win, uuids):
        profile = win.find_object_by_uuid(uuids["profile"])
        assert isinstance(profile, SignalObj)
        call = resolve_processing_call(win, profile)
        assert call.processor is win.imagepanel.processor
        assert (call.param.xc, call.param.yc) == (16, 16)
        param = sigima.params.RadialProfileParam.create(center="user", xc=10, yc=10)
        report = win.signalpanel.objprop.apply_processing_parameters(
            profile, interactive=False, param=param
        )
        assert report.success, report.message
        activity = win.provenance.ledger.activities[-1]
        assert inputs_of(win, activity) == [("source", uuids["image"])]
        assert output_uuid(win, activity) == uuids["profile"]


def test_missing_input_disables_the_reapplication() -> None:
    """A deleted input is never replaced; the reason is shown."""
    with app() as win:
        objs = [add_signal(win, Y * (i + 1), f"S{i}") for i in range(3)]
        (mean,) = compute(
            win.signalpanel, objs, "compute_n_to_1", sips.average, edit=False
        )
        panel = win.signalpanel
        panel.objview.set_current_object(objs[1])
        panel.remove_object(force=True)
        call = resolve_processing_call(win, mean)
        assert [i.obj is None for i in call.inputs] == [False, True, False]
        assert not call.can_reapply
        assert "no longer exist" in call.reason
        panel.objview.set_current_object(mean)
        assert not reapply_button(panel).isEnabled()
        count = len(win.provenance.ledger.activities)
        report = panel.processor.recompute_processing(mean, interactive=False)
        assert not report.success and "no longer exist" in report.message
        assert len(win.provenance.ledger.activities) == count


def test_unqualified_operation_is_still_applied_again() -> None:
    """Knowing an operation is enough to apply it again; verification refuses."""
    with app() as win:
        add_signal(win, Y, "S0")
        param = sigima.params.NormalizeParam.create(method="maximum")
        s0 = win.signalpanel.objview.get_current_object()
        (result,) = compute(
            win.signalpanel,
            [s0],
            "compute_1_to_1",
            sips.normalize,
            param=param,
            edit=False,
        )
        win.provenance.computation = dataclasses.replace(
            win.provenance.computation,
            qualification_status=lambda _contract: "not_qualified",
        )
        call = resolve_processing_call(win, result)
        assert call.origin == "ledger" and call.can_reapply
        report = win.signalpanel.processor.recompute_processing(
            result,
            param=sigima.params.NormalizeParam.create(method="amplitude"),
            interactive=False,
        )
        assert report.success, report.message
        assert np.array_equal(result.y, AMPLITUDE_Y)
        activity = win.provenance.ledger.activities[-1]
        report = win.verify_provenance_activity(activity["activity_id"])
        assert report["eligibility"] == "unsupported_contract"


def test_failure_and_cancellation_record_nothing(monkeypatch) -> None:
    """A failed or cancelled re-application changes nothing."""
    with app() as win:
        objs = [add_signal(win, Y * (i + 1), f"S{i}") for i in range(2)]
        (mean,) = compute(
            win.signalpanel, objs, "compute_n_to_1", sips.average, edit=False
        )
        processor = win.signalpanel.processor
        before = mean.y.copy()
        count = len(win.provenance.ledger.activities)
        monkeypatch.setattr(processor, "_BaseProcessor__exec_func", lambda *_args: None)
        report = processor.recompute_processing(mean, interactive=False)
        assert not report.success and report.cancelled
        monkeypatch.setattr(
            processor,
            "_BaseProcessor__exec_func",
            lambda *_args: CompOut(error_msg="simulated failure"),
        )
        report = processor.recompute_processing(mean, interactive=False)
        assert not report.success and not report.cancelled
        assert len(win.provenance.ledger.activities) == count
        assert np.array_equal(mean.y, before)


@pytest.mark.parametrize("method", ["compute_1_to_n", "compute_multiple_1_to_1"])
def test_one_output_of_a_grouped_command(method: str) -> None:
    """Each output restores its own call; re-applying one leaves the others."""
    with app() as win:
        s0 = add_signal(win, Y, "S0")
        maximum = sigima.params.NormalizeParam.create(method="maximum")
        amplitude = sigima.params.NormalizeParam.create(method="amplitude")
        if method == "compute_1_to_n":
            args = (sips.normalize, [maximum, amplitude])
        else:
            args = ([sips.normalize, sips.normalize], [maximum, amplitude])
        first, second = compute(win.signalpanel, [s0], method, *args)
        assert np.array_equal(first.y, MAXIMUM_Y)
        assert np.array_equal(second.y, AMPLITUDE_Y)
        ledger = win.provenance.ledger
        first_state = ledger.latest_state(get_uuid(first))["state_id"]
        call = resolve_processing_call(win, second)
        assert call.param.method == "amplitude"
        report = win.signalpanel.processor.recompute_processing(
            second,
            param=sigima.params.NormalizeParam.create(method="maximum"),
            interactive=False,
        )
        assert report.success, report.message
        assert np.array_equal(second.y, MAXIMUM_Y)
        assert np.array_equal(first.y, MAXIMUM_Y)
        assert ledger.latest_state(get_uuid(first))["state_id"] == first_state
        activity = ledger.activities[-1]
        assert inputs_of(win, activity) == [("source", get_uuid(s0))]
        assert output_uuid(win, activity) == get_uuid(second)
        assert len(win.signalpanel.objmodel) == 3


def test_web_files_are_described_and_applied_again() -> None:
    """Reference files written by DataLab-Web: consulted, restored, reapplied."""
    hdf5 = osp.join(osp.dirname(osp.dirname(__file__)), "hdf5")
    with app() as win:
        win.load_h5_workspace(
            [osp.join(hdf5, "provenance_web_chain.h5")], reset_all=True
        )
        ledger = win.provenance.ledger
        _a1, a2, _a3, _a4 = ledger.activities
        s1_uuid, s2_uuid = inputs_of(win, a2)[0][1], output_uuid(win, a2)
        s2 = win.find_object_by_uuid(s2_uuid)
        view = win.provenance.describe_object(s2_uuid, check_data=True)
        production = view["production"]
        assert production["activity_id"] == a2["activity_id"]
        assert production["environment"]["edition"] == "web"
        assert production["inputs"][0]["object_present"] is True
        assert view["anchor"]["content"] == "matches"
        call = resolve_processing_call(win, s2)
        assert (call.origin, call.pattern, call.param.method) == (
            "ledger",
            "1-to-1",
            "maximum",
        )
        assert call.inputs[0].object_uuid == s1_uuid and call.can_reapply
        report = win.signalpanel.objprop.apply_processing_parameters(
            s2,
            interactive=False,
            param=sigima.params.NormalizeParam.create(method="amplitude"),
        )
        assert report.success, report.message
        activity = ledger.activities[-1]
        assert (activity["edition"], activity["origin"]) == (
            "desktop",
            "recompute_in_place",
        )
        assert inputs_of(win, activity) == [("source", s1_uuid)]
        assert output_uuid(win, activity) == s2_uuid
    with app() as win:
        win.load_h5_workspace(
            [osp.join(hdf5, "provenance_web_difference.h5")], reset_all=True
        )
        act_ab, _act_ba = win.provenance.ledger.activities
        a_b = win.find_object_by_uuid(output_uuid(win, act_ab))
        call = resolve_processing_call(win, a_b)
        assert call.pattern == "2-to-1" and call.can_reapply
        assert [(i.role, i.object_uuid) for i in call.inputs] == inputs_of(win, act_ab)
        a_b.y[:] = 0.0
        report = win.signalpanel.processor.recompute_processing(a_b, interactive=False)
        assert report.success, report.message
        assert np.array_equal(a_b.y, A_MINUS_B)
        activity = win.provenance.ledger.activities[-1]
        assert activity["context"]["x_alignment"] == RULE
        assert (
            win.verify_provenance_activity(activity["activity_id"])["verdict"]
            == "exact"
        )
