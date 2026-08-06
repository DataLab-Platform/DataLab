# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History effects unit test
-------------------------

Test the analysis effects manifest (:mod:`datalab.history.effects`):
capture of metadata/ROI mutations during 1-to-0 analyses, persistence on
:class:`HistoryAction` and HDF5 round-trip (action schema v2).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import copy
import os
import tempfile
from unittest.mock import patch

import numpy as np
import sigima.params as sigima_param
from sigima.objects import Gauss2DParam, create_image_from_param, create_image_roi
from sigima.tests.data import create_peak_image

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.panel.history import recompute as hrec
from datalab.h5.native import NativeH5Writer
from datalab.history.effects import AnalysisEffects, capture_effects, merge_effects
from datalab.history.session import HistorySession
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    build_history_action,
    delete_hdf5_items_by_name,
    read_history_sessions,
)


def test_capture_effects_metadata_diff() -> None:
    """Capture added/replaced metadata keys, excluding bookkeeping keys."""
    obj = create_image_from_param(Gauss2DParam.create(height=16, width=16))
    obj.metadata["untouched"] = 1
    obj.metadata["changed_scalar"] = 5
    obj.metadata["changed_array"] = np.arange(3)
    with capture_effects(obj) as effects:
        obj.metadata["new_key"] = "hello"
        obj.metadata["changed_scalar"] = 6
        obj.metadata["changed_array"] = np.arange(4)
        obj.metadata["__uuid"] = "synthetic-uuid"
        obj.metadata["__number"] = 42
    assert effects.metadata_added == ["new_key"]
    assert effects.metadata_replaced == ["changed_array", "changed_scalar"]
    assert "untouched" not in effects.metadata_added
    assert "untouched" not in effects.metadata_replaced
    assert "__uuid" not in effects.metadata_added
    assert "__number" not in effects.metadata_added
    assert effects.roi_modified is False
    execenv.print("test_capture_effects_metadata_diff: ✓")


def test_capture_effects_roi_modified() -> None:
    """Detect ROI creation inside the capture window."""
    obj = create_image_from_param(Gauss2DParam.create(height=16, width=16))
    with capture_effects(obj) as effects:
        pass
    assert effects.roi_modified is False
    with capture_effects(obj) as effects:
        obj.roi = create_image_roi("rectangle", [2, 2, 5, 5])
    assert effects.roi_modified is True
    execenv.print("test_capture_effects_roi_modified: ✓")


def test_analysis_effects_dict_round_trip() -> None:
    """Round-trip AnalysisEffects through to_dict/from_dict."""
    effects = AnalysisEffects(
        metadata_added=["Geometry_peak_detection_dict"],
        metadata_replaced=["analysis_parameters"],
        roi_modified=True,
    )
    payload = effects.to_dict()
    assert payload == {
        "metadata_added": ["Geometry_peak_detection_dict"],
        "metadata_replaced": ["analysis_parameters"],
        "roi_modified": True,
    }
    assert AnalysisEffects.from_dict(payload) == effects
    assert AnalysisEffects.from_dict({}) == AnalysisEffects()
    execenv.print("test_analysis_effects_dict_round_trip: ✓")


def test_analysis_effects_populated_by_1_to_0_compute() -> None:
    """Populate the action effects manifest through a real 1-to-0 analysis."""
    with datalab_test_app_context(console=False, history=True) as win:
        execenv.print("History effects integration test (image peak detection):")
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.imagepanel
        img = create_peak_image()
        panel.add_object(img)

        det_param = sigima_param.Peak2DDetectionParam.create(
            create_rois=True, threshold=0.5
        )
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("peak_detection", det_param)

        action = history[len(history)]
        assert action.effects is not None, "1-to-0 action must carry effects"
        src_uuid = get_uuid(img)
        assert src_uuid in action.effects
        manifest = AnalysisEffects.from_dict(action.effects[src_uuid])
        assert any(
            key.startswith("Geometry_") and key.endswith("_dict")
            for key in manifest.metadata_added
        ), f"Expected a Geometry_*_dict key, got {manifest.metadata_added}"
        assert manifest.roi_modified is True, "Detection ROIs must flag roi_modified"
        execenv.print("  ✓ Effects manifest populated with source uuid and keys")


def test_merge_effects() -> None:
    """Merge manifests: added-stays-added, sticky roi_modified, sorted output."""
    new = AnalysisEffects(
        metadata_added=["b", "a"], metadata_replaced=["c"], roi_modified=False
    )
    merged = merge_effects(None, new)
    assert merged == AnalysisEffects(["a", "b"], ["c"], False)
    # Keys created on the first run stay "added" even though a recompute
    # observes them as replaced
    previous = AnalysisEffects(metadata_added=["result"], roi_modified=True)
    recomputed = AnalysisEffects(metadata_replaced=["result", "params"])
    merged = merge_effects(previous, recomputed)
    assert merged.metadata_added == ["result"]
    assert merged.metadata_replaced == ["params"]
    # roi_modified is sticky: True on first run stays True after a recompute
    assert merged.roi_modified is True
    execenv.print("test_merge_effects: ✓")


def test_recompute_updates_effects_manifest() -> None:
    """Keep first-run keys under metadata_added after a history recompute."""
    with datalab_test_app_context(console=False, history=True) as win:
        execenv.print("History effects recompute integration test:")
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.imagepanel
        img = create_peak_image()
        panel.add_object(img)
        det_param = sigima_param.Peak2DDetectionParam.create(
            create_rois=True, threshold=0.5
        )
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("peak_detection", det_param)
        action = history[len(history)]
        src_uuid = get_uuid(img)
        added_before = AnalysisEffects.from_dict(
            action.effects[src_uuid]
        ).metadata_added
        assert added_before, "First run must have recorded added keys"

        success = hrec.recompute_1_to_0_in_place(history, action)

        assert success is True
        manifest = AnalysisEffects.from_dict(action.effects[src_uuid])
        assert set(added_before) <= set(manifest.metadata_added), (
            "First-run keys must stay under metadata_added after recompute"
        )
        assert not set(added_before) & set(manifest.metadata_replaced)
        execenv.print("  ✓ Manifest keys stayed 'added' after recompute")


def test_targeted_rollback_preserves_unrelated_metadata() -> None:
    """Roll back only manifest keys when a manifest-driven recompute fails."""
    with datalab_test_app_context(console=False, history=True) as win:
        execenv.print("History effects targeted rollback test:")
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.imagepanel
        img = create_peak_image()
        panel.add_object(img)
        det_param = sigima_param.Peak2DDetectionParam.create(
            create_rois=False, threshold=0.5
        )
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("peak_detection", det_param)
        action = history[len(history)]
        src_uuid = get_uuid(img)
        manifest = AnalysisEffects.from_dict(action.effects[src_uuid])
        manifest_keys = manifest.metadata_added + manifest.metadata_replaced
        geometry_key = next(key for key in manifest_keys if key.startswith("Geometry_"))
        # Simulate a user having deleted one analysis result key beforehand
        del img.metadata[geometry_key]
        present_key = next(key for key in manifest_keys if key in img.metadata)
        value_before = copy.deepcopy(img.metadata[present_key])
        img.metadata["user_marker"] = 123
        effects_before = copy.deepcopy(action.effects)

        def failing_recompute(_func_name, obj, _param, plugin_origin=None):
            del plugin_origin
            obj.metadata[geometry_key] = "recreated-by-failed-attempt"
            obj.metadata[present_key] = "corrupted"
            raise RuntimeError("forced recompute failure")

        with patch.object(
            panel.processor, "recompute_1_to_0", side_effect=failing_recompute
        ):
            try:
                hrec.recompute_1_to_0_in_place(history, action)
            except RuntimeError:
                pass
            else:
                raise AssertionError("RuntimeError should have propagated")

        assert img.metadata["user_marker"] == 123, "Unrelated key must be untouched"
        assert geometry_key not in img.metadata, (
            "Manifest key absent before the recompute must be deleted on rollback"
        )
        assert img.metadata[present_key] == value_before, (
            "Manifest key must be restored to its pre-recompute value"
        )
        assert action.effects == effects_before, "Manifest must be unchanged"
        execenv.print("  ✓ Targeted rollback restored only manifest keys")


def test_action_effects_hdf5_round_trip_and_legacy() -> None:
    """Round-trip effects through HDF5 and tolerate legacy files without it."""
    action = build_history_action()
    action.effects = {
        "source-uuid": AnalysisEffects(
            metadata_added=["Geometry_peak_detection_dict"],
            metadata_replaced=["analysis_parameters"],
            roi_modified=True,
        ).to_dict()
    }
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        loaded = read_history_sessions(path)[0].actions[0]
        assert loaded.effects == action.effects
        # Legacy tolerance: files without the effects group deserialize to None
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
            delete_hdf5_items_by_name(writer.h5, "effects")
        legacy = read_history_sessions(path)[0].actions[0]
        assert legacy.effects is None
    execenv.print("test_action_effects_hdf5_round_trip_and_legacy: ✓")


if __name__ == "__main__":
    execenv.unattended = True  # Auto-close dialogs and event loops (standalone run)
    test_capture_effects_metadata_diff()
    test_capture_effects_roi_modified()
    test_analysis_effects_dict_round_trip()
    test_merge_effects()
    test_analysis_effects_populated_by_1_to_0_compute()
    test_recompute_updates_effects_manifest()
    test_targeted_rollback_preserves_unrelated_metadata()
    test_action_effects_hdf5_round_trip_and_legacy()
