# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Detection ROI replacement confirmation test

Testing the following:
  - When create_rois=True and image already has ROIs, the preprocess hook
    runs before the progress bar and can abort the computation
  - In unattended mode (automated tests) the dialog is skipped and ROIs
    are always replaced
  - When create_rois=False, existing ROIs are left untouched
  - When no existing ROIs are present, ROI creation proceeds normally
  - When the user cancels the confirmation dialog, existing ROIs are preserved
  - The confirmation dialog is shown only when ROIs already exist
"""

# guitest: show

from __future__ import annotations

from unittest.mock import patch

import sigima.params
import sigima.proc.image as sipi
from qtpy import QtWidgets as QW
from sigima.objects import NewImageParam, create_image_roi
from sigima.tests.data import create_multigaussian_image, create_peak_image

from datalab.env import execenv
from datalab.tests import datalab_test_app_context
from datalab.tests.features.image.roi_app_test import IROI1, IROI2


def _create_image_with_roi():
    """Return a multigaussian image that already has ROIs defined."""
    newparam = NewImageParam.create(height=200, width=200)
    ima = create_multigaussian_image(newparam)
    roi = create_image_roi("rectangle", IROI1)
    roi.add_roi(create_image_roi("circle", IROI2))
    ima.roi = roi
    return ima


def test_create_rois_no_existing_roi():
    """ROIs are created normally when the image has no pre-existing ROI."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = create_peak_image()
        assert ima.roi is None
        panel.add_object(ima)

        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        result = panel.processor.compute_peak_detection(param)
        assert result is not None, "Peak detection should return results"

        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "ROI should be created when create_rois=True"
        assert not obj.roi.is_empty(), "Created ROI should not be empty"


def test_create_rois_with_existing_roi_unattended():
    """In unattended mode, existing ROIs are silently replaced (no dialog)."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = _create_image_with_roi()
        initial_roi = ima.roi
        panel.add_object(ima)

        # execenv.unattended is True in the test suite: the dialog is skipped
        assert execenv.unattended, "This test requires unattended mode"

        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        result = panel.processor.compute_peak_detection(param)
        assert result is not None, "Peak detection should return results"

        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "ROI should be present after detection"
        assert obj.roi != initial_roi, (
            "Existing ROI should have been replaced in unattended mode"
        )


def test_create_rois_false_preserves_existing_roi():
    """When create_rois=False, existing ROIs are never touched."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = _create_image_with_roi()
        panel.add_object(ima)

        obj = panel.objview.get_current_object()
        roi_before = obj.roi

        param = sigima.params.Peak2DDetectionParam.create(create_rois=False)
        panel.processor.compute_peak_detection(param)

        obj = panel.objview.get_current_object()
        assert obj.roi == roi_before, (
            "Existing ROI must not be modified when create_rois=False"
        )


def test_dialog_shown_only_when_roi_exists():
    """The confirmation dialog is triggered only when the image already has ROIs.

    - With existing ROIs and create_rois=True: preprocess_1_to_0 calls the dialog
    - Without existing ROIs and create_rois=True: preprocess_1_to_0 returns True
      directly, without opening any dialog
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)

        # Case 1: image with no ROI — dialog must NOT be shown
        ima_no_roi = create_peak_image()
        panel.add_object(ima_no_roi)
        objs_no_roi = panel.objview.get_sel_objects(include_groups=True)

        with patch.object(QW.QMessageBox, "question") as mock_question:
            execenv.unattended = False
            try:
                result = panel.processor.preprocess_1_to_0(
                    sipi.peak_detection, param, objs_no_roi
                )
            finally:
                execenv.unattended = True

        assert result is True, "Should proceed when no existing ROIs"
        mock_question.assert_not_called()

        # Case 2: image with existing ROI — dialog MUST be shown
        ima_with_roi = _create_image_with_roi()
        panel.add_object(ima_with_roi)
        objs_with_roi = panel.objview.get_sel_objects(include_groups=True)

        with patch.object(
            QW.QMessageBox, "question", return_value=QW.QMessageBox.Yes
        ) as mock_question:
            execenv.unattended = False
            try:
                result = panel.processor.preprocess_1_to_0(
                    sipi.peak_detection, param, objs_with_roi
                )
            finally:
                execenv.unattended = True

        assert result is True, "Should proceed when user confirms"
        mock_question.assert_called_once()


def test_cancel_dialog_preserves_existing_roi():
    """When the user cancels the confirmation dialog, existing ROIs are preserved."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = _create_image_with_roi()
        panel.add_object(ima)

        obj = panel.objview.get_current_object()
        roi_before = obj.roi

        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)

        # Simulate the user clicking "No" in the confirmation dialog
        with patch.object(QW.QMessageBox, "question", return_value=QW.QMessageBox.No):
            execenv.unattended = False
            try:
                result = panel.processor.compute_peak_detection(param)
            finally:
                execenv.unattended = True

        assert result is None, "Computation should be aborted when user cancels"
        obj = panel.objview.get_current_object()
        assert obj.roi == roi_before, (
            "Existing ROI must be preserved when user cancels the dialog"
        )


def test_preprocess_hook_abort_skipped_in_unattended():
    """preprocess_1_to_0 returns True in unattended mode (no blocking dialog)."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = _create_image_with_roi()
        panel.add_object(ima)

        assert execenv.unattended, "This test requires unattended mode"

        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        objs = panel.objview.get_sel_objects(include_groups=True)

        # In unattended mode the hook must always return True (no dialog shown)
        result = panel.processor.preprocess_1_to_0(sipi.peak_detection, param, objs)
        assert result is True, "preprocess_1_to_0 must return True in unattended mode"


def test_auto_recompute_does_not_replace_rois():
    """auto_recompute_analysis must not recreate ROIs deleted by the user.

    Scenario:
    1. Run peak detection with create_rois=True → ROIs are created and the
       analysis parameters (including create_rois=True) are stored in the
       object's metadata.
    2. The user deletes the ROIs manually.
    3. auto_recompute_analysis is triggered (e.g. after a data change).
    4. The ROIs must NOT be recreated: auto_recompute_analysis disables
       create_rois before calling compute_1_to_0.
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = create_peak_image()
        panel.add_object(ima)

        # Step 1: run detection with ROI creation to store analysis params
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        panel.processor.compute_peak_detection(param)

        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "Peak detection should have created ROIs"

        # Step 2: user deletes the ROIs
        obj.roi = None

        # Step 3 & 4: auto-recompute must NOT recreate the ROIs
        panel.processor.auto_recompute_analysis(obj)

        obj = panel.objview.get_current_object()
        assert obj.roi is None, (
            "auto_recompute_analysis must not recreate ROIs "
            "(create_rois is disabled during auto-recompute)"
        )


if __name__ == "__main__":
    test_create_rois_no_existing_roi()
    test_create_rois_with_existing_roi_unattended()
    test_create_rois_false_preserves_existing_roi()
    test_dialog_shown_only_when_roi_exists()
    test_cancel_dialog_preserves_existing_roi()
    test_preprocess_hook_abort_skipped_in_unattended()
    test_auto_recompute_does_not_replace_rois()
