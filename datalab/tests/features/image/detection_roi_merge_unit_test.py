# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Detection ROI merge test

Testing the following:
  - When create_rois=True and the image has no ROI, detection creates ROIs
  - When create_rois=True and the image already has ROIs, the newly detected
    ROIs are appended to the existing ones (non-destructive, no dialog)
  - When create_rois=False, existing ROIs are left untouched
  - contour_shape with create_rois=True creates ROIs in DataLab integration
  - auto_recompute_analysis does not recreate ROIs deleted/edited by the user
  - The ROI modification loop is broken (no infinite re-creation cycle)
"""

# guitest: show

from __future__ import annotations

import sigima.params
from sigima.enums import ContourShape
from sigima.objects import NewImageParam, create_image_roi
from sigima.tests.data import create_multigaussian_image, create_peak_image

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


def test_create_rois_appends_to_existing_roi():
    """Newly detected ROIs are appended to existing ones, not replacing them."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = _create_image_with_roi()
        panel.add_object(ima)

        obj = panel.objview.get_current_object()
        existing_rois = list(obj.roi.single_rois)
        n_existing = len(existing_rois)
        assert n_existing > 0, "Test image must have pre-existing ROIs"

        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        result = panel.processor.compute_peak_detection(param)
        assert result is not None, "Peak detection should return results"

        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "ROI should be present after detection"
        # Pre-existing ROIs must still be there
        for roi in existing_rois:
            assert roi in obj.roi.single_rois, (
                "Existing ROIs must be preserved when detection creates new ROIs"
            )
        # New ROIs must have been appended on top of the existing ones
        assert len(obj.roi.single_rois) > n_existing, (
            "Detected ROIs must be appended to the existing ROIs"
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


def test_contour_shape_creates_rois_in_datalab():
    """Integration test: contour_shape with create_rois=True creates ROIs via DataLab.

    This verifies the full DataLab integration path: the contour_shape
    computation function is called through run_feature, and the postprocess
    hook (apply_detection_rois) creates the ROIs on the image object.
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        newparam = NewImageParam.create(height=200, width=200)
        ima = create_multigaussian_image(newparam)
        panel.add_object(ima)

        for shape in ContourShape:
            # Reset: remove ROIs from previous iteration
            obj = panel.objview.get_current_object()
            obj.roi = None

            param = sigima.params.ContourShapeParam.create(
                shape=shape, create_rois=True
            )
            panel.processor.run_feature("contour_shape", param)

            obj = panel.objview.get_current_object()
            assert obj.roi is not None, (
                f"contour_shape({shape.name}) with create_rois=True "
                "must create ROIs in DataLab"
            )
            assert not obj.roi.is_empty(), (
                f"contour_shape({shape.name}) ROI must not be empty"
            )


def test_no_infinite_roi_recreation_loop():
    """The ROI creation → auto-recompute cycle must not loop infinitely.

    Full scenario matching the real user workflow:
    1. Run detection with create_rois=True → ROIs are created.
    2. Simulate what happens when the user edits ROIs: auto_recompute_analysis
       is called (as DataLab does after ROI graphical editing).
    3. Verify that auto_recompute_analysis does NOT recreate ROIs.
    4. Repeat step 2 a second time to confirm stability.

    This test guards against the semi-infinite loop described in issue #329:
    modifying ROIs triggers auto-recompute, which re-runs the detection
    function. If create_rois stays True in the recompute path, the detection
    would overwrite the user's ROI edit, triggering another recompute, etc.
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = create_peak_image()
        panel.add_object(ima)

        # Step 1: detection with ROI creation
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        panel.processor.compute_peak_detection(param)

        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "Initial detection should create ROIs"

        # Step 2: simulate user editing ROIs (replace with a single rectangle)
        obj.roi = create_image_roi("rectangle", [10, 10, 50, 50])
        user_roi = obj.roi

        # Step 3: auto-recompute fires (as DataLab does after ROI edit)
        panel.processor.auto_recompute_analysis(obj)

        obj = panel.objview.get_current_object()
        # The ROI must be the user's edited ROI, NOT a new set from detection
        assert obj.roi is user_roi, (
            "auto_recompute must not replace the user's manually edited ROI"
        )

        # Step 4: a second auto-recompute must also be stable
        panel.processor.auto_recompute_analysis(obj)

        obj = panel.objview.get_current_object()
        assert obj.roi is user_roi, (
            "Second auto_recompute must still preserve the user's ROI (no oscillation)"
        )


if __name__ == "__main__":
    test_create_rois_no_existing_roi()
    test_create_rois_appends_to_existing_roi()
    test_create_rois_false_preserves_existing_roi()
    test_auto_recompute_does_not_replace_rois()
    test_contour_shape_creates_rois_in_datalab()
    test_no_infinite_roi_recreation_loop()
