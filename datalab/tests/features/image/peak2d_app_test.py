# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
2D peak detection test

Testing the following:
  - Create a test image with multiple peaks
  - Compute 2D peak detection and show points on image
"""

# guitest: show

import sigima.params
from sigima.tests.data import create_peak_image

from datalab.tests import datalab_test_app_context, take_plotwidget_screenshot


def test_peak2d(screenshot: bool = False) -> None:
    """Run 2D peak detection scenario"""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = create_peak_image()
        panel.add_object(ima)

        # Test with ROI creation enabled (default)
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        results = panel.processor.compute_peak_detection(param)
        assert results is not None, "Peak detection should return results"

        # Get the processed image object
        obj = panel.objview.get_current_object()
        assert obj.roi is not None, "ROI should be created when create_rois=True"
        assert not obj.roi.is_empty(), "ROI should not be empty"

        # Test with ROI creation disabled
        ima2 = create_peak_image()
        panel.add_object(ima2)
        param2 = sigima.params.Peak2DDetectionParam.create(create_rois=False)
        panel.processor.compute_peak_detection(param2)

        obj2 = panel.objview.get_current_object()
        assert obj2.roi is None or obj2.roi.is_empty(), (
            "ROI should not be created when create_rois=False"
        )

        win.toggle_show_titles(False)
        if screenshot:
            take_plotwidget_screenshot(panel, "peak2d_test")


if __name__ == "__main__":
    test_peak2d()
