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


def test_peak2d():
    """Run 2D peak detection scenario"""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = create_peak_image()
        panel.add_object(ima)
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
        panel.processor.run_feature("peak_detection", param)
        win.toggle_show_titles(False)
        take_plotwidget_screenshot(panel, "peak2d_test")


if __name__ == "__main__":
    test_peak2d()
