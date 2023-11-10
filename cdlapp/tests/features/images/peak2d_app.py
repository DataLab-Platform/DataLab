# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
2D peak detection test

Testing the following:
  - Create a test image with multiple peaks
  - Compute 2D peak detection and show points on image
"""

# guitest: show

import cdlapp.param
from cdlapp.tests import cdl_app_context, take_plotwidget_screenshot
from cdlapp.tests.data import create_peak2d_image


def test():
    """Run 2D peak detection scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        ima = create_peak2d_image()
        panel.add_object(ima)
        param = cdlapp.param.Peak2DDetectionParam.create(create_rois=True)
        panel.processor.compute_peak_detection(param)
        panel.toggle_show_titles(False)
        take_plotwidget_screenshot(panel, "peak2d_test")


if __name__ == "__main__":
    test()
