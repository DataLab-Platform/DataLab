# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image ROI application test"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sigima.params as sigima_param
from sigima.objects import ImageObj, ImageROI, NewImageParam, create_image_roi
from sigima.tests.data import create_multigaussian_image
from sigima.tests.helpers import print_obj_data_dimensions

from datalab.env import execenv
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.panel.image import ImagePanel

SIZE = 200

# Image ROIs:
IROI1 = [100, 100, 75, 100]  # Rectangle
IROI2 = [66, 100, 50]  # Circle
# Polygon (triangle, that is intentionally inside the rectangle, so that this ROI
# has no impact on the mask calculations in the tests)
IROI3 = [100, 100, 100, 150, 150, 133]


def __run_image_computations(panel: ImagePanel, singleobj: bool | None = None):
    """Test all image features related to ROI"""
    panel.processor.run_feature("centroid")
    panel.processor.run_feature("histogram", sigima_param.HistogramParam())
    panel.processor.run_feature(
        "peak_detection", sigima_param.Peak2DDetectionParam.create(create_rois=False)
    )
    roi = ImageROI(singleobj=singleobj)
    panel.processor.compute_roi_extraction(roi)


def create_test_image_with_roi(newimageparam: NewImageParam) -> ImageObj:
    """Create test image with ROIs

    Args:
        newimageparam (sigima.NewImageParam): Image parameters

    Returns:
        sigima.ImageObj: Image object with ROIs
    """
    ima = create_multigaussian_image(newimageparam)
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    roi = create_image_roi("rectangle", IROI1)
    roi.add_roi(create_image_roi("circle", IROI2))
    roi.add_roi(create_image_roi("polygon", IROI3))
    ima.roi = roi
    return ima


def test_image_roi_app(screenshots: bool = False):
    """Run Image ROI application test scenario"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Image ROI application test:")
        panel = win.imagepanel
        param = NewImageParam.create(height=SIZE, width=SIZE)
        ima1 = create_multigaussian_image(param)
        panel.add_object(ima1)
        __run_image_computations(panel)
        ima2 = create_test_image_with_roi(param)
        for singleobj in (False, True):
            ima2_i = ima2.copy()
            panel.add_object(ima2_i)
            print_obj_data_dimensions(ima2_i)
            panel.processor.edit_roi_graphically()
            if screenshots:
                win.statusBar().hide()
                win.take_screenshot("i_roi_image")
            __run_image_computations(panel, singleobj=singleobj)


@pytest.mark.skip(reason="This test is only for manual testing")
def test_image_roi_basic_app():
    """Run Image ROI basic application test scenario"""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel
        param = NewImageParam.create(height=SIZE, width=SIZE)
        ima1 = create_multigaussian_image(param)
        panel.add_object(ima1)
        panel.processor.edit_roi_graphically()


if __name__ == "__main__":
    test_image_roi_app(screenshots=True)
    test_image_roi_basic_app()
