# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.param
from sigima.obj import NewImageParam
from sigima.tests.data import create_multigauss_image

from datalab.env import execenv
from datalab.tests import cdltest_app_context, take_plotwidget_screenshot
from datalab.tests.features.common.roi_app_test import create_test_image_with_roi


def test_contour_app():
    """Run contour finding application test scenario"""
    newparam = NewImageParam.create(height=200, width=200)
    with cdltest_app_context() as win:
        panel = win.imagepanel
        for shape in ("polygon", "circle", "ellipse"):
            ima1 = create_multigauss_image(newparam)
            ima1.set_metadata_option("colormap", "gray")
            panel.add_object(ima1)
            param = sigima.param.ContourShapeParam.create(shape=shape)
            panel.processor.run_feature("contour_shape", param)
            take_plotwidget_screenshot(panel, "contour_test")
            ima2 = create_test_image_with_roi(newparam)
            ima2.set_metadata_option("colormap", "gray")
            panel.add_object(ima2)
            panel.processor.edit_regions_of_interest()
            panel.processor.run_feature("contour_shape", param)
            if not execenv.unattended:
                break


if __name__ == "__main__":
    test_contour_app()
