# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding application test with Fabry-Perot images
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.obj
import cdl.param
from cdl.tests import cdltest_app_context, take_plotwidget_screenshot
from cdl.tests.data import get_test_image


def test_contour_app():
    """Run contour finding application test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel

        shape = "circle"

        ima1 = get_test_image("fabry-perot1.jpg")
        ima1.metadata["colormap"] = "gray"
        xc, yc, r = 601.0, 556.0, 457.0
        roi = cdl.obj.create_image_roi("circle", [xc, yc, r])
        ima1.roi = roi
        panel.add_object(ima1)
        param = cdl.param.ContourShapeParam.create(shape=shape)
        panel.processor.compute_contour_shape(param)
        take_plotwidget_screenshot(panel, "contour_fabryperot_test")

        ima2 = get_test_image("fabry-perot2.jpg")
        ima2.metadata["colormap"] = "gray"
        ima2.roi = roi
        panel.add_object(ima2)
        panel.processor.compute_contour_shape(param)

        param = cdl.param.LineProfileParam.create(direction="horizontal", row=554)
        panel.processor.compute_line_profile(param)

        param = cdl.param.AverageProfileParam.create(
            direction="horizontal", row1=550, row2=560
        )
        panel.processor.compute_average_profile(param)


if __name__ == "__main__":
    test_contour_app()
