# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding application test with Fabry-Perot images
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima_.obj as so
import sigima_.param
from cdl.tests import cdltest_app_context, take_plotwidget_screenshot
from cdl.tests.data import get_test_image


def test_contour_app() -> None:
    """Run contour finding application test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel

        shape = "circle"

        ima1 = get_test_image("fabry-perot1.jpg")
        ima1.set_metadata_option("colormap", "gray")
        xc, yc, r = 601.0, 556.0, 457.0
        roi = so.create_image_roi("circle", [xc, yc, r])
        ima1.roi = roi
        panel.add_object(ima1)
        param = sigima_.param.ContourShapeParam.create(shape=shape)
        panel.processor.run_feature("contour_shape", param)
        take_plotwidget_screenshot(panel, "contour_fabryperot_test")

        ima2 = get_test_image("fabry-perot2.jpg")
        ima2.set_metadata_option("colormap", "gray")
        ima2.roi = roi
        panel.add_object(ima2)
        panel.processor.run_feature("contour_shape", param)

        param = sigima_.param.LineProfileParam.create(direction="horizontal", row=554)
        panel.processor.run_feature("line_profile", param)

        param = sigima_.param.AverageProfileParam.create(
            direction="horizontal", row1=550, row2=560
        )
        panel.processor.run_feature("average_profile", param)


if __name__ == "__main__":
    test_contour_app()
