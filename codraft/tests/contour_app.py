# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Contour finding application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from codraft.core.gui.processor.image import ContourShapeParam
from codraft.tests import codraft_app_context, take_plotwidget_screenshot
from codraft.tests.data import create_test_image3
from codraft.tests.roi_app import create_test_image_with_roi

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run ROI unit test scenario"""
    size = 200
    with codraft_app_context() as win:
        panel = win.imagepanel
        ima1 = create_test_image3(size)
        panel.add_object(ima1)
        param = ContourShapeParam()
        panel.processor.compute_contour_shape(param)
        take_plotwidget_screenshot(panel, "contour_test")
        ima2 = create_test_image_with_roi(size)
        panel.add_object(ima2)
        panel.processor.edit_regions_of_interest()
        panel.processor.compute_contour_shape(param)


if __name__ == "__main__":
    test()
