# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

import numpy as np

from cdl.obj import create_image
from cdl.param import FWHMParam
from cdl.tests import cdl_app_context
from cdl.tests import data as test_data

SHOW = True  # Show test in GUI-based test launcher


def create_image_with_resultshapes():
    """Create test image with resultshapes"""
    data = test_data.create_2d_gaussian(600, np.uint16, x0=2.0, y0=3.0)
    image = create_image("Test image with metadata", data)
    for mshape in test_data.create_resultshapes():
        mshape.add_to(image)
    return image


def test():
    """Result shapes test"""
    obj1 = test_data.create_test_image1()
    obj2 = create_image_with_resultshapes()
    obj2.roi = np.array([[10, 10, 60, 400]], int)
    with cdl_app_context(console=False) as win:
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_test_signal2(noised=noised)
            panel.add_object(sig)
            panel.processor.compute_fwhm(FWHMParam())
            panel.processor.compute_fw1e2()
        panel.objview.select_objects((0, 1))
        panel.show_results()
        win.switch_to_panel("image")
        panel = win.imagepanel
        for obj in (obj1, obj2):
            panel.add_object(obj)
        panel.show_results()


if __name__ == "__main__":
    test()
