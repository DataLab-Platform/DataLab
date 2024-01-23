# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

import numpy as np

import cdl.obj
import cdl.param
from cdl.tests import cdltest_app_context
from cdl.tests import data as test_data


def create_image_with_resultshapes():
    """Create test image with resultshapes"""
    newparam = cdl.obj.new_image_param(
        height=600,
        width=600,
        title="Test image (with result shapes)",
        itype=cdl.obj.ImageTypes.GAUSS,
        dtype=cdl.obj.ImageDatatypes.UINT16,
    )
    addparam = cdl.obj.Gauss2DParam.create(x0=2, y0=3)
    image = cdl.obj.create_image_from_param(newparam, addparam)
    for mshape in test_data.create_resultshapes():
        mshape.add_to(image)
    return image


def test_resultshapes():
    """Result shapes test"""
    obj1 = test_data.create_sincos_image()
    obj2 = create_image_with_resultshapes()
    obj2.roi = np.array([[10, 10, 60, 400]], int)
    with cdltest_app_context(console=False) as win:
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_noisy_signal(noised=noised)
            panel.add_object(sig)
            panel.processor.compute_fwhm(cdl.param.FWHMParam())
            panel.processor.compute_fw1e2()
        panel.objview.select_objects((1, 2))
        panel.show_results()
        panel.plot_results()
        win.set_current_panel("image")
        panel = win.imagepanel
        for obj in (obj1, obj2):
            panel.add_object(obj)
        panel.show_results()
        panel.plot_results()


if __name__ == "__main__":
    test_resultshapes()
