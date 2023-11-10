# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

import numpy as np

import cdlapp.obj
import cdlapp.param
from cdlapp.tests import cdl_app_context
from cdlapp.tests import data as test_data


def create_image_with_resultshapes():
    """Create test image with resultshapes"""
    newparam = cdlapp.obj.new_image_param(
        height=600,
        width=600,
        title="Test image (with result shapes)",
        itype=cdlapp.obj.ImageTypes.GAUSS,
        dtype=cdlapp.obj.ImageDatatypes.UINT16,
    )
    addparam = cdlapp.obj.Gauss2DParam.create(x0=2, y0=3)
    image = cdlapp.obj.create_image_from_param(newparam, addparam)
    for mshape in test_data.create_resultshapes():
        mshape.add_to(image)
    return image


def test():
    """Result shapes test"""
    obj1 = test_data.create_sincos_image()
    obj2 = create_image_with_resultshapes()
    obj2.roi = np.array([[10, 10, 60, 400]], int)
    with cdl_app_context(console=False) as win:
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_noisy_signal(noised=noised)
            panel.add_object(sig)
            panel.processor.compute_fwhm(cdlapp.param.FWHMParam())
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
