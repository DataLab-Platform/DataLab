# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Basic application launcher test 3

- Iterate over all image creation types and data types
- For each image, test all 1->1 computations
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from cdl.core.model.image import (
    Gauss2DParam,
    ImageDatatypes,
    ImageTypes,
    new_image_param,
)
from cdl.tests import cdl_app_context
from cdl.tests.newobject_unit import iterate_image_creation
from cdl.tests.scenario_sig_app import test_compute_11_operations

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Simple test"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        newparam = new_image_param(itype=ImageTypes.GAUSS, dtype=ImageDatatypes.UINT8)
        addparam = Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
        panel.new_object(newparam, addparam=addparam, edit=False)
        panel.processor.compute_wiener()

        for image in iterate_image_creation(500, non_zero=True):
            panel.add_object(image)
            test_compute_11_operations(panel, 0)
            panel.remove_all_objects()


if __name__ == "__main__":
    test()
