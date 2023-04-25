# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Flat field test

Testing the following:
  - Create a gaussian image (raw data)
  - Create a random noised image (flat data)
  - Compute the flat field image
"""

from cdl.config import _
from cdl.core.gui.processor.image import FlatFieldParam
from cdl.core.model.base import UniformRandomParam
from cdl.core.model.image import Gauss2DParam, ImageTypes, new_image_param
from cdl.tests import cdl_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run flat field test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel

        i0p = new_image_param(_("Raw data (2D-Gaussian)"), itype=ImageTypes.GAUSS)
        panel.new_object(i0p, addparam=Gauss2DParam(), edit=False)
        i1p = new_image_param(
            _("Flat data (Uniform random)"), itype=ImageTypes.UNIFORMRANDOM
        )
        addp = UniformRandomParam()
        addp.vmax = 5
        panel.new_object(i1p, addparam=addp, edit=False)

        panel.objview.select_nums((0, 1))
        ffp = FlatFieldParam()
        ffp.threshold = 80
        panel.processor.flat_field_correction(ffp)


if __name__ == "__main__":
    test()
