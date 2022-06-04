# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Flat field test

Testing the following:
  - Create a gaussian image (raw data)
  - Create a random noised image (flat data)
  - Compute the flat field image
"""

from codraft.config import _
from codraft.core.gui.processor.image import FlatFieldParam
from codraft.core.model.base import UniformRandomParam
from codraft.core.model.image import Gauss2DParam, ImageTypes, new_image_param
from codraft.tests import codraft_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run flat field test scenario"""
    with codraft_app_context() as win:
        panel = win.imagepanel

        i0p = new_image_param(_("Raw data (2D-Gaussian)"), itype=ImageTypes.GAUSS)
        panel.new_object(i0p, addparam=Gauss2DParam(), edit=False)
        i1p = new_image_param(
            _("Flat data (Uniform random)"), itype=ImageTypes.UNIFORMRANDOM
        )
        addp = UniformRandomParam()
        addp.vmax = 5
        panel.new_object(i1p, addparam=addp, edit=False)

        panel.objlist.select_rows((0, 1))
        ffp = FlatFieldParam()
        ffp.threshold = 80
        panel.processor.flat_field_correction(ffp)


if __name__ == "__main__":
    test()
