# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Flat field test

Testing the following:
  - Create a gaussian image (raw data)
  - Create a random noised image (flat data)
  - Compute the flat field image
"""

# guitest: show

from cdl.config import _
from cdl.obj import Gauss2DParam, ImageTypes, UniformRandomParam, new_image_param
from cdl.param import FlatFieldParam
from cdl.tests import cdltest_app_context


def test_flatfield():
    """Run flat field test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel

        ima0 = panel.new_object(
            new_image_param(_("Raw data (2D-Gaussian)"), itype=ImageTypes.GAUSS),
            addparam=Gauss2DParam(),
            edit=False,
        )
        addp = UniformRandomParam()
        addp.vmax = 5
        ima1 = panel.new_object(
            new_image_param(
                _("Flat data (Uniform random)"), itype=ImageTypes.UNIFORMRANDOM
            ),
            addparam=addp,
            edit=False,
        )

        panel.objview.select_objects([ima0])
        ffp = FlatFieldParam()
        ffp.threshold = 80
        panel.processor.compute_flatfield(ima1, ffp)


if __name__ == "__main__":
    test_flatfield()
