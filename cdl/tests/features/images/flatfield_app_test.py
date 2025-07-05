# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Flat field test

Testing the following:
  - Create a gaussian image (raw data)
  - Create a random noised image (flat data)
  - Compute the flat field image
"""

# guitest: show

from sigima.computation.image import FlatFieldParam
from sigima.obj import Gauss2DParam, ImageTypes, NewImageParam, UniformRandomParam

from cdl.config import _
from cdl.tests import cdltest_app_context


def test_flatfield():
    """Run flat field test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel

        ima0 = panel.new_object(
            NewImageParam.create(
                title=_("Raw data (2D-Gaussian)"), itype=ImageTypes.GAUSS
            ),
            extra_param=Gauss2DParam(),
            edit=False,
        )
        addp = UniformRandomParam()
        addp.vmax = 5
        ima1 = panel.new_object(
            NewImageParam.create(
                title=_("Flat data (Uniform random)"), itype=ImageTypes.UNIFORMRANDOM
            ),
            extra_param=addp,
            edit=False,
        )

        panel.objview.select_objects([ima0])
        ffp = FlatFieldParam()
        ffp.threshold = 80
        panel.processor.run_feature("flatfield", ima1, ffp)


if __name__ == "__main__":
    test_flatfield()
