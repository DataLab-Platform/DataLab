# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Flat field test

Testing the following:
  - Create a gaussian image (raw data)
  - Create a random noised image (flat data)
  - Compute the flat field image
"""

# guitest: show

from datalab.config import _
from datalab.tests import datalab_test_app_context
from sigima.objects import Gauss2DParam, UniformRandom2DParam
from sigima.proc.image import FlatFieldParam


def test_flatfield():
    """Run flat field test scenario"""
    with datalab_test_app_context() as win:
        panel = win.imagepanel

        param0 = Gauss2DParam.create(title=_("Raw data (2D-Gaussian)"))
        ima0 = panel.new_object(param0, edit=False)
        param1 = UniformRandom2DParam.create(
            title=_("Flat data (Uniform random)"), vmax=5
        )
        ima1 = panel.new_object(param1, edit=False)

        panel.objview.select_objects([ima0])
        ffp = FlatFieldParam()
        ffp.threshold = 80
        panel.processor.run_feature("flatfield", ima1, ffp)


if __name__ == "__main__":
    test_flatfield()
