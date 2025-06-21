# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Basic application launcher test 2

Create signal and image objects (with circles, rectangles, segments and markers),
then open DataLab to show them.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.app import run
from cdl.tests import data as test_data
from sigima_.obj import NewImageParam


def test_launcher2():
    """Simple test"""
    sig1 = test_data.create_paracetamol_signal()
    sig2 = test_data.create_noisy_signal()
    param = NewImageParam.create(height=2000, width=2000)
    ima1 = test_data.create_sincos_image(param)
    ima2 = test_data.create_noisygauss_image(param, add_annotations=True)
    run(objects=(sig1, sig2, ima1, ima2), size=(1200, 550))


if __name__ == "__main__":
    test_launcher2()
