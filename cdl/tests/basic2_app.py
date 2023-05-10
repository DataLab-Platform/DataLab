# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Basic application launcher test 2

Create signal and image objects (with circles, rectangles, segments and markers),
then open DataLab to show them.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from cdl.app import run
from cdl.tests import data as test_data

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Simple test"""
    sig1 = test_data.create_test_signal1()
    sig2 = test_data.create_test_signal2()
    size = 2000
    ima1 = test_data.create_test_image1(size)
    ima2 = test_data.create_test_image2(size, with_annotations=True)
    run(objects=(sig1, sig2, ima1, ima2), size=(1200, 550))


if __name__ == "__main__":
    test()
