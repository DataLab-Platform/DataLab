# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Image tools test

Simple image dialog for testing all image tools available in CobraDataLab
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
from guiqwt.builder import make

from cdl.tests.data import create_2d_gaussian, create_2d_random
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def image_tools_test(size=2000, dtype=np.uint16):
    """Image tools test"""
    with qt_app_context():
        znoise = create_2d_random(size, dtype)
        zgauss = create_2d_gaussian(size, dtype, x0=5.0, y0=-3.0)
        data = zgauss + znoise
        items = [make.image(data, interpolation="nearest", eliminate_outliers=2.0)]
        view_image_items(items)


if __name__ == "__main__":
    image_tools_test()
