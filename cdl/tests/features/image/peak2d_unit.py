# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image peak detection test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import time

from guiqwt.builder import make

from cdl.algorithms.image import get_2d_peaks_coords
from cdl.env import execenv
from cdl.tests.data import get_peak2d_data
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items


def exec_image_peak_detection_func(data):
    """Compare image peak detection methods"""
    items = [make.image(data, interpolation="linear", colormap="hsv")]
    t0 = time.time()
    coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    for x, y in coords:
        items.append(make.marker((x, y)))
    execenv.print(f"Calculation time: {int(dt * 1e3):d} ms")
    execenv.print(f"  => {coords.tolist()}")
    view_image_items(items)


def peak2d_test():
    """2D peak detection test"""
    with qt_app_context():
        exec_image_peak_detection_func(get_peak2d_data(multi=False))


if __name__ == "__main__":
    peak2d_test()
