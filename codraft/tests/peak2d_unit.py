# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Image peak detection test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import time

from guiqwt.builder import make

from codraft.core.computation.image import get_2d_peaks_coords
from codraft.tests.data import get_peak2d_data
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def exec_image_peak_detection_func(data):
    """Compare image peak detection methods"""
    items = [make.image(data, interpolation="linear", colormap="hsv")]
    t0 = time.time()
    coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    for x, y in coords:
        items.append(make.marker((x, y)))
    print(f"Calculation time: {int(dt * 1e3):d} ms\n")
    print(coords)
    view_image_items(items)


def peak2d_test():
    """2D peak detection test"""
    with qt_app_context():
        exec_image_peak_detection_func(get_peak2d_data())


if __name__ == "__main__":
    peak2d_test()
