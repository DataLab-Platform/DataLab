# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import time

from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl.env import execenv
from sigima_.algorithms.image import get_2d_peaks_coords
from sigima_.tests.data import get_peak2d_data
from sigima_.tests.vistools import view_image_items


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


def test_peak2d_unit():
    """2D peak detection test"""
    with qt_app_context():
        exec_image_peak_detection_func(get_peak2d_data(multi=False))


if __name__ == "__main__":
    test_peak2d_unit()
