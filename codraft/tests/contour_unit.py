# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Contour finding test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import sys
import time

from guiqwt.builder import make

from codraft.core.computation.image import get_2d_peaks_coords, get_contour_shapes
from codraft.env import execenv
from codraft.tests.data import get_peak2d_data
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def make_ellipse(x1, y1, x2, y2, x3, y3, x4, y4):
    """Make ellipse shape plot item"""
    item = make.ellipse(x1, y1, x2, y2)
    item.switch_to_ellipse()
    item.set_ydiameter(x3, y3, x4, y4)
    return item


def exec_contour_test(data):
    """Find contours"""
    items = [make.image(data, interpolation="linear", colormap="hsv")]
    t0 = time.time()
    peak_coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    for x, y in peak_coords:
        items.append(make.marker((x, y)))
    execenv.print(f"Calculation time: {int(dt * 1e3):d} ms\n", file=sys.stderr)
    execenv.print(f"Peak coordinates: {peak_coords}")
    for shape in ("circle", "ellipse"):
        coords = get_contour_shapes(data, shape=shape)
        execenv.print(f"Coordinates ({shape}s): {coords}")
        for shapeargs in coords:
            if shape == "circle":
                item = make.circle(*shapeargs)
            else:
                item = make_ellipse(*shapeargs)
            items.append(item)
    view_image_items(items)


def contour_test():
    """2D peak detection test"""
    with qt_app_context():
        exec_contour_test(get_peak2d_data())


if __name__ == "__main__":
    contour_test()
