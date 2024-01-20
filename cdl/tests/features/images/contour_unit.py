# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Contour finding test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import sys
import time

from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl.algorithms import coordinates
from cdl.algorithms.image import get_2d_peaks_coords, get_contour_shapes
from cdl.env import execenv
from cdl.tests.data import get_peak2d_data
from cdl.utils.vistools import view_image_items


def find_contours(data):
    """Find contours"""
    items = [make.image(data, interpolation="linear", colormap="hsv")]
    t0 = time.time()
    peak_coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    for x, y in peak_coords:
        items.append(make.marker((x, y)))
    execenv.print(f"Calculation time: {int(dt * 1e3):d} ms\n", file=sys.stderr)
    execenv.print(f"Peak coordinates: {peak_coords}")
    for shape in ("circle", "ellipse", "polygon"):
        coords = get_contour_shapes(data, shape=shape)
        execenv.print(f"Coordinates ({shape}s): {coords}")
        for shapeargs in coords:
            if shape == "circle":
                xc, yc, r = shapeargs
                x0, y0, x1, y1 = coordinates.circle_center_radius_to_diameter(xc, yc, r)
                item = make.circle(x0, y0, x1, y1)
            elif shape == "ellipse":
                xc, yc, a, b, theta = shapeargs
                coords = coordinates.ellipse_center_axes_angle_to_diameters(
                    xc, yc, a, b, theta
                )
                x0, y0, x1, y1, x2, y2, x3, y3 = coords
                item = make.ellipse(x0, y0, x1, y1, x2, y2, x3, y3)
            else:
                # `shapeargs` is a flattened array of x, y coordinates
                x, y = shapeargs[::2], shapeargs[1::2]
                item = make.polygon(x, y, closed=False)
            items.append(item)
    view_image_items(items)


def test_contour():
    """2D peak detection test"""
    with qt_app_context():
        find_contours(get_peak2d_data())


if __name__ == "__main__":
    test_contour()
