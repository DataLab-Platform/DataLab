# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Contour finding test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import sys
import time

import pytest

from sigima_.algorithms import coordinates
from sigima_.algorithms.image import get_2d_peaks_coords, get_contour_shapes
from sigima_.env import execenv
from sigima_.tests.data import get_peak2d_data


def find_contours(data):
    """Find contours"""
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    from sigima_.tests import vistools

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
                x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
                item = make.circle(x0, y0, x1, y1)
            elif shape == "ellipse":
                xc, yc, a, b, theta = shapeargs
                coords = coordinates.ellipse_to_diameters(xc, yc, a, b, theta)
                x0, y0, x1, y1, x2, y2, x3, y3 = coords
                item = make.ellipse(x0, y0, x1, y1, x2, y2, x3, y3)
            else:
                # `shapeargs` is a flattened array of x, y coordinates
                x, y = shapeargs[::2], shapeargs[1::2]
                item = make.polygon(x, y, closed=False)
            items.append(item)
    vistools.view_image_items(items)


@pytest.mark.gui
def test_contour_interactive():
    """2D peak detection test"""
    data, _coords = get_peak2d_data()
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        find_contours(data)


if __name__ == "__main__":
    test_contour_interactive()
