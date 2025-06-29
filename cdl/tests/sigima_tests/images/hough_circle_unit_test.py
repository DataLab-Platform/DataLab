# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test using circle Hough transform
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np
import pytest
from skimage.feature import canny

from sigima_.algorithms.image import get_hough_circle_peaks
from sigima_.env import execenv
from sigima_.tests.data import get_peak2d_data


def __exec_hough_circle_test(data):
    """Peak detection using circle Hough transform"""
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    from sigima_.tests import vistools

    edges = canny(
        data,
        sigma=30,
        low_threshold=0.6,
        high_threshold=0.8,
        use_quantiles=True,
    )
    items = [
        make.image(
            data, interpolation="linear", colormap="gray", eliminate_outliers=2.0
        ),
        make.image(
            np.array(edges, dtype=np.uint8),
            interpolation="linear",
            colormap="hsv",
            alpha_function="tanh",
        ),
    ]
    coords = get_hough_circle_peaks(
        edges, min_radius=25, max_radius=35, min_distance=70
    )
    execenv.print(f"Coordinates: {coords}")
    for shapeargs in coords:
        xc, yc, r = shapeargs
        item = make.circle(xc - r, yc, xc + r, yc)
        items.append(item)
    vistools.view_image_items(items)


@pytest.mark.gui
def test_hough_circle():
    """2D peak detection test"""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        __exec_hough_circle_test(get_peak2d_data(multi=False))


if __name__ == "__main__":
    test_hough_circle()
