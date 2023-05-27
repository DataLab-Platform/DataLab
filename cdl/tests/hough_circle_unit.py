# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image peak detection test using circle Hough transform
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np
from guiqwt.builder import make
from skimage.feature import canny

from cdl.algorithms.image import get_hough_circle_peaks
from cdl.env import execenv
from cdl.tests.data import get_peak2d_data
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def exec_hough_circle_test(data):
    """Peak detection using circle Hough transform"""
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
            alpha_mask=True,
        ),
    ]
    coords = get_hough_circle_peaks(
        edges, min_radius=25, max_radius=35, min_distance=70
    )
    execenv.print(f"Coordinates: {coords}")
    for shapeargs in coords:
        item = make.circle(*shapeargs)
        items.append(item)
    view_image_items(items)


def hough_circle_test():
    """2D peak detection test"""
    with qt_app_context():
        exec_hough_circle_test(get_peak2d_data(multi=False))


if __name__ == "__main__":
    hough_circle_test()
