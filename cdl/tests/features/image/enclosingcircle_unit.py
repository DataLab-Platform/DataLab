# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Enclosing circle test

Testing enclsoing circle function on various test images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from guiqwt.builder import make

from cdl.algorithms.image import get_centroid_fourier, get_enclosing_circle
from cdl.config import _
from cdl.env import execenv
from cdl.tests.data import get_laser_spot_data
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def test_enclosingcircle(data):
    """Enclosing circle test function"""
    items = []
    items += [make.image(data, interpolation="nearest", eliminate_outliers=1.0)]

    # Computing centroid coordinates
    row, col = get_centroid_fourier(data)
    label = _("Centroid") + " (%d, %d)"
    execenv.print(label % (row, col))
    cursor = make.xcursor(col, row, label=label)
    cursor.set_resizable(False)
    cursor.set_movable(False)
    items.append(cursor)

    x, y, radius = get_enclosing_circle(data)
    circle = make.circle(x - radius, y - radius, x + radius, y + radius)
    circle.set_readonly(True)
    circle.set_resizable(False)
    circle.set_movable(False)
    items.append(circle)
    execenv.print(x, y, radius)
    execenv.print("")

    view_image_items(items)


def enclosing_circle_test():
    """Test"""
    with qt_app_context():
        for data in get_laser_spot_data():
            test_enclosingcircle(data)


if __name__ == "__main__":
    enclosing_circle_test()
