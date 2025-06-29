# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Enclosing circle test

Testing enclsoing circle function on various test images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl.config import _
from cdl.env import execenv
from sigima_.algorithms.image import get_centroid_fourier, get_enclosing_circle
from sigima_.tests.data import get_laser_spot_data
from sigima_.tests.vistools import view_image_items


def __enclosingcircle_test(data):
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


def test_enclosing_circle():
    """Test"""
    with qt_app_context():
        for data in get_laser_spot_data():
            __enclosingcircle_test(data)


if __name__ == "__main__":
    test_enclosing_circle()
