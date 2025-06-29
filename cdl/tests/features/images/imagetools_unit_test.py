# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image tools test

Simple image dialog for testing all image tools available in DataLab
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from sigima_.tests.data import create_noisygauss_image
from sigima_.tests.vistools import view_image_items


def test_image_tools_unit():
    """Image tools test"""
    with qt_app_context():
        data = create_noisygauss_image().data
        items = [make.image(data, interpolation="nearest", eliminate_outliers=2.0)]
        view_image_items(items)


if __name__ == "__main__":
    test_image_tools_unit()
