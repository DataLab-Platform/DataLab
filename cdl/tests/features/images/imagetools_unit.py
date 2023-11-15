# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image tools test

Simple image dialog for testing all image tools available in DataLab
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl.tests.data import create_noisygauss_image
from cdl.utils.vistools import view_image_items


def image_tools_test():
    """Image tools test"""
    with qt_app_context():
        data = create_noisygauss_image().data
        items = [make.image(data, interpolation="nearest", eliminate_outliers=2.0)]
        view_image_items(items)


if __name__ == "__main__":
    image_tools_test()
