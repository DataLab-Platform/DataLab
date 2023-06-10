# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image contrast test on problematic data

Emphasizing contrast calculation issue when data contains outlier points (e.g. hot
spots) thus preventing from cleaning up contrast histogram or showing valid Z-axis.
"""

# guitest: show

from cdl.tests.data import get_test_image
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_images


def contrast_test():
    """Contrats test"""
    with qt_app_context():
        view_images(get_test_image("contrast_test_data.npy").data)


if __name__ == "__main__":
    contrast_test()
