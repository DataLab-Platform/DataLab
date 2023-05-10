# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image contrast test on problematic data

Emphasizing contrast calculation issue when data contains outlier points (e.g. hot
spots) thus preventing from cleaning up contrast histogram or showing valid Z-axis.
"""

import numpy as np

from cdl.utils.qthelpers import qt_app_context
from cdl.utils.tests import get_test_fnames
from cdl.utils.vistools import view_images

SHOW = True  # Show test in GUI-based test launcher


def contrast_test():
    """Contrats test"""
    with qt_app_context():
        view_images(np.load(get_test_fnames("contrast_test_data.npy")[0]))


if __name__ == "__main__":
    contrast_test()
