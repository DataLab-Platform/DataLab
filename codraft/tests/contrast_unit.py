# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Image contrast test on problematic data

Emphasizing contrast calculation issue when data contains outlier points (e.g. hot
spots) thus preventing from cleaning up contrast histogram or showing valid Z-axis.
"""

import numpy as np

from codraft.utils.qthelpers import qt_app_context
from codraft.utils.tests import get_test_fnames
from codraft.utils.vistools import view_images

SHOW = True  # Show test in GUI-based test launcher


def contrast_test():
    """Contrats test"""
    with qt_app_context():
        view_images(np.load(get_test_fnames("contrast_test_data.npy")[0]))


if __name__ == "__main__":
    contrast_test()
