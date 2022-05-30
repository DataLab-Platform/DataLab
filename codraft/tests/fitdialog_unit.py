# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""Curve fitting dialog test

Testing the multi-Gaussian fit dialog.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


import numpy as np

from codraft.core.computation.signal import peak_indexes
from codraft.tests.data import get_test_fnames
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.tests import get_default_test_name
from codraft.widgets.fitdialog import multigaussianfit

SHOW = True  # Show test in GUI-based test launcher

# TODO: [P2] Check out curve fit parameters in GUI
#  (min/max values appear to be incorrect)


def test():
    """Test function"""
    with qt_app_context():
        x, y = np.loadtxt(get_test_fnames("paracetamol.txt")[0], delimiter=",").T
        peakindexes = peak_indexes(y)
        print(multigaussianfit(x, y, peakindexes, name=get_default_test_name("00")))


if __name__ == "__main__":
    test()
