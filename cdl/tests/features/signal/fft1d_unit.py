# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np

from cdl.algorithms.signal import xy_fft, xy_ifft
from cdl.core.model.signal import (
    SignalTypes,
    create_signal_from_param,
    new_signal_param,
)
from cdl.env import execenv
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_curves

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Signal peak dialog test"""
    with qt_app_context():
        newparam = new_signal_param(stype=SignalTypes.COSINUS, size=10000)
        s1 = create_signal_from_param(newparam)
        t, y = s1.xydata
        f, s = xy_fft(t, y)
        t2, y2 = xy_ifft(f, s)
        execenv.print("Comparing original and FFT/IFFT signals:", end=" ")
        np.testing.assert_almost_equal(t, t2, decimal=3)
        np.testing.assert_almost_equal(y, y2, decimal=10)
        execenv.print("OK")
        view_curves([(t, y), (t2, y2)])


if __name__ == "__main__":
    test()