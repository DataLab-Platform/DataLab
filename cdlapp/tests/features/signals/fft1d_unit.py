# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Signal FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np

from cdlapp.algorithms.signal import xy_fft, xy_ifft
from cdlapp.env import execenv
from cdlapp.obj import SignalTypes, create_signal_from_param, new_signal_param
from cdlapp.utils.qthelpers import qt_app_context
from cdlapp.utils.vistools import view_curves


def test():
    """1D FFT unit test."""
    with qt_app_context():
        newparam = new_signal_param(stype=SignalTypes.COSINUS, size=10000)
        s1 = create_signal_from_param(newparam)
        t, y = s1.xydata
        f, s = xy_fft(t, y)
        t2, y2 = xy_ifft(f, s)
        execenv.print("Comparing original and FFT/iFFT signals...", end=" ")
        np.testing.assert_almost_equal(t, t2, decimal=3)
        np.testing.assert_almost_equal(y, y2, decimal=10)
        execenv.print("OK")
        view_curves([(t, y), (t2, y2)])


if __name__ == "__main__":
    test()
