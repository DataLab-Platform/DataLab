# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context

import cdl.core.computation.signal as cps
import cdl.obj
import cdl.param
from cdl.algorithms.signal import xy_fft, xy_ifft
from cdl.env import execenv
from cdl.obj import SignalTypes, create_signal_from_param, new_signal_param
from cdl.utils.vistools import view_curves


def test_signal_fft_interactive() -> None:
    """1D FFT interactive test."""
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


def create_cosinus_signal(size: int = 10000) -> cdl.obj.SignalObj:
    """Create a cosinus signal."""
    newparam = new_signal_param(stype=SignalTypes.COSINUS, size=size)
    return create_signal_from_param(newparam)


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    s1 = create_cosinus_signal()
    fft = cps.compute_fft(s1)
    ifft = cps.compute_ifft(fft)
    assert np.allclose(s1.y, np.real(ifft.y))


@pytest.mark.validation
def test_signal_abs() -> None:
    """Absolute value validation test."""
    s1 = create_cosinus_signal()
    abs_signal = cps.compute_abs(s1)
    assert np.allclose(np.abs(s1.y), abs_signal.y)


if __name__ == "__main__":
    test_signal_fft_interactive()
    test_signal_fft()
    test_signal_abs()
