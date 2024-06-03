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

import cdl.algorithms.signal as alg
import cdl.core.computation.signal as cps
import cdl.obj
import cdl.tests.data as ctd
from cdl.env import execenv
from cdl.utils.tests import check_array_result, check_scalar_result
from cdl.utils.vistools import view_curves


def test_signal_fft_interactive() -> None:
    """1D FFT interactive test."""
    with qt_app_context():
        newparam = cdl.obj.new_signal_param(stype=cdl.obj.SignalTypes.COSINUS, size=500)

        # *** Note ***
        #
        # We set xmin to 0.0 to be able to compare the X data of the original and
        # reconstructed signals, because the FFT do not preserve the X data (phase is
        # lost, sampling rate is assumed to be constant), so that comparing the X data
        # is not meaningful if xmin is different.
        newparam.xmin = 0.0

        s1 = cdl.obj.create_signal_from_param(newparam)
        t, y = s1.xydata
        f, s = alg.fft1d(t, y)
        t2, y2 = alg.ifft1d(f, s)
        execenv.print("Comparing original and FFT/iFFT signals...", end=" ")
        execenv.print("OK")
        check_array_result("Signal FFT/iFFT X data", t2, t)
        check_array_result("Signal FFT/iFFT Y data", y2, y)
        view_curves([(t, y), (t2, y2)])


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    freq = 50.0
    size = 10000

    # See note in the interactive test above
    xmin = 0.0

    s1 = ctd.create_periodic_signal(
        cdl.obj.SignalTypes.COSINUS, freq=freq, size=size, xmin=xmin
    )
    fft = cps.compute_fft(s1)
    ifft = cps.compute_ifft(fft)

    # Check that the inverse FFT reconstructs the original signal
    check_array_result("Cosine signal FFT/iFFT X reconstruction", s1.y, ifft.y.real)
    check_array_result("Cosine signal FFT/iFFT Y reconstruction", s1.x, ifft.x)

    # Check FFT properties
    mag = np.abs(fft.y)

    # Find the peak in the FFT
    ipk1, ipk2 = np.argmax(mag[: size // 2]), np.argmax(mag[size // 2 :]) + size // 2
    fpk1, fpk2 = fft.x[ipk1], fft.x[ipk2]

    # Verify the peak frequencies are correct
    check_scalar_result("Cosine negative frequency", fpk1, -freq, rtol=0.001)
    check_scalar_result("Cosine positive frequency", fpk2, freq, rtol=0.001)

    # Verify the magnitude at the peak
    exp_mag = size / 2
    check_scalar_result("Cosine peak magnitude", mag[ipk1], exp_mag, rtol=0.05)

    # Verify the symmetry of the FFT
    check_array_result(
        "FFT symmetry",
        mag[1 : size // 2],
        mag[1 + size // 2 :][::-1],
    )


@pytest.mark.skip(reason="Already covered by the `test_signal_fft` test.")
@pytest.mark.validation
def test_signal_ifft() -> None:
    """1D iFFT validation test."""
    # This is just a way of marking the iFFT test as a validation test because it is
    # already covered by the FFT test above (there is no need to repeat the same test).


if __name__ == "__main__":
    test_signal_fft_interactive()
    test_signal_fft()
