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
from cdl.utils.vistools import view_curves


def test_signal_fft_interactive() -> None:
    """1D FFT interactive test."""
    with qt_app_context():
        newparam = cdl.obj.new_signal_param(stype=cdl.obj.SignalTypes.COSINUS, size=500)
        s1 = cdl.obj.create_signal_from_param(newparam)
        t, y = s1.xydata
        f, s = alg.fft1d(t, y)
        t2, y2 = alg.ifft1d(f, s)
        execenv.print("Comparing original and FFT/iFFT signals...", end=" ")
        np.testing.assert_almost_equal(t, t2, decimal=3)
        np.testing.assert_almost_equal(y, y2, decimal=10)
        execenv.print("OK")
        view_curves([(t, y), (t2, y2)])


@pytest.mark.validation
def test_signal_fft() -> None:
    """1D FFT validation test."""
    freq = 50.0
    size = 10000
    s1 = ctd.create_periodic_signal(cdl.obj.SignalTypes.COSINUS, freq=freq, size=size)
    fft = cps.compute_fft(s1)
    ifft = cps.compute_ifft(fft)

    # Check that the inverse FFT reconstructs the original signal
    assert np.allclose(
        s1.y, np.real(ifft.y)
    ), "Cosine signal FFT/IFFT reconstruction failed"

    # Check FFT properties
    fft_magnitude = np.abs(fft.y)
    fft_phase = np.angle(fft.y)
    freqs = fft.x

    # Find the peak in the FFT
    peak_index = np.argmax(fft_magnitude)
    peak_freq = freqs[peak_index]

    # Verify the peak frequency is correct
    assert np.isclose(
        peak_freq, freq, rtol=1e-3
    ), f"Expected peak at {freq} Hz, found at {peak_freq} Hz"

    # Verify the magnitude at the peak
    expected_magnitude = size / 2
    assert np.isclose(
        fft_magnitude[peak_index], expected_magnitude, rtol=0.05
    ), f"Expected magnitude {expected_magnitude}, found {fft_magnitude[peak_index]}"

    # Verify the symmetry of the FFT
    assert np.allclose(
        fft_magnitude[1 : size // 2], fft_magnitude[1 + size // 2 :][::-1]
    ), "FFT is not symmetric"

    # Verify the phase at the peak (should be 0 or π for a cosine)
    assert np.isclose(fft_phase[peak_index], 0, atol=0.5) or np.isclose(
        fft_phase[peak_index], np.pi, atol=0.5
    ), f"Expected phase at peak: 0 or π, found {fft_phase[peak_index]}"


@pytest.mark.skip(reason="Already covered by the `test_signal_fft` test.")
@pytest.mark.validation
def test_signal_ifft() -> None:
    """1D iFFT validation test."""
    # This is just a way of marking the iFFT test as a validation test because it is
    # already covered by the FFT test above (there is no need to repeat the same test).


if __name__ == "__main__":
    test_signal_fft_interactive()
    test_signal_fft()
