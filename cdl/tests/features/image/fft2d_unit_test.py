# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context

import cdl.algorithms.image as alg
import cdl.core.computation.image as cpi
import cdl.tests.data as ctd
from cdl.env import execenv
from cdl.utils.vistools import view_images_side_by_side


def test_image_fft_interactive():
    """2D FFT interactive test."""
    with qt_app_context():
        # Create a 2D ring image
        execenv.print("Generating 2D ring image...", end=" ")
        data = ctd.create_ring_image(ctd.RingParam()).data
        execenv.print("OK")

        # FFT
        execenv.print("Computing FFT of image...", end=" ")
        f = alg.fft2d(data)
        data2 = alg.ifft2d(f)
        execenv.print("OK")
        execenv.print("Comparing original and FFT/iFFT images...", end=" ")
        np.testing.assert_almost_equal(data, data2, decimal=10)
        execenv.print("OK")

        images = [data, f.real, f.imag, np.abs(f), data2.real, data2.imag]
        titles = ["Original", "Re(FFT)", "Im(FFT)", "Abs(FFT)", "Re(iFFT)", "Im(iFFT)"]
        view_images_side_by_side(images, titles, rows=2, title="2D FFT/iFFT")


@pytest.mark.validation
def test_image_fft() -> None:
    """2D FFT validation test."""
    ima1 = ctd.create_checkerboard()
    fft = cpi.compute_fft(ima1)
    ifft = cpi.compute_ifft(fft)

    # Check that the inverse FFT reconstructs the original image
    assert np.allclose(
        ima1.data, ifft.data.real
    ), "Checkerboard image FFT/iFFT reconstruction failed"

    # Parseval's Theorem Validation
    original_energy = np.sum(np.abs(ima1.data) ** 2)
    transformed_energy = np.sum(np.abs(fft.data) ** 2) / (ima1.data.size)
    assert np.isclose(original_energy, transformed_energy)


@pytest.mark.skip(reason="Already covered by the `test_image_fft` test.")
@pytest.mark.validation
def test_image_ifft() -> None:
    """2D iFFT validation test."""
    # This is just a way of marking the iFFT test as a validation test because it is
    # already covered by the FFT test above (there is no need to repeat the same test).


if __name__ == "__main__":
    test_image_fft_interactive()
    test_image_fft()
