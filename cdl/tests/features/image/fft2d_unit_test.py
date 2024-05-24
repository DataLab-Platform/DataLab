# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.algorithms.image import fft2d, ifft2d
from cdl.env import execenv
from cdl.tests.data import RingParam, create_ring_image
from cdl.utils.vistools import view_images_side_by_side


def test_fft2d_unit():
    """2D FFT unit test."""
    with qt_app_context():
        # Create a 2D ring image
        execenv.print("Generating 2D ring image...", end=" ")
        image = create_ring_image(RingParam())
        execenv.print("OK")
        data = image.data

        # FFT
        execenv.print("Computing FFT of image...", end=" ")
        f = fft2d(data)
        data2 = ifft2d(f)
        execenv.print("OK")
        execenv.print("Comparing original and FFT/iFFT images...", end=" ")
        np.testing.assert_almost_equal(data, data2, decimal=10)
        execenv.print("OK")

        images = [data, f.real, f.imag, np.abs(f), data2.real, data2.imag]
        titles = ["Original", "Re(FFT)", "Im(FFT)", "Abs(FFT)", "Re(iFFT)", "Im(iFFT)"]
        view_images_side_by_side(images, titles, rows=2, title="2D FFT/iFFT")


if __name__ == "__main__":
    test_fft2d_unit()
