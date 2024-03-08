# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.algorithms.image import z_fft, z_ifft
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
        f = z_fft(data)
        data2 = z_ifft(f)
        execenv.print("OK")
        execenv.print("Comparing original and FFT/iFFT images...", end=" ")
        np.testing.assert_almost_equal(data, data2, decimal=10)
        execenv.print("OK")

        images = [data, f.real, f.imag, np.abs(f), data2.real, data2.imag]
        titles = ["Original", "Re(FFT)", "Im(FFT)", "Abs(FFT)", "Re(iFFT)", "Im(iFFT)"]
        view_images_side_by_side(images, titles, rows=2, title="2D FFT/iFFT")


if __name__ == "__main__":
    test_fft2d_unit()
