# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.obj import ImageTypes, create_image_from_param, new_image_param
from cdl.tests import cdltest_app_context


def test_fft2d_app():
    """FFT application test."""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        newparam = new_image_param(itype=ImageTypes.GAUSS, width=100, height=100)
        i1 = create_image_from_param(newparam)
        panel.add_object(i1)
        panel.processor.compute("fft")
        panel.processor.compute("ifft")


if __name__ == "__main__":
    test_fft2d_app()
