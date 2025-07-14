# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.objects

from datalab.tests import cdltest_app_context


def test_fft2d_app() -> None:
    """FFT application test."""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        newparam = sigima.objects.NewImageParam.create(
            itype=sigima.objects.ImageTypes.GAUSS, width=100, height=100
        )
        extra_param = sigima.objects.Gauss2DParam()
        i1 = sigima.objects.create_image_from_param(newparam, extra_param=extra_param)
        panel.add_object(i1)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("ifft")


if __name__ == "__main__":
    test_fft2d_app()
