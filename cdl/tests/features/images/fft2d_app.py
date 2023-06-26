# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.obj import ImageTypes, create_image_from_param, new_image_param
from cdl.tests import cdl_app_context


def test():
    """FFT application test."""
    with cdl_app_context() as win:
        panel = win.imagepanel
        newparam = new_image_param(itype=ImageTypes.GAUSS, width=100, height=100)
        i1 = create_image_from_param(newparam)
        panel.add_object(i1)
        panel.processor.compute_fft()
        panel.processor.compute_ifft()


if __name__ == "__main__":
    test()
