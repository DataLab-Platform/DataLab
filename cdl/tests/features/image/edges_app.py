# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Edges processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from skimage.data import human_mitosis  # pylint: disable=no-name-in-module

import cdl.param
from cdl.obj import create_image
from cdl.tests import cdl_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run ROI unit test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        proc = panel.processor
        for paramclass, compute_method, name in (
            (cdl.param.CannyParam, proc.compute_canny, "Canny filter"),
            (None, proc.compute_roberts, "Roberts filter"),
            (None, proc.compute_prewitt, "Prewitt filter"),
            (None, proc.compute_prewitt_h, "Prewitt horizontal filter"),
            (None, proc.compute_prewitt_v, "Prewitt vertical filter"),
            (None, proc.compute_sobel, "Sobel filter"),
            (None, proc.compute_sobel_h, "Sobel horizontal filter"),
            (None, proc.compute_sobel_v, "Sobel vertical filter"),
            (None, proc.compute_scharr, "Scharr filter"),
            (None, proc.compute_scharr_h, "Scharr horizontal filter"),
            (None, proc.compute_scharr_v, "Scharr vertical filter"),
            (None, proc.compute_farid, "Farid filter"),
            (None, proc.compute_farid_h, "Farid horizontal filter"),
            (None, proc.compute_farid_v, "Farid vertical filter"),
            (None, proc.compute_laplace, "Laplace filter"),
        ):
            image = create_image(f"Testing {name}", human_mitosis())
            panel.add_object(image)
            param = None if paramclass is None else paramclass()
            if param is None:
                compute_method()
            else:
                compute_method(param)


if __name__ == "__main__":
    test()
