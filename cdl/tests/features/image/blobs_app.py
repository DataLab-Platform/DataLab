# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Blob detection application test
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
        data = human_mitosis()

        # Testing blob detection
        # ======================
        for paramclass, compute_method, name in (
            (cdl.param.BlobDOGParam, proc.compute_blob_dog, "BlobDOG"),
            (cdl.param.BlobDOHParam, proc.compute_blob_doh, "BlobDOH"),
            (cdl.param.BlobLOGParam, proc.compute_blob_log, "BlobLOG"),
            (cdl.param.BlobOpenCVParam, proc.compute_blob_opencv, "BlobOpenCV"),
        ):
            param = paramclass()
            image = create_image(name, data)
            image.add_label_with_title()
            panel.add_object(image)
            compute_method(param)

        # Testing distribute_on_grid and reset_positions
        # ==============================================
        # We begin by selecting all objects, then we reset their positions. This does
        # not make sense except for coverage (because the objects are already at their
        # default positions) and it allows to finish by testing the distribution on
        # grid, which is more appropriate for the eventual final screenshot.
        panel.objview.selectAll()
        proc.reset_positions()  # No sense except for coverage
        param = cdl.param.GridParam()
        param.cols = 2
        param.colspac = param.rowspac = 10
        proc.distribute_on_grid(param)


if __name__ == "__main__":
    test()
