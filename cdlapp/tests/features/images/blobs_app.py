# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Blob detection application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdlapp.param
from cdlapp.obj import create_image
from cdlapp.tests import cdl_app_context
from cdlapp.tests.data import get_test_image


def test():
    """Run blob detection application test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        proc = panel.processor
        data = get_test_image("flower.npy").data

        # Testing blob detection
        # ======================
        for paramclass, compute_method, name in (
            (cdlapp.param.BlobDOGParam, proc.compute_blob_dog, "BlobDOG"),
            (cdlapp.param.BlobDOHParam, proc.compute_blob_doh, "BlobDOH"),
            (cdlapp.param.BlobLOGParam, proc.compute_blob_log, "BlobLOG"),
            (cdlapp.param.BlobOpenCVParam, proc.compute_blob_opencv, "BlobOpenCV"),
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
        param = cdlapp.param.GridParam.create(cols=2, colspac=10, rowspac=10)
        proc.distribute_on_grid(param)


if __name__ == "__main__":
    test()
