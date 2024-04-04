# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Blob detection application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.obj import create_image
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image


def test_blobs():
    """Run blob detection application test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        proc = panel.processor
        data = get_test_image("flower.npy").data

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
        panel.objview.selectAll()
        param = cdl.param.GridParam.create(cols=2, colspac=10, rowspac=10)
        proc.distribute_on_grid(param)


if __name__ == "__main__":
    test_blobs()
