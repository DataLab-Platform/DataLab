# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Blob detection application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.param
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
            (sigima.param.BlobDOGParam, "blob_dog", "BlobDOG"),
            (sigima.param.BlobDOHParam, "blob_doh", "BlobDOH"),
            (sigima.param.BlobLOGParam, "blob_log", "BlobLOG"),
            (sigima.param.BlobOpenCVParam, "blob_opencv", "BlobOpenCV"),
        ):
            param = paramclass()
            image = create_image(name, data)
            image.add_label_with_title()
            panel.add_object(image)
            proc.run_feature(compute_method, param)

        # Testing distribute_on_grid and reset_positions
        # ==============================================
        panel.objview.selectAll()
        param = sigima.param.GridParam.create(cols=2, colspac=10, rowspac=10)
        proc.distribute_on_grid(param)


if __name__ == "__main__":
    test_blobs()
