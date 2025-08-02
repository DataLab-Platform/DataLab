# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Blob detection application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.tests import datalab_test_app_context, skip_if_opencv_missing
from sigima.objects import create_image
from sigima.tests.data import get_test_image


def test_blobs():
    """Run blob detection application test scenario"""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        proc = panel.processor
        data = get_test_image("flower.npy").data

        # Testing blob detection
        # ======================
        for paramclass, compute_method, name in (
            (sigima.params.BlobDOGParam, "blob_dog", "BlobDOG"),
            (sigima.params.BlobDOHParam, "blob_doh", "BlobDOH"),
            (sigima.params.BlobLOGParam, "blob_log", "BlobLOG"),
            (sigima.params.BlobOpenCVParam, "blob_opencv", "BlobOpenCV"),
        ):
            param = paramclass()
            image = create_image(name, data)
            create_adapter_from_object(image).add_label_with_title()
            panel.add_object(image)
            with skip_if_opencv_missing():
                proc.run_feature(compute_method, param)

        # Testing distribute_on_grid and reset_positions
        # ==============================================
        panel.objview.selectAll()
        param = sigima.params.GridParam.create(cols=2, colspac=10, rowspac=10)
        proc.distribute_on_grid(param)


if __name__ == "__main__":
    test_blobs()
