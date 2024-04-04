# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Example of high-level test scenario

Create an image object from Scikit-image human mitosis sample,
then open DataLab to show it.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.obj import create_image
from cdl.proxy import proxy_context
from cdl.tests.data import get_test_image


def test_example_app():
    """Example of high-level test scenario using proxy interface, so that it may
    be run remotely inside an already running DataLab instance, or in a new
    dedicated instance."""
    with proxy_context("local") as proxy:
        data = get_test_image("flower.npy").data
        image = create_image("Test image with peaks", data)
        proxy.add_object(image)
        proxy.compute_roberts()
        data_size = data.shape[0]
        n = data_size // 5
        m = int(n * 1.25)
        param = cdl.param.ROIDataParam.create([[n, m, data_size - n, data_size - m]])
        proxy.compute_roi_extraction(param)
        param = cdl.param.BlobOpenCVParam.create(
            min_dist_between_blobs=0.1,
            filter_by_color=False,
            min_area=500,
            max_area=2000,
            filter_by_circularity=True,
            min_circularity=0.2,
        )
        proxy.compute_blob_opencv(param)


if __name__ == "__main__":
    test_example_app()
