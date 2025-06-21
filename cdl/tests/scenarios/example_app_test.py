# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Example of high-level test scenario

Create an image object from Scikit-image human mitosis sample,
then open DataLab to show it.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima_.obj as so
import sigima_.param as sp
from cdl.proxy import proxy_context
from cdl.tests import skip_if_opencv_missing
from cdl.tests.data import get_test_image


def test_example_app() -> None:
    """Example of high-level test scenario using proxy interface, so that it may
    be run remotely inside an already running DataLab instance, or in a new
    dedicated instance."""
    with proxy_context("local") as proxy:
        data = get_test_image("flower.npy").data
        image = so.create_image("Test image with peaks", data)
        proxy.add_object(image)
        proxy.calc("roberts")
        data_size = data.shape[0]
        n = data_size // 5
        roi = so.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        proxy.compute_roi_extraction(roi)
        param = sp.BlobOpenCVParam.create(
            min_dist_between_blobs=0.1,
            filter_by_color=False,
            min_area=500,
            max_area=2000,
            filter_by_circularity=True,
            min_circularity=0.2,
        )
        with skip_if_opencv_missing():
            proxy.calc("blob_opencv", param)


if __name__ == "__main__":
    test_example_app()
