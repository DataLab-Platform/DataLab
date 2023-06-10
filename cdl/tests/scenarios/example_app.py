# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Example of high-level test scenario

Create an image object from Scikit-image human mitosis sample,
then open DataLab to show it.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show,skip

import cdl.param
from cdl.obj import create_image
from cdl.tests import cdl_app_context
from cdl.tests.data import get_test_image


def test():
    """Example of high-level test scenario"""
    with cdl_app_context(console=False) as win:
        panel = win.imagepanel
        data = get_test_image("flower.npy").data
        image = create_image("Test image with peaks", data)
        panel.add_object(image)
        panel.processor.compute_roberts()
        data_size = data.shape[0]
        n = data_size // 5
        m = int(n * 1.25)
        panel.processor.extract_roi([[n, m, data_size - n, data_size - m]])
        param = cdl.param.BlobOpenCVParam()
        param.min_dist_between_blobs = 0
        param.filter_by_color = False
        param.min_area = 500
        param.max_area = 2000
        param.filter_by_circularity = True
        param.min_circularity = 0.2
        panel.processor.compute_blob_opencv(param)


if __name__ == "__main__":
    test()
