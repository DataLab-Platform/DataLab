# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Average application test
========================

The purpose of this test is to check that we can average a set of images without
overflowing the data type.

This test was written following a regression where the average of 10 images of
size 256x256 with a Gaussian distribution of values was overflowing the data type
uint8.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import numpy as np

import cdl.tests.data as ctd
import sigima_.obj as so
from cdl.tests import cdltest_app_context
from cdl.utils.tests import check_array_result


def test_image_average() -> None:
    """Average application test."""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        N, size = 10, 256
        dtype = so.ImageDatatypes.UINT8
        p = so.NewImageParam.create(height=size, width=size, dtype=dtype)
        data = ctd.create_2d_gaussian(size, np.dtype(dtype.value))
        for _idx in range(N):
            obj = so.create_image_from_param(p)
            obj.data = data
            panel.add_object(obj)
        panel.objview.select_groups([0])
        panel.processor.run_feature("average")
        res_data = panel.objview.get_sel_objects(include_groups=True)[0].data
    exp_data = np.array(data, dtype=float)
    check_array_result("Average", res_data, exp_data)


if __name__ == "__main__":
    test_image_average()
