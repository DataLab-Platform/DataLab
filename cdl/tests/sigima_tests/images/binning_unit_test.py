# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image pixel binning computation test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import time

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context
from numpy import ma
from plotpy.builder import make

import sigima_.computation.image as sigima_image
import sigima_.param
from sigima_.algorithms.image import BINNING_OPERATIONS, binning
from sigima_.env import execenv
from sigima_.tests.data import get_test_image
from sigima_.tests.vistools import view_image_items


def compare_binning_images(data: ma.MaskedArray) -> None:
    """Compare binning images

    Args:
        data: Image data
    """
    items = []
    items += [make.image(data, interpolation="nearest", eliminate_outliers=2.0)]
    # Computing pixel binning
    oa_t0 = time.time()
    for ix in range(1, 5):
        sx = 2**ix
        for iy in range(1, 5):
            sy = 2**iy
            for operation in BINNING_OPERATIONS:
                t0 = time.time()
                bdata = binning(data, sx=sx, sy=sy, operation=operation)
                title = f"[{sx}x{sy},{operation}]"
                item = make.image(
                    bdata,
                    title=title,
                    interpolation="nearest",
                    eliminate_outliers=2.0,
                    xdata=[0, data.shape[1]],
                    ydata=[0, data.shape[0]],
                )
                item.hide()
                items.append(item)
                dt = time.time() - t0
                execenv.print(f"    {title}: {int(dt * 1e3):d} ms")
    oa_dt = time.time() - oa_t0
    execenv.print(f"    Overall calculation time: {int(oa_dt * 1e3):d} ms")
    view_image_items(items, title="Binning test", show_itemlist=True)


def test_binning_graphically() -> None:
    """Test binning computation and show results"""
    with qt_app_context():
        data = get_test_image("*.scor-data").data[:500, :500]
        execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
        compare_binning_images(data.view(ma.MaskedArray))


@pytest.mark.validation
def test_binning() -> None:
    """Validation test for binning computation"""

    # Implementation note:
    # ---------------------
    #
    # Pixel binning algorithm is validated graphically by comparing the results of
    # different binning operations and sizes: that is the purpose of the
    # `test_binning_graphically`` function.
    # Formal validation is not possible without reimplementation of the algorithm
    # here, which would be redundant and proove nothing. Instead, as a complementary
    # test, we only validate some basic properties of the binning algorithm:
    # - The output shape is correct
    # - The output data type is correct
    # - Some basic properties of the output data are correct (e.g. min, max, mean)

    src = get_test_image("*.scor-data")
    src.data = data = np.array(src.data[:500, :500], dtype=float)
    ny, nx = data.shape

    p = sigima_.param.BinningParam()
    for operation in p.operations:
        p.operation = operation
        for sx in range(1, 3):
            for sy in range(1, 5):
                p.sx = sx
                p.sy = sy
                rdata = data[: ny - (ny % sy), : nx - (nx % sx)]
                dst = sigima_image.binning(src, p)
                bdata = dst.data
                assert bdata.shape == (data.shape[0] // sy, data.shape[1] // sx)
                assert bdata.dtype == data.dtype
                if operation == "min":
                    assert bdata.min() == rdata.min()
                elif operation == "max":
                    assert bdata.max() == rdata.max()
                elif operation == "sum":
                    assert bdata.sum() == rdata.sum()
                elif operation == "average":
                    assert bdata.mean() == rdata.mean()
    for src_dtype in (float, np.uint8, np.uint16, np.int16):
        src.data = data = np.array(src.data[:500, :500], dtype=src_dtype)
        for dtype_str in p.dtypes:
            p.dtype_str = dtype_str
            dst = sigima_image.binning(src, p)
            bdata = dst.data
            if dtype_str == "dtype":
                assert bdata.dtype is data.dtype
            else:
                assert bdata.dtype is np.dtype(dtype_str)


if __name__ == "__main__":
    test_binning_graphically()
    test_binning()
