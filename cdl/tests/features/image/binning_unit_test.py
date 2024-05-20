# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image pixel binning computation test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import time

from guidata.qthelpers import qt_app_context
from numpy import ma
from plotpy.builder import make

from cdl.algorithms.image import BINNING_OPERATIONS, binning
from cdl.env import execenv
from cdl.tests.data import get_laser_spot_data
from cdl.utils.vistools import view_image_items


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
        binning_x = 2**ix
        for iy in range(1, 5):
            binning_y = 2**iy
            for operation in BINNING_OPERATIONS:
                t0 = time.time()
                bdata = binning(
                    data, binning_x=binning_x, binning_y=binning_y, operation=operation
                )
                title = f"[{binning_x}x{binning_y},{operation}]"
                item = make.image(
                    bdata,
                    title=title,
                    interpolation="nearest",
                    eliminate_outliers=2.0,
                    xdata=[0, data.shape[1]],
                    ydata=[0, data.shape[0]],
                )
                items.append(item)
                dt = time.time() - t0
                execenv.print(f"    {title}: {int(dt * 1e3):d} ms")
    oa_dt = time.time() - oa_t0
    execenv.print(f"    Overall calculation time: {int(oa_dt * 1e3):d} ms")
    for idx, operation in enumerate(BINNING_OPERATIONS):
        title = f"Binning test (operation: {operation})"
        view_image_items(items[idx + 1 :: 5], title=title, show_itemlist=True)


def test_binning_graphically() -> None:
    """Test binning computation and show results"""
    with qt_app_context():
        for data in get_laser_spot_data()[1:]:
            data = data[:500, :500]
            execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
            compare_binning_images(data.view(ma.MaskedArray))


if __name__ == "__main__":
    test_binning_graphically()
