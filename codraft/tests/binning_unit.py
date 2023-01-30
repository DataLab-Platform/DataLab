# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Image pixel binning computation test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import time

from guiqwt.builder import make
from numpy import ma

from codraft.core.computation.image import BINNING_OPERATIONS, binning
from codraft.env import execenv
from codraft.tests.data import get_laser_spot_data
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def compare_binning_images(data):
    """Compare binning images"""
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
                title = f"Binning[{binning_x}x{binning_y},{operation}]"
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
    view_image_items(items)


def binning_test():
    """Centroid test"""
    with qt_app_context():
        for data in get_laser_spot_data()[2:]:
            execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
            compare_binning_images(data.view(ma.MaskedArray))


if __name__ == "__main__":
    binning_test()
