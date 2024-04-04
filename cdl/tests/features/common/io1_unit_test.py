# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from __future__ import annotations

import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.core.io.image import funcs as image_funcs
from cdl.env import execenv
from cdl.obj import ImageObj, SignalObj, read_images, read_signals
from cdl.utils.tests import try_open_test_data
from cdl.utils.vistools import view_curve_items, view_images


def __read_objs(fname: str) -> list[ImageObj] | list[SignalObj]:
    """Read objects from a file"""
    if "curve_formats" in fname:
        objs = read_signals(fname)
    else:
        objs = read_images(fname)
    for obj in objs:
        if np.all(np.isnan(obj.data)):
            raise ValueError("Data is all NaNs")
    return objs


@try_open_test_data("Testing TXT file reader", "*.txt")
def open_txt(fname: str | None = None) -> None:
    """Testing TXT files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    view_curve_items([obj.make_item() for obj in objs])


@try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname: str | None = None) -> None:
    """Testing CSV files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    view_curve_items([obj.make_item() for obj in objs])


@try_open_test_data("Testing SIF file handler", "*.sif")
def open_sif(fname: str | None = None) -> None:
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    datalist = image_funcs.imread_sif(fname)
    view_images(datalist)


@try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def open_scordata(fname: str | None = None) -> None:
    """Testing SCOR-DATA files"""
    execenv.print(image_funcs.SCORFile(fname))
    data = image_funcs.imread_scor(fname)
    view_images(data)


def test_io1() -> None:
    """I/O test"""
    with qt_app_context():
        open_txt()
        open_csv()
        open_sif()
        open_scordata()


if __name__ == "__main__":
    test_io1()
