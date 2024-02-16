# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from __future__ import annotations

from guidata.qthelpers import qt_app_context

from cdl.core.io.image import funcs as image_funcs
from cdl.env import execenv
from cdl.obj import ImageObj, SignalObj, read_image, read_signal
from cdl.utils.tests import try_open_test_data
from cdl.utils.vistools import view_curve_items, view_images


def __read_obj(fname: str) -> SignalObj | ImageObj:
    """Read an object from a file"""
    if "curve_formats" in fname:
        return read_signal(fname)
    return read_image(fname)


@try_open_test_data("Testing TXT file reader", "*.txt")
def open_txt(fname: str | None = None) -> None:
    """Testing TXT files"""
    obj = __read_obj(fname)
    execenv.print(obj)
    view_curve_items([obj.make_item()])


@try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname: str | None = None) -> None:
    """Testing CSV files"""
    obj = __read_obj(fname)
    execenv.print(obj)
    view_curve_items([obj.make_item()])


@try_open_test_data("Testing SIF file handler", "*.sif")
def open_sif(fname: str | None = None) -> None:
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    data = image_funcs.imread_sif(fname)[0]
    view_images(data)


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
