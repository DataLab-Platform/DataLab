# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from __future__ import annotations

import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.adapters_plotpy.factories import create_adapter_from_object
from cdl.env import execenv
from sigima_.io import read_images, read_signals
from sigima_.io.image import funcs as image_funcs
from sigima_.obj import ImageObj, SignalObj
from sigima_.tests.helpers import try_open_test_data
from sigima_.tests.vistools import view_curve_items, view_images


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
    view_curve_items([create_adapter_from_object(obj).make_item() for obj in objs])


@try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname: str | None = None) -> None:
    """Testing CSV files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    view_curve_items([create_adapter_from_object(obj).make_item() for obj in objs])


@try_open_test_data("Testing MAT-File reader", "*.mat")
def open_mat(fname: str | None = None) -> None:
    """Testing MAT files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    if isinstance(objs[0], SignalObj):
        view_curve_items([create_adapter_from_object(obj).make_item() for obj in objs])
    else:
        view_images([obj.data for obj in objs])


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
        open_mat()
        open_sif()
        open_scordata()


if __name__ == "__main__":
    test_io1()
