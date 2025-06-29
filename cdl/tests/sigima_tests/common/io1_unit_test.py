# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing `sigima_` specific formats.
"""

from __future__ import annotations

import numpy as np
from guidata.qthelpers import qt_app_context

from sigima_.env import execenv
from sigima_.io import read_images, read_signals
from sigima_.io.image import funcs as image_funcs
from sigima_.obj import ImageObj, SignalObj
from sigima_.tests import helpers, vistools


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


@helpers.try_open_test_data("Testing TXT file reader", "*.txt")
def open_txt(fname: str | None = None) -> None:
    """Testing TXT files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    vistools.view_curves_and_images(objs)


@helpers.try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname: str | None = None) -> None:
    """Testing CSV files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    vistools.view_curves_and_images(objs)


@helpers.try_open_test_data("Testing MAT-File reader", "*.mat")
def open_mat(fname: str | None = None) -> None:
    """Testing MAT files"""
    objs = __read_objs(fname)
    for obj in objs:
        execenv.print(obj)
    if isinstance(objs[0], SignalObj):
        vistools.view_curves(objs)
    else:
        vistools.view_images(objs)


@helpers.try_open_test_data("Testing SIF file handler", "*.sif")
def open_sif(fname: str | None = None) -> None:
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    datalist = image_funcs.imread_sif(fname)
    vistools.view_images(datalist)


@helpers.try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def open_scordata(fname: str | None = None) -> None:
    """Testing SCOR-DATA files"""
    execenv.print(image_funcs.SCORFile(fname))
    data = image_funcs.imread_scor(fname)
    vistools.view_images(data)


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
