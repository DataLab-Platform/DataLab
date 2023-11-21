# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from cdl.core.io.image import funcs as image_funcs
from cdl.env import execenv
from cdl.obj import read_signal
from cdl.utils.tests import try_open_test_data
from cdl.utils.vistools import view_curve_items, view_images


@try_open_test_data("Testing TXT file reader", "*.txt")
def open_txt(fname=None):
    """Testing TXT files"""
    signal = read_signal(fname)
    execenv.print(signal)
    view_curve_items([signal.make_item()])


@try_open_test_data("Testing CSV file reader", "*.csv")
def open_csv(fname=None):
    """Testing CSV files"""
    signal = read_signal(fname)
    execenv.print(signal)
    view_curve_items([signal.make_item()])


@try_open_test_data("Testing SIF file handler", "*.sif")
def open_sif(fname=None):
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    data = image_funcs.imread_sif(fname)[0]
    view_images(data)


@try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def open_scordata(fname=None):
    """Testing SCOR-DATA files"""
    execenv.print(image_funcs.SCORFile(fname))
    data = image_funcs.imread_scor(fname)
    view_images(data)


def test_io1():
    """I/O test"""
    with qt_app_context():
        open_txt()
        open_csv()
        open_sif()
        open_scordata()


if __name__ == "__main__":
    test_io1()
