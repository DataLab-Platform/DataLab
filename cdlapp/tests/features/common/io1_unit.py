# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from cdlapp.core.io.image import funcs as image_funcs
from cdlapp.env import execenv
from cdlapp.obj import read_signal
from cdlapp.utils.qthelpers import qt_app_context
from cdlapp.utils.tests import try_open_test_data
from cdlapp.utils.vistools import view_curve_items, view_images


@try_open_test_data("Testing TXT file reader", "*.txt")
def test_txt(fname=None):
    """Testing TXT files"""
    signal = read_signal(fname)
    execenv.print(signal)
    view_curve_items([signal.make_item()])


@try_open_test_data("Testing CSV file reader", "*.csv")
def test_csv(fname=None):
    """Testing CSV files"""
    signal = read_signal(fname)
    execenv.print(signal)
    view_curve_items([signal.make_item()])


@try_open_test_data("Testing SIF file handler", "*.sif")
def test_sif(fname=None):
    """Testing SIF files"""
    execenv.print(image_funcs.SIFFile(fname))
    data = image_funcs.imread_sif(fname)[0]
    view_images(data)


@try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def test_scordata(fname=None):
    """Testing SCOR-DATA files"""
    execenv.print(image_funcs.SCORFile(fname))
    data = image_funcs.imread_scor(fname)
    view_images(data)


def io_test():
    """I/O test"""
    with qt_app_context():
        test_txt()
        test_csv()
        test_sif()
        test_scordata()


if __name__ == "__main__":
    io_test()
