# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
I/O test

Testing CodraFT specific formats.
"""

from codraft.core.io.image import (
    FXDFile,
    SCORFile,
    SIFFile,
    imread_fxd,
    imread_scor,
    imread_sif,
)
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.tests import try_open_test_data
from codraft.utils.vistools import view_images

SHOW = True  # Show test in GUI-based test launcher


@try_open_test_data("Testing SIF file handler", "*.sif")
def test_sif(fname=None):
    """Testing SIF files"""
    print(SIFFile(fname))
    data = imread_sif(fname)[0]
    view_images(data)


@try_open_test_data("Testing FXD file handler", "*.fxd")
def test_fxd(fname=None):
    """Testing FXD files"""
    print(FXDFile(fname))
    data = imread_fxd(fname)
    view_images(data)


@try_open_test_data("Testing SCOR-DATA file handler", "*.scor-data")
def test_scordata(fname=None):
    """Testing SCOR-DATA files"""
    print(SCORFile(fname))
    data = imread_scor(fname)
    view_images(data)


def io_test():
    """I/O test"""
    with qt_app_context():
        test_sif()
        test_fxd()
        test_scordata()


if __name__ == "__main__":
    io_test()
