# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Native I/O module (native HDF5/JSON formats)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from guidata.io import HDF5Reader, HDF5Writer

from cdl.info import get_version

DATALAB_VERSION_NAME = "DataLab_Version"


class NativeH5Writer(HDF5Writer):
    """DataLab signal/image objects HDF5 guidata Dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.h5[DATALAB_VERSION_NAME] = get_version()


class NativeH5Reader(HDF5Reader):
    """DataLab signal/image objects HDF5 guidata dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.version = self.h5[DATALAB_VERSION_NAME]
