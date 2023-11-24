# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O signal formats
"""

import numpy as np

from cdl.config import _
from cdl.core.io.base import FormatInfo
from cdl.core.io.conv import convert_array_to_standard_type
from cdl.core.io.signal import funcs
from cdl.core.io.signal.base import SignalFormatBase
from cdl.core.model.signal import SignalObj


class NumPySignalFormat(SignalFormatBase):
    """Object representing a NumPy signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("NumPy binary files"),
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read_xydata(self, filename: str, obj: SignalObj) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename (str): Name of file to read
            obj (SignalObj): Signal object to write metadata to

        Returns:
            np.ndarray: xydata
        """
        return convert_array_to_standard_type(np.load(filename))

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename (str): Name of file to write
            obj (SignalObj): Signal object to read data from
        """
        np.save(filename, obj.xydata.T)


class CSVSignalFormat(SignalFormatBase):
    """Object representing a CSV signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("CSV files"),
        extensions="*.csv *.txt",
        readable=True,
        writeable=True,
    )

    def read_xydata(self, filename: str, obj: SignalObj) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename (str): Name of file to read
            obj (SignalObj): Signal object to write metadata to

        Returns:
            np.ndarray: xydata
        """
        xydata, xlabel, xunit, ylabel, yunit, header = funcs.read_csv(filename)
        obj.xlabel = xlabel
        obj.xunit = xunit
        obj.ylabel = ylabel
        obj.yunit = yunit
        if header:
            obj.metadata[self.HEADER_KEY] = header
        return xydata

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename (str): Name of file to write
            obj (SignalObj): Signal object to read data from
        """
        funcs.write_csv(
            filename,
            obj.xydata.T,
            obj.xlabel,
            obj.xunit,
            obj.ylabel,
            obj.yunit,
            obj.metadata.get(self.HEADER_KEY, ""),
        )
