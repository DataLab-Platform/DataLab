# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab I/O signal formats
"""

import numpy as np

from cdl.config import _
from cdl.core.io.base import FormatInfo
from cdl.core.io.signal import funcs
from cdl.core.io.signal.base import SignalFormatBase
from cdl.core.model.signal import SignalParam


class NumPySignalFormat(SignalFormatBase):
    """Object representing a NumPy signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("NumPy binary files"),
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read_xydata(self, filename: str, obj: SignalParam) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata"""
        return np.load(filename)

    def write(self, filename: str, obj: SignalParam) -> None:
        """Write data to file"""
        np.save(filename, obj.xydata.T)


class CSVSignalFormat(SignalFormatBase):
    """Object representing a CSV signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("CSV files"),
        extensions="*.csv *.txt",
        readable=True,
        writeable=True,
    )

    def read_xydata(self, filename: str, obj: SignalParam) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata"""
        xydata, xlabel, xunit, ylabel, yunit, header = funcs.read_csv(filename)
        obj.xlabel = xlabel
        obj.xunit = xunit
        obj.ylabel = ylabel
        obj.yunit = yunit
        if header:
            obj.metadata[self.HEADER_KEY] = header
        return xydata

    def write(self, filename: str, obj: SignalParam) -> None:
        """Write data to file"""
        funcs.write_csv(
            filename,
            obj.xydata.T,
            obj.xlabel,
            obj.xunit,
            obj.ylabel,
            obj.yunit,
            obj.metadata.get(self.HEADER_KEY, ""),
        )
