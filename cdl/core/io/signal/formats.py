# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab I/O signal formats
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from cdl.config import _
from cdl.core.io.base import FormatInfo
from cdl.core.io.conv import convert_array_to_standard_type
from cdl.core.io.signal import funcs
from cdl.core.io.signal.base import SignalFormatBase
from cdl.core.model.signal import SignalObj


class CSVSignalFormat(SignalFormatBase):
    """Object representing a CSV signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("CSV files"),
        extensions="*.csv *.txt",
        readable=True,
        writeable=True,
    )

    def read(self, filename: str, progress_callback: Callable) -> list[SignalObj]:
        """Read list of signal objects from file

        Args:
            filename: File name
            progress_callback: progress callback function (a function that takes a float
             between 0 and 1 as argument representing the progress, and returns a
             boolean indicating whether to cancel the operation)

        Returns:
            List of signal objects
        """
        xydata, xlabel, xunit, ylabels, yunits, header = funcs.read_csv(
            filename, progress_callback
        )
        if ylabels:
            # If y labels are present, we are sure that the data contains at least
            # two columns (x and y)
            objs = []
            for i, (ylabel, yunit) in enumerate(zip(ylabels, yunits)):
                obj = self.create_object(filename, i if len(ylabels) > 1 else None)
                obj.set_xydata(xydata[:, 0], xydata[:, i + 1])
                obj.xlabel = xlabel or ""
                obj.xunit = xunit or ""
                obj.ylabel = ylabel or ""
                obj.yunit = yunit or ""
                if header:
                    obj.metadata[self.HEADER_KEY] = header
                objs.append(obj)
            return objs
        return self.create_signals_from(xydata, filename)

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename (str): Name of file to write
            obj (SignalObj): Signal object to read data from
        """
        funcs.write_csv(
            filename,
            obj.xydata,
            obj.xlabel,
            obj.xunit,
            [obj.ylabel],
            [obj.yunit],
            obj.metadata.get(self.HEADER_KEY, ""),
        )


class NumPySignalFormat(SignalFormatBase):
    """Object representing a NumPy signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("NumPy binary files"),
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read_xydata(self, filename: str) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename (str): Name of file to read

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
