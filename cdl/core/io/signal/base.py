# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab signal I/O registry
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc

import numpy as np

from cdl.core.io.base import BaseIORegistry, FormatBase
from cdl.core.model.signal import SignalObj, create_signal
from cdl.utils.qthelpers import CallbackWorker
from cdl.utils.strings import reduce_path


class SignalIORegistry(BaseIORegistry):
    """Metaclass for registering signal I/O handler classes"""

    _io_format_instances: list[SignalFormatBase] = []


class SignalFormatBaseMeta(SignalIORegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class SignalFormatBase(abc.ABC, FormatBase, metaclass=SignalFormatBaseMeta):
    """Object representing a signal file type"""

    HEADER_KEY = "HEADER"

    @staticmethod
    def create_object(filename: str, index: int | None = None) -> SignalObj:
        """Create empty object

        Args:
            filename: File name
            index: Index of object in file

        Returns:
            Signal object
        """
        name = reduce_path(filename)
        if index is not None:
            name += f" {index:02d}"
        return create_signal(name)

    def create_signals_from(self, xydata: np.ndarray, filename: str) -> list[SignalObj]:
        """Create signal objects from xydata and filename

        Args:
            xydata: XY data
            filename: File name

        Returns:
            List of signal objects
        """
        assert isinstance(xydata, np.ndarray), "Data type not supported"
        assert len(xydata.shape) in (1, 2), "Data not supported"
        if len(xydata.shape) == 1:
            # 1D data
            obj = self.create_object(filename)
            obj.set_xydata(np.arange(xydata.size), xydata)
            return [obj]
        # 2D data: x, y1, y2, ...
        # Eventually transpose data:
        if xydata.shape[1] > xydata.shape[0]:
            xydata = xydata.T
        objs = []
        # Create objects for each y column
        for i in range(1, xydata.shape[1]):
            obj = self.create_object(filename, i)
            obj.set_xydata(xydata[:, 0], xydata[:, i])
            objs.append(obj)
        return objs

    def read(self, filename: str, worker: CallbackWorker) -> list[SignalObj]:
        """Read list of signal objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of signal objects
        """
        xydata = self.read_xydata(filename)
        return self.create_signals_from(xydata, filename)

    def read_xydata(self, filename: str) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename: File name

        Returns:
            XY data
        """
        raise NotImplementedError(f"Reading from {self.info.name} is not supported")
