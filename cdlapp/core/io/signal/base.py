# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab signal I/O registry
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc

import numpy as np

from cdlapp.core.io.base import BaseIORegistry, FormatBase
from cdlapp.core.io.conv import data_to_xy
from cdlapp.core.model.signal import SignalObj, create_signal
from cdlapp.utils.misc import reduce_path


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
            filename (str): File name
            index (int | None): Index of object in file

        Returns:
            SignalObj: Signal object
        """
        name = reduce_path(filename)
        if index is not None:
            name += f"_{index}"
        return create_signal(name)

    def read(self, filename: str) -> SignalObj:
        """Read data from file, return one or more objects

        Args:
            filename (str): File name

        Returns:
            SignalObj: Signal object
        """
        obj = self.create_object(filename)
        xydata = self.read_xydata(filename, obj)
        self.set_signal_xydata(obj, xydata)
        return obj

    @staticmethod
    def set_signal_xydata(signal: SignalObj, xydata: np.ndarray) -> None:
        """Set signal xydata

        Args:
            signal (SignalObj): Signal object
            xydata (numpy.ndarray): XY data
        """
        assert isinstance(xydata, np.ndarray), "Data type not supported"
        assert len(xydata.shape) in (1, 2), "Data not supported"
        if len(xydata.shape) == 1:
            signal.set_xydata(np.arange(xydata.size), xydata)
        else:
            x, y, dx, dy = data_to_xy(xydata)
            signal.set_xydata(x, y, dx, dy)

    @abc.abstractmethod
    def read_xydata(self, filename: str, obj: SignalObj) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename (str): File name
            obj (SignalObj): Signal object

        Returns:
            np.ndarray: XY data
        """
