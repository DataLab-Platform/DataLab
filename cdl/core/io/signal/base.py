# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
DataLab signal I/O registry
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
from typing import List, Optional

import numpy as np

from cdl.core.io.base import BaseIORegistry, FormatBase
from cdl.core.io.conv import data_to_xy
from cdl.core.model.signal import SignalParam, create_signal
from cdl.utils.misc import reduce_path


class SignalIORegistry(BaseIORegistry):
    """Metaclass for registering signal I/O handler classes"""

    _io_format_instances: List[SignalFormatBase] = []


class SignalFormatBaseMeta(SignalIORegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class SignalFormatBase(abc.ABC, FormatBase, metaclass=SignalFormatBaseMeta):
    """Object representing a signal file type"""

    HEADER_KEY = "HEADER"

    @staticmethod
    def create_object(filename: str, index: Optional[int] = None) -> SignalParam:
        """Create empty object"""
        name = reduce_path(filename)
        if index is not None:
            name += f"_{index}"
        return create_signal(name)

    def read(self, filename: str) -> SignalParam:
        """Read data from file, return one or more objects"""
        obj = self.create_object(filename)
        xydata = self.read_xydata(filename, obj)
        self.set_signal_xydata(obj, xydata)
        return obj

    @staticmethod
    def set_signal_xydata(signal: SignalParam, xydata: np.ndarray) -> None:
        """Set signal xydata"""
        assert isinstance(xydata, np.ndarray), "Data type not supported"
        assert len(xydata.shape) in (1, 2), "Data not supported"
        if len(xydata.shape) == 1:
            signal.set_xydata(np.arange(xydata.size), xydata)
        else:
            x, y, dx, dy = data_to_xy(xydata)
            signal.set_xydata(x, y, dx, dy)

    @abc.abstractmethod
    def read_xydata(self, filename: str, obj: SignalParam) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata"""
