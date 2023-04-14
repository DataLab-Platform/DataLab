# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab image I/O registry
"""

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
from typing import List, Optional

import numpy as np

from cdl.core.io.base import BaseIORegistry, FormatBase
from cdl.core.model.image import ImageParam, create_image
from cdl.utils.misc import reduce_path


class ImageIORegistry(BaseIORegistry):
    """Metaclass for registering image I/O handler classes"""

    _io_format_instances: List[ImageFormatBase] = []


class ImageFormatBaseMeta(ImageIORegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class ImageFormatBase(abc.ABC, FormatBase, metaclass=ImageFormatBaseMeta):
    """Object representing an image file type"""

    @staticmethod
    def create_object(filename: str, index: Optional[int] = None) -> ImageParam:
        """Create empty object"""
        name = reduce_path(filename)
        if index is not None:
            name += f"_{index}"
        return create_image(name)

    def read(self, filename: str) -> ImageParam:
        """Read data from file, return one or more objects"""
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        return obj

    @staticmethod
    @abc.abstractmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
