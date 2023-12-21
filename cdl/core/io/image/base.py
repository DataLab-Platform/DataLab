# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab image I/O registry
"""

from __future__ import annotations

import abc

import numpy as np

from cdl.core.io.base import BaseIORegistry, FormatBase
from cdl.core.model.image import ImageObj, create_image
from cdl.utils.strings import reduce_path


class ImageIORegistry(BaseIORegistry):
    """Metaclass for registering image I/O handler classes"""

    _io_format_instances: list[ImageFormatBase] = []


class ImageFormatBaseMeta(ImageIORegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class ImageFormatBase(abc.ABC, FormatBase, metaclass=ImageFormatBaseMeta):
    """Object representing an image file type"""

    @staticmethod
    def create_object(filename: str, index: int | None = None) -> ImageObj:
        """Create empty object

        Args:
            filename (str): File name
            index (int | None): Index of object in file

        Returns:
            ImageObj: Image object
        """
        name = reduce_path(filename)
        if index is not None:
            name += f"_{index}"
        return create_image(name)

    def read(self, filename: str) -> ImageObj:
        """Read data from file, return one or more objects

        Args:
            filename (str): File name

        Returns:
            ImageObj: Image object
        """
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        return obj

    @staticmethod
    @abc.abstractmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename (str): File name

        Returns:
            np.ndarray: Image data
        """
