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
            filename: File name
            index: Index of object in file

        Returns:
            Image object
        """
        name = reduce_path(filename)
        if index is not None:
            name += f" {index:02d}"
        return create_image(name)

    def read(self, filename: str) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name

        Returns:
            List of image objects
        """
        # Default implementation covers the case of a single image:
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        return [obj]

    @staticmethod
    @abc.abstractmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """
        data = obj.data
        self.write_data(filename, data)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        raise NotImplementedError(f"Writing to {filename} is not supported")
