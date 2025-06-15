# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image I/O registry
"""

from __future__ import annotations

import abc
import os.path as osp

import numpy as np

from cdl.config import _
from cdl.utils.qthelpers import CallbackWorker
from sigima_.io.base import BaseIORegistry, FormatBase
from sigima_.model.image import ImageObj, create_image


class ImageIORegistry(BaseIORegistry):
    """Metaclass for registering image I/O handler classes"""

    REGISTRY_INFO: str = _("Image I/O formats")

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
        name = osp.basename(filename)
        if index is not None:
            name += f" {index:02d}"
        return create_image(name, metadata={"source": filename})

    def read(
        self, filename: str, worker: CallbackWorker | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        # Default implementation covers the case of a single image:
        obj = self.create_object(filename)
        obj.data = self.read_data(filename)
        unique_values = np.unique(obj.data)
        if len(unique_values) == 2:
            # Binary image: set LUT range to unique values
            obj.zscalemin, obj.zscalemax = unique_values.tolist()
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


class MultipleImagesFormatBase(ImageFormatBase):
    """Base image format object for multiple images (e.g., SIF or SPE).

    Works with read function that returns a NumPy array of 3 dimensions, where
    the first dimension is the number of images.
    """

    def read(
        self, filename: str, worker: CallbackWorker | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        data = self.read_data(filename)
        if len(data.shape) == 3:
            objlist = []
            for idx in range(data.shape[0]):
                obj = self.create_object(filename, index=idx)
                obj.data = data[idx, ::]
                objlist.append(obj)
                if worker is not None:
                    worker.set_progress((idx + 1) / data.shape[0])
                    if worker.was_canceled():
                        break
            return objlist
        obj = self.create_object(filename)
        obj.data = data
        return [obj]
