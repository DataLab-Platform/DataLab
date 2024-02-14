# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O image formats
"""

import os.path as osp

import numpy as np
import plotpy.io
import skimage.io

from cdl.config import _
from cdl.core.io.base import FormatInfo
from cdl.core.io.conv import convert_array_to_standard_type
from cdl.core.io.image import funcs
from cdl.core.io.image.base import ImageFormatBase
from cdl.core.model.image import ImageObj


class ClassicsImageFormat(ImageFormatBase):
    """Object representing classic image file types"""

    FORMAT_INFO = FormatInfo(
        name="BMP, JPEG, PNG, TIFF JPEG2000",
        extensions="*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.jp2",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return skimage.io.imread(filename, as_gray=True)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        ext = osp.splitext(filename)[1].lower()
        if ext in (".bmp", ".jpg", ".jpeg", ".png"):
            if data.dtype is not np.uint8:
                data = data.astype(np.uint8)
        if ext in (".jp2",):
            if data.dtype not in (np.uint8, np.uint16):
                data = data.astype(np.uint16)
        skimage.io.imsave(filename, data, check_contrast=False)


class NumPyImageFormat(ImageFormatBase):
    """Object representing NumPy image file type"""

    FORMAT_INFO = FormatInfo(
        name="NumPy",
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return convert_array_to_standard_type(np.load(filename))

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        np.save(filename, data)


class TextImageFormat(ImageFormatBase):
    """Object representing text image file type"""

    FORMAT_INFO = FormatInfo(
        name=_("Text files"),
        extensions="*.txt *.csv *.asc",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            for delimiter in ("\t", ",", " ", ";"):
                try:
                    return np.loadtxt(filename, delimiter=delimiter, encoding=encoding)
                except ValueError:
                    continue
        raise ValueError(f"Could not read file {filename} as text file")

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        if data.dtype in (
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
        ):
            fmt = "%d"
        else:
            fmt = "%.18e"
        ext = osp.splitext(filename)[1]
        if ext.lower() in (".txt", ".asc", ""):
            np.savetxt(filename, data, fmt=fmt)
        elif ext.lower() == ".csv":
            np.savetxt(filename, data, fmt=fmt, delimiter=",")
        else:
            raise ValueError(f"Unknown text file extension {ext}")


class DICOMImageFormat(ImageFormatBase):
    """Object representing DICOM image file type"""

    FORMAT_INFO = FormatInfo(
        name="DICOM",
        extensions="*.dcm *.dicom",
        readable=True,
        writeable=False,
        requires=["pydicom"],
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return plotpy.io.imread(filename)


class AndorSIFImageFormat(ImageFormatBase):
    """Object representing an Andor SIF image file type"""

    FORMAT_INFO = FormatInfo(
        name="Andor SIF",
        extensions="*.sif",
        readable=True,
        writeable=False,
    )

    def read(self, filename: str) -> ImageObj:
        """Read data from file, return one or more objects"""
        data = self.read_data(filename)
        if len(data.shape) == 3:
            objlist = []
            for idx in range(data.shape[0]):
                obj = self.create_object(filename, index=idx)
                obj.data = data[idx, ::]
                objlist.append(obj)
            return objlist
        obj = self.create_object(filename)
        obj.data = data
        return obj

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return funcs.imread_sif(filename)


class SpiriconImageFormat(ImageFormatBase):
    """Object representing Spiricon image file type"""

    FORMAT_INFO = FormatInfo(
        name="Spiricon",
        extensions="*.scor-data",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return funcs.imread_scor(filename)
