# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O image formats
"""

import os.path as osp

import guiqwt.io
import numpy as np
import skimage.io

from cdl.config import _
from cdl.core.io.base import FormatInfo
from cdl.core.io.image import funcs
from cdl.core.io.image.base import ImageFormatBase
from cdl.core.model.image import ImageParam


class ClassicsImageFormat(ImageFormatBase):
    """Object representing a classic image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("BMP, JPEG, PNG and TIFF files"),
        extensions="*.bmp *.jpg *.jpeg *.png *.tif *.tiff",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return skimage.io.imread(filename, as_gray=True)

    def write(self, filename: str, obj: ImageParam) -> None:
        """Write data to file"""
        skimage.io.imsave(filename, obj.data, check_contrast=False)


class NumPyImageFormat(ImageFormatBase):
    """Object representing a NumPy image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("NumPy binary files"),
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return np.load(filename)

    def write(self, filename: str, obj: ImageParam) -> None:
        """Write data to file"""
        np.save(filename, obj.data)


class TextImageFormat(ImageFormatBase):
    """Object representing a text image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("Text files"),
        extensions="*.txt *.csv *.asc",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        for delimiter in ("\t", ",", " ", ";"):
            try:
                return np.loadtxt(filename, delimiter=delimiter)
            except ValueError:
                continue
        raise ValueError(f"Could not read file {filename} as text file")

    def write(self, filename: str, obj: ImageParam) -> None:
        """Write data to file"""
        if obj.data.dtype in (
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
            np.savetxt(filename, obj.data, fmt=fmt)
        elif ext.lower() == ".csv":
            np.savetxt(filename, obj.data, fmt=fmt, delimiter=",")
        else:
            raise ValueError(f"Unknown text file extension {ext}")


class DICOMImageFormat(ImageFormatBase):
    """Object representing a DICOM image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("DICOM files"),
        extensions="*.dcm *.dicom",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return guiqwt.io.imread(filename)


class AndorSIFImageFormat(ImageFormatBase):
    """Object representing an Andor SIF image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("Andor SIF files"),
        extensions="*.sif",
        readable=True,
        writeable=False,
    )

    def read(self, filename: str) -> ImageParam:
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
    """Object representing a SPIRICON image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("SPIRICON files"),
        extensions="*.scor-data",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return funcs.imread_scor(filename)


class FXDImageFormat(ImageFormatBase):
    """Object representing a FXD image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("FXD files"),
        extensions="*.fxd",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return funcs.imread_fxd(filename)


class XYZImageFormat(ImageFormatBase):
    """Object representing a XYZ image file types"""

    FORMAT_INFO = FormatInfo(
        name=_("XYZ files"),
        extensions="*.xyz",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it"""
        return funcs.imread_xyz(filename)
