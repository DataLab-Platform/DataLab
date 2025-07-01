# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test adding new I/O formats
"""

from __future__ import annotations

from typing import Type

import numpy as np

from sigima_.env import execenv
from sigima_.io import ImageIORegistry, SignalIORegistry
from sigima_.io.base import FormatInfo
from sigima_.io.image.base import ImageFormatBase
from sigima_.io.signal.base import SignalFormatBase


def _get_image_format_number() -> int:
    """Get the number of standard image formats"""
    return len(ImageIORegistry.get_formats())


def _add_image_format() -> Type[ImageFormatBase]:
    """Add a new image format to the registry"""

    class MyImageFormat(ImageFormatBase):
        """Object representing MyImageFormat image file type"""

        FORMAT_INFO = FormatInfo(
            name="MyImageFormat",
            extensions="*.myimg",
            readable=True,
            writeable=False,
        )

        @staticmethod
        def read_data(filename: str) -> np.ndarray:
            """Read data and return it

            Args:
                filename (str): path to MyImageFormat file

            Returns:
                np.ndarray: image data
            """
            # Implement reading logic here

    return MyImageFormat


def test_add_image_format() -> None:
    """Test adding a new image format"""
    n1 = _get_image_format_number()
    execenv.print(f"Number of standard image formats: {n1}")
    execenv.print("Adding MyImageFormat... ", end="")
    image_class = _add_image_format()
    n2 = _get_image_format_number()
    assert n2 == n1 + 1, "Image format was not added correctly"
    execenv.print("OK")
    execenv.print(f"New number of image formats:      {n2}")
    assert (
        sum(isinstance(fmt, image_class) for fmt in ImageIORegistry.get_formats()) == 1
    )
    finfo = image_class.FORMAT_INFO
    finfo_str = "\n".join([(" " * 4) + line for line in str(finfo).splitlines()])
    assert finfo_str in ImageIORegistry.get_format_info(rst=False)


def _get_signal_format_number() -> int:
    """Get the number of standard signal formats"""
    return len(SignalIORegistry.get_formats())


def _add_signal_format() -> Type[SignalFormatBase]:
    """Add a new signal format to the registry"""

    class MySignalFormat(SignalFormatBase):
        """Object representing MySignalFormat signal file type"""

        FORMAT_INFO = FormatInfo(
            name="MySignalFormat",
            extensions="*.mysig",
            readable=True,
            writeable=False,
        )

        def read_xydata(self, filename: str) -> np.ndarray:
            """Read data and metadata from file, write metadata to object, return xydata

            Args:
                filename: Name of file to read

            Returns:
                NumPy array xydata
            """
            # Implement reading logic here
            print(f"Reading data from {filename}")

    return MySignalFormat


def test_add_signal_format() -> None:
    """Test adding a new signal format"""
    n1 = _get_signal_format_number()
    execenv.print(f"Number of standard signal formats: {n1}")
    execenv.print("Adding MySignalFormat... ", end="")
    signal_class = _add_signal_format()
    n2 = _get_signal_format_number()
    assert n2 == n1 + 1, "Signal format was not added correctly"
    execenv.print("OK")
    execenv.print(f"New number of signal formats:      {n2}")
    assert (
        sum(isinstance(fmt, signal_class) for fmt in SignalIORegistry.get_formats())
        == 1
    )
    finfo = signal_class.FORMAT_INFO
    finfo_str = "\n".join([(" " * 4) + line for line in str(finfo).splitlines()])
    assert finfo_str in SignalIORegistry.get_format_info(rst=False)


if __name__ == "__main__":
    test_add_image_format()
    test_add_signal_format()
