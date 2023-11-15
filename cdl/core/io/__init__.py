# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O module
"""

# Registering dynamic I/O features:
import cdl.core.io.h5  # pylint: disable=unused-import
import cdl.core.io.image  # pylint: disable=unused-import
import cdl.core.io.signal  # pylint: disable=unused-import

# Other imports:
from cdl.core.io.image.base import ImageIORegistry
from cdl.core.io.signal.base import SignalIORegistry
from cdl.core.model.image import ImageObj
from cdl.core.model.signal import SignalObj


def read_signal(filename: str) -> SignalObj:
    """Read a signal from a file.

    Args:
        filename (str): File name.

    Returns:
        Signal: Signal.
    """
    return SignalIORegistry.read(filename)


def write_signal(filename: str, signal: SignalObj) -> None:
    """Write a signal to a file.

    Args:
        filename (str): File name.
        signal (Signal): Signal.
    """
    SignalIORegistry.write(filename, signal)


def read_image(filename: str) -> ImageObj:
    """Read an image from a file.

    Args:
        filename (str): File name.

    Returns:
        Image: Image.
    """
    return ImageIORegistry.read(filename)


def write_image(filename: str, image: ImageObj) -> None:
    """Write an image to a file.

    Args:
        filename (str): File name.
        image (Image): Image.
    """
    ImageIORegistry.write(filename, image)
