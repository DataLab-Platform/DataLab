# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O module
"""

from __future__ import annotations

# Registering dynamic I/O features:
import cdl.core.io.h5  # pylint: disable=unused-import
import cdl.core.io.image  # pylint: disable=unused-import
import cdl.core.io.signal  # pylint: disable=unused-import

# Other imports:
from cdl.core.io.image.base import ImageIORegistry
from cdl.core.io.signal.base import SignalIORegistry
from cdl.core.model.image import ImageObj
from cdl.core.model.signal import SignalObj


def read_signals(filename: str) -> list[SignalObj]:
    """Read a list of signals from a file.

    Args:
        filename: File name.

    Returns:
        List of signals.
    """
    return SignalIORegistry.read(filename)


def read_signal(filename: str) -> SignalObj:
    """Read a signal from a file.

    Args:
        filename: File name.

    Returns:
        Signal.
    """
    return read_signals(filename)[0]


def write_signal(filename: str, signal: SignalObj) -> None:
    """Write a signal to a file.

    Args:
        filename: File name.
        signal: Signal.
    """
    SignalIORegistry.write(filename, signal)


def read_images(filename: str) -> list[ImageObj]:
    """Read a list of images from a file.

    Args:
        filename: File name.

    Returns:
        List of images.
    """
    return ImageIORegistry.read(filename)


def read_image(filename: str) -> ImageObj:
    """Read an image from a file.

    Args:
        filename: File name.

    Returns:
        Image.
    """
    return read_images(filename)[0]


def write_image(filename: str, image: ImageObj) -> None:
    """Write an image to a file.

    Args:
        filename: File name.
        image: Image.
    """
    ImageIORegistry.write(filename, image)
