# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O module
"""

from __future__ import annotations

from sigima_.io.image.base import ImageIORegistry
from sigima_.io.signal.base import SignalIORegistry
from sigima_.obj.image import ImageObj
from sigima_.obj.signal import SignalObj


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
