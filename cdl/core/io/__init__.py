# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
DataLab I/O module
"""

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

# Registering dynamic I/O features:
import cdl.core.io.h5  # pylint: disable=unused-import
import cdl.core.io.image  # pylint: disable=unused-import
import cdl.core.io.signal  # pylint: disable=unused-import

# Other imports:
from cdl.core.io.image.base import ImageIORegistry
from cdl.core.io.signal.base import SignalIORegistry
from cdl.core.model.image import ImageParam
from cdl.core.model.signal import SignalParam


def read_signal(filename: str) -> SignalParam:
    """
    Read a signal from a file.

    Parameters
    ----------
    filename: str
        File name.

    Returns
    -------
    signal: Signal
        Signal.
    """
    return SignalIORegistry.read(filename)


def write_signal(filename: str, signal: SignalParam) -> None:
    """
    Write a signal to a file.

    Parameters
    ----------
    filename: str
        File name.
    signal: Signal
        Signal.
    """
    SignalIORegistry.write(filename, signal)


def read_image(filename: str) -> ImageParam:
    """
    Read an image from a file.

    Parameters
    ----------
    filename: str
        File name.

    Returns
    -------
    image: Image
        Image.
    """
    return ImageIORegistry.read(filename)


def write_image(filename: str, image: ImageParam) -> None:
    """
    Write an image to a file.

    Parameters
    ----------
    filename: str
        File name.
    image: Image
        Image.
    """
    ImageIORegistry.write(filename, image)
