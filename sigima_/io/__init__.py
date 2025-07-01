# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O (:mod:`sigima_.io`)
-----------------------

This package provides input/output functionality for reading and writing
signals and images in various formats. It includes a registry for managing
the available formats and their associated read/write functions.

General purpose I/O functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package provides functions to read and write signals and images, allowing users
to easily handle different file formats without needing to know the specifics
of each format.

It includes the following main functions:

- :py:func:`read_signals`: Read a list of signals from a file.
- :py:func:`read_signal`: Read a single signal from a file.
- :py:func:`write_signal`: Write a single signal to a file.
- :py:func:`read_images`: Read a list of images from a file.
- :py:func:`read_image`: Read a single image from a file.
- :py:func:`write_image`: Write a single image to a file.

Supported formats
^^^^^^^^^^^^^^^^^

.. autodata:: sigima_.config.SIGNAL_FORMAT_INFO

.. autodata:: sigima_.config.IMAGE_FORMAT_INFO

Adding new formats
^^^^^^^^^^^^^^^^^^

To add new formats, you can create a new class that inherits from
:py:class:`sigima_.io.image.base.ImageFormatBase` or
:py:class:`sigima_.io.signal.base.SignalFormatBase` and implement the required methods.

.. note::

    Thanks to the plugin system, you can add new formats simply by defining a new class
    in a separate module, and it will be automatically discovered and registered, as
    long as it is imported in your application or library.

Example of a new image format plugin:

.. code-block:: python

    from sigima_.io.image.base import ImageFormatBase
    from sigima_.io.base import FormatInfo

    class MyImageFormat(ImageFormatBase):
        \"\"\"Object representing MyImageFormat image file type\"\"\"

        FORMAT_INFO = FormatInfo(
            name="MyImageFormat",
            extensions="*.myimg",
            readable=True,
            writeable=False,
        )

        @staticmethod
        def read_data(filename: str) -> np.ndarray:
            \"\"\"Read data and return it

            Args:
                filename (str): path to MyImageFormat file

            Returns:
                np.ndarray: image data
            \"\"\"
            # Implement reading logic here
            pass
"""

from __future__ import annotations

from sigima_.io.image.base import ImageIORegistry
from sigima_.io.signal.base import SignalIORegistry
from sigima_.obj.image import ImageObj
from sigima_.obj.signal import SignalObj

SIGNAL_FORMAT_INFO = SignalIORegistry.get_format_info(rst=True)
IMAGE_FORMAT_INFO = ImageIORegistry.get_format_info(rst=True)


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
