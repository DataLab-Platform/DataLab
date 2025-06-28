# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image I/O formats test
"""

from __future__ import annotations


def get_image_formats():
    """Get image formats"""
    from sigima_.io.image import formats

    return [
        class_name for class_name in dir(formats) if class_name.endswith("ImageFormat")
    ]


def test_imageio_formats_option():
    """Set other image I/O formats"""
    from sigima_.config import options
    from sigima_.io.image import formats

    # Set custom image I/O formats
    options.imageio_formats.set((("*.rec", "PCO Camera REC"),))
    # Check if the formats are set correctly
    assert hasattr(formats, "RECImageFormat"), "RECImageFormat not found in formats"


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_image_formats())
    test_imageio_formats_option()
    pprint(get_image_formats())
