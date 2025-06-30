# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image I/O formats test
"""

# Note about the modules imported outside top-level:
#
# We want to keep the import order under control, so we import the modules only when
# they are needed, especially for the dynamically defined image formats in the `formats`
# module. This way, we can ensure that the formats are defined just before we use them
# in the tests. This is particularly useful for testing the `imageio_formats` option
# that allows users to set custom image I/O formats.


def get_image_formats():
    """Get image formats"""
    # pylint: disable=import-outside-toplevel
    from sigima_.io.image import formats

    return [clname for clname in dir(formats) if clname.endswith("ImageFormat")]


def test_imageio_formats_option():
    """Set other image I/O formats"""
    # pylint: disable=import-outside-toplevel
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
