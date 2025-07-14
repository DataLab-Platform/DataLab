# -*- coding: utf-8 -*-

"""
Image format Plugin for DataLab (example)
=========================================

This plugin is an example of how to add a new image file format to DataLab.

It provides a new image file format, Dürr NDT XYZ, from Dürr NDT GmbH & Co. KG
(see `Dürr NDT website <https://www.duerr-ndt.com/>`_).
"""

import numpy as np
from sigima.io.base import FormatInfo
from sigima.io.image.base import SingleImageFormatBase


class XYZImageFormat(SingleImageFormatBase):
    """Object representing Dürr NDT XYZ image file type"""

    FORMAT_INFO = FormatInfo(
        name="Dürr NDT",
        extensions="*.xyz",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename (str): path to XYZ file

        Returns:
            np.ndarray: image data
        """
        with open(filename, "rb") as fdesc:
            cols = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            rows = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            arr = np.fromfile(fdesc, dtype=np.uint16, count=cols * rows)
            arr = arr.reshape((rows, cols))
        return np.fliplr(arr)
