# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Base I/O common module (native HDF5 format)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from guidata.hdf5io import HDF5Reader, HDF5Writer
from guidata.jsonio import JSONReader, JSONWriter

from codraft import __version__

H5_VERSION = "CodraFT_Version"


LIST_LENGTH_STR = "__list_length__"


class NativeH5Writer(HDF5Writer):
    """CodraFT signal/image objects HDF5 guidata Dataset Writer class,
    supporting dictionary serialization"""

    def __init__(self, filename):
        super().__init__(filename)
        self.h5[H5_VERSION] = __version__

    def write_dict(self, val):
        """Write dictionary to h5 file"""
        # Keys must be strings
        # Values must be h5py supported data types
        group = self.get_parent_group()
        dict_group = group.create_group(self.option[-1])
        for key, value in val.items():
            if isinstance(value, dict):
                with self.group(key):
                    self.write_dict(value)
            elif isinstance(value, list):
                with self.group(key):
                    with self.group(LIST_LENGTH_STR):
                        self.write(len(value))
                    for index, i_val in enumerate(value):
                        with self.group("elt" + str(index)):
                            self.write(i_val)
            else:
                try:
                    dict_group.attrs[key] = value
                except TypeError:
                    pass


class NativeH5Reader(HDF5Reader):
    """CodraFT signal/image objects HDF5 guidata dataset Writer class,
    supporting dictionary deserialization"""

    def __init__(self, filename):
        super().__init__(filename)
        self.version = self.h5[H5_VERSION]

    def read_dict(self):
        """Read dictionary from h5 file"""
        group = self.get_parent_group()
        dict_group = group[self.option[-1]]
        dict_val = {}
        for key, value in dict_group.attrs.items():
            dict_val[key] = value
        for key in dict_group:
            with self.group(key):
                if "__list_length__" in dict_group[key].attrs:
                    with self.group(LIST_LENGTH_STR):
                        list_len = self.read()
                    dict_val[key] = [
                        dict_group[key]["elt" + str(index)][:]
                        for index in range(list_len)
                    ]
                else:
                    dict_val[key] = self.read_dict()
        return dict_val


class NativeJSONWriter(JSONWriter):
    """CodraFT signal/image objects JSON guidata Dataset Writer class,
    supporting dictionary serialization"""

    write_dict = JSONWriter.write_any


class NativeJSONReader(JSONReader):
    """CodraFT signal/image objects JSON guidata Dataset Reader class,
    supporting dictionary deserialization"""

    read_dict = JSONReader.read_any
