# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Generic HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import h5py
import numpy as np

from cdl.h5 import common, utils
from sigima_.io.converters import to_string
from sigima_.obj import create_image, create_signal


class BaseGenericNode(common.BaseNode):
    """Object representing a generic HDF5 data node"""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""
        return not isinstance(dset, h5py.Group)

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "h5scalar.svg"

    @property
    def data(self):
        """Data associated to node, if available"""
        return self.dset[()]

    @property
    def dtype_str(self):
        """Return string representation of node data type, if any"""
        return str(self.data.dtype)

    @property
    def text(self):
        """Return node textual representation"""
        return to_string(self.data)


class GenericScalarNode(BaseGenericNode):
    """Object representing a generic scalar HDF5 data node"""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""
        if not super().match(dset):
            return False
        data = dset[()]
        return isinstance(data, np.generic) and utils.is_supported_num_dtype(data)


common.NODE_FACTORY.register(GenericScalarNode, is_generic=True)


class GenericTextNode(BaseGenericNode):
    """Object representing a generic text HDF5 data node"""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""
        if not super().match(dset):
            return False
        data = dset[()]
        return isinstance(data, bytes) or utils.is_supported_str_dtype(data)

    @property
    def dtype_str(self):
        """Return string representation of node data type, if any"""
        return "string"

    @property
    def text(self):
        """Return node textual representation"""
        if utils.is_single_str_array(self.data):
            return self.data[0]
        return to_string(self.data)


common.NODE_FACTORY.register(GenericTextNode, is_generic=True)


class GenericArrayNode(BaseGenericNode):
    """Object representing a generic array HDF5 data node"""

    IS_ARRAY = True

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""
        if not super().match(dset):
            return False
        data = dset[()]
        return (
            utils.is_supported_num_dtype(data)
            and isinstance(data, np.ndarray)
            and len(data.shape) in (1, 2)
        )

    def is_supported(self) -> bool:
        """Return True if node is associated to supported data"""
        return self.data.size > 1

    @property
    def __is_signal(self):
        """Return True if array represents a signal"""
        shape = self.data.shape
        return len(shape) == 1 or shape[0] in (1, 2) or shape[1] in (1, 2)

    @property
    def icon_name(self):
        """Icon name associated to node"""
        if self.is_supported():
            return "signal.svg" if self.__is_signal else "image.svg"
        return "h5array.svg"

    @property
    def shape_str(self):
        """Return string representation of node shape, if any"""
        return " x ".join([str(size) for size in self.data.shape])

    @property
    def dtype_str(self):
        """Return string representation of node data type, if any"""
        return str(self.data.dtype)

    @property
    def text(self):
        """Return node textual representation"""
        return str(self.data)

    def create_native_object(self):
        """Create native object, if supported"""
        if self.__is_signal:
            obj = create_signal(self.object_title)
            try:
                self.set_signal_data(obj)
            except ValueError:
                obj = None
        else:
            obj = create_image(self.object_title)
            try:
                self.set_image_data(obj)
            except ValueError:
                obj = None
        return obj


common.NODE_FACTORY.register(GenericArrayNode, is_generic=True)
