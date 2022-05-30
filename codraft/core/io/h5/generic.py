# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT Generic HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import h5py
import numpy as np

from codraft.core.io.h5 import common, utils
from codraft.core.model.image import create_image
from codraft.core.model.signal import create_signal
from codraft.utils.misc import to_string


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
        return np.issctype(data) and utils.is_supported_num_dtype(data)


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

    @property
    def is_signal(self):
        """Return True if array represents a signal"""
        shape = self.data.shape
        return len(shape) == 1 or shape[0] in (1, 2) or shape[1] in (1, 2)

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "signal.svg" if self.is_signal else "image.svg"

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

    def create_object(self):
        """Create native object, if supported"""
        if self.is_signal:
            obj = create_signal(self.object_title)
            self.set_signal_data(obj)
        else:
            obj = create_image(self.object_title)
            self.set_image_data(obj)
        return obj


common.NODE_FACTORY.register(GenericArrayNode, is_generic=True)
