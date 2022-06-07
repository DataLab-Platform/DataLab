# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Common tools for exogenous HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc
import os.path as osp

import h5py
import numpy as np

from codraft.utils.misc import to_string


class BaseNode(metaclass=abc.ABCMeta):
    """Object representing a HDF5 node"""

    IS_ARRAY = False

    def __init__(self, h5file, dname):
        self.h5file = h5file
        self.dset = h5file[dname]
        self.metadata = {}
        self.__obj = None
        self.children = []
        self.uint32_wng = False

    def id(self):
        """Return node id"""
        return self.dset.name

    @property
    def name(self):
        """Return node name, constructed from dataset name"""
        return self.dset.name.split("/")[-1]

    @property
    def data(self):
        """Data associated to node, if available"""
        return None

    @property
    def icon_name(self):
        """Icon name associated to node"""

    @property
    def shape_str(self):
        """Return string representation of node shape, if any"""
        return ""

    @property
    def dtype_str(self):
        """Return string representation of node data type, if any"""
        return ""

    @property
    def text(self):
        """Return node textual representation"""

    @property
    def description(self):
        """Return node description"""
        return ""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""

    def create_object(self):  # pylint: disable=no-self-use
        """Create native object, if supported"""
        return None

    def get_object(self):
        """Return native object, if supported"""
        if self.__obj is None:
            obj = self.create_object()  # pylint: disable=assignment-from-none
            if obj is not None:
                self.__process_metadata(obj)
            self.__obj = obj
        return self.__obj

    def __process_metadata(self, obj):
        """Process metadata from dataset to obj"""
        obj.metadata = {}
        for key, value in self.dset.attrs.items():
            if isinstance(value, bytes):
                value = to_string(value)
            obj.metadata[key] = value
        obj.metadata.update(self.metadata)

    @property
    def object_title(self):
        """Return signal/image object title"""
        return f"{self.name} ({osp.basename(self.h5file.filename)})"

    def set_signal_data(self, obj):
        """Set signal data (handles various issues)"""
        data = self.data
        if data.dtype not in (float, np.complex128):
            data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            obj.set_xydata(np.arange(data.size), data)
        else:
            for colnb in (2, 3, 4):
                if data.shape[1] == colnb and data.shape[0] > colnb:
                    data = data.T
                    break
            obj.xydata = np.array(data)

    def set_image_data(self, obj):
        """Set image data (handles various issues)"""
        data = self.data
        if data.dtype == np.uint32:
            self.uint32_wng = data.max() > np.iinfo(np.int32).max
            clipped_data = data.clip(0, np.iinfo(np.int32).max)
            data = np.array(clipped_data, dtype=np.int32)
        obj.data = data


class NodeFactory:
    """Factory for node classes"""

    def __init__(self):
        self.ignored_datasets = []
        self.generic_classes = []
        self.thirdparty_classes = []

    def add_ignored_datasets(self, names):
        """Add h5 dataset name to ignore list"""
        self.ignored_datasets.extend(names)

    def register(self, cls, is_generic=False):
        """Register node class.
        Generic classes are processed after specific classes (as a fallback solution)"""
        if is_generic:
            self.generic_classes.append(cls)
        else:
            self.thirdparty_classes.append(cls)

    def get(self, dset):
        """Return node class that matches h5 dataset"""
        for name in dset.name.split("/"):
            if name in self.ignored_datasets:
                return None
        for cls in self.thirdparty_classes + self.generic_classes:
            if cls.match(dset):
                return cls
        if isinstance(dset, h5py.Group):
            return GroupNode
        return None


NODE_FACTORY = NodeFactory()


class GroupNode(BaseNode):
    """Object representing a HDF5 group node"""

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "h5group.svg"

    def collect_children(self, node_names):
        """Construct tree"""
        for dset in self.dset.values():
            child_cls = NODE_FACTORY.get(dset)
            if child_cls is not None:
                child = child_cls(self.h5file, dset.name)
                node_names[child.id] = child
                self.children.append(child)
                if isinstance(child, GroupNode):
                    child.collect_children(node_names)

    @property
    def text(self):
        """Return node textual representation"""
        return self.dset.name


class RootNode(GroupNode):
    """Object representing a HDF5 root node"""

    def __init__(self, h5file):
        super().__init__(h5file, "/")

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "h5file.svg"

    @property
    def name(self):
        """Return node name, constructed from dataset name"""
        return osp.basename(self.h5file.filename)

    @property
    def description(self):
        """Return node description"""
        return self.h5file.filename
