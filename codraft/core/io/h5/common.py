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
from typing import Callable, Dict

import h5py
import numpy as np

from codraft.core.io.conv import data_to_xy
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

    @property
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
        obj.metadata["HDF5Path"] = self.h5file.filename
        obj.metadata["HDF5Dataset"] = self.id
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
            x, y, dx, dy = data_to_xy(data)
            obj.set_xydata(x, y, dx, dy)

    def set_image_data(self, obj):
        """Set image data (handles various issues)"""
        data = self.data
        if data.dtype == np.uint32:
            self.uint32_wng = data.max() > np.iinfo(np.int32).max
            clipped_data = data.clip(0, np.iinfo(np.int32).max)
            data = np.array(clipped_data, dtype=np.int32)
        obj.data = data


class H5Importer:
    """CodraFT HDF5 importer class"""

    def __init__(self, filename):
        self.h5file = h5py.File(filename)
        self.__nodes = {}
        self.root = RootNode(self.h5file)
        self.__nodes[self.root.id] = self.root.dset
        self.root.collect_children(self.__nodes)
        NODE_FACTORY.run_post_triggers(self)

    @property
    def nodes(self):
        """Return all nodes"""
        return self.__nodes.values()

    def get(self, node_id: str):
        """Return node associated to id"""
        return self.__nodes[node_id]

    def get_relative(self, node: BaseNode, relpath: str, ancestor: int = 0):
        """Return node using relative path to another node"""
        path = "/" + (
            "/".join(node.id.split("/")[:-ancestor]) + "/" + relpath.strip("/")
        ).strip("/")
        return self.__nodes[path]

    def close(self):
        """Close HDF5 file"""
        self.__nodes = {}
        self.h5file.close()


class NodeFactory:
    """Factory for node classes"""

    def __init__(self):
        self.__ignored_datasets = []
        self.__generic_classes = []
        self.__thirdparty_classes = []
        self.__post_triggers = {}

    def add_ignored_datasets(self, names):
        """Add h5 dataset name to ignore list"""
        self.__ignored_datasets.extend(names)

    def add_post_trigger(self, nodecls: BaseNode, callback: Callable):
        """Add post trigger function, to be called at the end of the collect process.
        Callbacks take only one argument: H5Importer instance."""
        triggers = self.__post_triggers.setdefault(nodecls, [])
        triggers.append(callback)

    def register(self, cls, is_generic=False):
        """Register node class.
        Generic classes are processed after specific classes (as a fallback solution)"""
        if is_generic:
            self.__generic_classes.append(cls)
        else:
            self.__thirdparty_classes.append(cls)

    def get(self, dset):
        """Return node class that matches h5 dataset"""
        for name in dset.name.split("/"):
            if name in self.__ignored_datasets:
                return None
        for cls in self.__thirdparty_classes + self.__generic_classes:
            if cls.match(dset):
                return cls
        if isinstance(dset, h5py.Group):
            return GroupNode
        return None

    def run_post_triggers(self, importer: H5Importer):
        """Run post-collect callbacks"""
        for node in importer.nodes:
            for nodecls, triggers in self.__post_triggers.items():
                if isinstance(node, nodecls):
                    for func in triggers:
                        func(node, importer)


NODE_FACTORY = NodeFactory()


class GroupNode(BaseNode):
    """Object representing a HDF5 group node"""

    @property
    def icon_name(self):
        """Icon name associated to node"""
        return "h5group.svg"

    def collect_children(self, node_dict: Dict):
        """Construct tree"""
        for dset in self.dset.values():
            child_cls = NODE_FACTORY.get(dset)
            if child_cls is not None:
                child = child_cls(self.h5file, dset.name)
                node_dict[child.id] = child
                self.children.append(child)
                if isinstance(child, GroupNode):
                    child.collect_children(node_dict)

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
