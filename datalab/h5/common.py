# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Common tools for exogenous HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import os.path as osp
from collections.abc import Callable

import h5py
import numpy as np
from guidata.utils.misc import to_string
from sigima.io.common.converters import convert_array_to_valid_dtype
from sigima.objects import ImageObj, SignalObj

from datalab.config import Conf


def data_to_xy(data: np.ndarray) -> list[np.ndarray]:
    """Convert 2-D array into a list of 1-D array data (x, y, dx, dy).
    This is useful for importing data and creating a DataLab signal with it.

    Args:
        data (numpy.ndarray): 2-D array of data

    Returns:
        list[np.ndarray]: list of 1-D array data (x, y, dx, dy)
    """
    if len(data.ravel()) == len(data):
        return np.arange(len(data)), data.ravel(), None, None
    rows, cols = data.shape
    for colnb in (2, 3, 4):
        if cols == colnb and rows > colnb:
            data = data.T
            break
    if len(data) == 1:
        data = data.T
    if len(data) not in (2, 3, 4):
        raise ValueError(f"Invalid data: len(data)={len(data)} (expected 2, 3 or 4)")
    x, y = data[:2]
    dx, dy = None, None
    if len(data) == 3:
        dy = data[2]
    if len(data) == 4:
        dx, dy = data[2:]
    return x, y, dx, dy


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
        return to_string(self.dset.name).split("/")[-1]

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
        return ""

    @property
    def description(self):
        """Return node description"""
        return ""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset match node pattern"""

    def is_supported(self) -> bool:
        """Return True if node is associated to supported data"""
        return False

    def create_native_object(self):
        """Create native object, if supported"""
        return None

    def get_native_object(self):
        """Return native object, if supported"""
        if self.__obj is None:
            obj = self.create_native_object()  # pylint: disable=assignment-from-none
            if obj is not None:
                self.__process_metadata(obj)
            self.__obj = obj
        return self.__obj

    def collect_attributes(self):
        """Collect attributes from node"""
        for key, value in self.dset.attrs.items():
            if isinstance(value, bytes):
                value = to_string(value)
            if isinstance(value, (np.ndarray, str, float, int, bool)) or np.isscalar(
                value
            ):
                self.metadata[key] = value

    def __process_metadata(self, obj):
        """Process metadata from dataset to obj"""
        obj.reset_metadata_to_defaults()
        obj.set_metadata_option("HDF5Path", self.h5file.filename)
        obj.set_metadata_option("HDF5Dataset", self.id)
        obj.metadata.update(self.metadata)

    @property
    def object_title(self):
        """Return signal/image object title"""
        if Conf.io.h5_fullpath_in_title.get():
            title = self.id
        else:
            title = self.name
        if Conf.io.h5_fname_in_title.get():
            title += f" ({osp.basename(self.h5file.filename)})"
        return title

    def set_signal_data(self, obj: SignalObj) -> None:
        """Set signal data (handles various issues)"""
        data = self.data
        if data.dtype not in (float, np.complex128):
            data = np.array(data, dtype=float)
        data = convert_array_to_valid_dtype(data, SignalObj.VALID_DTYPES)
        if len(data.shape) == 1:
            obj.set_xydata(np.arange(data.size), data)
        else:
            x, y, dx, dy = data_to_xy(data)
            obj.set_xydata(x, y, dx, dy)

    def set_image_data(self, obj: ImageObj) -> None:
        """Set image data (handles various issues)"""
        data = self.data
        if data.dtype == np.uint32:
            self.uint32_wng = data.max() > np.iinfo(np.int32).max
            clipped_data = data.clip(0, np.iinfo(np.int32).max)
            data = np.array(clipped_data, dtype=np.int32)
        obj.data = convert_array_to_valid_dtype(data, ImageObj.VALID_DTYPES)


class H5Importer:
    """DataLab HDF5 importer class"""

    def __init__(self, filename):
        self.h5file = h5py.File(filename)
        self.__nodes = {}
        self.root = RootNode(self.h5file)
        self.__nodes[self.root.id] = self.root
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
        for name in to_string(dset.name).split("/"):
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

    def collect_children(self, node_dict: dict[str, BaseNode]):
        """Construct tree"""
        for dset in self.dset.values():
            child_cls = NODE_FACTORY.get(dset)
            if child_cls is not None:
                child = child_cls(self.h5file, dset.name)
                node_dict[child.id] = child
                self.children.append(child)
                if isinstance(child, GroupNode):
                    child.collect_children(node_dict)
                child.collect_attributes()


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
