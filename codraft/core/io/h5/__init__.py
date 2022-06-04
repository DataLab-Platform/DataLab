# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)


"""
CodraFT HDF5 importer module
"""

import h5py

# Registering dynamic I/O features:
from codraft.core.io.h5 import generic, mos07636  # pylint: disable=W0611
from codraft.core.io.h5.common import RootNode


class H5Importer:
    """CodraFT HDF5 importer class"""

    def __init__(self, filename):
        self.h5file = h5py.File(filename)
        self.__nodes = {}
        self.root = RootNode(self.h5file)
        self.__nodes[self.root.id] = self.root.dset
        self.root.collect_children(self.__nodes)

    def get(self, node_id):
        """Return node associated to id"""
        return self.__nodes[node_id]

    def close(self):
        """Close HDF5 file"""
        self.__nodes = {}
        self.h5file.close()
