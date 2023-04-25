# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)


"""
DataLab HDF5 importer module
"""

# Registering dynamic I/O features:
from cdl.core.io.h5 import generic  # pylint: disable=W0611
from cdl.core.io.h5.common import H5Importer  # pylint: disable=W0611
