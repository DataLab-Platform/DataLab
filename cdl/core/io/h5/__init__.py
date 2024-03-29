# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)


"""
DataLab HDF5 importer module
"""

# pylint: disable=unused-import

# Registering dynamic I/O features:
from cdl.core.io.h5 import generic  # noqa: F401
from cdl.core.io.h5.common import H5Importer  # noqa: F401
