# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)


"""
CodraFT HDF5 importer module
"""

# Registering dynamic I/O features:
from codraft.core.io.h5 import generic, mos07636  # pylint: disable=W0611
from codraft.core.io.h5.common import H5Importer  # pylint: disable=W0611
