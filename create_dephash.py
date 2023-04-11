# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Create dependencies hash
"""

from codraft.config import DATAPATH
from codraft.utils import dephash

dephash.create_dependencies_file(DATAPATH, ("guidata", "guiqwt"))
