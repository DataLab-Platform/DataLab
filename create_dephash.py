# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Create dependencies hash
"""

from cdl.config import DATAPATH
from cdl.utils import dephash

dephash.create_dependencies_file(DATAPATH, ("guidata", "plotpy"))
