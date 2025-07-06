# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Create dependencies hash
"""

from datalab.config import DATAPATH
from datalab.utils import dephash

dephash.create_dependencies_file(DATAPATH, ("guidata", "plotpy"))
