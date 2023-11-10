# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Create dependencies hash
"""

from cdlapp.config import DATAPATH
from cdlapp.utils import dephash

dephash.create_dependencies_file(DATAPATH, ("guidata", "plotpy"))
