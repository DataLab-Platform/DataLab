# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab
=======

Starter script for DataLab.
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    # ----------------------------------------------------------------------------------
    # Macro command execution for the standalone version of DataLab
    exec(sys.argv[2])
    # ----------------------------------------------------------------------------------
else:
    from cdlapp.app import run

    run()
