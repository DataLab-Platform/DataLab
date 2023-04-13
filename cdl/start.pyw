# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab, the Codra Filtering Tool
Simple signal and image processing application based on guiqwt and guidata

Starter
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    # ----------------------------------------------------------------------------------
    # Macro command execution for the standalone version of CobraDataLab
    exec(sys.argv[2])
    # ----------------------------------------------------------------------------------
else:
    from cdl.app import run

    run()
