# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
    try:
        from cdl.app import run

        run()
    except Exception as exc:
        with open("datalab_error.log", "w") as stream:
            stream.write(str(exc))
