# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab
=======

Starter script for DataLab.
"""

import sys

# --------------------------------------------------------------------------------------
# Macro command execution for the standalone version of DataLab
if len(sys.argv) > 1 and sys.argv[1] == "-c":
    exec(sys.argv[2])
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Main DataLab application
#
# This part is executed when the script is run as a standalone application.
# It is protected by using `if __name__ == '__main__':` to avoid running it again when
# the multiprocessing pool is created (see cdl/gui/processor/base.py).
elif __name__ == "__main__":
    try:
        from cdl.app import run

        run()
    except Exception as exc:
        # At this point, we intercept exceptions raised before the application is
        # started and write them to a log file. This could happen if there is an error
        # occuring during the import of one of the modules used by the application.
        # Last example to date: NumPy V2.0.0-2.0.1 raises an exception when imported
        # if the script is executed using 'pythonw.exe' instead of 'python.exe'.
        import datetime
        import traceback

        with open("datalab_error.log", "w") as stream:
            stream.write(f"DataLab failed to start:\n")
            stream.write(f"  Date: {datetime.datetime.now()}\n")
            stream.write(f"  Python {sys.version}\n")
            stream.write(f"  Executable: {sys.executable}\n")
            comment = ""
            if sys.platform != "win32":
                comment = " - <!!!> This script is intended to be run on Windows only."
            stream.write(f"  Platform: {sys.platform}{comment}\n")
            stream.write(f"  Python path:\n")
            for path in sys.path:
                stream.write(f"    {path}\n")
            stream.write("\n")
            traceback.print_exc(file=stream)
# --------------------------------------------------------------------------------------
