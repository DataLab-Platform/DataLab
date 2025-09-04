# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab
=======

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `PlotPyStack`_ libraries).

.. _PlotPyStack: https://github.com/PlotPyStack
"""

import multiprocessing
import os

# Set multiprocessing start method to 'spawn' early to avoid fork-related warnings
# on Linux systems when using Qt and multithreading. This must be done before
# any multiprocessing.Pool is created.
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # This exception is raised if the method is already set (this may happen because
    # this module is imported more than once, e.g. when running tests)
    pass

__version__ = "0.21.0"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/DataLab/issues/new/choose"

os.environ["DATALAB_VERSION"] = __version__

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
