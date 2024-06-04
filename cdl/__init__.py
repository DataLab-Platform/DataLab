# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab
=======

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `PlotPyStack`_ libraries).

.. _PlotPyStack: https://github.com/PlotPyStack
"""

import os

from cdl.info import __version__

__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/DataLab/issues/new/choose"

os.environ["CDL_VERSION"] = __version__

try:
    import cdl.patch  # analysis:ignore  # noqa: F401
except ImportError:
    if not os.environ.get("CDL_DOC"):
        raise

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
