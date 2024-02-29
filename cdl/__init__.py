# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab
=======

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `PlotPyStack`_ libraries).

.. _PlotPyStack: https://github.com/PlotPyStack
"""

import os

__version__ = "0.14.0"
__docurl__ = __homeurl__ = "https://DataLab-Platform.github.io/"
__supporturl__ = "https://github.com/DataLab-Platform/DataLab/issues/new/choose"

os.environ["CDL_VERSION"] = __version__

try:
    import cdl.core.io  # analysis:ignore
    import cdl.patch  # analysis:ignore  # noqa: F401
except ImportError:
    if not os.environ.get("CDL_DOC"):
        raise

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
