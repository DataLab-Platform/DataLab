# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab
============

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `guidata`_ and `guiqwt`_ libraries).

.. _guidata: https://pypi.python.org/pypi/guidata
.. _guiqwt: https://pypi.python.org/pypi/guiqwt
"""

import os

__version__ = "1.0.0"
__docurl__ = "https://cdl.readthedocs.io/en/latest/"
__homeurl__ = "https://codra-ingenierie-informatique.github.io/DataLab/"
__supporturl__ = (
    "https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/new/choose"
)

os.environ["CDL_VERSION"] = __version__

try:
    import cdl.core.io  # analysis:ignore
    import cdl.patch  # analysis:ignore
except ImportError:
    if not os.environ.get("CDL_DOC"):
        raise

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
