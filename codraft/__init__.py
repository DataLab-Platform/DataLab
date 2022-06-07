# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT
=======

CodraFT is a generic signal and image processing software based on Python scientific
libraries (such as NumPy, SciPy or OpenCV) and Qt graphical user interfaces (thanks to
`guidata`_ and `guiqwt`_ libraries).

.. _guidata: https://pypi.python.org/pypi/guidata
.. _guiqwt: https://pypi.python.org/pypi/guiqwt
"""

import os

__version__ = "2.0.2"
__docurl__ = "https://codraft.readthedocs.io/en/latest/"
__homeurl__ = "https://codra-ingenierie-informatique.github.io/CodraFT/"

try:
    import codraft.core.io  # analysis:ignore
    import codraft.patch  # analysis:ignore
except ImportError:
    if not os.environ.get("CODRAFT_DOC"):
        raise

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""

#    Copyright © 2009-2010 CEA
#    Pierre Raybaut
#    Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
#    (see included file `Licence_CeCILL_V2.1-en.txt`)
