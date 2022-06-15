# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT
=======

CodraFT is a generic signal and image processing software based on Python scientific
libraries (such as NumPy, SciPy or OpenCV) and Qt graphical user interfaces (thanks to
guidata and guiqwt libraries).

CodraFT is Copyright © 2022 CEA-CODRA, Pierre Raybaut, and Licensed under the
terms of the BSD 3-Clause License or the CeCILL-B License.
"""

from distutils.core import setup

import setuptools  # pylint: disable=unused-import
from guidata.configtools import get_module_data_path
from guidata.utils import get_package_data, get_subpackages

from codraft import __docurl__, __homeurl__
from codraft import __version__ as version
from codraft.utils import dephash

LIBNAME = "CodraFT"
MODNAME = LIBNAME.lower()

DESCRIPTION = "Signal and image processing software"
LONG_DESCRIPTION = f"""\
CodraFT: Signal and Image Processing Software
=============================================

CodraFT is a **generic signal and image processing software**.
It is based on Python scientific libraries (such as NumPy,
SciPy or OpenCV) and Qt graphical user interfaces (thanks to
`guidata`_ and `guiqwt`_ libraries).

.. image:: https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/CodraFT/master/doc/images/dark_light_modes.png

CodraFT stands for "CODRA Filtering Tool".

CodraFT features are available not only using the **stand-alone application**
(easily installed thanks to the Windows installer or the Python package)
but also by **embedding it into your own application**
(see the "embedded tests" for detailed examples of how to do so).

See `homepage`_ or `documentation`_ for more details on the library
and `changelog`_ for recent history of changes.

Copyrights and licensing
------------------------

- Copyright © 2022 `CEA`_ - `CODRA`_, Pierre Raybaut
- Licensed under the terms of the `BSD / CeCILL-B License`_

.. _guidata: https://pypi.python.org/pypi/guidata
.. _guiqwt: https://pypi.python.org/pypi/guiqwt
.. _homepage: {__homeurl__}
.. _documentation: {__docurl__}
.. _changelog: https://github.com/CODRA-Ingenierie-Informatique/CodraFT/blob/master/CHANGELOG.md
.. _CEA: http://www.cea.fr
.. _CODRA: https://codra.net/
.. _BSD / CeCILL-B License: https://github.com/CODRA-Ingenierie-Informatique/CodraFT/blob/master/LICENSE
"""

KEYWORDS = ""
CLASSIFIERS = [
    "Topic :: Scientific/Engineering",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: OS Independent",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
if "beta" in version or "b" in version:
    CLASSIFIERS += ["Development Status :: 4 - Beta"]
elif "alpha" in version or "a" in version:
    CLASSIFIERS += ["Development Status :: 3 - Alpha"]
else:
    CLASSIFIERS += ["Development Status :: 5 - Production/Stable"]


dephash.create_dependencies_file(
    get_module_data_path("codraft", "data"), ("guidata", "guiqwt")
)

setup(
    name=LIBNAME,
    version=version,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=get_subpackages(MODNAME),
    package_data={
        MODNAME: get_package_data(
            MODNAME,
            (
                ".png",
                ".svg",
                ".mo",
                ".chm",
                ".txt",
                ".h5",
                ".sig",
                ".csv",
                ".json",
                ".npy",
                ".fxd",
                ".scor-data",
            ),
        )
    },
    entry_points={
        "gui_scripts": [f"{MODNAME} = {MODNAME}.app:run"],
        "console_scripts": [
            f"{MODNAME}-tests = {MODNAME}.tests:run",
            f"{MODNAME}-alltests = {MODNAME}.tests.all_tests:run",
        ],
    },
    author="Pierre Raybaut",
    author_email="p.raybaut@codra.fr",
    url=__homeurl__,
    license="BSD",
    classifiers=CLASSIFIERS,
)
