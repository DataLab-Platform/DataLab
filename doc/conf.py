# -*- coding: utf-8 -*-

# pylint: skip-file

import os
import sys

from guidata.utils.genreqs import generate_requirement_tables

sys.path.insert(0, os.path.abspath(".."))

import cdl  # noqa: E402

generate_requirement_tables(cdl, ["Python>=3.8", "PyQt5>=5.15"])
os.environ["CDL_DOC"] = "1"

# -- Project information -----------------------------------------------------

project = "DataLab"
author = "Pierre Raybaut"
copyright = "2023, Codra - " + author
html_logo = latex_logo = "_static/DataLab-title.png"
release = cdl.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
htmlhelp_basename = project
html_static_path = ["_static"]

# -- Options for sphinx-intl package -----------------------------------------

locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional.

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qwt": ("https://pythonqwt.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "guidata": ("https://guidata.readthedocs.io/en/latest/", None),
    "guiqwt": ("https://guiqwt.readthedocs.io/en/latest/", None),
}
