# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import cdl

os.environ["CDL_DOC"] = "1"

# -- Project information -----------------------------------------------------

project = "DataLab"
author = ""
copyright = "2023, DataLab Platform Developers"
release = cdl.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_sitemap",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "guidata.dataset.autodoc",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for sitemap extension -------------------------------------------
html_baseurl = cdl.__homeurl__  # for sitemap extension
sitemap_locales = ["en", "fr"]
sitemap_filename = "../sitemap.xml"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/DataLab-Title.svg"
html_title = project
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
templates_path = ["_templates"]
if "language=fr" in sys.argv:
    ann = "DataLab sera présenté à <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> et <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>! 🚀"  # noqa: E501
else:
    ann = "DataLab will be presented at <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> and <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>! 🚀"  # noqa: E501
html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/DataLab-Platform/DataLab/",
    "logo": {
        "text": f"v{cdl.__version__}",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cdl",
            "icon": "_static/pypi.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "Codra",
            "url": "https://codra.net",
            "icon": "_static/codra.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "announcement": ann,
}
html_static_path = ["_static"]

# -- Options for LaTeX output ------------------------------------------------

latex_logo = "_static/DataLab-Frontpage.png"

# -- Options for sphinx-intl package -----------------------------------------

locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional.

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qwt": ("https://pythonqwt.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "guidata": ("https://guidata.readthedocs.io/en/latest/", None),
    "plotpy": ("https://plotpy.readthedocs.io/en/latest/", None),
}
