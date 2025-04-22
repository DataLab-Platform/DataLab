# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import os.path as osp
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))

import cdl

os.environ["CDL_DOC"] = "1"


# -- Copy CHANGELOG.md to doc/contributing folder ------------------------
#
# Note: An alternative to this could be to create a 'contributing/changelog.rst' file
# containing the following:
#
# .. include:: ../../CHANGELOG.md
#    :parser: myst_parser.sphinx_
#
# But, due to the on-the-fly parsing of the markdown file, this alternative approach
# is not compatible with the internationalization process of the documentation (see
# https://github.com/DataLab-Platform/DataLab/issues/108). That is why we copy the
# CHANGELOG.md file to the doc/contributing folder and remove it after the build.


def copy_changelog(app):
    """Copy CHANGELOG.md to doc/contributing folder."""
    docpath = osp.abspath(osp.dirname(__file__))
    dest_fname = osp.join(docpath, "contributing", "changelog.md")
    if osp.exists(dest_fname):
        os.remove(dest_fname)
    shutil.copyfile(osp.join(docpath, "..", "CHANGELOG.md"), dest_fname)
    app.env.temp_changelog_path = dest_fname


def cleanup_changelog(app, exception):
    """Remove CHANGELOG.md from doc/contributing folder."""
    try:
        path = getattr(app.env, "temp_changelog_path", None)
        if path and osp.exists(path):
            os.remove(path)
    except Exception as exc:
        print(f"Warning: failed to remove {path}: {exc}")
    finally:
        del app.env.temp_changelog_path


def setup(app):
    """Setup function for Sphinx."""
    app.connect("builder-inited", copy_changelog)
    app.connect("build-finished", cleanup_changelog)


# -- Project information -----------------------------------------------------

project = "DataLab"
author = ""
copyright = "2023, DataLab Platform Developers"
release = cdl.__version__
rst_prolog = f"""
.. |download_link1| raw:: html

    <a href="https://github.com/DataLab-Platform/DataLab/releases/download/v{release}/DataLab-{release}.msi">DataLab {release} | Windows 7 SP1, 8, 10, 11</a>
"""  # noqa: E501

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
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
    ann = "DataLab a été dévoilé à <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> 🐍 (Etats-Unis) et présenté en détails à <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>, puis à <a href='https://www.youtube.com/watch?v=lBEu-DeHyz0&list=PLJjbbmRgu6RqGMOhahm2iE6NUkIYIaEDK'>Open Source Experience 2024</a> ! 🚀"  # noqa: E501
else:
    ann = "DataLab has been introduced at <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> 🐍 (Tacoma, WA) and presented thoroughly at <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>! 🚀"  # noqa: E501
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
gettext_compact = False
gettext_location = False

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
