# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import os.path as osp
import sys
import zipfile

import guidata.config as gcfg

sys.path.insert(0, os.path.abspath(".."))

# Importing sigima to avoid re-enabling guidata validation mode
import sigima  # noqa

import datalab

os.environ["DATALAB_DOC"] = "1"

# Turn off validation of guidata config
# (documentation build is not the right place for validation)
gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)


def compress_tutorials_data(app):
    """Compress tutorials data folders to zip files in doc/_download directory."""
    docpath = osp.abspath(osp.dirname(__file__))
    tutorials_path = osp.join(docpath, "..", "datalab", "data", "tutorials")
    download_dir = osp.join(docpath, "_download")

    if not osp.exists(tutorials_path):
        print(f"Warning: tutorials directory not found: {tutorials_path}")
        return

    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Look for folders in tutorials directory
    try:
        for item in os.listdir(tutorials_path):
            item_path = osp.join(tutorials_path, item)
            if osp.isdir(item_path):
                zip_filename = osp.join(download_dir, f"{item}.zip")
                print(f"Compressing tutorial folder: {item} -> {zip_filename}")

                # Create zip file
                with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = osp.join(root, file)
                            arcname = osp.join(item, osp.relpath(file_path, item_path))
                            zipf.write(file_path, arcname)

                print(f"Created: {zip_filename}")
    except Exception as exc:
        print(f"Warning: failed to compress tutorials: {exc}")


def setup(app):
    """Setup function for Sphinx."""
    app.connect("builder-inited", compress_tutorials_data)

    # Exclude outreach directory from LaTeX/PDF builds
    def exclude_outreach_from_latex(app):
        """Exclude outreach directory from LaTeX builds to keep it HTML-only."""
        if app.builder.format == "latex":
            # Exclude the entire outreach directory for PDF builds
            patterns_to_exclude = ["outreach/*"]
            for pattern in patterns_to_exclude:
                if pattern not in app.config.exclude_patterns:
                    app.config.exclude_patterns.append(pattern)
            # Suppress warnings about excluded outreach documents during latex builds
            warnings_to_suppress = ["toc.excluded", "ref.doc"]
            for warning_type in warnings_to_suppress:
                if warning_type not in app.config.suppress_warnings:
                    app.config.suppress_warnings.append(warning_type)

    # Exclude detailed API documentation from gettext extraction, but keep api/index.rst
    def exclude_api_from_gettext(app):
        if app.builder.name == "gettext":
            # Get all RST files in the api directory
            api_rel_dir = "features/advanced/api"
            api_dir = osp.join(app.srcdir, *api_rel_dir.split("/"))
            if osp.exists(api_dir):
                for filename in os.listdir(api_dir):
                    if filename.endswith(".rst") and filename != "index.rst":
                        # Remove .rst extension and add wildcard
                        pattern = f"{api_rel_dir}/{filename[:-4]}*"
                        if pattern not in app.config.exclude_patterns:
                            app.config.exclude_patterns.append(pattern)

                # Also check subdirectories like api/gui/
                for dirname in os.listdir(api_dir):
                    subdir_path = osp.join(api_dir, dirname)
                    if osp.isdir(subdir_path):
                        # Exclude entire subdirectories except their index files
                        pattern = f"{api_rel_dir}/{dirname}/*"
                        if pattern not in app.config.exclude_patterns:
                            app.config.exclude_patterns.append(pattern)

            # Suppress warnings about excluded API documents during gettext builds
            app.config.suppress_warnings.extend(["toc.excluded", "ref.doc"])

    app.connect("builder-inited", exclude_outreach_from_latex)
    app.connect("builder-inited", exclude_api_from_gettext)


# -- Project information -----------------------------------------------------

project = "DataLab"
author = ""
copyright = "2023, DataLab Platform Developers"
release = datalab.__version__
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
html_baseurl = datalab.__homeurl__  # for sitemap extension
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
    ann = "DataLab a été présenté à <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> 🐍, <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>, <a href='https://www.youtube.com/watch?v=lBEu-DeHyz0&list=PLJjbbmRgu6RqGMOhahm2iE6NUkIYIaEDK'>OSXP 2024</a> et <a href='https://datalab-platform.com/fr/outreach/osxp2025.html'>OSXP 2025</a> 🚀 — <a href='https://datalab-platform.com/fr/outreach/index.html'>En savoir plus</a>"  # noqa: E501
else:
    ann = "DataLab has been presented at <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> 🐍, <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>, <a href='https://www.opensource-experience.com/'>OSXP 2024</a>, and <a href='https://datalab-platform.com/en/outreach/osxp2025.html'>OSXP 2025</a> 🚀 — <a href='https://datalab-platform.com/en/outreach/index.html'>Learn more</a>"  # noqa: E501
html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/DataLab-Platform/DataLab/",
    "logo": {
        "text": f"v{datalab.__version__}",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/datalab-platform",
            "icon": "_static/pypi.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "CODRA",
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

# -- Latex macros for math in docstrings -------------------------------------
macros = {
    "FFT": r"\operatorname{FFT}",
    "PSD": r"\operatorname{PSD}",
    "sgn": r"\operatorname{sgn}",
    "sinc": r"\operatorname{sinc}",
    "sawtooth": r"\operatorname{sawtooth}",
    "erfc": r"\operatorname{erfc}",
    "erf": r"\operatorname{erf}",
}

latex_elements = {
    "preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{mathrsfs}"""
    + "\n".join(f"\\newcommand{{\\{cmd}}}{{{defn}}}" for cmd, defn in macros.items()),
}

# -- MathJax configuration for HTML output -----------------------------------
mathjax3_config = {
    "loader": {"load": ["[tex]/ams"]},
    "tex": {
        "macros": macros,
    },
}
