# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import os.path as osp
import sys
import warnings
import zipfile

import guidata.config as gcfg

# Silence Sphinx 10 deprecation warning emitted from cairocffi (third-party,
# used by sphinxcontrib-svg2pdfconverter during LaTeX builds).
warnings.filterwarnings(
    "ignore",
    message=r".*Sphinx 10 will drop support for representing paths as strings.*",
)

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
    "sphinxcontrib.cairosvgconverter",
    "sphinx_sitemap",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "guidata.dataset.autodoc",
]
templates_path = ["_templates"]
exclude_patterns = []

# The HTML/LaTeX builds define per-output image substitutions with the same
# names (px for HTML, cm for LaTeX) inside `only::` blocks -- see
# doc/index.rst and doc/intro/index.rst. docutils registers substitution
# definitions at read time, before `only` filtering, so it flags them as
# duplicates ("Duplicate substitution definition name"). They are harmless
# (each builder only ever renders one definition); suppress the docutils
# system messages so every build -- HTML, LaTeX and the `-W` gettext build --
# stays clean.
suppress_warnings = ["docutils"]

# Per-language figure resolution: if e.g. ``foo.png`` is referenced, Sphinx
# will use ``foo.<language>.png`` when available, falling back to ``foo.png``
# otherwise. This is how the maintainer-refreshed UI screenshots under
# ``doc/images/shots/`` (foo.fr.png / foo.en.png) get picked up automatically
# in each language build.
figure_language_filename = "{root}.{language}{ext}"

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
    ann = "DataLab a été présenté à <a href='https://cfp.scipy.org/2024/talk/G3MC9L/'>SciPy 2024</a> 🐍, <a href='https://pretalx.com/pydata-paris-2024/talk/WTDVCC/'>PyData Paris 2024</a>, <a href='https://www.youtube.com/watch?v=lBEu-DeHyz0&list=PLJjbbmRgu6RqGMOhahm2iE6NUkIYIaEDK'>OSXP 2024</a> et <a href='https://www.youtube.com/watch?v=0D4ffBJIc5Q&list=PLJjbbmRgu6RoVze2tajiPe3zJB5muBcuC'>OSXP 2025</a> 🚀 — <a href='https://datalab-platform.com/fr/outreach/index.html'>En savoir plus</a>"  # noqa: E501
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
    "sigima": ("https://sigima.readthedocs.io/en/latest/", None),
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
    # Drop the cmap package: it is a pdflatex-only helper, emits a noisy
    # "pdftex not detected" warning under xelatex, and the resulting PDF
    # remains fully searchable thanks to fontspec/XeTeX.
    "cmappkg": "",
    # Use xelatex (set via ``latex_engine`` below): pdflatex chokes on the
    # emoji / box-drawing / arrow glyphs sprinkled across the docs. The
    # ``ucharclasses`` package automatically routes whole Unicode blocks
    # to the Noto fallback fonts (SIL OFL, fully redistributable):
    #   * Noto Sans Symbols 2 -- miscellaneous symbols, arrows, dingbats,
    #     box drawing, geometric shapes...
    #   * Noto Emoji -- monochrome emoji (XeLaTeX does not render the
    #     COLR/CPAL color tables of Noto Color Emoji reliably).
    # Install:
    #   * Debian/Ubuntu: ``apt install fonts-noto-core fonts-noto-mono
    #     fonts-noto-extra`` (``fonts-noto-extra`` ships the monochrome
    #     ``NotoEmoji-Regular.ttf``; ``fonts-noto-color-emoji`` is *not*
    #     usable because XeTeX rejects CBDT/CBLC color-bitmap fonts).
    #   * Windows: MiKTeX fetches the ``noto`` and ``noto-emoji`` packages
    #     on demand (``xelatex -enable-installer ...``).
    "preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{mathrsfs}
    \usepackage{fontspec}
    % Three complementary Noto fallbacks -- a single one does NOT cover every
    % Unicode sub-block we hit in the docs:
    %   * Noto Sans Symbols  -- arrows, box drawing, geometric shapes,
    %     misc. symbols (info, check mark, sparkles, ...), dingbats.
    %   * Noto Sans Symbols 2 -- box drawing extensions, transport, misc
    %     technical (⏱), and a number of pictographs (🛠 🏗 📁 ...) the
    %     monochrome Noto Emoji does NOT include.
    %   * Noto Sans Mono     -- box drawing characters (─..╿), which
    %     none of the Symbols fonts cover.
    %   * Noto Emoji (mono)  -- emoji-presentation chars (✅ ✨ ❌ ⚠ ℹ).
    %     Noto Color Emoji is not used because XeTeX rejects CBDT/CBLC bitmap
    %     fonts. A few extended pictographs (🧱 🧠 🧩 🧹) live ONLY in Noto
    %     Color Emoji and remain reported as missing -- acceptable cosmetic
    %     limitation.
    \newfontfamily\symbolsfallback{NotoSansSymbols-Regular.ttf}[Scale=MatchLowercase]
    \newfontfamily\symbolstwofallback{NotoSansSymbols2-Regular.ttf}[Scale=MatchLowercase]
    \newfontfamily\monofallback{NotoSansMono-Regular.ttf}[Scale=MatchLowercase]
    \newfontfamily\emojifallback{NotoEmoji-Regular.ttf}[Scale=MatchLowercase]
    \usepackage[Latin,Arrows,LetterlikeSymbols,BoxDrawing,GeometricShapes,Dingbats,MiscellaneousSymbols,MiscellaneousSymbolsAndArrows,MiscellaneousTechnical,Emoticons,MiscellaneousSymbolsAndPictographs,SupplementalSymbolsAndPictographs,TransportAndMapSymbols,SymbolsAndPictographsExtendedA]{ucharclasses}
    % Restore the *current* family (rm / sf / tt) instead of an unconditional
    % \normalfont, which would break monospace rendering inside verbatim and
    % inline code whenever a Unicode symbol appears nearby.
    \makeatletter
    \newcommand{\dlrestorefont}{%
      \ifx\f@family\ttdefault\ttfamily\else
      \ifx\f@family\sfdefault\sffamily\else
      \normalfont\fi\fi}
    \makeatother
    % Force an explicit transition back to the surrounding family whenever we
    % re-enter a Latin block. Without this, XeTeX keeps the last font set by
    % a Symbols/Emoji transition for every following Latin character,
    % producing thousands of "Missing character: There is no <letter> in font
    % Noto Emoji" warnings and a broken PDF.
    \setTransitionsForLatin{\dlrestorefont}{}
    % Route blocks to the font that actually covers them. Coverage notes:
    %   * Box Drawing / Block Elements / Misc Technical live in Symbols 2.
    %   * ⚠ ✅ ✨ ➝ (Misc Symbols / Dingbats with emoji presentation)
    %     are only in Noto Emoji.
    \setTransitionsFor{Arrows}{\symbolsfallback}{\dlrestorefont}
    \setTransitionsFor{LetterlikeSymbols}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{BoxDrawing}{\monofallback}{\dlrestorefont}
    \setTransitionsFor{GeometricShapes}{\symbolsfallback}{\dlrestorefont}
    \setTransitionsFor{Dingbats}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{MiscellaneousSymbols}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{MiscellaneousSymbolsAndArrows}{\symbolstwofallback}{\dlrestorefont}
    \setTransitionsFor{MiscellaneousTechnical}{\symbolstwofallback}{\dlrestorefont}
    \setTransitionsFor{Emoticons}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{MiscellaneousSymbolsAndPictographs}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{SupplementalSymbolsAndPictographs}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{TransportAndMapSymbols}{\emojifallback}{\dlrestorefont}
    \setTransitionsFor{SymbolsAndPictographsExtendedA}{\emojifallback}{\dlrestorefont}
    % Individual overrides for codepoints that are NOT in the block-routed
    % font but exist in another installed Noto. The strategy is:
    %   1. Reset the codepoint's XeTeX charclass to 0 so the broader block's
    %      \setTransitionsFor (which would switch to Noto Emoji and miss
    %      the glyph) no longer fires for it.
    %   2. Use \newunicodechar to redefine the character as a macro that
    %      locally switches to Symbols 2 and emits the glyph via \char to
    %      avoid the infinite recursion that would occur if the macro body
    %      contained the active character itself.
    %   * ➝ HEAVY ROUND-TIPPED RIGHTWARDS ARROW (Dingbats).
    %   * 🛠 🏗 🗃 🖼: pictographs missing from Noto Emoji but present
    %     in Symbols 2.
    \XeTeXcharclass"279D=0
    \XeTeXcharclass"1F6E0=0
    \XeTeXcharclass"1F3D7=0
    \XeTeXcharclass"1F5C3=0
    \XeTeXcharclass"1F5BC=0
    \usepackage{newunicodechar}
    \newunicodechar{➝}{{\symbolstwofallback\char"279D\relax}}
    \newunicodechar{🛠}{{\symbolstwofallback\char"1F6E0\relax}}
    \newunicodechar{🏗}{{\symbolstwofallback\char"1F3D7\relax}}
    \newunicodechar{🗃}{{\symbolstwofallback\char"1F5C3\relax}}
    \newunicodechar{🖼}{{\symbolstwofallback\char"1F5BC\relax}}
    % Discard U+FE0F (VARIATION SELECTOR-16) at the input layer: it has no
    % visible glyph, only requests the emoji presentation of the preceding
    % codepoint. catcode 9 means "ignored character", so XeTeX drops it before
    % font selection -- no more "Missing character" warnings for it.
    \catcode"FE0F=9\relax
    % Silence cosmetic xelatex/LaTeX warnings that are inherent to the
    % current Sphinx 9 + XeTeX + Noto fallback setup and do not affect the
    % visible PDF output. We deliberately keep real errors and undefined
    % cross-references visible.
    %   * \tracinglostchars=0 mutes the XeTeX "Missing character" terminal
    %     messages for the handful of color-bitmap-only emoji (🧱🧩🧠🧹)
    %     that no monochrome Noto font ships.
    %   * \hbadness / \vbadness raise the threshold so the engine no longer
    %     reports Underfull \hbox/\vbox notices caused by long identifier
    %     names in narrow table columns and code-style paragraphs.
    %   * The silence filters drop predictable, harmless package messages.
    \tracinglostchars=0
    % Sphinx resets \hbadness/\vbadness at \begin{document}, so we re-apply
    % the thresholds inside an \AtBeginDocument hook to silence Underfull
    % \hbox/\vbox reports. Note: residual Overfull \hbox notices inside
    % Sphinx tabulary/varwidth cells are not suppressible from here (Sphinx
    % resets \hfuzz locally) and remain as informative typographic notices.
    \AtBeginDocument{%
      \hbadness=99999\relax
      \vbadness=99999\relax
    }
    \usepackage{silence}
    \WarningFilter{latexfont}{Font shape}
    \WarningFilter{latexfont}{Some font shapes were not available}
    \WarningFilter{cmap}{pdftex not detected}
    \WarningFilter{longtable}{Table widths have changed}
    \WarningFilter{rerunfilecheck}{File}
    % Prevent orphan section headings at the bottom of a page: force a page
    % break if there is not enough room for the heading plus a few lines of
    % its following paragraph.
    \usepackage{needspace}
    \usepackage{etoolbox}
    \pretocmd{\section}{\Needspace{12\baselineskip}}{}{}
    \pretocmd{\subsection}{\Needspace{10\baselineskip}}{}{}
    \pretocmd{\subsubsection}{\Needspace{8\baselineskip}}{}{}
    % Sphinx 9 wraps every table cell content in \begin{varwidth}[t]{...}.
    % Combined with our `m{}` column specs (e.g. intro/operating-modes and
    % stakeholders tables), the [t] reference point makes short single-line
    % cells sit at the row bottom instead of the visual centre. Forcing the
    % varwidth alignment to [c] makes m{} truly centre the content; it is a
    % no-op for p{}-style columns where Sphinx ignores the box reference.
    \usepackage{varwidth}
    \let\dlorigvarwidth\varwidth
    \let\dlorigendvarwidth\endvarwidth
    \renewenvironment{varwidth}[2][c]{\dlorigvarwidth[c]{#2}}{\dlorigendvarwidth}
    """
    + "\n".join(f"\\newcommand{{\\{cmd}}}{{{defn}}}" for cmd, defn in macros.items()),
}
latex_engine = "xelatex"
# Sphinx 9 emits the Python Module Index with a \detokenize/\sphinxstyleindexpageref
# pattern whose key does not match the corresponding \label definitions, producing
# spurious "undefined reference" warnings on every entry. The HTML build keeps a
# fully functional modindex; the PDF one is not worth its bogus warnings.
latex_domain_indices = False

# -- MathJax configuration for HTML output -----------------------------------
mathjax3_config = {
    "loader": {"load": ["[tex]/ams"]},
    "tex": {
        "macros": macros,
    },
}
