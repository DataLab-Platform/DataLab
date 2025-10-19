# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration module
----------------------------

This module handles `DataLab` configuration (options, images and icons).
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import Literal

from guidata import configtools
from plotpy.config import CONF as PLOTPY_CONF
from plotpy.config import MAIN_BG_COLOR, MAIN_FG_COLOR
from plotpy.constants import LUTAlpha
from sigima.config import options as sigima_options
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    set_default_title_formatter,
)

from datalab import __version__
from datalab.utils import conf

# Configure Sigima to use DataLab-compatible placeholder title formatting
set_default_title_formatter(PlaceholderTitleFormatter())

CONF_VERSION = "1.0.0"

APP_NAME = "DataLab"
MOD_NAME = "datalab"


def get_config_app_name() -> str:
    """Get configuration application name with major version suffix.

    This function returns the application name used for configuration storage.
    Starting from v1.0, the major version is appended to allow different major
    versions to coexist on the same machine without interfering with each other.

    Returns:
        str: Configuration application name (e.g., "DataLab" for v0.x,
             "DataLab_v1" for v1.x)

    Examples:
        - v0.20.x → "DataLab" (configuration stored in ~/.DataLab)
        - v1.0.x → "DataLab_v1" (configuration stored in ~/.DataLab_v1)
        - v2.0.x → "DataLab_v2" (configuration stored in ~/.DataLab_v2)
    """
    major_version = __version__.split(".", maxsplit=1)[0]

    # Keep v0.x configuration folder unchanged for backward compatibility
    if major_version == "0":
        return APP_NAME

    return f"{APP_NAME}_v{major_version}"


_ = configtools.get_translation(MOD_NAME)

APP_DESC = _("""DataLab is a generic signal and image processing platform""")
APP_PATH = osp.dirname(__file__)

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true")
if DEBUG:
    print("*** DEBUG mode *** [Reset configuration file, do not redirect std I/O]")

TEST_SEGFAULT_ERROR = len(os.environ.get("TEST_SEGFAULT_ERROR", "")) > 0
if TEST_SEGFAULT_ERROR:
    print('*** TEST_SEGFAULT_ERROR mode *** [Enabling test action in "?" menu]')
DATETIME_FORMAT = "%d/%m/%Y - %H:%M:%S"


configtools.add_image_module_path(MOD_NAME, osp.join("data", "logo"))
configtools.add_image_module_path(MOD_NAME, osp.join("data", "icons"))

DATAPATH = configtools.get_module_data_path(MOD_NAME, "data")
SHOTPATH = osp.join(
    configtools.get_module_data_path(MOD_NAME), os.pardir, "doc", "images", "shots"
)
OTHER_PLUGINS_PATHLIST = [configtools.get_module_data_path(MOD_NAME, "plugins")]


def is_frozen(module_name: str) -> bool:
    """Test if module has been frozen (py2exe/cx_Freeze/pyinstaller)

    Args:
        module_name (str): module name

    Returns:
        bool: True if module has been frozen (py2exe/cx_Freeze/pyinstaller)
    """
    datapath = configtools.get_module_path(module_name)
    parentdir = osp.normpath(osp.join(datapath, osp.pardir))
    return not osp.isfile(__file__) or osp.isfile(parentdir)  # library.zip


IS_FROZEN = is_frozen(MOD_NAME)
if IS_FROZEN:
    OTHER_PLUGINS_PATHLIST.append(osp.join(osp.dirname(sys.executable), "plugins"))
    try:
        os.mkdir(OTHER_PLUGINS_PATHLIST[-1])
    except OSError:
        pass


def get_mod_source_dir() -> str | None:
    """Return module source directory

    Returns:
        str | None: module source directory, or None if not found
    """
    if IS_FROZEN:
        devdir = osp.abspath(osp.join(sys.prefix, os.pardir, os.pardir))
    else:
        devdir = osp.abspath(osp.join(osp.dirname(__file__), os.pardir))
    if osp.isfile(osp.join(devdir, MOD_NAME, "__init__.py")):
        return devdir
    # Unhandled case (this should not happen, but just in case):
    return None


class MainSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the main configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    color_mode = conf.EnumOption(["auto", "dark", "light"], default="auto")
    process_isolation_enabled = conf.Option()
    rpc_server_enabled = conf.Option()
    rpc_server_port = conf.Option()
    traceback_log_path = conf.ConfigPathOption()
    traceback_log_available = conf.Option()
    faulthandler_enabled = conf.Option()
    faulthandler_log_path = conf.ConfigPathOption()
    faulthandler_log_available = conf.Option()
    window_maximized = conf.Option()
    window_position = conf.Option()
    window_size = conf.Option()
    window_state = conf.Option()
    base_dir = conf.WorkingDirOption()
    available_memory_threshold = conf.Option()
    current_tab = conf.Option()
    plugins_enabled = conf.Option()
    plugins_path = conf.Option()
    tour_enabled = conf.Option()
    v020_plugins_warning_ignore = conf.Option()  # True: do not warn, False: warn


class ConsoleSection(conf.Section, metaclass=conf.SectionMeta):
    """Classs defining the console configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    console_enabled = conf.Option()
    show_console_on_error = conf.Option()
    max_line_count = conf.Option()
    external_editor_path = conf.Option()
    external_editor_args = conf.Option()


class IOSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the I/O configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # HDF5 file format options
    # ------------------------
    # When opening an HDF5 file, ask user for confirmation if the current workspace
    # has to be cleared before loading the file:
    h5_clear_workspace = conf.Option()  # True: clear workspace, False: do not clear
    h5_clear_workspace_ask = conf.Option()  # True: ask user, False: do not ask
    # Signal or image title when importing from HDF5 file:
    # - True: use HDF5 full dataset path in signal or image title
    # - False: use HDF5 dataset name in signal or image title
    h5_fullpath_in_title = conf.Option()
    # Signal or image title when importing from HDF5 file:
    # - True: add HDF5 file name in signal or image title
    # - False: do not add HDF5 file name in signal or image title
    h5_fname_in_title = conf.Option()

    # ImageIO supported file formats:
    imageio_formats = conf.Option()


class ProcSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the Processing configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # Operation mode:
    # - "single": single operand mode
    # - "pairwise": pairwise operation mode
    operation_mode = conf.EnumOption(["single", "pairwise"], default="single")

    # ROI extraction strategy:
    # - True: extract all ROIs in a single signal or image
    # - False: extract each ROI in a separate signal or image
    extract_roi_singleobj = conf.Option()

    # Keep analysis results after processing:
    # - True: keep analysis results (dangerous because results may not be valid anymore)
    # - False: do not keep analysis results (default)
    keep_results = conf.Option()

    # Use xmin and xmax bounds from current signal when creating a new signal:
    use_signal_bounds = conf.Option()

    # Use dimensions from current image when creating a new image:
    use_image_dims = conf.Option()

    # FFT shift enabled state for signal/image processing:
    # - True: FFT shift is enabled (default)
    # - False: FFT shift is disabled
    fft_shift_enabled = conf.Option()

    # Ignore warnings during computation:
    # - True: ignore warnings
    # - False: do not ignore warnings
    ignore_warnings = conf.Option()

    # X-array compatibility behavior for multi-signal computations:
    # - "ask": ask user for confirmation when x-arrays are incompatible (default)
    # - "interpolate": automatically interpolate when x-arrays are incompatible
    xarray_compat_behavior = conf.EnumOption(["ask", "interpolate"], default="ask")


class ViewSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the view configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # Toolbar position:
    # - "top": top
    # - "bottom": bottom
    # - "left": left
    # - "right": right
    plot_toolbar_position = conf.Option()

    # Ignore information message when inserting object title as annotation label:
    ignore_title_insertion_msg = conf.Option()

    # String formatting for shape legends
    sig_format = conf.Option()
    ima_format = conf.Option()

    show_label = conf.Option()
    auto_refresh = conf.Option()
    show_first_only = conf.Option()  # Show only first selected item
    show_contrast = conf.Option()
    sig_antialiasing = conf.Option()
    sig_autodownsampling = conf.Option()
    sig_autodownsampling_maxpoints = conf.Option()

    # Autoscale margin settings for plots (percentage values)
    sig_autoscale_margin_percent = conf.Option()
    ima_autoscale_margin_percent = conf.Option()

    # If True, lock aspect ratio of images to 1:1 (ignore physical pixel size)
    ima_aspect_ratio_1_1 = conf.Option()

    # If True, images are shown with the same LUT range as the first selected image
    ima_ref_lut_range = conf.Option()

    # Default visualization settings at item creation
    # (e.g. see adapter's `make_item` methods in datalab/adapters_plotpy/*.py)
    ima_eliminate_outliers = conf.Option()

    # Default visualization settings, persisted in object metadata
    # (e.g. see `BaseDataPanel.update_metadata_view_settings`)
    sig_def_shade = conf.Option()
    sig_def_curvestyle = conf.Option()
    sig_def_baseline = conf.Option()
    # ⚠️ Do not add "sig_def_use_dsamp" and "sig_def_dsamp_factor" options here
    # because it would not be compatible with the auto-downsampling feature.

    # Default visualization settings, persisted in object metadata
    # (e.g. see `BaseDataPanel.update_metadata_view_settings`)
    ima_def_colormap = conf.Option()
    ima_def_invert_colormap = conf.Option()
    ima_def_interpolation = conf.Option()
    ima_def_alpha = conf.Option()
    ima_def_alpha_function = conf.Option()

    # Datetime axis format strings for different time units
    # Format strings use Python's strftime format codes
    sig_datetime_format_s = conf.Option()  # Format for s, min, h
    sig_datetime_format_ms = conf.Option()  # Format for ms, us, ns

    @classmethod
    def get_def_dict(cls, category: Literal["ima", "sig"]) -> dict:
        """Get default visualization settings as a dictionary

        Args:
            category: category ("ima" or "sig", respectively for image and signal)

        Returns:
            Default visualization settings as a dictionary
        """
        assert category in ("ima", "sig")
        prefix = f"{category}_def_"
        def_dict = {}
        for attrname in dir(cls):
            if attrname.startswith(prefix):
                name = attrname[len(prefix) :]
                opt = getattr(cls, attrname)
                defval = opt.get(None)
                if defval is not None:
                    def_dict[name] = defval
        return def_dict

    @classmethod
    def set_def_dict(cls, category: Literal["ima", "sig"], def_dict: dict) -> None:
        """Set default visualization settings from a dictionary

        Args:
            category: category ("ima" or "sig", respectively for image and signal)
            def_dict: default visualization settings as a dictionary
        """
        assert category in ("ima", "sig")
        prefix = f"{category}_def_"
        for attrname in dir(cls):
            if attrname.startswith(prefix):
                name = attrname[len(prefix) :]
                opt = getattr(cls, attrname)
                if name in def_dict:
                    opt.set(def_dict[name])


# Usage (example): Conf.console.console_enabled.get(True)
class Conf(conf.Configuration, metaclass=conf.ConfMeta):
    """Class defining DataLab configuration structure.
    Each class attribute is a section (metaclass is automatically affecting
    section names in .INI file based on class attribute names)."""

    main = MainSection()
    console = ConsoleSection()
    view = ViewSection()
    proc = ProcSection()
    io = IOSection()


def get_old_log_fname(fname):
    """Return old log fname from current log fname"""
    return osp.splitext(fname)[0] + ".1.log"


def initialize():
    """Initialize application configuration"""
    config_app_name = get_config_app_name()
    Conf.initialize(config_app_name, CONF_VERSION, load=not DEBUG)

    # Set default values:
    # -------------------
    # (do not use "set" method here to avoid overwriting user settings in .INI file)
    # Setting here the default values only for the most critical options. The other
    # options default values are set when used in the application code.
    #
    # Main section
    Conf.main.color_mode.get("auto")
    Conf.main.process_isolation_enabled.get(True)
    Conf.main.rpc_server_enabled.get(True)
    Conf.main.traceback_log_path.get(f".{APP_NAME}_traceback.log")
    Conf.main.faulthandler_log_path.get(f".{APP_NAME}_faulthandler.log")
    Conf.main.available_memory_threshold.get(500)
    Conf.main.plugins_enabled.get(True)
    Conf.main.plugins_path.get(Conf.get_path("plugins"))
    Conf.main.tour_enabled.get(True)
    Conf.main.v020_plugins_warning_ignore.get(False)
    # Console section
    Conf.console.console_enabled.get(True)
    Conf.console.show_console_on_error.get(False)
    Conf.console.external_editor_path.get("code")
    Conf.console.external_editor_args.get("-g {path}:{line_number}")
    # IO section
    Conf.io.h5_clear_workspace.get(False)
    Conf.io.h5_clear_workspace_ask.get(True)
    Conf.io.h5_fullpath_in_title.get(False)
    Conf.io.h5_fname_in_title.get(True)
    iofmts = Conf.io.imageio_formats.get(())
    if len(iofmts) > 0:
        sigima_options.imageio_formats.set(iofmts)  # Sync with sigima config
    # Proc section
    Conf.proc.operation_mode.get("single")
    Conf.proc.use_signal_bounds.get(False)
    Conf.proc.use_image_dims.get(True)
    Conf.proc.fft_shift_enabled.get(True)
    sigima_options.fft_shift_enabled.set(True)  # Sync with sigima config
    Conf.proc.extract_roi_singleobj.get(False)
    Conf.proc.keep_results.get(False)
    Conf.proc.ignore_warnings.get(False)
    Conf.proc.xarray_compat_behavior.get("ask")
    # View section
    tb_pos = Conf.view.plot_toolbar_position.get("left")
    assert tb_pos in ("top", "bottom", "left", "right")
    Conf.view.ignore_title_insertion_msg.get(False)
    Conf.view.sig_autodownsampling.get(True)
    Conf.view.sig_autodownsampling_maxpoints.get(100000)
    Conf.view.sig_autoscale_margin_percent.get(2.0)
    Conf.view.ima_autoscale_margin_percent.get(1.0)
    Conf.view.ima_aspect_ratio_1_1.get(False)
    Conf.view.ima_ref_lut_range.get(False)
    Conf.view.ima_eliminate_outliers.get(0.1)
    Conf.view.sig_def_shade.get(0.0)
    Conf.view.sig_def_curvestyle.get("Lines")
    Conf.view.sig_def_baseline.get(0.0)
    Conf.view.ima_def_colormap.get("viridis")
    Conf.view.ima_def_invert_colormap.get(False)
    Conf.view.ima_def_interpolation.get(5)
    Conf.view.ima_def_alpha.get(1.0)
    Conf.view.ima_def_alpha_function.get(LUTAlpha.NONE.value)
    # Datetime format strings: % must be escaped as %% for ConfigParser
    Conf.view.sig_datetime_format_s.get("%%H:%%M:%%S")
    Conf.view.sig_datetime_format_ms.get("%%H:%%M:%%S.%%f")

    # Initialize PlotPy configuration with versioned app name
    PLOTPY_CONF.set_application(
        osp.join(config_app_name, "plotpy"), CONF_VERSION, load=False
    )


def reset():
    """Reset application configuration"""
    Conf.reset()
    initialize()


initialize()

ROI_LINE_COLOR = "#5555ff"
ROI_SEL_LINE_COLOR = "#9393ff"

PLOTPY_DEFAULTS = {
    "plot": {
        #
        # XXX: If needed in the future, add here the default settings for PlotPy:
        # that will override the PlotPy settings.
        # That is the right way to customize the PlotPy settings for shapes and
        # annotations when they are added using tools from the DataLab application
        # (see `BaseDataPanel.ANNOTATION_TOOLS`).
        # For example, for shapes:
        # "shape/drag/line/color": "#00ffff",
        #
        # Overriding default plot settings from PlotPy
        "title/font/size": 11,
        "title/font/bold": False,
        "marker/curve/text/textcolor": "black",
        "marker/cross/text/textcolor": "black",
        "marker/cross/text/background_alpha": 0.7,
    },
    "results": {  # Annotated shape style for result shapes
        # Signals:
        "s/line/style": "SolidLine",
        "s/line/color": MAIN_FG_COLOR,
        "s/line/width": 1,
        "s/fill/style": "SolidPattern",
        "s/fill/color": MAIN_BG_COLOR,
        "s/fill/alpha": 0.1,
        "s/symbol/marker": "XCross",
        "s/symbol/size": 7,
        "s/symbol/edgecolor": MAIN_FG_COLOR,
        "s/symbol/facecolor": MAIN_FG_COLOR,
        "s/symbol/alpha": 1.0,
        "s/sel_line/style": "SolidLine",
        "s/sel_line/color": "#00ff00",
        "s/sel_line/width": 1,
        "s/sel_fill/style": "SolidPattern",
        "s/sel_fill/color": MAIN_BG_COLOR,
        "s/sel_fill/alpha": 0.1,
        "s/sel_symbol/marker": "Rect",
        "s/sel_symbol/size": 9,
        "s/sel_symbol/edgecolor": "#00aa00",
        "s/sel_symbol/facecolor": "#00ff00",
        "s/sel_symbol/alpha": 0.7,
        # Images:
        "i/line/style": "SolidLine",
        "i/line/color": "#ffff00",
        "i/line/width": 2,
        "i/fill/style": "SolidPattern",
        "i/fill/color": MAIN_BG_COLOR,
        "i/fill/alpha": 0.1,
        "i/symbol/marker": "Rect",
        "i/symbol/size": 3,
        "i/symbol/edgecolor": "#ffff00",
        "i/symbol/facecolor": "#ffff00",
        "i/symbol/alpha": 1.0,
        "i/sel_line/style": "SolidLine",
        "i/sel_line/color": "#00ff00",
        "i/sel_line/width": 2,
        "i/sel_fill/style": "SolidPattern",
        "i/sel_fill/color": MAIN_BG_COLOR,
        "i/sel_fill/alpha": 0.1,
        "i/sel_symbol/marker": "Rect",
        "i/sel_symbol/size": 9,
        "i/sel_symbol/edgecolor": "#00aa00",
        "i/sel_symbol/facecolor": "#00ff00",
        "i/sel_symbol/alpha": 0.7,
        # Style for labels:
        "label/symbol/marker": "NoSymbol",
        "label/symbol/size": 0,
        "label/symbol/edgecolor": MAIN_BG_COLOR,
        "label/symbol/facecolor": MAIN_BG_COLOR,
        "label/border/style": "SolidLine",
        "label/border/color": "#cbcbcb",
        "label/border/width": 1,
        "label/font/size": 8,
        "label/font/family/nt": ["Cascadia Code", "Consolas", "Courier New"],
        "label/font/family/posix": "Bitstream Vera Sans Mono",
        "label/font/family/mac": "Monaco",
        "label/font/bold": False,
        "label/font/italic": False,
        "label/color": MAIN_FG_COLOR,
        "label/bgcolor": MAIN_BG_COLOR,
        "label/bgalpha": 0.8,
        "label/anchor": "TL",
        "label/xc": 10,
        "label/yc": 10,
        "label/abspos": True,
        "label/absg": "TL",
        "label/xg": 0.0,
        "label/yg": 0.0,
    },
    "roi": {  # Shape style for ROI
        # Signals:
        # - Editable ROI (ROI editor):
        "s/editable/line/style": "SolidLine",
        "s/editable/line/color": "#ffff00",
        "s/editable/line/width": 1,
        "s/editable/fill/style": "SolidPattern",
        "s/editable/fill/color": MAIN_BG_COLOR,
        "s/editable/fill/alpha": 0.1,
        "s/editable/symbol/marker": "Rect",
        "s/editable/symbol/size": 3,
        "s/editable/symbol/edgecolor": "#ffff00",
        "s/editable/symbol/facecolor": "#ffff00",
        "s/editable/symbol/alpha": 1.0,
        "s/editable/sel_line/style": "SolidLine",
        "s/editable/sel_line/color": "#00ff00",
        "s/editable/sel_line/width": 1,
        "s/editable/sel_fill/style": "SolidPattern",
        "s/editable/sel_fill/color": MAIN_BG_COLOR,
        "s/editable/sel_fill/alpha": 0.1,
        "s/editable/sel_symbol/marker": "Rect",
        "s/editable/sel_symbol/size": 9,
        "s/editable/sel_symbol/edgecolor": "#00aa00",
        "s/editable/sel_symbol/facecolor": "#00ff00",
        "s/editable/sel_symbol/alpha": 0.7,
        # - Readonly ROI (plot):
        "s/readonly/line/style": "SolidLine",
        "s/readonly/line/color": ROI_LINE_COLOR,
        "s/readonly/line/width": 1,
        "s/readonly/sel_line/style": "SolidLine",
        "s/readonly/sel_line/color": ROI_SEL_LINE_COLOR,
        "s/readonly/sel_line/width": 2,
        "s/readonly/fill": ROI_LINE_COLOR,
        "s/readonly/shade": 0.15,
        "s/readonly/symbol/marker": "Ellipse",
        "s/readonly/symbol/size": 7,
        "s/readonly/symbol/edgecolor": MAIN_BG_COLOR,
        "s/readonly/symbol/facecolor": ROI_LINE_COLOR,
        "s/readonly/symbol/alpha": 1.0,
        "s/readonly/sel_symbol/marker": "Ellipse",
        "s/readonly/sel_symbol/size": 9,
        "s/readonly/sel_symbol/edgecolor": MAIN_BG_COLOR,
        "s/readonly/sel_symbol/facecolor": ROI_SEL_LINE_COLOR,
        "s/readonly/sel_symbol/alpha": 0.9,
        "s/readonly/multi/color": "#806060",
        # Images:
        # - Editable ROI (ROI editor):
        "i/editable/line/style": "SolidLine",
        "i/editable/line/color": "#ffff00",
        "i/editable/line/width": 1,
        "i/editable/fill/style": "SolidPattern",
        "i/editable/fill/color": MAIN_BG_COLOR,
        "i/editable/fill/alpha": 0.1,
        "i/editable/symbol/marker": "Rect",
        "i/editable/symbol/size": 3,
        "i/editable/symbol/edgecolor": "#ffff00",
        "i/editable/symbol/facecolor": "#ffff00",
        "i/editable/symbol/alpha": 1.0,
        "i/editable/sel_line/style": "SolidLine",
        "i/editable/sel_line/color": "#00ff00",
        "i/editable/sel_line/width": 1,
        "i/editable/sel_fill/style": "SolidPattern",
        "i/editable/sel_fill/color": MAIN_BG_COLOR,
        "i/editable/sel_fill/alpha": 0.1,
        "i/editable/sel_symbol/marker": "Rect",
        "i/editable/sel_symbol/size": 9,
        "i/editable/sel_symbol/edgecolor": "#00aa00",
        "i/editable/sel_symbol/facecolor": "#00ff00",
        "i/editable/sel_symbol/alpha": 0.7,
        # - Readonly ROI (plot):
        "i/readonly/line/style": "DotLine",
        "i/readonly/line/color": ROI_LINE_COLOR,
        "i/readonly/line/width": 1,
        "i/readonly/fill/style": "SolidPattern",
        "i/readonly/fill/color": MAIN_BG_COLOR,
        "i/readonly/fill/alpha": 0.1,
        "i/readonly/symbol/marker": "NoSymbol",
        "i/readonly/symbol/size": 5,
        "i/readonly/symbol/edgecolor": ROI_LINE_COLOR,
        "i/readonly/symbol/facecolor": ROI_LINE_COLOR,
        "i/readonly/symbol/alpha": 0.6,
        "i/readonly/sel_line/style": "DotLine",
        "i/readonly/sel_line/color": "#0000ff",
        "i/readonly/sel_line/width": 1,
        "i/readonly/sel_fill/style": "SolidPattern",
        "i/readonly/sel_fill/color": MAIN_BG_COLOR,
        "i/readonly/sel_fill/alpha": 0.1,
        "i/readonly/sel_symbol/marker": "Rect",
        "i/readonly/sel_symbol/size": 8,
        "i/readonly/sel_symbol/edgecolor": "#0000aa",
        "i/readonly/sel_symbol/facecolor": "#0000ff",
        "i/readonly/sel_symbol/alpha": 0.7,
    },
}

# PlotPy configuration will be initialized in initialize() function
PLOTPY_CONF.update_defaults(PLOTPY_DEFAULTS)
