# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration module
----------------------------

This module handles `DataLab` configuration (options, images and icons).
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import sys

from guidata import configtools
from plotpy.config import CONF as PLOTPY_CONF
from plotpy.config import MAIN_BG_COLOR, MAIN_FG_COLOR
from plotpy.constants import LUTAlpha
from plotpy.styles import MarkerParam, ShapeParam
from sigima.config import options as sigima_options
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    set_default_title_formatter,
)
from sigimax.config import set_conf
from sigimax.utils import conf
from sigimax.utils.conf import Configuration

from datalab import __version__
from datalab.config.config_options import DataLabOptions
from datalab.config.config_persistence import load_options_from_ini

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

DEBUG = os.environ.get("DATALAB_DEBUG", "").lower() in ("1", "true")
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

# Additional third-party plugin directories provided via the `DATALAB_PLUGINS`
# environment variable. Multiple paths may be separated by `os.pathsep`
# (`;` on Windows, `:` on Unix), following the same convention as `PYTHONPATH`.
# Non-existent paths are skipped with a warning logged at startup.
DATALAB_PLUGINS_ENV_VAR = "DATALAB_PLUGINS"
#: Plugin directories declared through the ``DATALAB_PLUGINS`` env var
#: (subset of :data:`OTHER_PLUGINS_PATHLIST`, kept around so that consumers
#: such as the plugin configuration dialog can flag them as user-provided).
DATALAB_PLUGINS_ENV_PATHS: list[str] = []


def parse_datalab_plugins_env_var(
    env_value: str | None,
    pathlist: list[str],
    env_paths: list[str],
) -> None:
    """Parse ``DATALAB_PLUGINS`` and append valid directories to ``pathlist``.

    Args:
        env_value: Raw value of the ``DATALAB_PLUGINS`` environment variable
         (``None`` or empty string is a no-op).
        pathlist: Plugin search path list to extend in-place
         (typically :data:`OTHER_PLUGINS_PATHLIST`).
        env_paths: List of env-var-provided directories to extend in-place
         (typically :data:`DATALAB_PLUGINS_ENV_PATHS`), used by the GUI to
         flag entries originating from the environment variable.
    """
    if not env_value:
        return

    logger = logging.getLogger(__name__)
    for raw_path in env_value.split(os.pathsep):
        path = raw_path.strip()
        if not path:
            continue
        path = osp.normpath(osp.expanduser(path))
        if osp.isdir(path):
            if path not in pathlist:
                pathlist.append(path)
            if path not in env_paths:
                env_paths.append(path)
        else:
            logger.warning(
                "%s: ignoring non-existent plugin directory '%s'",
                DATALAB_PLUGINS_ENV_VAR,
                path,
            )


parse_datalab_plugins_env_var(
    os.environ.get(DATALAB_PLUGINS_ENV_VAR),
    OTHER_PLUGINS_PATHLIST,
    DATALAB_PLUGINS_ENV_PATHS,
)


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


#: Active typed DataLab configuration shared with reused SigimaX components.
Conf: DataLabOptions = DataLabOptions()  # pylint: disable=invalid-name
set_conf(Conf)


def normalize_plugin_paths(paths: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a list of plugin directories and drop duplicates/empties."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        if not raw_path:
            continue
        path = osp.normpath(osp.abspath(osp.expanduser(raw_path)))
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def get_user_plugin_paths() -> list[str]:
    """Return user-configured extra plugin directories.

    Reads from ``plugins_path_list`` (list of directories).  For backward
    compatibility, the deprecated ``plugins_path`` single-directory string is
    also merged into ``plugins_path_list`` if this is empty.
    """
    fixed_default = osp.normpath(get_config_path("plugins"))

    # New list-based option (primary)
    path_list = Conf.plugins_path_list.get([])
    if path_list is None:
        path_list = []
    candidates = list(path_list)

    # Migrate deprecated single-directory option into the list
    legacy_path = Conf.plugins_path.get("")
    if legacy_path and isinstance(legacy_path, str):
        norm_legacy = osp.normpath(osp.abspath(osp.expanduser(legacy_path)))
        if not candidates and norm_legacy != fixed_default:
            candidates.append(legacy_path)
            Conf.plugins_path_list.set(candidates)

    normalized = normalize_plugin_paths(candidates)
    return [path for path in normalized if path != fixed_default]


def set_user_plugin_paths(paths: list[str] | tuple[str, ...]) -> None:
    """Persist user-configured extra plugin directories.

    Writes to ``plugins_path_list``.  The deprecated ``plugins_path`` is left
    untouched so that older DataLab versions can still find at least one
    user-configured directory.
    """
    normalized = normalize_plugin_paths(list(paths))
    Conf.plugins_path_list.set(normalized)


def get_old_log_fname(fname):
    """Return old log fname from current log fname"""
    return osp.splitext(fname)[0] + ".1.log"


def reload_from_ini() -> None:
    """Reload the active DataLab options from the INI backend."""
    Conf.set_ini_persist_enabled(False)
    try:
        load_options_from_ini(Conf, conf.CONF)
    finally:
        Conf.set_ini_persist_enabled(True)


def get_config_path(basename: str) -> str:
    """Return a path inside the DataLab configuration directory."""
    return Configuration.get_path(basename)


def get_config_filename() -> str:
    """Return the DataLab INI configuration file name."""
    return Configuration.get_filename()


def initialize():
    """Initialize application configuration"""
    config_app_name = get_config_app_name()
    Conf.set_ini_persist_enabled(False)
    try:
        Configuration.initialize(config_app_name, CONF_VERSION, load=not DEBUG)
        if not DEBUG:
            load_options_from_ini(Conf, conf.CONF)
    finally:
        Conf.set_ini_persist_enabled(True)

    # Set default values:
    # -------------------
    # (do not use "set" method here to avoid overwriting user settings in .INI file)
    # Setting here the default values only for the most critical options. The other
    # options default values are set when used in the application code.
    #
    defaults = {
        "color_mode": "auto",
        "process_isolation_enabled": True,
        "rpc_server_enabled": True,
        "webapi_localhost_no_token": True,
        "traceback_log_path": f".{APP_NAME}_traceback.log",
        "faulthandler_log_path": f".{APP_NAME}_faulthandler.log",
        "available_memory_threshold": 500,
        "plugins_enabled": True,
        "plugins_enabled_list": None,
        "plugins_path": "",
        "plugins_path_list": [],
        "tour_enabled": True,
        "v020_plugins_warning_ignore": False,
        "console_enabled": True,
        "show_console_on_error": False,
        "external_editor_path": "code",
        "external_editor_args": "-g {path}:{line_number}",
        "h5_clear_workspace": True,
        "h5_clear_workspace_ask": True,
        "h5_fullpath_in_title": False,
        "h5_fname_in_title": True,
        "imageio_formats": (),
        "macro_console_max_lines": 5000,
        "macro_close_tab_keeps_macro": True,
        "macro_templates_path": get_config_path("macro_templates"),
        "operation_mode": "single",
        "use_signal_bounds": False,
        "use_image_dims": True,
        "fft_shift_enabled": True,
        "auto_normalize_kernel": False,
        "extract_roi_singleobj": False,
        "keep_results": False,
        "show_result_dialog": True,
        "ignore_warnings": False,
        "xarray_compat_behavior": "ask",
        "small_mono_font": (configtools.MONOSPACE, 8, False),
        "plot_toolbar_position": "left",
        "ignore_title_insertion_msg": False,
        "sig_linewidth": 1.0,
        "sig_linewidth_perfs_threshold": 1000,
        "sig_autodownsampling": True,
        "sig_autodownsampling_maxpoints": 100000,
        "sig_autoscale_margin_percent": 2.0,
        "ima_autoscale_margin_percent": 1.0,
        "ima_aspect_ratio_1_1": False,
        "ima_eliminate_outliers": 0.1,
        "sig_def_shade": 0.0,
        "sig_def_curvestyle": "Lines",
        "sig_def_baseline": 0.0,
        "ima_def_colormap": "viridis",
        "ima_def_invert_colormap": False,
        "ima_def_interpolation": 5,
        "ima_def_alpha": 1.0,
        "ima_def_alpha_function": LUTAlpha.NONE.value,
        "ima_def_keep_lut_range": False,
        "sig_datetime_format_s": "%H:%M:%S",
        "sig_datetime_format_ms": "%H:%M:%S.%f",
        "max_shapes_to_draw": 1000,
        "max_cells_in_label": 100,
        "max_cols_in_label": 15,
        "show_result_label": True,
        "show_marker_labels_in_table": True,
    }
    for field_name, default in defaults.items():
        getattr(Conf, field_name).get(default)

    iofmts = Conf.imageio_formats.get()
    if len(iofmts) > 0:
        sigima_options.imageio_formats.set(iofmts)  # Sync with sigima config
    sigima_options.fft_shift_enabled.set(True)  # Sync with sigima config
    sigima_options.auto_normalize_kernel.set(False)  # Sync with sigima config
    tb_pos = Conf.plot_toolbar_position.get()
    assert tb_pos in ("top", "bottom", "left", "right")


def reset():
    """Reset application configuration"""
    Conf.set_ini_persist_enabled(False)
    Configuration.reset()
    Conf.reset_to_defaults()
    initialize()


ROI_LINE_COLOR = "#5555ff"
ROI_SEL_LINE_COLOR = "#9393ff"
MARKER_LINE_COLOR = "#A11818"
MARKER_TEXT_COLOR = "#440909"

PLUGIN_OK_COLOR = "#2ecc71"
PLUGIN_ERROR_COLOR = "#e74c3c"

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
        "selected_curve_symbol/marker": "Ellipse",
        "selected_curve_symbol/edgecolor": "#a0a0a4",
        "selected_curve_symbol/facecolor": MAIN_FG_COLOR,
        "selected_curve_symbol/alpha": 0.3,
        "selected_curve_symbol/size": 5,
        "marker/curve/text/textcolor": "black",
        # Cross marker style (shown when pressing Alt key on plot)
        "marker/cross/symbol/marker": "Cross",
        "marker/cross/symbol/edgecolor": MAIN_FG_COLOR,
        "marker/cross/symbol/facecolor": "#ff0000",
        "marker/cross/symbol/alpha": 1.0,
        "marker/cross/symbol/size": 8,
        "marker/cross/text/font/family": "default",
        "marker/cross/text/font/size": 8,
        "marker/cross/text/font/bold": False,
        "marker/cross/text/font/italic": False,
        "marker/cross/text/textcolor": "#000000",
        "marker/cross/text/background_color": "#ffffff",
        "marker/cross/text/background_alpha": 0.7,
        "marker/cross/line/style": "DashLine",
        "marker/cross/line/color": MARKER_LINE_COLOR,
        "marker/cross/line/width": 1.0,
        "marker/cross/markerstyle": "Cross",
        "marker/cross/spacing": 7,
        # Cursor line and symbol style
        "marker/cursor/line/style": "SolidLine",
        "marker/cursor/line/color": MARKER_LINE_COLOR,
        "marker/cursor/line/width": 1.0,
        "marker/cursor/symbol/marker": "NoSymbol",
        "marker/cursor/symbol/size": 11,
        "marker/cursor/symbol/edgecolor": MAIN_BG_COLOR,
        "marker/cursor/symbol/facecolor": "#ff9393",
        "marker/cursor/symbol/alpha": 1.0,
        "marker/cursor/sel_line/style": "SolidLine",
        "marker/cursor/sel_line/color": MARKER_LINE_COLOR,
        "marker/cursor/sel_line/width": 2.0,
        "marker/cursor/sel_symbol/marker": "NoSymbol",
        "marker/cursor/sel_symbol/size": 11,
        "marker/cursor/sel_symbol/edgecolor": MAIN_BG_COLOR,
        "marker/cursor/sel_symbol/facecolor": MARKER_LINE_COLOR,
        "marker/cursor/sel_symbol/alpha": 0.8,
        "marker/cursor/text/font/size": 9,
        "marker/cursor/text/font/family": "default",
        "marker/cursor/text/font/bold": False,
        "marker/cursor/text/font/italic": False,
        "marker/cursor/text/textcolor": MARKER_TEXT_COLOR,
        "marker/cursor/text/background_color": "#ffffff",
        "marker/cursor/text/background_alpha": 0.7,
        "marker/cursor/sel_text/font/size": 9,
        "marker/cursor/sel_text/font/family": "default",
        "marker/cursor/sel_text/font/bold": False,
        "marker/cursor/sel_text/font/italic": False,
        "marker/cursor/sel_text/textcolor": MARKER_TEXT_COLOR,
        "marker/cursor/sel_text/background_color": "#ffffff",
        "marker/cursor/sel_text/background_alpha": 0.7,
        # Default annotation text style for segments:
        "shape/segment/line/style": "SolidLine",
        "shape/segment/line/color": "#00ff55",
        "shape/segment/line/width": 1.0,
        "shape/segment/sel_line/style": "SolidLine",
        "shape/segment/sel_line/color": "#00ff55",
        "shape/segment/sel_line/width": 2.0,
        "shape/segment/fill/style": "NoBrush",
        "shape/segment/sel_fill/style": "NoBrush",
        "shape/segment/symbol/marker": "XCross",
        "shape/segment/symbol/size": 9,
        "shape/segment/symbol/edgecolor": "#00ff55",
        "shape/segment/symbol/facecolor": "#00ff55",
        "shape/segment/symbol/alpha": 1.0,
        "shape/segment/sel_symbol/marker": "XCross",
        "shape/segment/sel_symbol/size": 12,
        "shape/segment/sel_symbol/edgecolor": "#00ff55",
        "shape/segment/sel_symbol/facecolor": "#00ff55",
        "shape/segment/sel_symbol/alpha": 0.7,
        # Default style for drag shapes: (global annotations style)
        "shape/drag/line/style": "SolidLine",
        "shape/drag/line/color": "#00ff55",
        "shape/drag/line/width": 1.0,
        "shape/drag/fill/style": "SolidPattern",
        "shape/drag/fill/color": MAIN_BG_COLOR,
        "shape/drag/fill/alpha": 0.1,
        "shape/drag/symbol/marker": "Rect",
        "shape/drag/symbol/size": 3,
        "shape/drag/symbol/edgecolor": "#00ff55",
        "shape/drag/symbol/facecolor": "#00ff55",
        "shape/drag/symbol/alpha": 1.0,
        "shape/drag/sel_line/style": "SolidLine",
        "shape/drag/sel_line/color": "#00ff55",
        "shape/drag/sel_line/width": 2.0,
        "shape/drag/sel_fill/style": "SolidPattern",
        "shape/drag/sel_fill/color": MAIN_BG_COLOR,
        "shape/drag/sel_fill/alpha": 0.1,
        "shape/drag/sel_symbol/marker": "Rect",
        "shape/drag/sel_symbol/size": 7,
        "shape/drag/sel_symbol/edgecolor": "#00ff55",
        "shape/drag/sel_symbol/facecolor": "#00ff00",
        "shape/drag/sel_symbol/alpha": 0.7,
    },
    "results": {
        # Annotated shape style for result shapes:
        #   Signals:
        "s/annotation/line/style": "SolidLine",
        "s/annotation/line/color": "#00aa00",
        "s/annotation/line/width": 2,
        "s/annotation/fill/style": "NoBrush",
        "s/annotation/fill/color": MAIN_BG_COLOR,
        "s/annotation/fill/alpha": 0.1,
        "s/annotation/symbol/marker": "XCross",
        "s/annotation/symbol/size": 7,
        "s/annotation/symbol/edgecolor": "#00aa00",
        "s/annotation/symbol/facecolor": "#00aa00",
        "s/annotation/symbol/alpha": 1.0,
        "s/annotation/sel_line/style": "DashLine",
        "s/annotation/sel_line/color": "#00ff00",
        "s/annotation/sel_line/width": 1,
        "s/annotation/sel_fill/style": "SolidPattern",
        "s/annotation/sel_fill/color": MAIN_BG_COLOR,
        "s/annotation/sel_fill/alpha": 0.1,
        "s/annotation/sel_symbol/marker": "Rect",
        "s/annotation/sel_symbol/size": 9,
        "s/annotation/sel_symbol/edgecolor": "#00aa00",
        "s/annotation/sel_symbol/facecolor": "#00ff00",
        "s/annotation/sel_symbol/alpha": 0.7,
        #   Images:
        "i/annotation/line/style": "SolidLine",
        "i/annotation/line/color": "#ffff00",
        "i/annotation/line/width": 2,
        "i/annotation/fill/style": "SolidPattern",
        "i/annotation/fill/color": MAIN_BG_COLOR,
        "i/annotation/fill/alpha": 0.1,
        "i/annotation/symbol/marker": "Rect",
        "i/annotation/symbol/size": 3,
        "i/annotation/symbol/edgecolor": "#ffff00",
        "i/annotation/symbol/facecolor": "#ffff00",
        "i/annotation/symbol/alpha": 1.0,
        "i/annotation/sel_line/style": "SolidLine",
        "i/annotation/sel_line/color": "#00ff00",
        "i/annotation/sel_line/width": 2,
        "i/annotation/sel_fill/style": "SolidPattern",
        "i/annotation/sel_fill/color": MAIN_BG_COLOR,
        "i/annotation/sel_fill/alpha": 0.1,
        "i/annotation/sel_symbol/marker": "Rect",
        "i/annotation/sel_symbol/size": 9,
        "i/annotation/sel_symbol/edgecolor": "#00aa00",
        "i/annotation/sel_symbol/facecolor": "#00ff00",
        "i/annotation/sel_symbol/alpha": 0.7,
        # Marker styles for results:
        #   Signals:
        "s/marker/cursor/line/style": "DashLine",
        "s/marker/cursor/line/color": MARKER_LINE_COLOR,
        "s/marker/cursor/line/width": 1.0,
        "s/marker/cursor/symbol/marker": "Ellipse",
        "s/marker/cursor/symbol/size": 11,
        "s/marker/cursor/symbol/edgecolor": MAIN_BG_COLOR,
        "s/marker/cursor/symbol/facecolor": MARKER_LINE_COLOR,
        "s/marker/cursor/symbol/alpha": 0.7,
        "s/marker/cursor/sel_line/style": "DashLine",
        "s/marker/cursor/sel_line/color": MARKER_LINE_COLOR,
        "s/marker/cursor/sel_line/width": 2.0,
        "s/marker/cursor/sel_symbol/marker": "Ellipse",
        "s/marker/cursor/sel_symbol/size": 11,
        "s/marker/cursor/sel_symbol/edgecolor": MARKER_LINE_COLOR,
        "s/marker/cursor/sel_symbol/facecolor": MARKER_LINE_COLOR,
        "s/marker/cursor/sel_symbol/alpha": 0.7,
        "s/marker/cursor/text/font/size": 9,
        "s/marker/cursor/text/font/family": "default",
        "s/marker/cursor/text/font/bold": False,
        "s/marker/cursor/text/font/italic": False,
        "s/marker/cursor/text/textcolor": MARKER_TEXT_COLOR,
        "s/marker/cursor/text/background_color": "#ffffff",
        "s/marker/cursor/text/background_alpha": 0.7,
        "s/marker/cursor/sel_text/font/size": 9,
        "s/marker/cursor/sel_text/font/family": "default",
        "s/marker/cursor/sel_text/font/bold": False,
        "s/marker/cursor/sel_text/font/italic": False,
        "s/marker/cursor/sel_text/textcolor": MARKER_TEXT_COLOR,
        "s/marker/cursor/sel_text/background_color": "#ffffff",
        "s/marker/cursor/sel_text/background_alpha": 0.7,
        "s/marker/cursor/markerstyle": "Cross",
        #   Images:
        "i/marker/cursor/line/style": "DashLine",
        "i/marker/cursor/line/color": MARKER_LINE_COLOR,
        "i/marker/cursor/line/width": 1.0,
        "i/marker/cursor/symbol/marker": "Diamond",
        "i/marker/cursor/symbol/size": 11,
        "i/marker/cursor/symbol/edgecolor": MARKER_LINE_COLOR,
        "i/marker/cursor/symbol/facecolor": MARKER_LINE_COLOR,
        "i/marker/cursor/symbol/alpha": 0.7,
        "i/marker/cursor/sel_line/style": "DashLine",
        "i/marker/cursor/sel_line/color": MARKER_LINE_COLOR,
        "i/marker/cursor/sel_line/width": 2.0,
        "i/marker/cursor/sel_symbol/marker": "Diamond",
        "i/marker/cursor/sel_symbol/size": 11,
        "i/marker/cursor/sel_symbol/edgecolor": MARKER_LINE_COLOR,
        "i/marker/cursor/sel_symbol/facecolor": MARKER_LINE_COLOR,
        "i/marker/cursor/sel_symbol/alpha": 0.7,
        "i/marker/cursor/text/font/size": 9,
        "i/marker/cursor/text/font/family": "default",
        "i/marker/cursor/text/font/bold": False,
        "i/marker/cursor/text/font/italic": False,
        "i/marker/cursor/text/textcolor": MARKER_TEXT_COLOR,
        "i/marker/cursor/text/background_color": "#ffffff",
        "i/marker/cursor/text/background_alpha": 0.7,
        "i/marker/cursor/sel_text/font/size": 9,
        "i/marker/cursor/sel_text/font/family": "default",
        "i/marker/cursor/sel_text/font/bold": False,
        "i/marker/cursor/sel_text/font/italic": False,
        "i/marker/cursor/sel_text/textcolor": MARKER_TEXT_COLOR,
        "i/marker/cursor/sel_text/background_color": "#ffffff",
        "i/marker/cursor/sel_text/background_alpha": 0.7,
        "i/marker/cursor/markerstyle": "Cross",
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
        "s/editable/fill": "#ffff00",
        "s/editable/shade": 0.10,
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
        "s/readonly/shade": 0.10,
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
PLOTPY_CONF.set_application(
    osp.join(get_config_app_name(), "plotpy"), CONF_VERSION, load=False
)


class DataLabShapeParam(ShapeParam):
    """ShapeParam subclass with internal items hidden from settings dialog"""

    def __init__(self):
        super().__init__()
        # Hide internal items that should not appear in settings dialog
        for item in self._items:
            if item._name in ("label", "readonly", "private"):
                item.set_prop("display", hide=True)


def initialize_default_plotpy_instances():
    """Initialize default PlotPy instances for DataLab configuration options"""
    # Initialize default instances for DataSetOptions now that PLOTPY_DEFAULTS exists
    _sig_shapeparam = DataLabShapeParam()
    _sig_shapeparam.read_config(PLOTPY_CONF, "results", "s/annotation")
    Conf.sig_shape_param.set_default_instance(_sig_shapeparam)
    Conf.sig_shape_param.get()

    _sig_markerparam = MarkerParam()
    _sig_markerparam.read_config(PLOTPY_CONF, "results", "s/marker/cursor")
    Conf.sig_marker_param.set_default_instance(_sig_markerparam)
    Conf.sig_marker_param.get()

    _ima_shapeparam = DataLabShapeParam()
    _ima_shapeparam.read_config(PLOTPY_CONF, "results", "i/annotation")
    Conf.ima_shape_param.set_default_instance(_ima_shapeparam)
    Conf.ima_shape_param.get()

    _ima_markerparam = MarkerParam()
    _ima_markerparam.read_config(PLOTPY_CONF, "results", "i/marker/cursor")
    Conf.ima_marker_param.set_default_instance(_ima_markerparam)
    Conf.ima_marker_param.get()


initialize_default_plotpy_instances()
initialize()
