# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab configuration module
----------------------------

This module handles `DataLab` configuration (options, images and icons).
"""

from __future__ import annotations

import os
import os.path as osp
import sys

from guidata import configtools
from plotpy.config import CONF as PLOTPY_CONF
from plotpy.config import MAIN_BG_COLOR, MAIN_FG_COLOR
from plotpy.constants import LUTAlpha

from cdl.utils import conf, tests

CONF_VERSION = "1.0.0"

APP_NAME = "DataLab"
MOD_NAME = "cdl"
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
    base_dir = conf.WorkingDirOption()
    available_memory_threshold = conf.Option()
    current_tab = conf.Option()
    plugins_enabled = conf.Option()
    plugins_path = conf.Option()
    tour_enabled = conf.Option()


class ConsoleSection(conf.Section, metaclass=conf.SectionMeta):
    """Classs defining the console configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    console_enabled = conf.Option()
    max_line_count = conf.Option()
    external_editor_path = conf.Option()
    external_editor_args = conf.Option()


class IOSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the I/O configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # HDF5 file format options
    # ------------------------
    # Signal or image title when importing from HDF5 file:
    # - True: use HDF5 full dataset path in signal or image title
    # - False: use HDF5 dataset name in signal or image title
    h5_fullpath_in_title = conf.Option()
    # Signal or image title when importing from HDF5 file:
    # - True: add HDF5 file name in signal or image title
    # - False: do not add HDF5 file name in signal or image title
    h5_fname_in_title = conf.Option()


class ProcSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the Processing configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # ROI extraction strategy:
    # - True: extract all ROIs in a single signal or image
    # - False: extract each ROI in a separate signal or image
    extract_roi_singleobj = conf.Option()

    # FFT shift enabled state for signal/image processing:
    # - True: FFT shift is enabled (default)
    # - False: FFT shift is disabled
    fft_shift_enabled = conf.Option()

    # Ignore warnings during computation:
    # - True: ignore warnings
    # - False: do not ignore warnings
    ignore_warnings = conf.Option()


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

    # String formatting for shape legends
    sig_format = conf.Option()
    ima_format = conf.Option()

    show_label = conf.Option()
    auto_refresh = conf.Option()
    show_contrast = conf.Option()
    sig_antialiasing = conf.Option()

    # If True, images are shown with the same LUT range as the first selected image
    ima_ref_lut_range = conf.Option()

    # Default visualization settings at item creation
    # (e.g. see `ImageObj.make_item` in cdl/core/model/image.py)
    ima_eliminate_outliers = conf.Option()

    # Default visualization settings, persisted in object metadata
    # (e.g. see `SignalObj.update_metadata_view_settings`)
    sig_def_shade = conf.Option()
    sig_def_curvestyle = conf.Option()
    sig_def_baseline = conf.Option()

    # Default visualization settings, persisted in object metadata
    # (e.g. see `ImageObj.update_metadata_view_settings`)
    ima_def_colormap = conf.Option()
    ima_def_interpolation = conf.Option()
    ima_def_alpha = conf.Option()
    ima_def_alpha_function = conf.Option()

    @classmethod
    def get_def_dict(cls, category: str) -> dict:
        """Get default visualization settings as a dictionary

        Args:
            category (str): category ("ima" or "sig", respectively for image
                and signal)

        Returns:
            dict: default visualization settings as a dictionary
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
    def set_def_dict(cls, category: str, def_dict: dict) -> None:
        """Set default visualization settings from a dictionary

        Args:
            category (str): category ("ima" or "sig", respectively for image
                and signal)
            def_dict (dict): default visualization settings as a dictionary
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
    Conf.initialize(APP_NAME, CONF_VERSION, load=not DEBUG)

    # Set default values:
    # -------------------
    # (do not use "set" method here to avoid overwriting user settings in .INI file)
    # Setting here the default values only for the most critical options. The other
    # options default values are set when used in the application code.
    #
    # Main section
    Conf.main.process_isolation_enabled.get(True)
    Conf.main.rpc_server_enabled.get(True)
    Conf.main.traceback_log_path.get(f".{APP_NAME}_traceback.log")
    Conf.main.faulthandler_log_path.get(f".{APP_NAME}_faulthandler.log")
    Conf.main.available_memory_threshold.get(500)
    Conf.main.plugins_enabled.get(True)
    Conf.main.plugins_path.get(Conf.get_path("plugins"))
    Conf.main.tour_enabled.get(True)
    # Console section
    Conf.console.console_enabled.get(True)
    Conf.console.external_editor_path.get("code")
    Conf.console.external_editor_args.get("-g {path}:{line_number}")
    # IO section
    Conf.io.h5_fullpath_in_title.get(False)
    Conf.io.h5_fname_in_title.get(True)
    # Proc section
    Conf.proc.fft_shift_enabled.get(True)
    Conf.proc.extract_roi_singleobj.get(False)
    Conf.proc.ignore_warnings.get(False)
    # View section
    tb_pos = Conf.view.plot_toolbar_position.get("left")
    assert tb_pos in ("top", "bottom", "left", "right")
    Conf.view.ima_ref_lut_range.get(False)
    Conf.view.ima_eliminate_outliers.get(0.1)
    Conf.view.sig_def_shade.get(0.0)
    Conf.view.sig_def_curvestyle.get("Lines")
    Conf.view.sig_def_baseline.get(0.0)
    Conf.view.ima_def_colormap.get("jet")
    Conf.view.ima_def_interpolation.get(0)
    Conf.view.ima_def_alpha.get(1.0)
    Conf.view.ima_def_alpha_function.get(LUTAlpha.NONE.value)


def reset():
    """Reset application configuration"""
    Conf.reset()
    initialize()


initialize()
tests.add_test_module_path(MOD_NAME, osp.join("data", "tests"))


PLOTPY_DEFAULTS = {
    "plot": {
        # "antialiasing": False,
        # "title/font/size": 12,
        # "title/font/bold": False,
        # "marker/curve/text/font/size": 8,
        # "marker/curve/text/font/family": "default",
        # "marker/curve/text/font/bold": False,
        # "marker/curve/text/font/italic": False,
        "marker/curve/text/textcolor": "black",
        # "marker/curve/text/background_color": "#ffffff",
        # "marker/curve/text/background_alpha": 0.8,
        # "marker/cross/text/font/family": "default",
        # "marker/cross/text/font/size": 8,
        # "marker/cross/text/font/bold": False,
        # "marker/cross/text/font/italic": False,
        "marker/cross/text/textcolor": "black",
        # "marker/cross/text/background_color": "#ffffff",
        "marker/cross/text/background_alpha": 0.7,
        # "marker/cross/line/style": "DashLine",
        # "marker/cross/line/color": "yellow",
        # "marker/cross/line/width": 1,
        # "marker/cursor/text/font/size": 8,
        # "marker/cursor/text/font/family": "default",
        # "marker/cursor/text/font/bold": False,
        # "marker/cursor/text/font/italic": False,
        # "marker/cursor/text/textcolor": "#ff9393",
        # "marker/cursor/text/background_color": "#ffffff",
        # "marker/cursor/text/background_alpha": 0.8,
        "shape/drag/symbol/marker": "NoSymbol",
        "shape/mask/symbol/size": 5,
        "shape/mask/sel_symbol/size": 8,
        # -----------------------------------------------------------------------------
        # Annotated shape style for annotations:
        "shape/annotation/line/style": "SolidLine",
        "shape/annotation/line/color": "#ffff00",
        "shape/annotation/line/width": 1,
        "shape/annotation/fill/style": "SolidPattern",
        "shape/annotation/fill/color": MAIN_BG_COLOR,
        "shape/annotation/fill/alpha": 0.1,
        "shape/annotation/symbol/marker": "Rect",
        "shape/annotation/symbol/size": 3,
        "shape/annotation/symbol/edgecolor": "#ffff00",
        "shape/annotation/symbol/facecolor": "#ffff00",
        "shape/annotation/symbol/alpha": 1.0,
        "shape/annotation/sel_line/style": "SolidLine",
        "shape/annotation/sel_line/color": "#00ff00",
        "shape/annotation/sel_line/width": 1,
        "shape/annotation/sel_fill/style": "SolidPattern",
        "shape/annotation/sel_fill/color": MAIN_BG_COLOR,
        "shape/annotation/sel_fill/alpha": 0.1,
        "shape/annotation/sel_symbol/marker": "Rect",
        "shape/annotation/sel_symbol/size": 9,
        "shape/annotation/sel_symbol/edgecolor": "#00aa00",
        "shape/annotation/sel_symbol/facecolor": "#00ff00",
        "shape/annotation/sel_symbol/alpha": 0.7,
        # -----------------------------------------------------------------------------
        # Annotated shape style for result shapes / signals:
        "shape/result/s/line/style": "SolidLine",
        "shape/result/s/line/color": MAIN_FG_COLOR,
        "shape/result/s/line/width": 1,
        "shape/result/s/fill/style": "SolidPattern",
        "shape/result/s/fill/color": MAIN_BG_COLOR,
        "shape/result/s/fill/alpha": 0.1,
        "shape/result/s/symbol/marker": "XCross",
        "shape/result/s/symbol/size": 7,
        "shape/result/s/symbol/edgecolor": MAIN_FG_COLOR,
        "shape/result/s/symbol/facecolor": MAIN_FG_COLOR,
        "shape/result/s/symbol/alpha": 1.0,
        "shape/result/s/sel_line/style": "SolidLine",
        "shape/result/s/sel_line/color": "#00ff00",
        "shape/result/s/sel_line/width": 1,
        "shape/result/s/sel_fill/style": "SolidPattern",
        "shape/result/s/sel_fill/color": MAIN_BG_COLOR,
        "shape/result/s/sel_fill/alpha": 0.1,
        "shape/result/s/sel_symbol/marker": "Rect",
        "shape/result/s/sel_symbol/size": 9,
        "shape/result/s/sel_symbol/edgecolor": "#00aa00",
        "shape/result/s/sel_symbol/facecolor": "#00ff00",
        "shape/result/s/sel_symbol/alpha": 0.7,
        # -----------------------------------------------------------------------------
        # Annotated shape style for result shapes / images:
        "shape/result/i/line/style": "SolidLine",
        "shape/result/i/line/color": "#ffff00",
        "shape/result/i/line/width": 1,
        "shape/result/i/fill/style": "SolidPattern",
        "shape/result/i/fill/color": MAIN_BG_COLOR,
        "shape/result/i/fill/alpha": 0.1,
        "shape/result/i/symbol/marker": "Rect",
        "shape/result/i/symbol/size": 3,
        "shape/result/i/symbol/edgecolor": "#ffff00",
        "shape/result/i/symbol/facecolor": "#ffff00",
        "shape/result/i/symbol/alpha": 1.0,
        "shape/result/i/sel_line/style": "SolidLine",
        "shape/result/i/sel_line/color": "#00ff00",
        "shape/result/i/sel_line/width": 1,
        "shape/result/i/sel_fill/style": "SolidPattern",
        "shape/result/i/sel_fill/color": MAIN_BG_COLOR,
        "shape/result/i/sel_fill/alpha": 0.1,
        "shape/result/i/sel_symbol/marker": "Rect",
        "shape/result/i/sel_symbol/size": 9,
        "shape/result/i/sel_symbol/edgecolor": "#00aa00",
        "shape/result/i/sel_symbol/facecolor": "#00ff00",
        "shape/result/i/sel_symbol/alpha": 0.7,
        # -----------------------------------------------------------------------------
    },
}

PLOTPY_CONF.update_defaults(PLOTPY_DEFAULTS)
PLOTPY_CONF.set_application(osp.join(APP_NAME, "plotpy"), CONF_VERSION, load=False)
