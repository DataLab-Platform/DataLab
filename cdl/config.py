# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
cdl.config
----------

The `config` module handles `DataLab` configuration
(options, images and icons).
"""

import os
import os.path as osp

from guidata import configtools
from guiqwt.config import CONF as GUIQWT_CONF

from cdl.utils import conf, tests

CONF_VERSION = "1.0.0"

APP_NAME = "DataLab"
MOD_NAME = "cdl"
_ = configtools.get_translation(MOD_NAME)

APP_DESC = _("""DataLab is a generic signal and image processing platform""")
APP_PATH = osp.dirname(__file__)

DEBUG = len(os.environ.get("DEBUG", "")) > 0
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


def is_frozen(module_name):
    """Test if module has been frozen (py2exe/cx_Freeze)"""
    datapath = configtools.get_module_path(module_name)
    parentdir = osp.normpath(osp.join(datapath, osp.pardir))
    return not osp.isfile(__file__) or osp.isfile(parentdir)  # library.zip


IS_FROZEN = is_frozen(MOD_NAME)


class MainSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the main configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

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
    ignore_dependency_check = conf.Option()
    current_tab = conf.Option()


class ConsoleSection(conf.Section, metaclass=conf.SectionMeta):
    """Classs defining the console configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    enable = conf.Option()
    max_line_count = conf.Option()
    external_editor_path = conf.Option()
    external_editor_args = conf.Option()


class IOSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the I/O configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    h5_fname_in_title = conf.Option()
    h5_fullpath_in_title = conf.Option()


class ProcSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the Processing configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    extract_roi_singleobj = conf.Option()


class ViewSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the view configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    # String formatting for shape legends
    sig_format = conf.Option()
    ima_format = conf.Option()

    show_label = conf.Option()
    show_contrast = conf.Option()

    # If True, images are shown with the same LUT range as the first selected image
    ima_ref_lut_range = conf.Option()

    # Default visualization settings at item creation
    # (e.g. see `ImageParam.make_item` in cdl/core/model/image.py)
    ima_eliminate_outliers = conf.Option()

    # Default visualization settings, persisted in object metadata
    # (e.g. see `create_image` in cdl/core/model/image.py)
    ima_def_colormap = conf.Option()
    ima_def_interpolation = conf.Option()


# Usage (example): Conf.console.enable.get(True)
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
    Conf.main.traceback_log_path.set(f".{APP_NAME}_traceback.log")
    Conf.main.faulthandler_log_path.set(f".{APP_NAME}_faulthandler.log")
    Conf.console.external_editor_path.set("code")
    Conf.console.external_editor_args.set("-g {path}:{line_number}")


def reset():
    """Reset application configuration"""
    Conf.reset()
    initialize()


initialize()
tests.add_test_module_path(MOD_NAME, osp.join("data", "tests"))


GUIQWT_DEFAULTS = {
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
    },
}

GUIQWT_CONF.update_defaults(GUIQWT_DEFAULTS)
GUIQWT_CONF.set_application(osp.join(APP_NAME, "guiqwt"), CONF_VERSION, load=False)
