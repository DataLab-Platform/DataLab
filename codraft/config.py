# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
codraft.config
--------------

The `config` module handles `codraft` configuration
(options, images and icons).
"""

import os
import os.path as osp

from guidata import configtools

from codraft.utils import conf, tests

_ = configtools.get_translation("codraft")

CONF_VERSION = "1.0.0"
APP_NAME = "CodraFT"
APP_DESC = _(
    """<b>Codra</b> <b>F</b>iltering <b>T</b>ool<br>
Generic signal and image processing software based on Python and Qt"""
)
APP_PATH = osp.dirname(__file__)
DEBUG = len(os.environ.get("DEBUG", "")) > 0
if DEBUG:
    print("*** DEBUG mode *** [Reset configuration file, do not redirect std I/O]")
TEST_SEGFAULT_ERROR = len(os.environ.get("TEST_SEGFAULT_ERROR", "")) > 0
if TEST_SEGFAULT_ERROR:
    print('*** TEST_SEGFAULT_ERROR mode *** [Enabling test action in "?" menu]')
DATETIME_FORMAT = "%d/%m/%Y - %H:%M:%S"

configtools.add_image_module_path("codraft", osp.join("data", "logo"))
configtools.add_image_module_path("codraft", osp.join("data", "icons"))


class MainSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the main configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

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


class ConsoleSection(conf.Section, metaclass=conf.SectionMeta):
    """Classs defining the console configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    enable = conf.Option()
    max_line_count = conf.Option()


class ViewSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the view configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

    sig_format = conf.Option()
    ima_format = conf.Option()
    show_label = conf.Option()


# Usage (example): Conf.console.enable.get(True)
class Conf(conf.Configuration, metaclass=conf.ConfMeta):
    """Class defining CodraFT configuration structure.
    Each class attribute is a section (metaclass is automatically affecting
    section names in .INI file based on class attribute names)."""

    main = MainSection()
    console = ConsoleSection()
    view = ViewSection()


def get_old_log_fname(fname):
    """Return old log fname from current log fname"""
    return osp.splitext(fname)[0] + ".1.log"


def initialize():
    """Initialize application configuration"""
    Conf.initialize(APP_NAME, CONF_VERSION, load=not DEBUG)
    Conf.main.traceback_log_path.set(f".{APP_NAME}_traceback.log")
    Conf.main.faulthandler_log_path.set(f".{APP_NAME}_faulthandler.log")


def reset():
    """Reset application configuration"""
    Conf.reset()
    initialize()


initialize()
tests.add_test_module_path("codraft", osp.join("data", "tests"))
