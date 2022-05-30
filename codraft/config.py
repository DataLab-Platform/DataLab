# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
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
APP_NAME = _("CodraFT")
APP_DESC = _(
    """<b>Codra</b> <b>F</b>iltering <b>T</b>ool<br>
Generic signal and image processing software based on Python and Qt"""
)
APP_PATH = osp.dirname(__file__)
DEBUG = len(os.environ.get("DEBUG", "")) > 0
if DEBUG:
    print("*** DEBUG mode *** [Reset configuration file, do not redirect std I/O]")

configtools.add_image_module_path("codraft", osp.join("data", "logo"))
configtools.add_image_module_path("codraft", osp.join("data", "icons"))


class MainSection(conf.Section, metaclass=conf.SectionMeta):
    """Class defining the main configuration section structure.
    Each class attribute is an option (metaclass is automatically affecting
    option names in .INI file based on class attribute names)."""

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


Conf.initialize(APP_NAME, CONF_VERSION, load=not DEBUG)

tests.add_test_module_path("codraft", osp.join("data", "tests"))
