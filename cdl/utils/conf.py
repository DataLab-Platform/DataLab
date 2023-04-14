# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab Configuration utilities
"""

import os
import os.path as osp

from guidata.userconfig import NoDefault, UserConfig, get_config_dir


class AppUserConfig(UserConfig):
    """Application user configuration"""

    def get_path(self, basename: str) -> str:
        """Return filename path inside configuration directory"""
        app_config_dir = osp.join(get_config_dir(), f".{self.name}")
        if not osp.isdir(app_config_dir):
            os.makedirs(app_config_dir)
        return osp.join(app_config_dir, basename)

    def filename(self) -> str:
        """Return configuration file name"""
        return self.get_path(f"{self.name}.ini")


CONF = AppUserConfig({})


class Configuration:
    """Configuration file"""

    @classmethod
    def initialize(cls, name: str, version: str, load: bool) -> None:
        """Initialize configuration"""
        CONF.set_application(name, version, load=load)

    @classmethod
    def reset(cls) -> None:
        """Reset configuration"""
        global CONF  # pylint: disable=global-statement
        CONF.cleanup()  # Remove configuration file
        CONF = AppUserConfig({})

    @classmethod
    def get_path(cls, basename: str) -> str:
        """Return filename path inside configuration directory"""
        return CONF.get_path(basename)


class Section:
    """Configuration section"""

    @classmethod
    def set_name(cls, section: str) -> None:
        """Set section name"""
        cls._name = section

    @classmethod
    def get_name(cls) -> str:
        """Return section name"""
        return cls._name


class Option:
    """Configuration option handler"""

    def __init__(self):
        self.section = None
        self.option = None

    def get(self, default=NoDefault):
        """Get configuration option value"""
        return CONF.get(self.section, self.option, default)

    def set(self, value):
        """Set configuration option value"""
        CONF.set(self.section, self.option, value)

    def reset(self):
        """Reset configuration option"""
        CONF.remove_option(self.section, self.option)


class ConfigPathOption(Option):
    """Configuration file path configuration option handler"""

    def get(self, default=NoDefault):
        """Get configuration file path from configuration"""
        if default is NoDefault:
            default = ""
        fname = super().get(default)
        if osp.basename(fname) != fname:
            raise ValueError("Invalid configuration file name {fname}")
        return CONF.get_path(osp.basename(fname))


class WorkingDirOption(Option):
    """Working directory configuration option handler"""

    def get(self, default=NoDefault):
        """Get working directory from configuration"""
        if default is NoDefault:
            default = ""
        path = super().get(default)
        if osp.isdir(path):
            return path
        return ""

    def set(self, value):
        """Set working directory in configuration"""
        if not osp.isdir(value):
            value = osp.dirname(value)
            if not osp.isdir(value):
                raise FileNotFoundError(f"Invalid working directory name {value}")
        os.chdir(value)
        super().set(value)


class SectionMeta(type):
    """Configuration metaclass"""

    def __new__(cls, name, bases, dct):
        optlist = []
        for attrname, obj in list(dct.items()):
            if isinstance(obj, Option):
                obj.option = attrname
                optlist.append(obj)
        dct["_options"] = optlist
        return type.__new__(cls, name, bases, dct)


class ConfMeta(type):
    """Configuration metaclass"""

    def __new__(cls, name, bases, dct):
        for attrname, obj in list(dct.items()):
            if isinstance(obj, Section):
                obj.set_name(attrname)
                for option in obj._options:
                    option.section = attrname
        return type.__new__(cls, name, bases, dct)
