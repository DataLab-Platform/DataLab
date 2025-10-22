# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Configuration utilities
"""

from __future__ import annotations

import os
import os.path as osp
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from guidata.configtools import MONOSPACE, get_family
from guidata.userconfig import NoDefault, UserConfig

if TYPE_CHECKING:
    import qtpy.QtGui as QG


class AppUserConfig(UserConfig):
    """Application user configuration"""

    def to_dict(self) -> dict:
        """Return configuration as a dictionary"""
        confdict = {}
        for section in self.sections():
            secdict = {}
            for option, value in self.items(section, raw=self.raw):
                secdict[option] = value
            confdict[section] = secdict
        return confdict


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
    def get_filename(cls) -> str:
        """Return configuration file name"""
        return CONF.filename()

    @classmethod
    def get_path(cls, basename: str) -> str:
        """Return filename path inside configuration directory"""
        return CONF.get_path(basename)

    @classmethod
    def to_dict(cls) -> dict:
        """Return configuration as a dictionary"""
        return CONF.to_dict()


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

    def __init__(self) -> None:
        self.section = None
        self.option = None

    def get(self, default=NoDefault) -> Any:
        """Get configuration option value"""
        return CONF.get(self.section, self.option, default)

    def set(self, value: Any) -> None:
        """Set configuration option value"""
        CONF.set(self.section, self.option, value)

    def remove(self) -> None:
        """Remove configuration option"""
        # No use case for this method yet (quite dangerous!)
        CONF.remove_option(self.section, self.option)

    @contextmanager
    def temp(self, value: Any) -> Generator[None, None, None]:
        """Temporarily set configuration option value

        Args:
            value: new value
        """
        old_value = self.get()
        self.set(value)
        yield
        self.set(old_value)


class FontOption(Option):
    """Font configuration option handler"""

    def get(self, default=NoDefault) -> str:
        """Get font name from configuration"""
        if default is NoDefault:
            default = (MONOSPACE, 9, False)
        family = CONF.get(self.section, self.option + "_family", default[0])
        size = CONF.get(self.section, self.option + "_size", default[1])
        bold = CONF.get(self.section, self.option + "_bold", default[2])
        return family, size, bold

    def set(self, value: tuple[str | list[str], int, bool]) -> None:
        """Set font name in configuration"""
        assert isinstance(value, tuple), (
            "Font value must be a tuple (family, size, bold)"
        )
        CONF.set(self.section, self.option + "_family", value[0])
        CONF.set(self.section, self.option + "_size", value[1])
        CONF.set(self.section, self.option + "_bold", value[2])

    def get_font(self) -> QG.QFont:
        """Get QFont from configuration"""
        # Import here to avoid having to create a Qt application when
        # just manipulating configuration files
        from qtpy import QtGui as QG  # pylint: disable=import-outside-toplevel

        family, size, bold = self.get()
        if isinstance(family, (list, tuple)):
            family = get_family(family)
        return QG.QFont(family, size, QG.QFont.Bold if bold else QG.QFont.Normal)


class ConfigPathOption(Option):
    """Configuration file path configuration option handler"""

    def get(self, default=NoDefault) -> str:
        """Get configuration file path from configuration"""
        if default is NoDefault:
            default = ""
        fname = super().get(default)
        if osp.basename(fname) != fname:
            raise ValueError(f"Invalid configuration file name {fname}")
        return CONF.get_path(osp.basename(fname))


class WorkingDirOption(Option):
    """Working directory configuration option handler"""

    def get(self, default=NoDefault) -> str:
        """Get working directory from configuration"""
        if default is NoDefault:
            default = ""
        path = super().get(default)
        if osp.isdir(path):
            return path
        return ""

    def set(self, value: str) -> None:
        """Set working directory in configuration"""
        if not osp.isdir(value):
            value = osp.dirname(value)
            if not osp.isdir(value):
                raise FileNotFoundError(f"Invalid working directory name {value}")
        os.chdir(value)
        super().set(value)


class EnumOption(Option):
    """Enumeration option handler"""

    def __init__(self, values: list[Any], default: Any = NoDefault) -> None:
        super().__init__()
        if default is NoDefault:
            default = values[0]
        self.values = values
        self.default = default

    def get(self, default: Any = NoDefault) -> Any:
        """Get configuration option value"""
        value = super().get(default)
        if value not in self.values:
            # Only show a warning here, as the configuration file may be edited manually
            warnings.warn(
                f"Invalid value {value} for option {self.option}, "
                f"expected {self.values}"
            )
            return self.default
        return value

    def set(self, value: Any) -> None:
        """Set configuration option value"""
        if value not in self.values:
            raise ValueError(
                f"Invalid value {value} for option {self.option}, "
                f"expected {self.values}"
            )
        super().set(value)


class SectionMeta(type):
    """Configuration metaclass"""

    #  pylint: disable=bad-mcs-classmethod-argument
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

    #  pylint: disable=bad-mcs-classmethod-argument
    def __new__(cls, name, bases, dct):
        for attrname, obj in list(dct.items()):
            if isinstance(obj, Section):
                obj.set_name(attrname)
                for option in obj._options:
                    option.section = attrname
        return type.__new__(cls, name, bases, dct)
