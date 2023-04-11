# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT plugin system
"""

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
import dataclasses
import importlib
import pkgutil
import traceback
from typing import TYPE_CHECKING, List, Optional

from qtpy import QtWidgets as QW

from codraft.config import MOD_NAME, _

if TYPE_CHECKING:
    import codraft.core.gui.main as main
    import codraft.core.gui.panel as panel


class PluginRegistry(type):
    """Metaclass for registering plugins"""

    _plugins: List[PluginBase] = []

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != "PluginBase":
            cls._plugins.append(cls)

    @classmethod
    def get_plugin_classes(cls) -> List[PluginBase]:
        """Return plugin classes"""
        return cls._plugins

    @classmethod
    def get_plugin_instances(cls) -> List[PluginBase]:
        """Return plugin instances"""
        instances = []
        for plugin in cls._plugins:
            try:
                instances.append(plugin.get_instance())
            except Exception:  # pylint: disable=broad-except
                cls._plugins.remove(plugin)
                traceback.print_exc()
        return instances

    @classmethod
    def get_plugin_infos(cls) -> str:
        """Return plugin infos (names, versions, descriptions) in html format"""
        plugins = cls.get_plugin_instances()
        if plugins:
            html = "<i>" + _("Registered plugins:") + "</i><ul>"
            for plugin in plugins:
                html += f"<li>{plugin.info.name} ({plugin.info.version})"
                if plugin.info.description:
                    html += f": {plugin.info.description}"
                html += "</li>"
            html += "</ul>"
        else:
            html = "<i>" + _("No plugins available") + "</i>"
        return html

    @classmethod
    def get_plugin(cls, name_or_class) -> Optional[PluginBase]:
        """Return plugin instance by name"""
        for plugin in cls.get_plugin_instances():
            if isinstance(name_or_class, str):
                if name_or_class == plugin.info.name:
                    return plugin
            else:
                if isinstance(plugin, name_or_class):
                    return plugin
        return None


@dataclasses.dataclass
class PluginInfo:
    """Plugin info"""

    name: str = None
    version: str = "0.0.0"
    description: str = ""
    icon: str = None


class PluginBaseMeta(PluginRegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class PluginBase(abc.ABC, metaclass=PluginBaseMeta):
    """Plugin base class"""

    __instance = None
    PLUGIN_INFO: PluginInfo = None

    @classmethod
    def get_instance(cls) -> PluginBase:
        """Return plugin instance"""
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        self.main: main.CodraFTMainWindow = None
        self._is_registered = False
        self.info = self.PLUGIN_INFO
        if self.info is None:
            raise ValueError(f"Plugin info not set for {self.__class__.__name__}")

    @property
    def signalpanel(self) -> panel.SignalPanel:
        """Return signal panel"""
        return self.main.signalpanel

    @property
    def imagepanel(self) -> panel.ImagePanel:
        """Return image panel"""
        return self.main.imagepanel

    def show_warning(self, message: str):
        """Show warning message"""
        QW.QMessageBox.warning(self.main, _("Warning"), message)

    def show_error(self, message: str):
        """Show error message"""
        QW.QMessageBox.critical(self.main, _("Error"), message)

    def show_info(self, message: str):
        """Show info message"""
        QW.QMessageBox.information(self.main, _("Information"), message)

    def ask_yesno(
        self, message: str, title: Optional[str] = None, cancelable: bool = False
    ) -> bool:
        """Ask yes/no question"""
        if title is None:
            title = _("Question")
        buttons = QW.QMessageBox.Yes | QW.QMessageBox.No
        if cancelable:
            buttons |= QW.QMessageBox.Cancel
        answer = QW.QMessageBox.question(self.main, title, message, buttons)
        if answer == QW.QMessageBox.Yes:
            return True
        if answer == QW.QMessageBox.No:
            return False
        return None

    def is_registered(self):
        """Return True if plugin is registered"""
        return self._is_registered

    def register(self, main: main.CodraFTMainWindow) -> None:
        """Register plugin"""
        if self._is_registered:
            return
        self._is_registered = True
        self.main = main
        self.register_hooks()

    def unregister(self):
        """Unregister plugin"""
        if not self._is_registered:
            return
        self._is_registered = False
        self.unregister_hooks()

    def register_hooks(self):
        """Register plugin hooks"""
        pass

    def unregister_hooks(self):
        """Unregister plugin hooks"""
        pass

    @abc.abstractmethod
    def create_actions(self):
        """Create actions"""


def discover_plugins() -> List[PluginBase]:
    """Discover plugins using naming convention"""
    return [
        importlib.import_module(name)
        for _finder, name, _ispkg in pkgutil.iter_modules()
        if name.startswith(f"{MOD_NAME}_")
    ]
