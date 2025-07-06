# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab plugin system
---------------------

DataLab plugin system provides a way to extend the application with new
functionalities.

Plugins are Python modules that relies on two classes:

    - :class:`PluginInfo`, which stores information about the plugin
    - :class:`PluginBase`, which is the base class for all plugins

Plugins may also extends DataLab I/O features by providing new image or
signal formats. To do so, they must provide a subclass of :class:`ImageFormatBase`
or :class:`SignalFormatBase`, in which format information is defined using the
:class:`FormatInfo` class.
"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import os
import os.path as osp
import pkgutil
import sys
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW

# pylint: disable=unused-import
from sigima.io.base import FormatInfo  # noqa: F401
from sigima.io.image.base import ImageFormatBase  # noqa: F401
from sigima.io.image.formats import ClassicsImageFormat  # noqa: F401
from sigima.io.signal.base import SignalFormatBase  # noqa: F401

from cdl.config import MOD_NAME, OTHER_PLUGINS_PATHLIST, Conf, _
from cdl.env import execenv
from cdl.proxy import LocalProxy

if TYPE_CHECKING:
    from sigima.obj import NewImageParam, NewSignalParam

    from cdl.gui import main
    from cdl.gui.panel.image import ImagePanel
    from cdl.gui.panel.signal import SignalPanel


PLUGINS_DEFAULT_PATH = Conf.get_path("plugins")

if not osp.isdir(PLUGINS_DEFAULT_PATH):
    os.makedirs(PLUGINS_DEFAULT_PATH)


#  pylint: disable=bad-mcs-classmethod-argument
class PluginRegistry(type):
    """Metaclass for registering plugins"""

    _plugin_classes: list[type[PluginBase]] = []
    _plugin_instances: list[PluginBase] = []

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != "PluginBase":
            cls._plugin_classes.append(cls)

    @classmethod
    def get_plugin_classes(cls) -> list[type[PluginBase]]:
        """Return plugin classes"""
        return cls._plugin_classes

    @classmethod
    def get_plugins(cls) -> list[PluginBase]:
        """Return plugin instances"""
        return cls._plugin_instances

    @classmethod
    def get_plugin(cls, name_or_class: str | type[PluginBase]) -> PluginBase | None:
        """Return plugin instance"""
        for plugin in cls._plugin_instances:
            if name_or_class in (plugin.info.name, plugin.__class__):
                return plugin
        return None

    @classmethod
    def register_plugin(cls, plugin: PluginBase):
        """Register plugin"""
        if plugin.info.name in [plug.info.name for plug in cls._plugin_instances]:
            raise ValueError(f"Plugin {plugin.info.name} already registered")
        cls._plugin_instances.append(plugin)
        execenv.log(cls, f"Plugin {plugin.info.name} registered")

    @classmethod
    def unregister_plugin(cls, plugin: PluginBase):
        """Unregister plugin"""
        cls._plugin_instances.remove(plugin)
        execenv.log(cls, f"Plugin {plugin.info.name} unregistered")
        execenv.log(cls, f"{len(cls._plugin_instances)} plugins left")

    @classmethod
    def unregister_all_plugins(cls):
        """Unregister all plugins"""
        for plugin in cls._plugin_instances:
            execenv.log(cls, f"Unregistering plugin {plugin.info.name}")
            plugin.unregister()
        cls._plugin_instances.clear()
        execenv.log(cls, "All plugins unregistered")

    @classmethod
    def get_plugin_info(cls, html: bool = True) -> str:
        """Return plugin information (names, versions, descriptions) in html format

        Args:
            html: return html formatted text (default: True)
        """
        linesep = "<br>" if html else os.linesep
        bullet = "• " if html else " " * 4

        def italic(text: str) -> str:
            """Return italic text"""
            return f"<i>{text}</i>" if html else text

        if Conf.main.plugins_enabled.get():
            plugins = cls.get_plugins()
            if plugins:
                text = italic(_("Registered plugins:"))
                text += linesep
                for plugin in plugins:
                    text += f"{bullet}{plugin.info.name} ({plugin.info.version})"
                    if plugin.info.description:
                        text += f": {plugin.info.description}"
                    text += linesep
            else:
                text = italic(_("No plugins available"))
        else:
            text = italic(_("Plugins are disabled (see DataLab settings)"))
        return text


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

    PLUGIN_INFO: PluginInfo = None

    def __init__(self):
        self.main: main.CDLMainWindow = None
        self.proxy: LocalProxy = None
        self._is_registered = False
        self.info = self.PLUGIN_INFO
        if self.info is None:
            raise ValueError(f"Plugin info not set for {self.__class__.__name__}")

    @property
    def signalpanel(self) -> SignalPanel:
        """Return signal panel"""
        return self.main.signalpanel

    @property
    def imagepanel(self) -> ImagePanel:
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
        self, message: str, title: str | None = None, cancelable: bool = False
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

    def edit_new_signal_parameters(
        self,
        title: str | None = None,
        size: int | None = None,
        hide_signal_type: bool = True,
    ) -> NewSignalParam:
        """Create and edit new signal parameter dataset

        Args:
            title: title of the new signal
            size: size of the new signal (default: None, get from current signal)
            hide_signal_type: hide signal type parameter (default: True)

        Returns:
            New signal parameter dataset (or None if canceled)
        """
        newparam = self.signalpanel.get_newparam_from_current(title=title)
        if size is not None:
            newparam.size = size
        newparam.hide_signal_type = hide_signal_type
        if newparam.edit(self.main):
            return newparam
        return None

    def edit_new_image_parameters(
        self,
        title: str | None = None,
        shape: tuple[int, int] | None = None,
        hide_image_height: bool = False,
        hide_image_type: bool = True,
        hide_image_dtype: bool = False,
    ) -> NewImageParam | None:
        """Create and edit new image parameter dataset

        Args:
            title: title of the new image
            shape: shape of the new image (default: None, get from current image)
            hide_image_height: hide image heigth parameter (default: False)
            hide_image_type: hide image type parameter (default: True)
            hide_image_dtype: hide image data type parameter (default: False)

        Returns:
            New image parameter dataset (or None if canceled)
        """
        newparam = self.imagepanel.get_newparam_from_current(title=title)
        if shape is not None:
            newparam.width, newparam.height = shape
        newparam.hide_image_height = hide_image_height
        newparam.hide_image_type = hide_image_type
        newparam.hide_image_dtype = hide_image_dtype
        if newparam.edit(self.main):
            return newparam
        return None

    def is_registered(self):
        """Return True if plugin is registered"""
        return self._is_registered

    def register(self, main: main.CDLMainWindow) -> None:
        """Register plugin"""
        if self._is_registered:
            return
        PluginRegistry.register_plugin(self)
        self._is_registered = True
        self.main = main
        self.proxy = LocalProxy(main)
        self.register_hooks()

    def unregister(self):
        """Unregister plugin"""
        if not self._is_registered:
            return
        PluginRegistry.unregister_plugin(self)
        self._is_registered = False
        self.unregister_hooks()
        self.main = None
        self.proxy = None

    def register_hooks(self):
        """Register plugin hooks"""

    def unregister_hooks(self):
        """Unregister plugin hooks"""

    @abc.abstractmethod
    def create_actions(self):
        """Create actions"""


def discover_plugins() -> list[type[PluginBase]]:
    """Discover plugins using naming convention

    Returns:
        List of discovered plugins (as classes)
    """
    if Conf.main.plugins_enabled.get():
        for path in [
            Conf.main.plugins_path.get(),
            PLUGINS_DEFAULT_PATH,
        ] + OTHER_PLUGINS_PATHLIST:
            rpath = osp.realpath(path)
            if rpath not in sys.path:
                sys.path.append(rpath)
        return [
            importlib.import_module(name)
            for _finder, name, _ispkg in pkgutil.iter_modules()
            if name.startswith(f"{MOD_NAME}_")
        ]
    return []


def get_available_plugins() -> list[PluginBase]:
    """Instantiate and get available plugins

    Returns:
        List of available plugins (as instances)
    """
    # Note: this function is not used by DataLab itself, but it is used by the
    #       test suite to get a list of available plugins
    discover_plugins()
    return [plugin_class() for plugin_class in PluginRegistry.get_plugin_classes()]
