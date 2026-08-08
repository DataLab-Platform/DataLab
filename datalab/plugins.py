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
import importlib.util
import logging
import os
import os.path as osp
import pkgutil
import sys
import traceback
from contextlib import ExitStack
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW

# pylint: disable=unused-import
from sigima.io.base import FormatInfo  # noqa: F401
from sigima.io.image.base import ImageFormatBase  # noqa: F401
from sigima.io.image.formats import ClassicsImageFormat  # noqa: F401
from sigima.io.signal.base import SignalFormatBase  # noqa: F401

from datalab.config import (
    MOD_NAME,
    OTHER_PLUGINS_PATHLIST,
    Conf,
    _,
    get_user_plugin_paths,
)
from datalab.control.proxy import LocalProxy
from datalab.env import execenv

if TYPE_CHECKING:
    from sigima.objects import NewImageParam, NewSignalParam

    from datalab.gui import main
    from datalab.gui.panel.image import ImagePanel
    from datalab.gui.panel.signal import SignalPanel


PLUGINS_DEFAULT_PATH = Conf.get_path("plugins")

if not osp.isdir(PLUGINS_DEFAULT_PATH):
    os.makedirs(PLUGINS_DEFAULT_PATH)


#  pylint: disable=bad-mcs-classmethod-argument
class PluginRegistry(type):
    """Metaclass for registering plugins"""

    _plugin_classes: list[type[PluginBase]] = []
    _plugin_instances: list[PluginBase] = []
    _discovery_errors: list[str] = []
    _failed_plugins: list[FailedPluginInfo] = []

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
        """Return a plugin by stable ID, legacy name, or class.

        Legacy display names are accepted only when they identify a single plugin.
        """
        for plugin in cls._plugin_instances:
            if name_or_class in (plugin.plugin_id, plugin.__class__):
                return plugin
        if isinstance(name_or_class, str):
            matching_plugins = [
                plugin
                for plugin in cls._plugin_instances
                if plugin.info.name == name_or_class
            ]
            if len(matching_plugins) == 1:
                return matching_plugins[0]
        return None

    @classmethod
    def register_plugin(cls, plugin: PluginBase):
        """Register plugin"""
        if plugin.plugin_id in [
            registered.plugin_id for registered in cls._plugin_instances
        ]:
            raise ValueError(f"Plugin ID {plugin.plugin_id!r} already registered")
        cls._plugin_instances.append(plugin)
        execenv.log(cls, f"Plugin {plugin.info.name} ({plugin.plugin_id}) registered")

    @classmethod
    def unregister_plugin(cls, plugin: PluginBase):
        """Unregister plugin"""
        cls._plugin_instances.remove(plugin)
        execenv.log(cls, f"Plugin {plugin.info.name} unregistered")
        execenv.log(cls, f"{len(cls._plugin_instances)} plugins left")

    @classmethod
    def unregister_all_plugins(cls):
        """Unregister all plugins"""
        try:
            with ExitStack() as stack:
                for plugin in reversed(list(cls._plugin_instances)):
                    stack.callback(plugin.unregister)
                    stack.callback(
                        execenv.log,
                        cls,
                        f"Unregistering plugin {plugin.info.name}",
                    )
        finally:
            cls._plugin_instances.clear()
            execenv.log(cls, "All plugins unregistered")

    @classmethod
    def clear_plugin_classes(cls) -> None:
        """Clear registered plugin classes.

        This is mainly useful when reloading plugin modules at runtime.
        """
        cls._plugin_classes.clear()

    @classmethod
    def add_discovery_error(cls, tb_text: str) -> None:
        """Record an error traceback that occurred during plugin discovery.

        Args:
            tb_text: Formatted traceback string
        """
        cls._discovery_errors.append(tb_text)

    @classmethod
    def get_discovery_errors(cls) -> list[str]:
        """Return error tracebacks collected during plugin discovery.

        Returns:
            List of formatted traceback strings (may be empty)
        """
        return list(cls._discovery_errors)

    @classmethod
    def clear_discovery_errors(cls) -> None:
        """Clear recorded discovery errors."""
        cls._discovery_errors.clear()

    @classmethod
    def add_failed_plugin(cls, name: str, filepath: str, tb_text: str) -> None:
        """Record a plugin that failed to load or instantiate.

        Args:
            name: Module or plugin class name
            filepath: File path of the plugin module
            tb_text: Formatted traceback string
        """
        cls._failed_plugins.append(FailedPluginInfo(name, filepath, tb_text))

    @classmethod
    def get_failed_plugins(cls) -> list[FailedPluginInfo]:
        """Return structured info about plugins that failed to load.

        Returns:
            List of FailedPluginInfo objects (may be empty)
        """
        return list(cls._failed_plugins)

    @classmethod
    def clear_failed_plugins(cls) -> None:
        """Clear recorded failed plugin info."""
        cls._failed_plugins.clear()

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
class FailedPluginInfo:
    """Information about a plugin that failed to load or instantiate."""

    name: str
    filepath: str
    traceback: str


@dataclasses.dataclass
class PluginInfo:
    """Plugin info"""

    name: str = None
    version: str = "0.0.0"
    description: str = ""
    icon: str = None
    id: str | None = None


class PluginBaseMeta(PluginRegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class PluginBase(abc.ABC, metaclass=PluginBaseMeta):
    """Plugin base class"""

    PLUGIN_INFO: PluginInfo = None

    def __init__(self):
        self.main: main.DLMainWindow = None
        self.proxy: LocalProxy = None
        self._is_registered = False
        self.info = self.PLUGIN_INFO
        if self.info is None:
            raise ValueError(f"Plugin info not set for {self.__class__.__name__}")
        self.get_plugin_id()

    @classmethod
    def get_plugin_id(cls) -> str:
        """Return the stable plugin ID or a deterministic legacy fallback."""
        if cls.PLUGIN_INFO is None:
            raise ValueError(f"Plugin info not set for {cls.__name__}")
        if cls.PLUGIN_INFO.id is not None:
            if not cls.PLUGIN_INFO.id.strip():
                raise ValueError(f"Plugin ID not set for {cls.__name__}")
            return cls.PLUGIN_INFO.id
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def plugin_id(self) -> str:
        """Return the stable plugin ID or a deterministic legacy fallback."""
        return self.get_plugin_id()

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
    ) -> NewSignalParam:
        """Create and edit new signal parameter dataset

        Args:
            title: title of the new signal
            size: size of the new signal (default: None, get from current signal)

        Returns:
            New signal parameter dataset (or None if canceled)
        """
        newparam = self.signalpanel.get_newparam_from_current(title=title)
        if size is not None:
            newparam.size = size
        if newparam.edit(self.main):
            return newparam
        return None

    def edit_new_image_parameters(
        self,
        title: str | None = None,
        shape: tuple[int, int] | None = None,
        hide_height: bool = False,
        hide_width: bool = False,
        hide_type: bool = True,
        hide_dtype: bool = False,
    ) -> NewImageParam | None:
        """Create and edit new image parameter dataset

        Args:
            title: title of the new image
            shape: shape of the new image (default: None, get from current image)
            hide_height: hide image heigth parameter (default: False)
            hide_width: hide image width parameter (default: False)
            hide_type: hide image type parameter (default: True)
            hide_dtype: hide image data type parameter (default: False)

        Returns:
            New image parameter dataset (or None if canceled)
        """
        newparam = self.imagepanel.get_newparam_from_current(title=title)
        if shape is not None:
            newparam.height, newparam.width = shape
        newparam.hide_height = hide_height
        newparam.hide_width = hide_width
        newparam.hide_type = hide_type
        newparam.hide_dtype = hide_dtype
        if newparam.edit(self.main):
            return newparam
        return None

    def is_registered(self):
        """Return True if plugin is registered"""
        return self._is_registered

    def _reset_registration_state(self) -> None:
        """Reset references and flags associated with plugin registration."""
        self._is_registered = False
        self.main = None
        self.proxy = None

    def register(self, main: main.DLMainWindow) -> None:
        """Register plugin"""
        if self._is_registered:
            return
        PluginRegistry.register_plugin(self)
        self._is_registered = True
        self.main = main
        self.proxy = LocalProxy(main)
        with ExitStack() as rollback_stack:
            rollback_stack.callback(self._reset_registration_state)
            rollback_stack.callback(PluginRegistry.unregister_plugin, self)
            rollback_stack.callback(self.remove_owned_features)
            self.register_hooks()
            rollback_stack.pop_all()

    def unregister(self):
        """Unregister plugin"""
        if not self._is_registered:
            return
        with ExitStack() as cleanup_stack:
            cleanup_stack.callback(self._reset_registration_state)
            cleanup_stack.callback(PluginRegistry.unregister_plugin, self)
            cleanup_stack.callback(self.remove_owned_features)
            self.unregister_hooks()

    def remove_owned_features(self) -> None:
        """Remove this plugin's computing features from available processors."""
        if self.main is None:
            return
        for panel_name in ("signalpanel", "imagepanel"):
            panel = getattr(self.main, panel_name, None)
            processor = getattr(panel, "processor", None)
            if processor is not None:
                processor.remove_features_by_owner(self.plugin_id)

    def register_hooks(self):
        """Register plugin hooks"""

    def unregister_hooks(self):
        """Unregister plugin hooks"""

    def register_computations(self) -> None:
        """Register owned computations after signal and image panels exist."""

    @abc.abstractmethod
    def create_actions(self):
        """Create actions"""


def migrate_enabled_plugin_ids(
    enabled_plugins: list[str] | None,
    plugin_classes: list[type[PluginBase]] | None = None,
) -> list[str] | None:
    """Migrate enabled plugin display names to stable IDs.

    Unknown entries are retained so temporarily unavailable plugins do not lose
    their activation state. A legacy name shared by multiple plugins enables all
    matching IDs because the old setting cannot distinguish between them.

    Args:
        enabled_plugins: Configured stable IDs or legacy display names
        plugin_classes: Classes available for migration (registry by default)

    Returns:
        Enabled stable IDs and retained unknown entries, or None for all plugins
    """
    if enabled_plugins is None:
        return None
    if plugin_classes is None:
        plugin_classes = PluginRegistry.get_plugin_classes()

    known_plugin_ids: set[str] = set()
    plugin_ids_by_name: dict[str, list[str]] = {}
    for plugin_class in plugin_classes:
        plugin_info = plugin_class.PLUGIN_INFO
        if plugin_info is None:
            continue
        try:
            plugin_id = plugin_class.get_plugin_id()
        except ValueError:
            continue
        known_plugin_ids.add(plugin_id)
        plugin_ids_by_name.setdefault(plugin_info.name, []).append(plugin_id)

    migrated_plugins: list[str] = []
    seen_plugins: set[str] = set()
    for configured_plugin in enabled_plugins:
        if configured_plugin in known_plugin_ids:
            resolved_ids = [configured_plugin]
        else:
            resolved_ids = plugin_ids_by_name.get(
                configured_plugin, [configured_plugin]
            )
        for plugin_id in resolved_ids:
            if plugin_id not in seen_plugins:
                migrated_plugins.append(plugin_id)
                seen_plugins.add(plugin_id)
    return migrated_plugins


def _set_plugin_class_filepaths(module) -> None:
    """Attach the module file path to plugin classes defined in that module."""
    filepath = getattr(module, "__file__", None)
    if not filepath:
        return

    filepath = osp.abspath(filepath)
    for plugin_class in PluginRegistry.get_plugin_classes():
        if plugin_class.__module__ == module.__name__:
            plugin_class.__plugin_filepath__ = filepath


def discover_plugins() -> list[type[PluginBase]]:
    """Discover plugins using naming convention

    This function reloads or imports all modules matching the DataLab
    plugin naming scheme (``"{MOD_NAME}_*"``). Plugin classes are then
    registered automatically via the :class:`PluginRegistry` metaclass.

    Import errors for individual plugins are captured and logged so that
    one broken plugin does not prevent the others from loading.  Error
    tracebacks are accumulated in :class:`PluginRegistry` class attributes
    so that callers (e.g. the main window) can replay them into the
    internal console once it is ready.

    Returns:
        List of imported/reloaded plugin modules
    """
    PluginRegistry.clear_discovery_errors()
    PluginRegistry.clear_failed_plugins()

    if not Conf.main.plugins_enabled.get():
        return []

    # Ensure plugin search paths are present in sys.path
    for path in (
        get_user_plugin_paths() + [PLUGINS_DEFAULT_PATH] + OTHER_PLUGINS_PATHLIST
    ):
        rpath = osp.realpath(path)
        if rpath not in sys.path:
            sys.path.append(rpath)

    modules: list[type[PluginBase]] = []
    for finder, name, _ispkg in pkgutil.iter_modules():
        if not name.startswith(f"{MOD_NAME}_"):
            continue
        try:
            # If module is already loaded, reload it so that code changes
            # are taken into account (useful for hot-reload in dev).
            if name in sys.modules:
                module = importlib.reload(sys.modules[name])
            else:
                module = importlib.import_module(name)
            _set_plugin_class_filepaths(module)
            modules.append(module)
        # Plugin discovery imports arbitrary third-party modules. We must catch
        # every failure here so discovery can continue and the error is exposed
        # through the console, log files, and plugin configuration dialog.
        except Exception as e:  # pylint: disable=broad-except
            tb_text = traceback.format_exc()
            print(f"Error loading plugin '{name}': {e}")
            traceback.print_exc()
            # Log to file so it appears in Log Files viewer
            logger = logging.getLogger(__name__)
            logger.error("Error loading plugin '%s'", name, exc_info=True)
            Conf.main.traceback_log_available.set(True)
            # Accumulate for replay in internal console
            PluginRegistry.add_discovery_error(tb_text)
            # Record structured info about the failed plugin
            filepath = ""
            try:
                spec = importlib.util.find_spec(name)
                if spec and spec.origin:
                    filepath = spec.origin
            # Best effort only: failing to resolve the file path must never mask
            # the original plugin import error already captured above.
            except Exception:  # pylint: disable=broad-except
                if hasattr(finder, "path"):
                    filepath = osp.join(finder.path, name)
            PluginRegistry.add_failed_plugin(name, filepath, tb_text)
    return modules


def reload_plugin_modules() -> None:
    """Reload plugin modules and reset plugin classes.

    This helper is intended for hot-reloading plugins at runtime. It:

    - Updates the plugin search path
    - Clears the plugin class registry
    - Reloads or imports all modules matching the plugin naming convention
    """
    if not Conf.main.plugins_enabled.get():
        return

    # Reset class registry before re-executing modules so that plugin
    # classes are rebuilt from freshly executed code.
    PluginRegistry.unregister_all_plugins()

    # Re-discover plugins; discover_plugins will reload modules that
    # are already imported.
    discover_plugins()


def discover_v020_plugins() -> list[tuple[str, str]]:
    """Discover v0.20 plugins (with ``cdl_`` prefix) without importing them

    Returns:
        List of tuples (plugin_name, directory_path) for discovered v0.20 plugins
    """
    v020_plugins = []
    if Conf.main.plugins_enabled.get():
        for path in (
            get_user_plugin_paths() + [PLUGINS_DEFAULT_PATH] + OTHER_PLUGINS_PATHLIST
        ):
            rpath = osp.realpath(path)
            if rpath not in sys.path:
                sys.path.append(rpath)
        for finder, name, _ispkg in pkgutil.iter_modules():
            if name.startswith("cdl_"):
                # Get the directory path from the module finder
                if hasattr(finder, "path"):
                    directory_path = finder.path
                    v020_plugins.append((name, directory_path))
                else:
                    # Fallback if path is not available
                    v020_plugins.append((name, ""))
    return v020_plugins


def get_available_plugins() -> list[PluginBase]:
    """Instantiate and get available plugins

    Returns:
        List of available plugins (as instances)
    """
    # Note: this function is not used by DataLab itself, but it is used by the
    #       test suite to get a list of available plugins
    discover_plugins()
    return [plugin_class() for plugin_class in PluginRegistry.get_plugin_classes()]
