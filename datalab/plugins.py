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
import enum
import importlib
import importlib.util
import logging
import os
import os.path as osp
import pkgutil
import sys
import traceback
from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack
from importlib import metadata as importlib_metadata
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING
from urllib.parse import urlparse

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
    get_config_path,
    get_user_plugin_paths,
)
from datalab.control.proxy import LocalProxy
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.plugin_examples import PluginExample, PluginExampleData
from datalab.plugin_instruments import PluginInstrument
from datalab.plugin_tiles import WelcomeTile
from datalab.plugin_tools import PluginTool, ToolSelection, tool_accepts_selection
from datalab.recipe_binding import (
    RecipeReadiness,
    assess_recipe_inputs,
    create_recipe_parameters,
    is_compatible,
)
from datalab.recipes import RecipeDescriptor, RecipeOutcome

if TYPE_CHECKING:
    from sigima.objects import ImageObj, NewImageParam, NewSignalParam, SignalObj

    from datalab.gui import main
    from datalab.gui.panel.image import ImagePanel
    from datalab.gui.panel.signal import SignalPanel


PLUGINS_DEFAULT_PATH = get_config_path("plugins")
PLUGIN_ENTRY_POINT_GROUP = "datalab.plugins"

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
    def add_failed_plugin(
        cls, name: str, filepath: str, tb_text: str, source: str = ""
    ) -> None:
        """Record a plugin that failed to load or instantiate.

        Args:
            name: Module or plugin class name
            filepath: File path of the plugin module
            tb_text: Formatted traceback string
            source: Discovery source description
        """
        cls._failed_plugins.append(FailedPluginInfo(name, filepath, tb_text, source))

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

        if Conf.plugins_enabled.get():
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
    source: str = ""


class PluginCapability(str, enum.Enum):
    """Capability exposed by a plugin to DataLab consumers."""

    PROCESSING = "processing"
    IO = "io"
    VISUALIZATION = "visualization"
    APPLICATION = "application"


@dataclasses.dataclass
class PluginInfo:
    """Plugin info"""

    name: str = None
    version: str = "0.0.0"
    description: str = ""
    icon: str = None
    id: str | None = None
    capabilities: Collection[PluginCapability] = dataclasses.field(
        default_factory=frozenset
    )
    documentation_url: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze declared plugin capabilities."""
        if self.id is not None and not self.id.strip():
            raise ValueError("Plugin ID must be None or a non-blank string")
        capabilities = frozenset(self.capabilities)
        if any(
            not isinstance(capability, PluginCapability) for capability in capabilities
        ):
            raise TypeError("Plugin capabilities must be PluginCapability values")
        self.capabilities = capabilities
        if self.documentation_url is not None:
            if not isinstance(self.documentation_url, str):
                raise TypeError("Plugin documentation URL must be a string or None")
            parsed_url = urlparse(self.documentation_url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                raise ValueError("Plugin documentation URL must use HTTP or HTTPS")


def format_tool_requirement(tool: PluginTool) -> str:
    """Return the message telling which objects to select to open a tool."""
    object_type = None if tool.object_type is None else tool.object_type.value
    messages = {
        (ToolSelection.EXACTLY_ONE, "signal"): _("Select one signal"),
        (ToolSelection.EXACTLY_ONE, "image"): _("Select one image"),
        (ToolSelection.EXACTLY_ONE, None): _("Select one object"),
        (ToolSelection.AT_LEAST_ONE, "signal"): _("Select at least one signal"),
        (ToolSelection.AT_LEAST_ONE, "image"): _("Select at least one image"),
        (ToolSelection.AT_LEAST_ONE, None): _("Select at least one object"),
        (ToolSelection.AT_LEAST_TWO, "signal"): _("Select at least two signals"),
        (ToolSelection.AT_LEAST_TWO, "image"): _("Select at least two images"),
        (ToolSelection.AT_LEAST_TWO, None): _("Select at least two objects"),
    }
    return messages.get((tool.selection, object_type), "")


class PluginBaseMeta(PluginRegistry, abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class PluginBase(abc.ABC, metaclass=PluginBaseMeta):
    """Plugin base class"""

    PLUGIN_INFO: PluginInfo = None
    RECIPES: tuple[RecipeDescriptor, ...] = ()
    EXAMPLES: tuple[PluginExample, ...] = ()
    RECIPE_LAUNCHERS: Mapping[str, str] = MappingProxyType({})
    WELCOME_TILES: tuple[WelcomeTile, ...] = ()
    TOOLS: tuple[PluginTool, ...] = ()
    #: Data of the last opened generated example (``None`` for packaged ones)
    last_example_data: PluginExampleData | None = None
    #: UUIDs of the objects of the last opened generated example
    _last_example_uuids: frozenset[str] = frozenset()

    def __init__(self):
        self.main: main.DLMainWindow = None
        self.proxy: LocalProxy = None
        self._is_registered = False
        self._instruments: dict[str, PluginInstrument] = {}
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

    @classmethod
    def get_recipes(cls) -> tuple[RecipeDescriptor, ...]:
        """Return validated recipe descriptors exposed by this plugin."""
        recipes = tuple(cls.RECIPES)
        if not all(isinstance(recipe, RecipeDescriptor) for recipe in recipes):
            raise TypeError("Plugin recipes must be RecipeDescriptor values")
        recipe_ids: set[str] = set()
        plugin_id = cls.get_plugin_id()
        for recipe in recipes:
            if recipe.plugin_id != plugin_id:
                raise ValueError(
                    f"Recipe {recipe.recipe_id!r} is not owned by plugin {plugin_id!r}"
                )
            if recipe.plugin_version != cls.PLUGIN_INFO.version:
                raise ValueError(
                    f"Recipe {recipe.recipe_id!r} plugin version does not match "
                    f"plugin {plugin_id!r} version {cls.PLUGIN_INFO.version!r}"
                )
            if recipe.recipe_id in recipe_ids:
                raise ValueError(f"Duplicate plugin recipe ID: {recipe.recipe_id!r}")
            recipe_ids.add(recipe.recipe_id)
        return recipes

    @classmethod
    def get_recipe(cls, recipe_id: str) -> RecipeDescriptor:
        """Return one recipe descriptor by its namespaced ID."""
        for recipe in cls.get_recipes():
            if recipe.recipe_id == recipe_id:
                return recipe
        raise KeyError(f"Plugin recipe {recipe_id!r} not found")

    @classmethod
    def get_recipe_launchers(cls) -> Mapping[str, str]:
        """Return validated recipe-to-method bindings overriding the Desktop UI."""
        if not isinstance(cls.RECIPE_LAUNCHERS, Mapping):
            raise TypeError("Plugin recipe launchers must be a mapping")
        launchers = dict(cls.RECIPE_LAUNCHERS)
        recipe_ids = {recipe.recipe_id for recipe in cls.get_recipes()}
        for recipe_id, method_name in launchers.items():
            if recipe_id not in recipe_ids:
                raise ValueError(
                    f"Recipe launcher references unknown recipe {recipe_id!r}"
                )
            if not isinstance(method_name, str) or not method_name.strip():
                raise TypeError("Plugin recipe launcher names must be strings")
            if not callable(getattr(cls, method_name, None)):
                raise ValueError(
                    f"Recipe launcher method {method_name!r} is not callable"
                )
        return MappingProxyType(launchers)

    def launch_recipe(self, recipe_id: str) -> RecipeOutcome | None:
        """Launch a recipe on the current selection.

        A plugin method declared in ``RECIPE_LAUNCHERS`` replaces the generic
        interaction of :meth:`start_recipe`.
        """
        if self.main is None:
            raise RuntimeError("Plugin must be registered before launching a recipe")
        self.get_recipe(recipe_id)
        method_name = self.get_recipe_launchers().get(recipe_id)
        if method_name is None:
            return self.start_recipe(recipe_id)
        outcome = getattr(self, method_name)()
        if outcome is not None and not isinstance(outcome, RecipeOutcome):
            raise TypeError("Plugin recipe launchers must return RecipeOutcome or None")
        return outcome

    def get_selected_objects(self) -> list[SignalObj | ImageObj]:
        """Return the objects selected in the current signal or image panel."""
        if self.main is None:
            raise RuntimeError("Plugin must be registered to access the selection")
        panel = {"signal": self.signalpanel, "image": self.imagepanel}.get(
            self.main.get_current_panel()
        )
        if panel is None:
            return []
        return list(panel.objview.get_sel_objects(include_groups=True))

    def example_parameter_values(
        self,
        recipe_id: str,
        objects: Sequence[SignalObj | ImageObj],
    ) -> Mapping[str, object]:
        """Return the last example's parameter values for a recipe and objects.

        Values are returned only when all objects come from the last opened
        generated example: they are suited to that data only.
        """
        data = self.last_example_data
        if data is None or not objects:
            return MappingProxyType({})
        if any(get_uuid(obj) not in self._last_example_uuids for obj in objects):
            return MappingProxyType({})
        return data.values_for(recipe_id)

    def assess_recipe(
        self,
        recipe_id: str,
        objects: Sequence[SignalObj | ImageObj] | None = None,
    ) -> RecipeReadiness:
        """Assess whether a recipe can run on objects (default: selection)."""
        recipe = self.get_recipe(recipe_id)
        if objects is None:
            objects = self.get_selected_objects()
        parameters = create_recipe_parameters(
            recipe, self.example_parameter_values(recipe_id, objects)
        )
        return assess_recipe_inputs(recipe, objects, parameters)

    def start_recipe(
        self,
        recipe_id: str,
        objects: Sequence[SignalObj | ImageObj] | None = None,
        parameter_values: Mapping[str, object] | None = None,
    ) -> RecipeOutcome | None:
        """Run a recipe through the generic DataLab interaction.

        DataLab assigns the objects (default: selection) to the recipe inputs,
        asking the user when needed, checks them, edits the parameters, then
        runs the recipe.

        Args:
            recipe_id: namespaced recipe ID
            objects: candidate objects (default: current selection)
            parameter_values: initial parameter values (default: those of the
             last opened example, when the objects come from it)

        Returns:
            Recipe outcome, or None if cancelled or failed
        """
        if self.main is None:
            raise RuntimeError("Plugin must be registered before starting a recipe")
        # pylint: disable=import-outside-toplevel
        from datalab.gui.recipe_launcher import RecipeLauncher

        return RecipeLauncher(self).start(recipe_id, objects, parameter_values)

    def try_example(self, example_id: str, recipe_id: str) -> RecipeOutcome | None:
        """Open an example, then start one of the recipes it was designed for."""
        if self.main is None:
            raise RuntimeError("Plugin must be registered before trying an example")
        example = self.get_example(example_id)
        if recipe_id not in example.recipe_ids:
            raise ValueError(
                f"Plugin example {example_id!r} is not designed for recipe "
                f"{recipe_id!r}"
            )
        recipe = self.get_recipe(recipe_id)
        if self.launch_example(example_id) is None:
            return None
        objects = [
            obj
            for panel in (self.signalpanel, self.imagepanel)
            for obj in panel.objmodel.get_all_objects()
            if any(is_compatible(slot, obj) for slot in recipe.inputs)
        ]
        return self.start_recipe(
            recipe_id,
            objects,
            self.example_parameter_values(recipe_id, objects),
        )

    @classmethod
    def get_tools(cls) -> tuple[PluginTool, ...]:
        """Return validated tools listed in the Applications catalog."""
        tools = tuple(cls.TOOLS)
        if not tools:
            return ()
        info = cls.PLUGIN_INFO
        if PluginCapability.APPLICATION not in info.capabilities:
            raise ValueError("Plugin tools require the APPLICATION capability")
        if not all(isinstance(tool, PluginTool) for tool in tools):
            raise TypeError("Plugin tools must be PluginTool values")
        tool_ids: set[str] = set()
        for tool in tools:
            if tool.id in tool_ids:
                raise ValueError(f"Duplicate plugin tool ID: {tool.id!r}")
            method_name = tool.launcher or tool.instrument
            if not callable(getattr(cls, method_name, None)):
                kind = "launcher" if tool.launcher else "instrument"
                raise ValueError(
                    f"Plugin tool {kind} method {method_name!r} is not callable"
                )
            tool_ids.add(tool.id)
        return tuple(
            dataclasses.replace(tool, icon=info.icon)
            if tool.icon is None and info.icon is not None
            else tool
            for tool in tools
        )

    @classmethod
    def get_tool(cls, tool_id: str) -> PluginTool:
        """Return one tool by its plugin-local ID."""
        for tool in cls.get_tools():
            if tool.id == tool_id:
                return tool
        raise KeyError(f"Plugin tool {tool_id!r} not found")

    def assess_tool(
        self,
        tool_id: str,
        objects: Sequence[SignalObj | ImageObj] | None = None,
    ) -> str | None:
        """Return why a tool cannot be opened on objects (default: selection).

        Returns:
            Message telling which objects to select, or None if the tool
            can be opened
        """
        tool = self.get_tool(tool_id)
        if tool.selection is ToolSelection.NONE:
            return None
        if objects is None:
            objects = self.get_selected_objects()
        if tool_accepts_selection(tool, objects):
            return None
        return format_tool_requirement(tool)

    def launch_tool(self, tool_id: str) -> object:
        """Open a plugin tool: call its launcher, or show its instrument.

        Raises:
            ValueError: if the selection does not allow opening the tool
        """
        if self.main is None:
            raise RuntimeError("Plugin must be registered before launching a tool")
        tool = self.get_tool(tool_id)
        issue = self.assess_tool(tool_id)
        if issue is not None:
            raise ValueError(issue)
        if tool.launcher is not None:
            return getattr(self, tool.launcher)()
        return self.main.open_plugin_instrument(self, tool)

    def get_instrument(self, tool_id: str) -> PluginInstrument:
        """Return the instrument of a tool, created once per registration."""
        tool = self.get_tool(tool_id)
        if tool.instrument is None:
            raise ValueError(f"Plugin tool {tool_id!r} has no instrument")
        instruments = self._instruments
        if tool_id not in instruments:
            instrument = getattr(self, tool.instrument)()
            if not isinstance(instrument, PluginInstrument):
                raise TypeError(
                    f"Plugin tool instrument method {tool.instrument!r} must "
                    "return a PluginInstrument"
                )
            instruments[tool_id] = instrument
        return instruments[tool_id]

    @classmethod
    def get_examples(cls) -> tuple[PluginExample, ...]:
        """Return validated packaged examples exposed by this plugin."""
        examples = tuple(cls.EXAMPLES)
        if not all(isinstance(example, PluginExample) for example in examples):
            raise TypeError("Plugin examples must be PluginExample values")
        example_ids: set[str] = set()
        recipe_ids = {recipe.recipe_id for recipe in cls.get_recipes()}
        for example in examples:
            if example.id in example_ids:
                raise ValueError(f"Duplicate plugin example ID: {example.id!r}")
            for recipe_id in example.recipe_ids:
                if recipe_id not in recipe_ids:
                    raise ValueError(
                        f"Plugin example {example.id!r} references unknown recipe "
                        f"{recipe_id!r}"
                    )
            example_ids.add(example.id)
        return examples

    @classmethod
    def get_example(cls, example_id: str) -> PluginExample:
        """Return one packaged example by its plugin-local ID."""
        for example in cls.get_examples():
            if example.id == example_id:
                return example
        raise KeyError(f"Plugin example {example_id!r} not found")

    @classmethod
    def materialize_example(cls, example_id: str) -> PluginExampleData | None:
        """Generate one example in memory, or defer to its package resource."""
        cls.get_example(example_id)
        return None

    def launch_example(self, example_id: str) -> PluginExample | None:
        """Confirm and open a generated or packaged example, then select it."""
        if self.main is None:
            raise RuntimeError("Plugin must be registered before launching an example")
        self.get_example(example_id)
        if not self.main.confirm_memory_state():
            return None
        if any(len(panel) for panel in (self.signalpanel, self.imagepanel)) and not (
            self.ask_yesno(
                _("Opening this example replaces the current workspace. Continue?"),
                title=_("Open example"),
            )
        ):
            return None
        example = self.open_example(example_id, reset_all=True)
        panel = {"signal": self.signalpanel, "image": self.imagepanel}.get(
            self.main.get_current_panel()
        )
        if panel is not None:
            panel.objview.select_objects(panel.objmodel.get_all_objects())
        return example

    def open_example(
        self,
        example_id: str,
        reset_all: bool = True,
    ) -> PluginExample:
        """Open a generated or packaged HDF5 example in the Desktop workspace.

        Generated examples (``materialize_example()`` returning data) are
        loaded directly into the signal/image panels; packaged examples are
        opened through their native HDF5 resource. The materialized data is
        kept in :attr:`last_example_data`, so that recipes started on these
        objects reuse the example's parameter values
        (see :meth:`example_parameter_values`).
        """
        if self.main is None:
            raise RuntimeError("Plugin must be registered before opening an example")
        example = self.get_example(example_id)
        data = self.materialize_example(example_id)
        if data is not None:
            unknown = set(data.parameter_values).difference(example.recipe_ids)
            if unknown:
                raise ValueError(
                    f"Plugin example {example_id!r} provides parameters for "
                    f"recipes it is not designed for: {', '.join(sorted(unknown))}"
                )
        self.last_example_data = data
        self._last_example_uuids = (
            frozenset()
            if data is None
            else frozenset(get_uuid(obj) for obj in data.objects)
        )
        if data is not None:
            from sigima.objects import SignalObj

            if reset_all:
                self.main.reset_all()
            modified_before = self.main.is_modified()
            current_panel_before = self.main.get_current_panel()
            objects_by_panel: dict[
                SignalPanel | ImagePanel, list[SignalObj | ImageObj]
            ] = {}
            for obj in data.objects:
                panel = (
                    self.signalpanel if isinstance(obj, SignalObj) else self.imagepanel
                )
                objects_by_panel.setdefault(panel, []).append(obj)
            groups_before = {
                panel: {get_uuid(group) for group in panel.objmodel.get_groups()}
                for panel in objects_by_panel
            }
            committed_batches: list[
                tuple[
                    SignalPanel | ImagePanel,
                    tuple[SignalObj | ImageObj, ...],
                    tuple[object, ...],
                ]
            ] = []
            # Suspend plot refresh: generated campaigns may hold hundreds
            # of objects.
            with self.main.context_no_refresh():
                try:
                    for panel, objects in objects_by_panel.items():
                        batch = tuple(objects)
                        panel._add_objects(batch)  # pylint: disable=protected-access
                        new_groups = tuple(
                            group
                            for group in panel.objmodel.get_groups()
                            if get_uuid(group) not in groups_before[panel]
                        )
                        committed_batches.append((panel, batch, new_groups))
                except Exception:
                    try:
                        with ExitStack() as rollback:
                            for panel, objects, new_groups in committed_batches:
                                rollback.callback(panel.SIG_OBJECT_REMOVED.emit)
                                rollback.callback(panel.objview.update_tree)
                                for group in new_groups:
                                    rollback.callback(
                                        panel.objmodel.remove_group,
                                        group,
                                    )
                                    rollback.callback(
                                        panel.objview.remove_item,
                                        get_uuid(group),
                                        False,
                                    )
                                for obj in objects:
                                    rollback.callback(
                                        panel._remove_added_object,  # pylint: disable=protected-access
                                        obj,
                                    )
                    finally:
                        self.main.set_modified(modified_before)
                        self.main.set_current_panel(current_panel_before)
                    raise
            self.main.set_current_panel(
                "signal" if isinstance(data.objects[0], SignalObj) else "image"
            )
            return example
        with example.as_file() as filename:
            self.main.load_h5_workspace(
                [os.fspath(filename)],
                reset_all=reset_all,
            )
        return example

    @classmethod
    def get_welcome_tiles(cls) -> tuple[WelcomeTile, ...]:
        """Return validated welcome page tiles exposed by this application plugin.

        Without declared ``WELCOME_TILES``, an application plugin exposes one
        default tile built from its :class:`PluginInfo`. Declared tiles without
        icon inherit the plugin icon.
        """
        tiles = tuple(cls.WELCOME_TILES)
        info = cls.PLUGIN_INFO
        if PluginCapability.APPLICATION not in info.capabilities:
            if tiles:
                raise ValueError("Welcome tiles require the APPLICATION capability")
            return ()
        if not tiles:
            return (
                WelcomeTile(
                    id="application",
                    title=info.name,
                    description=info.description,
                    icon=info.icon,
                ),
            )
        if not all(isinstance(tile, WelcomeTile) for tile in tiles):
            raise TypeError("Plugin welcome tiles must be WelcomeTile values")
        tile_ids: set[str] = set()
        for tile in tiles:
            if tile.id in tile_ids:
                raise ValueError(f"Duplicate welcome tile ID: {tile.id!r}")
            if tile.launcher is not None and not callable(
                getattr(cls, tile.launcher, None)
            ):
                raise ValueError(
                    f"Welcome tile launcher method {tile.launcher!r} is not callable"
                )
            tile_ids.add(tile.id)
        return tuple(
            dataclasses.replace(tile, icon=info.icon)
            if tile.icon is None and info.icon is not None
            else tile
            for tile in tiles
        )

    def launch_welcome_tile(self, tile_id: str) -> object:
        """Launch a welcome page tile.

        A tile without launcher opens the plugin page of the Applications
        catalog; otherwise, its plugin method is called without arguments.

        Returns:
            Value returned by the launcher method (None for the catalog page)
        """
        if self.main is None:
            raise RuntimeError(
                "Plugin must be registered before launching a welcome tile"
            )
        tile = next(
            (tile for tile in self.get_welcome_tiles() if tile.id == tile_id), None
        )
        if tile is None:
            raise KeyError(f"Welcome tile {tile_id!r} not found")
        if tile.launcher is None:
            self.main.show_applications(self.plugin_id)
            return None
        return getattr(self, tile.launcher)()

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
        self._instruments.clear()

    def register(self, main: main.DLMainWindow) -> None:
        """Register plugin"""
        if self._is_registered:
            return
        PluginRegistry.register_plugin(self)
        self._is_registered = True
        self.main = main
        self.proxy = LocalProxy(main, input_source="plugin")
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
        """Register plugin hooks.

        Called by :meth:`register` during application startup or plugin
        reload, before the panels' PLUGINS action category exists: use this
        for non-GUI side effects only (I/O format registration, listeners).
        ``self.main`` and ``self.proxy`` are already set. Raising here rolls
        back the whole registration (the plugin stays unregistered).
        """

    def unregister_hooks(self):
        """Unregister plugin hooks.

        Called by :meth:`unregister`; must undo :meth:`register_hooks`.
        Owned computing features are removed automatically afterwards.
        """

    def register_computations(self) -> None:
        """Register owned computations after signal and image panels exist.

        Called once per registered plugin, right before
        :meth:`create_actions`; raising here removes the plugin's owned
        features but keeps the plugin registered.
        """

    @abc.abstractmethod
    def create_actions(self):
        """Create actions.

        Called immediately after :meth:`register_computations`, inside the
        panels' PLUGINS action category: actions created here populate the
        *Plugins* menu.
        """


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


def _add_plugin_discovery_source(plugin_class: type[PluginBase], source: str) -> None:
    """Attach a discovery source to a plugin class without duplicates."""
    sources = getattr(plugin_class, "__plugin_discovery_sources__", ())
    plugin_class.__plugin_discovery_sources__ = tuple(dict.fromkeys((*sources, source)))


def _get_plugin_entry_points() -> list[importlib_metadata.EntryPoint]:
    """Return installed DataLab plugin entry points in deterministic order."""
    entry_points = importlib_metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=PLUGIN_ENTRY_POINT_GROUP)
    else:  # Python 3.9 compatibility
        selected = entry_points.get(PLUGIN_ENTRY_POINT_GROUP, ())
    return sorted(
        selected, key=lambda entry_point: (entry_point.name, entry_point.value)
    )


def _record_plugin_discovery_failure(
    name: str,
    source: str,
    tb_text: str,
    filepath: str = "",
) -> None:
    """Record and report an isolated plugin discovery failure."""
    print(f"Error loading plugin {name!r} from {source}")
    print(tb_text, file=sys.stderr)
    logging.getLogger(__name__).error(
        "Error loading plugin %r from %s\n%s", name, source, tb_text
    )
    Conf.traceback_log_available.set(True)
    PluginRegistry.add_discovery_error(tb_text)
    PluginRegistry.add_failed_plugin(name, filepath, tb_text, source)


def _discover_entry_point_plugins() -> list[ModuleType]:
    """Load and register plugin classes declared through package entry points."""
    try:
        entry_points = _get_plugin_entry_points()
    # Installed distribution metadata is external input. A malformed package
    # must not prevent convention-based plugins from being discovered.
    except Exception:  # pylint: disable=broad-except
        source = f"entry point group {PLUGIN_ENTRY_POINT_GROUP!r}"
        _record_plugin_discovery_failure(
            PLUGIN_ENTRY_POINT_GROUP, source, traceback.format_exc()
        )
        return []

    discovered_modules: list[ModuleType] = []
    reloadable_modules = set(sys.modules)
    reloaded_modules: set[str] = set()
    for entry_point in entry_points:
        source = f"entry point {entry_point.name!r} ({entry_point.value})"
        plugin_classes = PluginRegistry.get_plugin_classes()
        previous_classes = list(plugin_classes)
        try:
            try:
                module_name = getattr(entry_point, "module", None)
                if (
                    module_name in reloadable_modules
                    and module_name not in reloaded_modules
                ):
                    importlib.reload(sys.modules[module_name])
                    reloaded_modules.add(module_name)
                plugin_class = entry_point.load()
            finally:
                # Importing the target module may invoke PluginBaseMeta. The entry
                # point contract contributes only its explicit target class.
                plugin_classes[:] = previous_classes

            if not isinstance(plugin_class, type) or not issubclass(
                plugin_class, PluginBase
            ):
                raise TypeError(
                    f"DataLab plugin {source} must resolve to a PluginBase subclass"
                )
        # Entry points execute arbitrary third-party imports. Isolating failures
        # here lets convention-based and other installed plugins keep loading.
        except Exception:  # pylint: disable=broad-except
            _record_plugin_discovery_failure(
                entry_point.name, source, traceback.format_exc()
            )
            continue

        if plugin_class not in plugin_classes:
            plugin_classes.append(plugin_class)
        _add_plugin_discovery_source(plugin_class, source)
        module = sys.modules.get(plugin_class.__module__)
        if module is not None:
            _set_plugin_class_filepaths(module)
            if module not in discovered_modules:
                discovered_modules.append(module)
    return discovered_modules


def _normalize_discovered_plugin_classes() -> None:
    """Merge identical targets and reject stable plugin ID collisions."""
    plugin_classes = PluginRegistry.get_plugin_classes()
    unique_classes: list[type[PluginBase]] = []
    target_positions: dict[str, int] = {}
    for plugin_class in plugin_classes:
        target = f"{plugin_class.__module__}:{plugin_class.__qualname__}"
        if target not in target_positions:
            target_positions[target] = len(unique_classes)
            unique_classes.append(plugin_class)
            continue

        position = target_positions[target]
        previous_class = unique_classes[position]
        previous_sources = getattr(previous_class, "__plugin_discovery_sources__", ())
        current_sources = getattr(plugin_class, "__plugin_discovery_sources__", ())
        plugin_class.__plugin_discovery_sources__ = tuple(
            dict.fromkeys((*previous_sources, *current_sources))
        )
        unique_classes[position] = plugin_class

    classes_by_id: dict[str, list[type[PluginBase]]] = {}
    invalid_classes: set[type[PluginBase]] = set()
    for plugin_class in unique_classes:
        try:
            plugin_id = plugin_class.get_plugin_id()
        except ValueError as error:
            invalid_classes.add(plugin_class)
            tb_text = "".join(traceback.format_exception_only(type(error), error))
            PluginRegistry.add_discovery_error(tb_text)
            logging.getLogger(__name__).error(tb_text.rstrip())
            Conf.traceback_log_available.set(True)
            sources = getattr(plugin_class, "__plugin_discovery_sources__", ())
            PluginRegistry.add_failed_plugin(
                plugin_class.__name__,
                getattr(plugin_class, "__plugin_filepath__", ""),
                tb_text,
                ", ".join(sources),
            )
            continue
        classes_by_id.setdefault(plugin_id, []).append(plugin_class)

    conflicting_classes: set[type[PluginBase]] = set()
    for plugin_id, classes in classes_by_id.items():
        if len(classes) < 2:
            continue
        conflicting_classes.update(classes)
        descriptions = []
        for plugin_class in classes:
            target = f"{plugin_class.__module__}:{plugin_class.__qualname__}"
            sources = getattr(plugin_class, "__plugin_discovery_sources__", ())
            descriptions.append(f"{target} from {', '.join(sources)}")
        error = ValueError(
            f"Plugin ID collision for {plugin_id!r}: {'; '.join(descriptions)}"
        )
        tb_text = "".join(traceback.format_exception_only(type(error), error))
        PluginRegistry.add_discovery_error(tb_text)
        logging.getLogger(__name__).error(tb_text.rstrip())
        Conf.traceback_log_available.set(True)
        for plugin_class in classes:
            sources = getattr(plugin_class, "__plugin_discovery_sources__", ())
            PluginRegistry.add_failed_plugin(
                plugin_class.__name__,
                getattr(plugin_class, "__plugin_filepath__", ""),
                tb_text,
                ", ".join(sources),
            )

    plugin_classes[:] = [
        plugin_class
        for plugin_class in unique_classes
        if plugin_class not in conflicting_classes
        and plugin_class not in invalid_classes
    ]


def discover_plugins() -> list[ModuleType]:
    """Discover plugins through package entry points and naming convention.

    Installed packages may expose a :class:`PluginBase` subclass through the
    ``datalab.plugins`` entry-point group. This function also reloads or imports
    modules matching the historical ``"{MOD_NAME}_*"`` naming scheme.

    Import errors for individual plugins are captured and logged so that
    one broken plugin does not prevent the others from loading.  Error
    tracebacks are accumulated in :class:`PluginRegistry` class attributes
    so that callers (e.g. the main window) can replay them into the
    internal console once it is ready.

    This function mutates ``sys.modules`` and the plugin registry without
    synchronization: call it from the Qt main thread only.

    Returns:
        Imported/reloaded modules containing discovered plugins
    """
    PluginRegistry.clear_discovery_errors()
    PluginRegistry.clear_failed_plugins()

    if not Conf.plugins_enabled.get():
        return []

    # Ensure plugin search paths are present in sys.path
    for path in (
        get_user_plugin_paths() + [PLUGINS_DEFAULT_PATH] + OTHER_PLUGINS_PATHLIST
    ):
        rpath = osp.realpath(path)
        if rpath not in sys.path:
            sys.path.append(rpath)

    modules = _discover_entry_point_plugins()
    for finder, name, _ispkg in pkgutil.iter_modules():
        if not name.startswith(f"{MOD_NAME}_"):
            continue
        try:
            previous_classes = list(PluginRegistry.get_plugin_classes())
            # If module is already loaded, reload it so that code changes
            # are taken into account (useful for hot-reload in dev).
            if name in sys.modules:
                module = importlib.reload(sys.modules[name])
            else:
                module = importlib.import_module(name)
            source = f"module convention {name!r}"
            for plugin_class in PluginRegistry.get_plugin_classes():
                if plugin_class not in previous_classes:
                    _add_plugin_discovery_source(plugin_class, source)
            _set_plugin_class_filepaths(module)
            if module not in modules:
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
            Conf.traceback_log_available.set(True)
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
            PluginRegistry.add_failed_plugin(
                name, filepath, tb_text, f"module convention {name!r}"
            )
    _normalize_discovered_plugin_classes()
    return modules


def reload_plugin_modules() -> None:
    """Reload plugin modules and reset plugin classes.

    This helper is intended for hot-reloading plugins at runtime. It:

    - Updates the plugin search path
    - Clears the plugin class registry
    - Reloads or imports all modules matching the plugin naming convention

    Like :func:`discover_plugins`, this is not thread-safe: call it from
    the Qt main thread only.
    """
    if not Conf.plugins_enabled.get():
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
    if Conf.plugins_enabled.get():
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
