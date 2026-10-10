# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Declarations of plugin-owned tools listed in the Applications catalog.

DataLab Desktop and DataLab-Web share this module.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence

from sigima.objects import ImageObj, SignalObj

from datalab.plugins.recipes import RecipeObjectType
from datalab.plugins.resources import LOCAL_ID_PATTERN, split_package_resource

__all__ = ["PluginTool", "ToolSelection", "tool_accepts_selection"]


class ToolSelection(str, enum.Enum):
    """Selection a tool needs to be opened (names follow DataLab's conditions)."""

    NONE = "none"
    EXACTLY_ONE = "exactly_one"
    AT_LEAST_ONE = "at_least_one"
    AT_LEAST_TWO = "at_least_two"


@dataclasses.dataclass(frozen=True)
class PluginTool:
    """Tool with its own user interface, exposed by an application plugin

    Recipes cover headless analyses run by DataLab; a tool covers any other
    interaction owned by the plugin (wizard, interactive editor, data
    preparation, instrument...). DataLab lists tools in the Applications
    catalog and in the plugin's submenu of the *Plugins* menu.

    A tool is opened either by a ``launcher`` method, called without
    arguments, or by an ``instrument`` method returning a
    :class:`datalab.plugins.instruments.PluginInstrument`, which DataLab shows
    in a window with a live view and the instrument settings.

    Args:
        id: plugin-local tool ID (lowercase letters, digits, ``.``, ``_``, ``-``)
        title: tool title
        launcher: name of the plugin method called, without arguments, to open
         the tool
        description: short description of what the tool does
        icon: ``package:path`` resource (SVG or bitmap) or DataLab icon file
         name; ``None`` falls back to the plugin icon
        instrument: name of the plugin method returning the instrument shown
         by DataLab (exclusive with ``launcher``)
        object_type: type of the objects the tool works on or creates; the
         tool is listed in the menu of that panel (``None``: both panels)
        selection: selection needed to open the tool, counting objects of
         ``object_type`` only
    """

    id: str
    title: str
    launcher: str | None = None
    description: str = ""
    icon: str | None = None
    instrument: str | None = None
    object_type: RecipeObjectType | None = None
    selection: ToolSelection = ToolSelection.NONE

    def __post_init__(self) -> None:
        """Validate identity, opening method, selection, and icon resource."""
        if not isinstance(self.id, str) or not LOCAL_ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                "Plugin tool ID must contain lowercase letters, digits, '.', '_' or '-'"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Plugin tool title must be a non-empty string")
        if (self.launcher is None) == (self.instrument is None):
            raise ValueError("Plugin tool needs either a launcher or an instrument")
        for method_name in (self.launcher, self.instrument):
            if method_name is not None and (
                not isinstance(method_name, str) or not method_name.isidentifier()
            ):
                raise ValueError("Plugin tool launcher must be a method name")
        if not isinstance(self.description, str):
            raise TypeError("Plugin tool description must be a string")
        if self.icon is not None:
            if not isinstance(self.icon, str) or not self.icon.strip():
                raise ValueError("Plugin tool icon must be a non-empty string")
            if ":" in self.icon:
                split_package_resource(self.icon, "Plugin tool icon")
        if self.object_type is not None:
            object.__setattr__(self, "object_type", RecipeObjectType(self.object_type))
        object.__setattr__(self, "selection", ToolSelection(self.selection))


def tool_accepts_selection(
    tool: PluginTool, objects: Sequence[SignalObj | ImageObj]
) -> bool:
    """Return True if the selected objects allow opening the tool.

    Args:
        tool: tool to open
        objects: selected objects

    Returns:
        True if the number of selected objects of the tool's object type
        matches the tool's selection
    """
    object_class = {
        RecipeObjectType.SIGNAL: SignalObj,
        RecipeObjectType.IMAGE: ImageObj,
        None: (SignalObj, ImageObj),
    }[tool.object_type]
    count = sum(isinstance(obj, object_class) for obj in objects)
    return {
        ToolSelection.NONE: True,
        ToolSelection.EXACTLY_ONE: count == 1,
        ToolSelection.AT_LEAST_ONE: count >= 1,
        ToolSelection.AT_LEAST_TWO: count >= 2,
    }[tool.selection]
