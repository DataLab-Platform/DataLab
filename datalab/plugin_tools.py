# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Declarations of plugin-owned tools listed in the Applications catalog."""

from __future__ import annotations

import dataclasses

from datalab.plugin_resources import LOCAL_ID_PATTERN, split_package_resource

__all__ = ["PluginTool"]


@dataclasses.dataclass(frozen=True)
class PluginTool:
    """Tool with its own user interface, exposed by an application plugin

    Recipes cover headless analyses run by DataLab; a tool covers any other
    interaction owned by the plugin (wizard, interactive editor, data
    preparation...). Tools are a Desktop feature.

    Args:
        id: plugin-local tool ID (lowercase letters, digits, ``.``, ``_``, ``-``)
        title: tool title
        launcher: name of the plugin method called, without arguments, to open
         the tool
        description: short description of what the tool does
        icon: ``package:path`` resource (SVG or bitmap) or DataLab icon file
         name; ``None`` falls back to the plugin icon
    """

    id: str
    title: str
    launcher: str
    description: str = ""
    icon: str | None = None

    def __post_init__(self) -> None:
        """Validate identity, launcher name, and icon resource."""
        if not isinstance(self.id, str) or not LOCAL_ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                "Plugin tool ID must contain lowercase letters, digits, '.', '_' or '-'"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Plugin tool title must be a non-empty string")
        if not isinstance(self.launcher, str) or not self.launcher.isidentifier():
            raise ValueError("Plugin tool launcher must be a method name")
        if not isinstance(self.description, str):
            raise TypeError("Plugin tool description must be a string")
        if self.icon is not None:
            if not isinstance(self.icon, str) or not self.icon.strip():
                raise ValueError("Plugin tool icon must be a non-empty string")
            if ":" in self.icon:
                split_package_resource(self.icon, "Plugin tool icon")
