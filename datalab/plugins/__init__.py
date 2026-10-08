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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datalab.plugins.base import (
        PLUGIN_ENTRY_POINT_GROUP,
        PLUGINS_DEFAULT_PATH,
        ClassicsImageFormat,
        FailedPluginInfo,
        FormatInfo,
        ImageFormatBase,
        PluginBase,
        PluginBaseMeta,
        PluginCapability,
        PluginInfo,
        PluginRegistry,
        SignalFormatBase,
        discover_plugins,
        discover_v020_plugins,
        format_tool_requirement,
        get_available_plugins,
        migrate_enabled_plugin_ids,
        reload_plugin_modules,
    )

__all__ = [
    "PLUGINS_DEFAULT_PATH",
    "PLUGIN_ENTRY_POINT_GROUP",
    "ClassicsImageFormat",
    "FailedPluginInfo",
    "FormatInfo",
    "ImageFormatBase",
    "PluginBase",
    "PluginBaseMeta",
    "PluginCapability",
    "PluginInfo",
    "PluginRegistry",
    "SignalFormatBase",
    "discover_plugins",
    "discover_v020_plugins",
    "format_tool_requirement",
    "get_available_plugins",
    "migrate_enabled_plugin_ids",
    "reload_plugin_modules",
]


# Importing the Qt plugin host lazily keeps contract submodules Qt-free
def __getattr__(name: str) -> object:
    """Return a public name of the plugin host, importing it on first access."""
    if name in __all__:
        # pylint: disable=import-outside-toplevel
        from datalab.plugins import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes, including lazily imported public names."""
    return sorted({*globals(), *__all__})
