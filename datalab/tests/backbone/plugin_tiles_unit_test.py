# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for plugin welcome page tile declarations."""

from __future__ import annotations

import importlib
import sys
import zipfile
from types import SimpleNamespace

import pytest

from datalab.plugin_resources import resolve_package_resource
from datalab.plugin_tiles import WelcomeTile
from datalab.plugin_tools import PluginTool
from datalab.plugins import PluginBase, PluginCapability, PluginInfo, PluginRegistry

PLUGIN_ICON = "datalab:data/icons/libre-gui-plugin.svg"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": "Invalid ID"}, "tile ID"),
        ({"title": " "}, "title"),
        ({"description": None}, "description"),
        ({"icon": ""}, "icon"),
        ({"icon": "datalab:icons:logo.svg"}, "package:path"),
        ({"icon": "datalab:/absolute.svg"}, "relative"),
        ({"icon": "datalab:icons/../logo.svg"}, "relative"),
        ({"icon": "invalid-package:logo.svg"}, "package is invalid"),
        ({"launcher": "open quickstart"}, "method name"),
    ],
)
def test_welcome_tile_rejects_invalid_declarations(kwargs, message) -> None:
    """Invalid tile declarations fail before any resource resolution."""
    values = {"id": "quickstart", "title": "Quick start"}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        WelcomeTile(**values)


def test_package_resource_resolves_from_zip_package(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tile icons resolve from zipped packages, and missing ones are reported."""
    archive_path = tmp_path / "tile-plugin.zip"
    package_name = "zipped_plugin_tiles"
    payload = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{package_name}/__init__.py", "")
        archive.writestr(f"{package_name}/icons/logo.svg", payload)
    monkeypatch.syspath_prepend(str(archive_path))
    importlib.invalidate_caches()
    tile = WelcomeTile(
        id="application",
        title="Application",
        icon=f"{package_name}:icons/logo.svg",
    )

    try:
        assert tile.is_package_icon
        assert resolve_package_resource(tile.icon).read_bytes() == payload
        with pytest.raises(FileNotFoundError, match="Welcome tile icon not found"):
            resolve_package_resource(
                f"{package_name}:icons/missing.svg", "Welcome tile icon"
            )
    finally:
        sys.modules.pop(package_name, None)
    assert not WelcomeTile(id="named", title="Named", icon="logo.svg").is_package_icon


def test_plugin_validates_and_launches_welcome_tiles() -> None:
    """Application plugins expose a default tile or validated declared tiles."""

    class TilePlugin(PluginBase):
        """Application plugin used to check welcome tile contracts."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.tiles",
            name="Tile application",
            version="1.0.0",
            description="Tile application description",
            icon=PLUGIN_ICON,
            capabilities=(PluginCapability.APPLICATION,),
        )

        def open_quickstart(self) -> str:
            """Return a marker instead of opening an example."""
            return "quickstart opened"

        def create_actions(self) -> None:
            """Create no actions for this contract test."""

    try:
        assert TilePlugin.get_welcome_tiles() == (
            WelcomeTile(
                id="application",
                title="Tile application",
                description="Tile application description",
                icon=PLUGIN_ICON,
            ),
        )

        quickstart = WelcomeTile(
            id="quickstart",
            title="Quick start",
            icon="libre-gui-about.svg",
            launcher="open_quickstart",
        )
        TilePlugin.WELCOME_TILES = (
            WelcomeTile(id="application", title="Tile application"),
            quickstart,
        )
        application, declared_quickstart = TilePlugin.get_welcome_tiles()
        assert application.icon == PLUGIN_ICON
        assert declared_quickstart is quickstart

        plugin = TilePlugin()
        with pytest.raises(RuntimeError, match="registered"):
            plugin.launch_welcome_tile("application")
        shown_pages: list[str] = []
        plugin.main = SimpleNamespace(show_applications=shown_pages.append)
        assert plugin.launch_welcome_tile("application") is None
        assert shown_pages == ["org.example.tiles"]
        assert plugin.launch_welcome_tile("quickstart") == "quickstart opened"
        with pytest.raises(KeyError, match="missing"):
            plugin.launch_welcome_tile("missing")

        for tiles, error, message in (
            ((quickstart, quickstart), ValueError, "Duplicate welcome tile ID"),
            (
                (WelcomeTile(id="broken", title="Broken", launcher="missing"),),
                ValueError,
                "not callable",
            ),
            (("not a tile",), TypeError, "WelcomeTile"),
        ):
            TilePlugin.WELCOME_TILES = tiles
            with pytest.raises(error, match=message):
                TilePlugin.get_welcome_tiles()

        TilePlugin.PLUGIN_INFO = PluginInfo(
            id="org.example.tiles",
            name="Processing only",
            capabilities=(PluginCapability.PROCESSING,),
        )
        with pytest.raises(ValueError, match="APPLICATION capability"):
            TilePlugin.get_welcome_tiles()
        TilePlugin.WELCOME_TILES = ()
        assert TilePlugin.get_welcome_tiles() == ()
    finally:
        PluginRegistry.get_plugin_classes().remove(TilePlugin)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": "Invalid ID"}, "tool ID"),
        ({"title": " "}, "title"),
        ({"launcher": "open tool"}, "method name"),
        ({"description": None}, "description"),
        ({"icon": ""}, "icon"),
        ({"icon": "datalab:/absolute.svg"}, "relative"),
    ],
)
def test_plugin_tool_rejects_invalid_declarations(kwargs, message) -> None:
    """Invalid tool declarations fail at import time."""
    values = {"id": "annotate", "title": "Annotate", "launcher": "annotate"}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        PluginTool(**values)


def test_plugin_validates_and_launches_tools() -> None:
    """Application plugins expose validated, plugin-owned tools."""

    class ToolPlugin(PluginBase):
        """Application plugin used to check tool contracts."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.tools",
            name="Tool application",
            version="1.0.0",
            icon=PLUGIN_ICON,
            capabilities=(PluginCapability.APPLICATION,),
        )
        TOOLS = (PluginTool(id="annotate", title="Annotate", launcher="annotate"),)

        def annotate(self) -> str:
            """Return a marker instead of opening a tool."""
            return "tool opened"

        def create_actions(self) -> None:
            """Create no actions for this contract test."""

    try:
        (tool,) = ToolPlugin.get_tools()
        assert tool.icon == PLUGIN_ICON
        plugin = ToolPlugin()
        with pytest.raises(RuntimeError, match="registered"):
            plugin.launch_tool("annotate")
        plugin.main = SimpleNamespace()
        assert plugin.launch_tool("annotate") == "tool opened"
        with pytest.raises(KeyError, match="missing"):
            plugin.launch_tool("missing")

        for tools, error, message in (
            ((tool, tool), ValueError, "Duplicate plugin tool ID"),
            (
                (PluginTool(id="broken", title="Broken", launcher="missing"),),
                ValueError,
                "not callable",
            ),
            (("not a tool",), TypeError, "PluginTool"),
        ):
            ToolPlugin.TOOLS = tools
            with pytest.raises(error, match=message):
                ToolPlugin.get_tools()

        ToolPlugin.TOOLS = (tool,)
        ToolPlugin.PLUGIN_INFO = PluginInfo(
            id="org.example.tools",
            name="Processing only",
            capabilities=(PluginCapability.PROCESSING,),
        )
        with pytest.raises(ValueError, match="APPLICATION capability"):
            ToolPlugin.get_tools()
    finally:
        PluginRegistry.get_plugin_classes().remove(ToolPlugin)
