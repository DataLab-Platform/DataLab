# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for plugin welcome page tile declarations."""

from __future__ import annotations

import importlib
import sys
import zipfile
from types import SimpleNamespace

import guidata.dataset as gds
import numpy as np
import pytest
from sigima.objects import create_image, create_signal

from datalab.plugins import (
    PluginBase,
    PluginCapability,
    PluginInfo,
    PluginRegistry,
    format_tool_requirement,
)
from datalab.plugins.instruments import (
    InstrumentAcquisition,
    InstrumentFrame,
    PluginInstrument,
)
from datalab.plugins.resources import resolve_package_resource
from datalab.plugins.tiles import WelcomeTile
from datalab.plugins.tools import PluginTool, ToolSelection, tool_accepts_selection

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
        ({"launcher": None}, "either a launcher or an instrument"),
        ({"instrument": "open_instrument"}, "either a launcher or an instrument"),
        ({"description": None}, "description"),
        ({"icon": ""}, "icon"),
        ({"icon": "datalab:/absolute.svg"}, "relative"),
        ({"object_type": "curve"}, "curve"),
        ({"selection": "some"}, "some"),
    ],
)
def test_plugin_tool_rejects_invalid_declarations(kwargs, message) -> None:
    """Invalid tool declarations fail at import time."""
    values = {"id": "annotate", "title": "Annotate", "launcher": "annotate"}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        PluginTool(**values)


@pytest.mark.parametrize(
    ("selection", "object_type", "counts", "accepted"),
    [
        ("none", "image", (0, 0), True),
        ("exactly_one", "image", (0, 1), True),
        ("exactly_one", "image", (3, 2), False),
        ("at_least_one", "signal", (1, 0), True),
        ("at_least_one", "signal", (0, 3), False),
        ("at_least_two", None, (1, 1), True),
        ("at_least_two", "image", (2, 1), False),
    ],
)
def test_tool_selection_counts_objects_of_the_tool_type(
    selection, object_type, counts, accepted
) -> None:
    """Tools count the selected objects of their own type only."""
    tool = PluginTool(
        id="tool",
        title="Tool",
        launcher="open_tool",
        object_type=object_type,
        selection=selection,
    )
    assert tool.selection is ToolSelection(selection)
    signal_count, image_count = counts
    objects = [
        create_signal("S", np.arange(3.0), np.arange(3.0))
        for _index in range(signal_count)
    ] + [create_image("I", np.zeros((2, 2))) for _index in range(image_count)]
    assert tool_accepts_selection(tool, objects) is accepted


def test_plugin_validates_and_launches_tools() -> None:
    """Application plugins expose validated, plugin-owned tools."""

    class Instrument(PluginInstrument):
        """Instrument returning fixed frames."""

        def preview(self) -> InstrumentFrame:
            """Return a single-point signal."""
            return InstrumentFrame((create_signal("Live", [0.0], [1.0]),))

        def acquire(self) -> InstrumentAcquisition:
            """Return a single-point signal."""
            return InstrumentAcquisition(
                "Acquisition", (create_signal("Shot", [0.0], [1.0]),)
            )

    class ToolPlugin(PluginBase):
        """Application plugin used to check tool contracts."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.tools",
            name="Tool application",
            version="1.0.0",
            icon=PLUGIN_ICON,
            capabilities=(PluginCapability.APPLICATION,),
        )
        TOOLS = (
            PluginTool(id="annotate", title="Annotate", launcher="annotate"),
            PluginTool(
                id="inspect",
                title="Inspect",
                launcher="annotate",
                object_type="image",
                selection="at_least_two",
            ),
            PluginTool(id="generator", title="Generator", instrument="generator"),
        )

        def annotate(self) -> str:
            """Return a marker instead of opening a tool."""
            return "tool opened"

        def generator(self) -> PluginInstrument:
            """Return a new instrument."""
            return Instrument(gds.DataSet())

        def create_actions(self) -> None:
            """Create no actions for this contract test."""

    try:
        tool, inspect, generator = ToolPlugin.get_tools()
        assert tool.icon == PLUGIN_ICON
        assert ToolPlugin.get_tool("generator") == generator
        plugin = ToolPlugin()
        with pytest.raises(RuntimeError, match="registered"):
            plugin.launch_tool("annotate")
        opened: list[tuple[object, PluginTool]] = []
        plugin.main = SimpleNamespace(
            open_plugin_instrument=lambda *args: opened.append(args) or "window"
        )
        plugin.get_selected_objects = lambda: []
        assert plugin.launch_tool("annotate") == "tool opened"
        with pytest.raises(KeyError, match="missing"):
            plugin.launch_tool("missing")
        assert plugin.assess_tool("inspect") == format_tool_requirement(inspect)
        assert plugin.assess_tool("inspect") == "Select at least two images"
        with pytest.raises(ValueError, match="Select at least two images"):
            plugin.launch_tool("inspect")
        images = [create_image("I", np.zeros((2, 2))) for _index in range(2)]
        assert plugin.assess_tool("inspect", images) is None

        assert plugin.launch_tool("generator") == "window"
        assert opened == [(plugin, generator)]
        instrument = plugin.get_instrument("generator")
        assert plugin.get_instrument("generator") is instrument
        with pytest.raises(ValueError, match="no instrument"):
            plugin.get_instrument("annotate")
        plugin._reset_registration_state()  # pylint: disable=protected-access
        assert plugin.get_instrument("generator") is not instrument

        ToolPlugin.generator = lambda self: "not an instrument"
        with pytest.raises(TypeError, match="PluginInstrument"):
            ToolPlugin().get_instrument("generator")

        for tools, error, message in (
            ((tool, tool), ValueError, "Duplicate plugin tool ID"),
            (
                (PluginTool(id="broken", title="Broken", launcher="missing"),),
                ValueError,
                "launcher method 'missing' is not callable",
            ),
            (
                (PluginTool(id="broken", title="Broken", instrument="missing"),),
                ValueError,
                "instrument method 'missing' is not callable",
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


def test_instrument_frames_and_acquisitions_are_validated() -> None:
    """Frames hold signals or one image; acquisitions hold one object type."""
    signal = create_signal("S", [0.0, 1.0], [0.0, 1.0])
    image = create_image("I", np.zeros((2, 2)))
    frame = InstrumentFrame([signal, signal], value_range=(0, 1))
    assert frame.objects == (signal, signal)
    assert frame.value_range == (0.0, 1.0)
    for kwargs, error, message in (
        ({"objects": ()}, ValueError, "at least one"),
        ({"objects": (signal, image)}, TypeError, "signals only or images only"),
        ({"objects": (image, image)}, ValueError, "single image"),
        ({"objects": (signal,), "value_range": (1, 0)}, ValueError, "increasing"),
        ({"objects": (signal,), "summary": None}, TypeError, "summary"),
    ):
        with pytest.raises(error, match=message):
            InstrumentFrame(**kwargs)
    assert InstrumentAcquisition("Frames", [image, image]).objects == (image, image)
    with pytest.raises(ValueError, match="group title"):
        InstrumentAcquisition(" ", (image,))
    with pytest.raises(TypeError, match="DataSet"):
        PluginInstrument.__init__(SimpleNamespace(), settings=object())
