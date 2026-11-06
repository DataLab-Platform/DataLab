# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Static metadata inspection never imports plugin code."""

from __future__ import annotations

from pathlib import Path

import pytest

from datalab.plugins import modules
from datalab.plugins.modules import (
    ModuleInspectionError,
    apply_catalog_metadata,
    inspect_module,
)

SOURCE = b"""from datalab.plugins import PluginBase, PluginInfo
PLUGIN_ID = "org.example.module"
PLUGIN_NAME = "An example"
class Example(PluginBase):
    PLUGIN_INFO = PluginInfo(id=PLUGIN_ID, name=PLUGIN_NAME, version="1.2")
    def create_actions(self):
        pass
"""
FILENAME = "datalab_module_probe.py"


def test_literal_metadata_and_no_execution(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    source = SOURCE + f"\nopen({str(marker)!r}, 'w').write('executed')\n".encode()
    manifest = inspect_module(source, filename=FILENAME)
    assert manifest["plugin_id"] == "org.example.module"
    assert manifest["display_name"] == "An example"
    assert manifest["version"] == "1.2"
    assert not manifest["warnings"]
    assert not marker.exists()


@pytest.mark.parametrize(
    "source",
    [
        b"raise RuntimeError('must not run')",
        SOURCE.replace(b'version="1.2"', b"version=dangerous()"),
        SOURCE.replace(b'PLUGIN_NAME = "An example"', b"PLUGIN_NAME = dangerous()"),
        SOURCE + SOURCE.replace(b"class Example", b"class Other"),
        SOURCE.replace(b"PluginInfo(id=PLUGIN_ID", b"factory(id=PLUGIN_ID"),
    ],
)
def test_unknown_metadata_is_not_evaluated(source: bytes) -> None:
    manifest = inspect_module(source, filename=FILENAME)
    assert manifest["warnings"]
    explicit = apply_catalog_metadata(
        manifest, plugin_id="org.example.module", name="Example", version="1.2"
    )
    assert explicit["version"] == "1.2"


def test_generated_constants_and_class_constants() -> None:
    manifest = inspect_module(
        SOURCE.replace(
            b"class Example(PluginBase):",
            b'class Example(PluginBase):\n    VERSION = "2.0"',
        ).replace(b'version="1.2"', b"version=VERSION"),
        filename=FILENAME,
    )
    assert manifest["version"] == "2.0"


def test_encoding_and_invalid_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source = b"# coding: latin-1\n" + SOURCE.replace(b"An example", b"Caf\xe9")
    assert inspect_module(source, filename=FILENAME)["display_name"] == "Caf\u00e9"
    for data in (
        b"class Broken(",
        b"\x00",
        b"# coding: does-not-exist\n",
        b"return 1",
        b"from __future__ import made_up_feature",
    ):
        with pytest.raises(ModuleInspectionError, match="Invalid Python"):
            inspect_module(data, filename=FILENAME)
    monkeypatch.setattr(modules, "MAX_MODULE_BYTES", 1)
    with pytest.raises(ModuleInspectionError, match="size limit"):
        inspect_module(SOURCE, filename=FILENAME)


def test_empty_module_cannot_produce_an_undownloadable_release() -> None:
    with pytest.raises(ModuleInspectionError, match="empty"):
        inspect_module(b"", filename=FILENAME)


def test_explicit_version_must_be_valid_even_when_source_metadata_is_unknown() -> None:
    manifest = inspect_module(b"VALUE = 1\n", filename=FILENAME)
    with pytest.raises(ValueError, match="Invalid version"):
        apply_catalog_metadata(
            manifest,
            plugin_id="org.example.module",
            name="Example",
            version="not a version",
        )


@pytest.mark.parametrize(
    "name",
    [
        "other.py",
        "datalab_.py",
        "datalab_bad-name.py",
        "../datalab_x.py",
        r"..\datalab_x.py",
        "datalab_x.py.exe",
    ],
)
def test_invalid_names(name: str) -> None:
    with pytest.raises(ModuleInspectionError, match="datalab_<name>"):
        inspect_module(SOURCE, filename=name)


def test_explicit_metadata_cannot_hide_known_conflicts() -> None:
    manifest = inspect_module(SOURCE, filename=FILENAME)
    with pytest.raises(ModuleInspectionError, match="plugin_id"):
        apply_catalog_metadata(
            manifest, plugin_id="org.example.other", name="Example", version="1.2"
        )
    with pytest.raises(ModuleInspectionError, match="version"):
        apply_catalog_metadata(
            manifest, plugin_id="org.example.module", name="Example", version="2.0"
        )


@pytest.mark.parametrize(
    "before_class",
    [
        b"PLUGIN_NAME = dangerous()\n",
        b"if condition:\n    PLUGIN_NAME = dangerous()\n",
        b"from other import PLUGIN_NAME\n",
    ],
)
def test_reassigned_constants_are_not_stale_metadata(before_class: bytes) -> None:
    source = SOURCE.replace(b"class Example", before_class + b"class Example")
    manifest = inspect_module(source, filename=FILENAME)
    assert not manifest["display_name"]
    assert manifest["warnings"]


def test_shadowed_constructor_and_expansion_are_not_metadata() -> None:
    for source in (
        SOURCE.replace(b"class Example", b"PluginInfo = dangerous\nclass Example"),
        SOURCE.replace(
            b"class Example(PluginBase):",
            b"class Example(PluginBase):\n    PluginInfo = dangerous",
        ),
        SOURCE.replace(b"PluginInfo(id=", b"PluginInfo(**dynamic, id="),
    ):
        manifest = inspect_module(source, filename=FILENAME)
        assert not manifest["plugin_id"]
        assert manifest["warnings"]


def test_conditional_class_metadata_is_not_guessed() -> None:
    source = SOURCE.replace(
        b"class Example(PluginBase):",
        b"class Example(PluginBase):\n"
        b'    VERSION = "1.0"\n'
        b"    if condition:\n"
        b"        VERSION = dangerous()",
    ).replace(b'version="1.2"', b"version=VERSION")
    assert not inspect_module(source, filename=FILENAME)["version"]
