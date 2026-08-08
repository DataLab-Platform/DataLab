# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the minimal DataLab plugin project generator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from datalab.plugin_generator import main
from datalab.plugins import PluginCapability, PluginRegistry


def test_create_minimal_plugin_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI creates an importable project with stable owned metadata."""
    destination = tmp_path / "camera-plugin"

    assert (
        main(
            [
                "create",
                str(destination),
                "--name",
                "Camera Characterization",
                "--package",
                "datalab_camera_characterization",
                "--plugin-id",
                "org.datalab.camera-characterization",
                "--description",
                "Characterize scientific cameras",
                "--capability",
                "application",
                "--capability",
                "processing",
                "--object-kind",
                "image",
            ]
        )
        == 0
    )
    assert str(destination.resolve()) in capsys.readouterr().out

    expected_files = {
        ".gitignore",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/datalab_camera_characterization/__init__.py",
        "src/datalab_camera_characterization/plugin.py",
        "tests/test_plugin.py",
    }
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == expected_files

    for path in destination.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert all(len(line) <= 88 for line in source.splitlines())

    pyproject = (destination / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools >= 77"]' in pyproject
    assert 'name = "datalab-camera-characterization"' in pyproject
    assert 'license = "BSD-3-Clause"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '[project.entry-points."datalab.plugins"]' in pyproject
    assert (
        "datalab_camera_characterization = "
        '"datalab_camera_characterization.plugin:CameraCharacterizationPlugin"'
        in pyproject
    )

    monkeypatch.syspath_prepend(str(destination / "src"))
    module = importlib.import_module("datalab_camera_characterization")
    plugin_class = module.CameraCharacterizationPlugin
    try:
        assert plugin_class.get_plugin_id() == "org.datalab.camera-characterization"
        assert plugin_class.PLUGIN_INFO.capabilities == frozenset(
            {PluginCapability.APPLICATION, PluginCapability.PROCESSING}
        )
        plugin_source = destination / "src/datalab_camera_characterization/plugin.py"
        assert "self.imagepanel.processor.register_1_to_1" in plugin_source.read_text(
            encoding="utf-8"
        )
        assert '            _("Inverse"),' in plugin_source.read_text(encoding="utf-8")
    finally:
        PluginRegistry.get_plugin_classes().remove(plugin_class)
        sys.modules.pop("datalab_camera_characterization.plugin", None)
        sys.modules.pop("datalab_camera_characterization", None)


def test_create_refuses_existing_destination(tmp_path: Path) -> None:
    """The CLI never overwrites an existing project directory."""
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                str(destination),
                "--name",
                "Existing Plugin",
                "--package",
                "datalab_existing",
                "--plugin-id",
                "org.datalab.existing",
                "--description",
                "Existing project",
            ]
        )

    assert exc_info.value.code == 2
    assert not list(destination.iterdir())


def test_create_rejects_python_keyword_package(tmp_path: Path) -> None:
    """The generated package is always a valid Python import identifier."""
    destination = tmp_path / "keyword-package"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                str(destination),
                "--name",
                "Keyword Plugin",
                "--package",
                "class",
                "--plugin-id",
                "org.datalab.keyword",
                "--description",
                "Invalid package example",
            ]
        )

    assert exc_info.value.code == 2
    assert not destination.exists()


def test_create_rejects_multiline_metadata(tmp_path: Path) -> None:
    """Carriage returns cannot leak into generated project text files."""
    destination = tmp_path / "multiline-metadata"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                str(destination),
                "--name",
                "Multiline Plugin",
                "--package",
                "datalab_multiline",
                "--plugin-id",
                "org.datalab.multiline",
                "--description",
                "First line\rSecond line",
            ]
        )

    assert exc_info.value.code == 2
    assert not destination.exists()


def test_create_escapes_display_name_from_python_source(tmp_path: Path) -> None:
    """Quotes in display metadata never invalidate generated Python files."""
    destination = tmp_path / "quoted-name"

    assert (
        main(
            [
                "create",
                str(destination),
                "--name",
                'Camera """ Plugin',
                "--package",
                "datalab_quoted_name",
                "--plugin-id",
                "org.datalab.quoted-name",
                "--description",
                "Plugin with quoted metadata",
            ]
        )
        == 0
    )

    for path in destination.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_create_interactive_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bare create command infers safe defaults from the plugin name."""
    answers = iter(("Pulse Characterization", "", "", "", ""))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.chdir(tmp_path)

    assert main(["create"]) == 0

    destination = tmp_path / "datalab-pulse-characterization"
    pyproject = (destination / "pyproject.toml").read_text(encoding="utf-8")
    plugin = (destination / "src/datalab_pulse_characterization/plugin.py").read_text(
        encoding="utf-8"
    )
    assert 'name = "datalab-pulse-characterization"' in pyproject
    assert 'id="org.datalab.pulse-characterization"' in plugin
    assert 'description="Pulse Characterization plugin for DataLab"' in plugin
