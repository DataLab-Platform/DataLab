# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the layered DataLab plugin project generator."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from datalab.plugin_generator import main
from datalab.plugins import PluginCapability, PluginRegistry

PROJECT_ROOT = Path(__file__).parents[3]


def _run_generated_project_checks(destination: Path) -> None:
    """Run the quality checks advertised by a generated project."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
    commands = (
        ("ruff", "format", "--check", "."),
        ("ruff", "check", "."),
        ("pytest", "-q"),
    )
    for module_args in commands:
        completed = subprocess.run(
            [sys.executable, "-m", *module_args],
            cwd=destination,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_create_layered_plugin_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI creates an importable layered project with stable metadata."""
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
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "doc/architecture.md",
        "pyproject.toml",
        "src/datalab_camera_characterization/__init__.py",
        "src/datalab_camera_characterization/adapters/__init__.py",
        "src/datalab_camera_characterization/adapters/desktop.py",
        "src/datalab_camera_characterization/adapters/web.py",
        "src/datalab_camera_characterization/core/__init__.py",
        "src/datalab_camera_characterization/workflow/__init__.py",
        "src/datalab_camera_characterization/workflow/recipes.py",
        "tests/integration/test_desktop_adapter.py",
        "tests/unit/test_architecture.py",
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
        '"datalab_camera_characterization.adapters.desktop:'
        'CameraCharacterizationPlugin"' in pyproject
    )

    _run_generated_project_checks(destination)

    web_adapter = (
        destination / "src/datalab_camera_characterization/adapters/web.py"
    ).read_text(encoding="utf-8")
    assert 'WEB_STATUS = "unsupported"' in web_adapter

    package_name = "datalab_camera_characterization"
    existing_modules = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if module_name == package_name or module_name.startswith(f"{package_name}.")
    }
    for module_name in existing_modules:
        sys.modules.pop(module_name)

    plugin_class = None
    try:
        monkeypatch.syspath_prepend(str(destination / "src"))
        module = importlib.import_module(package_name)
        assert module.PLUGIN_ID == "org.datalab.camera-characterization"
        assert f"{package_name}.adapters.desktop" not in sys.modules
        desktop_adapter = importlib.import_module(f"{package_name}.adapters.desktop")
        plugin_class = desktop_adapter.CameraCharacterizationPlugin
        assert plugin_class.get_plugin_id() == "org.datalab.camera-characterization"
        assert plugin_class.PLUGIN_INFO.capabilities == frozenset(
            {PluginCapability.APPLICATION, PluginCapability.PROCESSING}
        )
        plugin_source = (
            destination / "src/datalab_camera_characterization/adapters/desktop.py"
        )
        assert "self.imagepanel.processor.register_1_to_1" in plugin_source.read_text(
            encoding="utf-8"
        )
        assert '            _("Inverse"),' in plugin_source.read_text(encoding="utf-8")
    finally:
        if plugin_class in PluginRegistry.get_plugin_classes():
            PluginRegistry.get_plugin_classes().remove(plugin_class)
        for module_name in tuple(sys.modules):
            is_generated_module = module_name == package_name or module_name.startswith(
                f"{package_name}."
            )
            if is_generated_module:
                sys.modules.pop(module_name)
        sys.modules.update(existing_modules)


def test_create_application_only_project(tmp_path: Path) -> None:
    """A project without sample processing passes its advertised checks."""
    destination = tmp_path / "application-plugin"

    assert (
        main(
            [
                "create",
                str(destination),
                "--name",
                "Application Plugin",
                "--package",
                "datalab_application_plugin",
                "--plugin-id",
                "org.datalab.application-plugin",
                "--description",
                "Application-only plugin",
                "--capability",
                "application",
            ]
        )
        == 0
    )

    _run_generated_project_checks(destination)
    desktop_adapter = (
        destination / "src/datalab_application_plugin/adapters/desktop.py"
    ).read_text(encoding="utf-8")
    assert "register_computations" not in desktop_adapter


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
    plugin = (
        destination / "src/datalab_pulse_characterization/adapters/desktop.py"
    ).read_text(encoding="utf-8")
    package = (
        destination / "src/datalab_pulse_characterization/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'name = "datalab-pulse-characterization"' in pyproject
    assert 'PLUGIN_ID = "org.datalab.pulse-characterization"' in package
    assert 'PLUGIN_DESCRIPTION = "Pulse Characterization plugin for DataLab"' in package
    assert "id=PLUGIN_ID" in plugin


def test_create_pulse_project_with_long_metadata_passes_checks(tmp_path: Path) -> None:
    """The hardened template formats the roadmap's full Pulse metadata."""
    destination = tmp_path / "datalab-pulse-characterization"

    assert (
        main(
            [
                "create",
                str(destination),
                "--name",
                "Pulse & Transient Characterization",
                "--package",
                "datalab_pulse_characterization",
                "--plugin-id",
                "org.datalab.pulse-characterization",
                "--description",
                (
                    "Analyze repeated pulse acquisitions, timing and "
                    "shot-to-shot stability"
                ),
                "--capability",
                "application",
                "--capability",
                "processing",
                "--object-kind",
                "signal",
            ]
        )
        == 0
    )

    _run_generated_project_checks(destination)
