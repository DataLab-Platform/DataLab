# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the DataLab plugin generator: single files and layered projects."""

from __future__ import annotations

import ast
import importlib
import os
import os.path as osp
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from datalab import __version__
from datalab.config import Conf
from datalab.plugins import (
    PluginBase,
    PluginCapability,
    PluginRegistry,
    discover_plugins,
    generator,
)
from datalab.plugins import base as plugin_base
from datalab.plugins.generator import (
    PluginProject,
    _default_plugin_id,
    create_single_file_plugin,
    get_default_plugin_folder,
    is_default_plugin_folder,
    main,
    render_single_file_plugin,
)
from datalab.plugins.store import InstalledPluginStore

PROJECT_ROOT = Path(__file__).parents[4]


def _discover_from_folder(
    folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> tuple[list[type[PluginBase]], list]:
    """Discover the plugins of a folder, isolated from any other plugin.

    No installed or user plugin is scanned, and the plugin registry and the
    generated modules are restored afterwards.

    Returns:
        Discovered plugin classes and failed plugins
    """
    iter_modules = pkgutil.iter_modules
    store = InstalledPluginStore(str(tmp_path / "installed_plugins"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(plugin_base, "get_user_plugin_paths", lambda: [str(folder)])
    monkeypatch.setattr(plugin_base, "PLUGINS_DEFAULT_PATH", str(tmp_path / "none"))
    monkeypatch.setattr(plugin_base, "OTHER_PLUGINS_PATHLIST", [])
    monkeypatch.setattr(plugin_base, "get_installed_plugin_store", lambda: store)
    monkeypatch.setattr(plugin_base, "_get_plugin_entry_points", lambda: [])
    monkeypatch.setattr(
        plugin_base.pkgutil,
        "iter_modules",
        # pylint: disable-next=protected-access
        lambda: iter_modules(plugin_base._get_plugin_search_paths()),
    )
    monkeypatch.setattr(Conf.plugins_enabled, "get", lambda: True)
    previous_classes = list(PluginRegistry.get_plugin_classes())
    discovery_errors = PluginRegistry.get_discovery_errors()
    previous_failed_plugins = PluginRegistry.get_failed_plugins()
    PluginRegistry.clear_plugin_classes()
    try:
        discover_plugins()
        return (
            list(PluginRegistry.get_plugin_classes()),
            PluginRegistry.get_failed_plugins(),
        )
    finally:
        PluginRegistry.clear_plugin_classes()
        PluginRegistry.get_plugin_classes().extend(previous_classes)
        PluginRegistry.clear_discovery_errors()
        for tb_text in discovery_errors:
            PluginRegistry.add_discovery_error(tb_text)
        PluginRegistry.clear_failed_plugins()
        for failed_plugin in previous_failed_plugins:
            PluginRegistry.add_failed_plugin(
                failed_plugin.name,
                failed_plugin.filepath,
                failed_plugin.traceback,
                failed_plugin.source,
            )
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name)


def _run_generated_project_checks(destination: Path) -> None:
    """Run the quality checks advertised by a generated project."""
    environment = os.environ.copy()
    # The checks run in the generated project: relative entries (e.g. those of
    # the development ``.env``, relative to the project root) must stay valid
    entries = [
        str((PROJECT_ROOT / entry).resolve())
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), *entries))
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
                "--project",
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
                "--github-account",
                "DataLab-Platform",
            ]
        )
        == 0
    )
    assert str(destination.resolve()) in capsys.readouterr().out

    expected_files = {
        ".github/workflows/release.yml",
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
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'target-version = "py310"' in pyproject
    major, minor = __version__.split(".")[:2]
    assert (
        f'dependencies = ["datalab-platform >= {major}.{minor}, < {int(major) + 1}"]'
        in pyproject
    )
    assert 'keywords = ["datalab", "datalab-plugin"]' in pyproject
    repository = "https://github.com/DataLab-Platform/datalab-camera-characterization"
    assert f'Homepage = "{repository}"' in pyproject
    assert f'Issues = "{repository}/issues"' in pyproject
    assert '[project.entry-points."datalab.plugins"]' in pyproject
    assert (
        "datalab_camera_characterization = "
        '"datalab_camera_characterization.adapters.desktop:'
        'CameraCharacterizationPlugin"' in pyproject
    )

    _run_generated_project_checks(destination)

    release = (destination / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    steps = yaml.safe_load(release)["jobs"]["release"]["steps"]
    assert "gh release create" in steps[4]["run"]
    assert "id: org.datalab.camera-characterization\n" in steps[5]["run"]
    assert 'name: "Camera Characterization"\n' in steps[5]["run"]
    assert "capabilities: [application, processing]" in steps[5]["run"]

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
                "--project",
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


def test_generated_project_is_loaded_from_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DataLab loads a freshly generated project from its folder (Issue #411)."""
    destination = tmp_path / "folder-plugin"
    package_name = "datalab_folder_plugin"
    assert (
        main(
            [
                "create",
                "--project",
                str(destination),
                "--name",
                "Folder Plugin",
                "--package",
                package_name,
                "--plugin-id",
                "org.example.folder-plugin",
                "--description",
                "Plugin loaded from its project folder",
            ]
        )
        == 0
    )
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "**Plugins > Configure plugins... > Advanced**" in readme

    plugin_classes, failed_plugins = _discover_from_folder(
        destination, tmp_path, monkeypatch, package_name
    )

    assert not [*destination.rglob("*.egg-info"), *destination.rglob("*.dist-info")]
    assert len(plugin_classes) == 1
    plugin_class = plugin_classes[0]
    assert plugin_class.get_plugin_id() == "org.example.folder-plugin"
    assert plugin_class.__module__ == f"{package_name}.adapters.desktop"
    assert not failed_plugins


def test_create_refuses_existing_destination(tmp_path: Path) -> None:
    """The CLI never overwrites an existing project directory."""
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                "--project",
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
                "--project",
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
                "--project",
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
                "--project",
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
    answers = iter(("Pulse Characterization", "DataLab-Platform", "", "", "", ""))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.chdir(tmp_path)

    assert main(["create", "--project"]) == 0

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
    assert (
        'Homepage = "https://github.com/DataLab-Platform/'
        'datalab-pulse-characterization"' in pyproject
    )


@pytest.mark.parametrize(
    ("github_account", "plugin_id"),
    [
        ("", "org.example.spectrum-tools"),
        ("Alice-Lab", "io.github.alice-lab.spectrum-tools"),
        ("datalab-platform", "org.datalab.spectrum-tools"),
    ],
)
def test_default_plugin_id_follows_the_github_account(
    github_account: str, plugin_id: str
) -> None:
    """Only DataLab-Platform projects default to the reserved org.datalab IDs."""
    assert _default_plugin_id("datalab_spectrum_tools", github_account) == plugin_id


def test_create_rejects_invalid_github_account(tmp_path: Path) -> None:
    """The account becomes part of URLs and of the default plugin ID."""
    destination = tmp_path / "invalid-account"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                "--project",
                str(destination),
                "--name",
                "Account Plugin",
                "--package",
                "datalab_account_plugin",
                "--plugin-id",
                "org.example.account-plugin",
                "--description",
                "Invalid account example",
                "--github-account",
                "alice/lab",
            ]
        )

    assert exc_info.value.code == 2
    assert not destination.exists()


def test_create_pulse_project_with_long_metadata_passes_checks(tmp_path: Path) -> None:
    """The hardened template formats the roadmap's full Pulse metadata."""
    destination = tmp_path / "datalab-pulse-characterization"

    assert (
        main(
            [
                "create",
                "--project",
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


# --- Single-file plugins (Issue #413) ---------------------------------------------


@pytest.fixture(name="user_plugins")
def fixture_user_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace the user plugin directory: tests never write to the real one."""
    folder = tmp_path / "user_plugins"
    monkeypatch.setattr(generator, "get_default_plugin_folder", lambda: folder)
    return folder


def _check_single_file(path: Path) -> str:
    """Check a single-file plugin as Ruff does in a plugin directory.

    Returns:
        Source code of the plugin
    """
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert all(len(line) <= 88 for line in source.splitlines())
    for command in (
        ("format", "--isolated", "--check"),
        ("check", "--isolated", "--select", "E,F,I"),
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "ruff", *command, "--line-length", "88", path.name],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return source


def _string_constants(source: str) -> dict[str, str]:
    """Return the module-level string constants of a plugin source."""
    return {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_create_single_file_plugin_by_default(
    user_plugins: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Without --project, a ready-to-use file is written to the plugin directory."""

    def unexpected_prompt(prompt: str) -> str:
        raise AssertionError(f"Unexpected question: {prompt}")

    monkeypatch.setattr("builtins.input", unexpected_prompt)

    assert (
        main(["create", "--name", "My Filters", "--description", "Remove noise"]) == 0
    )

    path = user_plugins / "datalab_my_filters.py"
    assert [item.name for item in user_plugins.iterdir()] == [path.name]
    output = capsys.readouterr().out
    assert str(path.resolve()) in output
    assert "org.example.my-filters" in output
    assert "Plugins > Reload plugins" in output
    source = _check_single_file(path)
    assert _string_constants(source) == {
        "PLUGIN_DESCRIPTION": "Remove noise",
        "PLUGIN_ID": "org.example.my-filters",
        "PLUGIN_NAME": "My Filters",
    }
    assert "def sample_processing(y: np.ndarray) -> np.ndarray:" in source
    # The NumPy function is registered as it is, wrapped with Wrap1to1Func
    assert "            sips.Wrap1to1Func(sample_processing),\n" in source
    assert "computation_function" not in source
    assert "handler.action_for(FEATURE_ID)" in source


def test_create_single_file_plugin_asks_two_questions(
    user_plugins: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the name and the description are asked: other values are derived."""
    prompts: list[str] = []
    answers = iter(("Filtres améliorés", ""))

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)

    assert main(["create"]) == 0

    assert len(prompts) == 2
    source = (user_plugins / "datalab_filtres_ameliores.py").read_text(encoding="utf-8")
    assert _string_constants(source) == {
        "PLUGIN_DESCRIPTION": "Filtres améliorés plugin for DataLab",
        "PLUGIN_ID": "org.example.filtres-ameliores",
        "PLUGIN_NAME": "Filtres améliorés",
    }
    assert "class FiltresAmelioresPlugin(PluginBase):" in source


@pytest.mark.parametrize(
    ("object_kind", "capability", "expected_code"),
    [
        ("signal", "processing", "return np.maximum(y, 0.0)"),
        ("image", "processing", "return data.max() - data"),
        ("image", "application", 'handler.new_action(\n                "Show message"'),
    ],
)
def test_single_file_plugin_is_loaded_from_its_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    user_plugins: Path,
    object_kind: str,
    capability: str,
    expected_code: str,
) -> None:
    """A generated file is valid and discovered in any plugin directory."""
    folder = tmp_path / "plugins"
    module_name = f"datalab_generated_{object_kind}_{capability}"
    assert (
        main(
            [
                "create",
                str(folder),
                "--name",
                "Generated Plugin",
                "--package",
                module_name,
                "--description",
                "Generated single-file plugin",
                "--capability",
                capability,
                "--object-kind",
                object_kind,
            ]
        )
        == 0
    )
    assert not user_plugins.exists()
    # Outside the user plugin directory, the folder must be added to DataLab
    assert "Configure plugins... > Advanced" in capsys.readouterr().out
    source = _check_single_file(folder / f"{module_name}.py")
    assert expected_code in source
    assert f"self.{object_kind}panel" in source

    plugin_classes, failed_plugins = _discover_from_folder(
        folder, tmp_path, monkeypatch, module_name
    )

    assert not failed_plugins
    assert len(plugin_classes) == 1
    plugin_class = plugin_classes[0]
    assert plugin_class.__module__ == module_name
    assert plugin_class.get_plugin_id() == (
        f"org.example.{module_name.removeprefix('datalab_').replace('_', '-')}"
    )
    assert plugin_class.PLUGIN_INFO.capabilities == frozenset(
        {PluginCapability[capability.upper()]}
    )
    assert plugin_class.__plugin_discovery_sources__ == (
        f"module convention {module_name!r}",
    )


def test_default_plugin_folder_is_the_datalab_plugin_directory() -> None:
    """Generated files go where DataLab looks for plugins."""
    folder = get_default_plugin_folder()

    assert osp.normcase(osp.realpath(folder)) == osp.normcase(
        osp.realpath(plugin_base.PLUGINS_DEFAULT_PATH)
    )
    assert is_default_plugin_folder(Path(plugin_base.PLUGINS_DEFAULT_PATH))


def test_single_file_plugin_is_never_overwritten(tmp_path: Path) -> None:
    """An existing plugin file is kept, from the API and from the CLI."""
    project = PluginProject(
        destination=tmp_path,
        name="Kept Plugin",
        package="datalab_kept_plugin",
        plugin_id="org.example.kept-plugin",
        description="Kept plugin",
    )
    path = create_single_file_plugin(project)
    path.write_text("# Edited by the user\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_single_file_plugin(project)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                str(tmp_path),
                "--name",
                "Kept Plugin",
                "--description",
                "Kept plugin",
            ]
        )

    assert exc_info.value.code == 2
    assert path.read_text(encoding="utf-8") == "# Edited by the user\n"


def test_single_file_module_name_must_start_with_datalab(tmp_path: Path) -> None:
    """DataLab would ignore a plugin module without the ``datalab_`` prefix."""
    destination = tmp_path / "plugins"
    project = PluginProject(
        destination=destination,
        name="Hidden Plugin",
        package="hidden_plugin",
        plugin_id="org.example.hidden-plugin",
        description="Hidden plugin",
    )

    with pytest.raises(ValueError, match="datalab_"):
        render_single_file_plugin(project)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "create",
                str(destination),
                "--name",
                "Hidden Plugin",
                "--package",
                "hidden_plugin",
                "--description",
                "Hidden plugin",
            ]
        )

    assert exc_info.value.code == 2
    assert not destination.exists()


def test_single_file_plugin_keeps_quotes_and_accents_readable(tmp_path: Path) -> None:
    """User values are kept as typed, as valid and formatted Python strings."""
    name = 'Caméra """ & co'
    description = 'It\'s a "quoted" description with a \\ backslash'
    path = create_single_file_plugin(
        PluginProject(
            destination=tmp_path,
            name=name,
            package="datalab_quoted_camera",
            plugin_id="org.example.quoted-camera",
            description=description,
        )
    )

    source = _check_single_file(path)

    assert 'Caméra """ & co' in source
    constants = _string_constants(source)
    assert (constants["PLUGIN_NAME"], constants["PLUGIN_DESCRIPTION"]) == (
        name,
        description,
    )


def test_single_file_plugin_can_be_installed_from_the_plugin_dialog(
    tmp_path: Path,
) -> None:
    """A generated file is accepted by Plugins > Configure plugins > Install."""
    path = create_single_file_plugin(
        PluginProject(
            destination=tmp_path / "shared",
            name="Shared Plugin",
            package="datalab_shared_generated",
            plugin_id="org.example.shared-generated",
            description="Plugin shared with colleagues",
        )
    )
    store = InstalledPluginStore(str(tmp_path / "installed_plugins"))

    plugin = store.install_module(str(path)).plugin

    assert (plugin.kind, plugin.name) == ("module", "datalab_shared_generated")
