# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Generate DataLab plugins: a single file, or a layered installable project.

A single-file plugin is a ``datalab_<name>.py`` module that DataLab loads from
a plugin directory. A layered project is installable and publishable as a wheel.

The ``datalab-plugin`` command also builds and checks local plugin catalogs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import keyword
import os.path as osp
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from datalab import __version__
from datalab.config import _, get_config_path
from datalab.plugins.catalogtool import (
    build_local_catalog,
    check_local_catalog,
    prepare_local_catalog,
)

_PACKAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)+$")
_GITHUB_ACCOUNT_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_CAPABILITIES = ("application", "processing", "io", "visualization")
#: GitHub organization whose plugins use the reserved ``org.datalab.`` IDs
OFFICIAL_GITHUB_ACCOUNT = "DataLab-Platform"
#: Prefix of the modules that DataLab loads from its plugin directories
PLUGIN_MODULE_PREFIX = "datalab_"
#: Import of the plugin contracts by single-file plugins
_PLUGIN_IMPORT = (
    "from datalab.plugins import PluginBase, PluginCapability, PluginInfo\n"
)
#: Sample processing of single-file plugins, by object kind: argument, summary and
#: returned expression of the NumPy function that users edit
_SAMPLE_FUNCTIONS = {
    "signal": (
        "y",
        "Return new Y values computed from those of the selected signal.",
        "np.maximum(y, 0.0)",
    ),
    "image": (
        "data",
        "Return new image data computed from that of the selected image.",
        "data.max() - data",
    ),
}


@dataclasses.dataclass(frozen=True)
class PluginProject:
    """Configuration of a generated plugin: single file or layered project.

    ``destination`` is the folder receiving the ``<package>.py`` file of a
    single-file plugin, or the folder of a layered project.
    """

    destination: Path
    name: str
    package: str
    plugin_id: str
    description: str
    capabilities: tuple[str, ...] = ("processing",)
    object_kind: str = "signal"
    license_id: str = "BSD-3-Clause"
    github_account: str = ""

    def __post_init__(self) -> None:
        """Validate values used as paths, Python identifiers, and metadata."""
        if not self.name.strip() or any(char in self.name for char in "\r\n"):
            raise ValueError(_("Plugin name must be a non-empty single line"))
        if not _PACKAGE_PATTERN.fullmatch(self.package):
            raise ValueError(
                _(
                    "Package name must start with a lowercase letter and contain "
                    "only lowercase letters, digits, and underscores"
                )
            )
        if keyword.iskeyword(self.package):
            raise ValueError(_("Package name must not be a Python keyword"))
        if not _PLUGIN_ID_PATTERN.fullmatch(self.plugin_id):
            raise ValueError(
                _(
                    "Plugin ID must be a lowercase dotted identifier such as "
                    "'org.example.my-plugin'"
                )
            )
        if any(char in self.description for char in "\r\n"):
            raise ValueError(_("Description must be a single line"))
        if not self.capabilities or any(
            capability not in _CAPABILITIES for capability in self.capabilities
        ):
            raise ValueError(_("Plugin capabilities are invalid"))
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError(_("Plugin capabilities must be unique"))
        if self.object_kind not in ("signal", "image"):
            raise ValueError(_("Object kind must be 'signal' or 'image'"))
        if self.license_id != "BSD-3-Clause":
            raise ValueError(_("The minimal template currently supports BSD-3-Clause"))
        if self.github_account and not _GITHUB_ACCOUNT_PATTERN.fullmatch(
            self.github_account
        ):
            raise ValueError(_("GitHub account must be a GitHub user or organization"))

    @property
    def repository_url(self) -> str:
        """Return the GitHub repository URL, if the account is known."""
        if not self.github_account:
            return ""
        return f"https://github.com/{self.github_account}/{self.distribution}"

    @property
    def distribution(self) -> str:
        """Return the Python distribution name."""
        return self.package.replace("_", "-")

    @property
    def class_name(self) -> str:
        """Return the generated plugin class name."""
        words = re.findall(r"[A-Za-z0-9]+", _strip_accents(self.name))
        class_name = "".join(word[0].upper() + word[1:] for word in words)
        if not class_name:
            raise ValueError(_("Plugin name must contain letters or digits"))
        if class_name[0].isdigit():
            class_name = f"Generated{class_name}"
        if not class_name.endswith("Plugin"):
            class_name += "Plugin"
        return class_name


def _strip_accents(text: str) -> str:
    """Return *text* without accents, for identifiers derived from names."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _toml_string(value: str) -> str:
    """Return *value* as a TOML-compatible quoted string."""
    return json.dumps(value, ensure_ascii=True)


def _python_string(value: str) -> str:
    """Return *value* as a readable Python string literal, quoted as Ruff does.

    Non-ASCII characters are kept as they are. Single quotes are used when the
    value holds more double quotes than single quotes.
    """
    literal = json.dumps(value, ensure_ascii=False)
    if value.count('"') <= value.count("'"):
        return literal
    body = literal[1:-1].replace('\\"', '"').replace("'", "\\'")
    return f"'{body}'"


def _render_python_string_assignment(
    name: str, value: str, literal_function: Callable[[str], str] = _toml_string
) -> str:
    """Render a string constant without requiring formatter changes."""
    literal = literal_function(value)
    assignment = f"{name} = {literal}"
    if len(assignment) <= 88:
        return assignment
    return f"{name} = (\n    {literal}\n)"


def _render_pyproject(project: PluginProject) -> str:
    """Render generated project metadata and plugin entry point."""
    major, minor = __version__.split(".")[:2]
    requirement = f"datalab-platform >= {major}.{minor}, < {int(major) + 1}"
    urls = ""
    if project.repository_url:
        urls = f"""
[project.urls]
Homepage = "{project.repository_url}"
Issues = "{project.repository_url}/issues"
"""
    return f'''[build-system]
requires = ["setuptools >= 77"]
build-backend = "setuptools.build_meta"

[project]
name = {_toml_string(project.distribution)}
version = "0.1.0"
description = {_toml_string(project.description)}
readme = "README.md"
license = "{project.license_id}"
license-files = ["LICENSE"]
requires-python = ">=3.10"
dependencies = ["{requirement}"]
keywords = ["datalab", "datalab-plugin"]
{urls}
[project.entry-points."datalab.plugins"]
{project.package} = "{project.package}.adapters.desktop:{project.class_name}"

[project.optional-dependencies]
test = ["pytest", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]
'''


def _render_capabilities(project: PluginProject) -> str:
    """Render the ``capabilities`` argument of the generated ``PluginInfo``."""
    if len(project.capabilities) == 1:
        return (
            "        capabilities=(PluginCapability."
            f"{project.capabilities[0].upper()},),"
        )
    capability_items = "\n".join(
        f"            PluginCapability.{capability.upper()},"
        for capability in project.capabilities
    )
    return f"""        capabilities=(
{capability_items}
        ),"""


def _render_identity(
    project: PluginProject, literal_function: Callable[[str], str] = _toml_string
) -> str:
    """Render the plugin identity constants."""
    description = _render_python_string_assignment(
        "PLUGIN_DESCRIPTION", project.description, literal_function
    )
    plugin_id = _render_python_string_assignment(
        "PLUGIN_ID", project.plugin_id, literal_function
    )
    plugin_name = _render_python_string_assignment(
        "PLUGIN_NAME", project.name, literal_function
    )
    return f"{description}\n{plugin_id}\n{plugin_name}"


def _render_plugin(project: PluginProject) -> str:
    """Render the Desktop adapter and optional owned sample processing."""
    capabilities = _render_capabilities(project)
    processing_import = ""
    processing_method = ""
    if "processing" in project.capabilities:
        module_alias = "sips" if project.object_kind == "signal" else "sipi"
        processing_function, processing_title = (
            ("derivative", "Derivative")
            if project.object_kind == "signal"
            else ("inverse", "Inverse")
        )
        processing_import = (
            f"import sigima.proc.{project.object_kind} as {module_alias}\n"
            "from datalab.config import _\n"
        )
        processing_method = f'''
    def register_computations(self) -> None:
        """Register the sample processing owned by this plugin."""
        self.{project.object_kind}panel.processor.register_1_to_1(
            {module_alias}.{processing_function},
            _("{processing_title}"),
            feature_id="{project.plugin_id}:sample-processing",
            owner_plugin_id=self.plugin_id,
        )
'''
    return f'''"""DataLab Desktop plugin adapter."""

from __future__ import annotations

{processing_import}from datalab.plugins import PluginBase, PluginCapability, PluginInfo

from .. import PLUGIN_DESCRIPTION, PLUGIN_ID, PLUGIN_NAME, __version__
from ..workflow import RECIPES as WORKFLOW_RECIPES


class {project.class_name}(PluginBase):
    """Expose the plugin to DataLab Desktop."""

    PLUGIN_INFO = PluginInfo(
        id=PLUGIN_ID,
        name=PLUGIN_NAME,
        version=__version__,
        description=PLUGIN_DESCRIPTION,
{capabilities}
    )
    RECIPES = WORKFLOW_RECIPES
{processing_method}
    def create_actions(self) -> None:
        """Create plugin actions after the DataLab panels are ready."""
'''


def _render_init(project: PluginProject) -> str:
    """Render host-independent package identity."""
    return f'''"""DataLab plugin package."""

{_render_identity(project)}
__version__ = "0.1.0"

__all__ = [
    "PLUGIN_DESCRIPTION",
    "PLUGIN_ID",
    "PLUGIN_NAME",
    "__version__",
]
'''


def _check_single_file_module(project: PluginProject) -> None:
    """Check that DataLab loads the module of a single-file plugin.

    Raises:
        ValueError: The module name does not start with ``datalab_``
    """
    if not project.package.startswith(PLUGIN_MODULE_PREFIX):
        raise ValueError(
            _("The module name of a single-file plugin must start with '%s'")
            % PLUGIN_MODULE_PREFIX
        )


def _render_sample_processing(project: PluginProject) -> tuple[str, str, str]:
    """Render the imports, function and methods of the sample processing.

    The NumPy function is registered wrapped with ``Wrap1to1Func``, which applies
    it to the data of the selected signals or images.
    """
    kind = project.object_kind
    alias = "sips" if kind == "signal" else "sipi"
    argument, summary, expression = _SAMPLE_FUNCTIONS[kind]
    imports = (
        f"import numpy as np\nimport sigima.proc.{kind} as {alias}\n{_PLUGIN_IMPORT}"
    )
    function = f'''FEATURE_ID = f"{{PLUGIN_ID}}:sample-processing"


def sample_processing({argument}: np.ndarray) -> np.ndarray:
    """{summary}

    Edit this function to implement your own processing.
    """
    return {expression}'''
    methods = f'''
    def register_computations(self) -> None:
        """Register the processing of this plugin."""
        self.{kind}panel.processor.register_1_to_1(
            {alias}.Wrap1to1Func(sample_processing),
            "Sample processing",
            feature_id=FEATURE_ID,
            owner_plugin_id=self.plugin_id,
        )

    def create_actions(self) -> None:
        """Add the submenu of this plugin to the Plugins menu."""
        handler = self.{kind}panel.acthandler
        with handler.new_menu(PLUGIN_NAME.replace("&", "&&")):
            handler.action_for(FEATURE_ID)
'''
    return imports, function, methods


def _render_sample_action(project: PluginProject) -> str:
    """Render the methods of a menu action showing a message."""
    return f'''
    def show_message(self) -> None:
        """Show a message: replace this method with the action of the plugin."""
        self.show_info(f"{{PLUGIN_NAME}}: edit this plugin to add features.")

    def create_actions(self) -> None:
        """Add the submenu of this plugin to the Plugins menu."""
        handler = self.{project.object_kind}panel.acthandler
        with handler.new_menu(PLUGIN_NAME.replace("&", "&&")):
            handler.new_action(
                "Show message",
                triggered=self.show_message,
                select_condition="always",
            )
'''


def render_single_file_plugin(project: PluginProject) -> str:
    """Return the source code of a single-file plugin.

    With the ``processing`` capability, the plugin registers a sample NumPy
    function as a processing of the signal or image panel, listed in its submenu
    of the *Plugins* menu. Otherwise, the submenu holds an action showing a
    message.

    Args:
        project: Plugin configuration

    Returns:
        Source code of the ``<package>.py`` file

    Raises:
        ValueError: The module name does not start with ``datalab_``
    """
    _check_single_file_module(project)
    if "processing" in project.capabilities:
        imports, function, methods = _render_sample_processing(project)
    else:
        imports = _PLUGIN_IMPORT
        function, methods = "", _render_sample_action(project)
    constants = _render_identity(project, _python_string)
    if function:
        constants += f"\n{function}"
    return f'''"""DataLab plugin created by ``datalab-plugin create``.

DataLab loads this file from its plugin directory. After editing it, select
**Plugins > Reload plugins** in DataLab to take the changes into account.
"""

from __future__ import annotations

{imports}
{constants}


class {project.class_name}(PluginBase):
    """DataLab plugin."""

    PLUGIN_INFO = PluginInfo(
        id=PLUGIN_ID,
        name=PLUGIN_NAME,
        version="0.1.0",
        description=PLUGIN_DESCRIPTION,
{_render_capabilities(project)}
    )
{methods}'''


def get_default_plugin_folder() -> Path:
    """Return the user plugin directory, which DataLab always scans."""
    return Path(get_config_path("plugins"))


def is_default_plugin_folder(folder: Path) -> bool:
    """Return True if a folder is the user plugin directory."""

    def normalized(path: Path) -> str:
        return osp.normcase(osp.realpath(str(path)))

    return normalized(folder) == normalized(get_default_plugin_folder())


def create_single_file_plugin(project: PluginProject) -> Path:
    """Write a single-file plugin and return its path.

    The ``<package>.py`` file is written to ``project.destination``, created if
    needed. An existing file is never overwritten.

    Args:
        project: Plugin configuration

    Returns:
        Path of the plugin file

    Raises:
        FileExistsError: The plugin file already exists
        ValueError: The module name does not start with ``datalab_``
    """
    source = render_single_file_plugin(project)
    folder = project.destination.resolve()
    path = folder / f"{project.package}.py"
    if path.exists():
        raise FileExistsError(_("File already exists: %s") % path)
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(source)
    return path


def _render_workflow_init() -> str:
    """Render the headless workflow public API."""
    return '''"""Headless plugin workflows."""

from .recipes import RECIPES

__all__ = ["RECIPES"]
'''


def _render_workflow_recipes() -> str:
    """Render an empty headless recipe registry."""
    return '''"""Headless recipe registry."""

from __future__ import annotations

from datalab.plugins.recipes import RecipeDescriptor

RECIPES: tuple[RecipeDescriptor, ...] = ()

__all__ = ["RECIPES"]
'''


def _render_web_adapter() -> str:
    """Render an explicit unsupported Web boundary."""
    return '''"""Reserved DataLab-Web integration boundary."""

WEB_STATUS = "unsupported"

__all__ = ["WEB_STATUS"]
'''


def _render_test(project: PluginProject) -> str:
    """Render a headless descriptor contract test."""
    expected_capabilities = "\n".join(
        f"            PluginCapability.{capability.upper()},"
        for capability in project.capabilities
    )
    return f'''"""Contract tests for the generated DataLab plugin."""

from datalab.plugins import PluginCapability

from {project.package}.adapters import desktop


def test_plugin_descriptor() -> None:
    """The generated entry point exposes stable SDK metadata."""
    plugin_class = desktop.{project.class_name}
    assert plugin_class.get_plugin_id() == "{project.plugin_id}"
    assert plugin_class.PLUGIN_INFO.version == "0.1.0"
    assert plugin_class.PLUGIN_INFO.capabilities == frozenset(
        {{
{expected_capabilities}
        }}
    )
'''


_ARCHITECTURE_TEST = '''"""Dependency rules for host-independent plugin layers."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "__PACKAGE__"
PACKAGE_NAME = "__PACKAGE__"
HOST_MODULES = (
    "PyQt5",
    "PyQt6",
    "PySide6",
    "datalab.gui",
    f"{PACKAGE_NAME}.adapters",
    "js",
    "pyodide",
    "qtpy",
)
FORBIDDEN_LAYER_IMPORTS = {
    "core": ("datalab", f"{PACKAGE_NAME}.workflow"),
    "workflow": (),
}


def _module_imports(filename: Path) -> set[str]:
    """Return direct imports declared by one Python source file."""
    tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    relative_parts = filename.relative_to(PACKAGE_ROOT).with_suffix("").parts
    package_parts = (PACKAGE_NAME, *relative_parts[:-1])
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    imports.add(node.module)
                continue
            base_parts = package_parts[: len(package_parts) - node.level + 1]
            if node.module is not None:
                imports.add(".".join((*base_parts, *node.module.split("."))))
            else:
                imports.update(
                    ".".join((*base_parts, *alias.name.split(".")))
                    for alias in node.names
                )
    return imports


def test_package_root_does_not_import_host_adapters() -> None:
    """Importing package identity must not load a host adapter."""
    imports = _module_imports(PACKAGE_ROOT / "__init__.py")
    assert not any(name.startswith(f"{PACKAGE_NAME}.adapters") for name in imports)


def test_import_scanner_resolves_relative_imports() -> None:
    """Relative imports cannot bypass the architecture checks."""
    imports = _module_imports(PACKAGE_ROOT / "workflow" / "__init__.py")
    assert f"{PACKAGE_NAME}.workflow.recipes" in imports


def test_headless_layers_do_not_import_host_modules() -> None:
    """Core and workflow remain independent from Desktop and Web hosts."""
    violations: list[str] = []
    for layer in ("core", "workflow"):
        forbidden = HOST_MODULES + FORBIDDEN_LAYER_IMPORTS[layer]
        for filename in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
            for imported_module in sorted(_module_imports(filename)):
                if any(
                    imported_module == prefix
                    or imported_module.startswith(f"{prefix}.")
                    for prefix in forbidden
                ):
                    violations.append(
                        f"{filename.relative_to(PACKAGE_ROOT)} imports "
                        f"{imported_module}"
                    )
    assert not violations, "\\n".join(violations)
'''


def _render_architecture_test(project: PluginProject) -> str:
    """Render dependency checks learned from the Camera pilot."""
    return _ARCHITECTURE_TEST.replace("__PACKAGE__", project.package)


def _render_readme(project: PluginProject) -> str:
    """Render concise setup and validation instructions."""
    return f"""# {project.name}

{project.description}

## Trying the plugin in DataLab

Add this project folder in **Plugins > Configure plugins... > Advanced**,
then reload the plugins: DataLab loads the plugin from
`src/{project.package}/adapters/desktop.py`, without installing the project.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m ruff check .
```

Installing the project also registers `{project.plugin_id}` through the
`datalab.plugins` entry-point group, as the published wheel does. Put
host-independent algorithms in `core`, compose them into headless recipes in
`workflow`, and keep DataLab or browser integration in `adapters`. The generated
architecture test preserves these dependency boundaries as the plugin grows.

## Publishing

Set the new version in `pyproject.toml` and in the package `__init__.py`, then
push a tag such as `v0.1.0`. The **Release** workflow builds the wheel and
attaches it to a GitHub release: DataLab users install it with
**Plugins > Configure plugins... > From file**.

To list the plugin in the DataLab plugin catalog, open a pull request on
<https://github.com/DataLab-Platform/plugins> with the entry printed in the
summary of the **Release** workflow run.
"""


_RELEASE_WORKFLOW = """name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: python -m pip install build

      - run: python -m build --wheel

      - name: Attach the wheel to a GitHub release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          wheel=$(ls dist/*-py3-none-any.whl)
          case "$wheel" in
            *-"${GITHUB_REF_NAME#v}"-py3-none-any.whl) ;;
            *) echo "Tag $GITHUB_REF_NAME does not match $wheel" >&2; exit 1 ;;
          esac
          gh release create "$GITHUB_REF_NAME" "$wheel" --verify-tag --generate-notes

      - name: Print the plugin catalog entry
        run: |
          sha256=$(sha256sum dist/*-py3-none-any.whl | cut -d " " -f 1)
          {
            echo "To list this release in the DataLab plugin catalog, add"
            echo "\\`plugins/__PLUGIN_ID__.yaml\\` to"
            echo "https://github.com/DataLab-Platform/plugins"
            echo "(or only the new release if the plugin is already listed):"
            echo
            echo '```yaml'
            cat <<'EOF'
          id: __PLUGIN_ID__
          name: __PLUGIN_NAME__
          EOF
            echo "repository: $GITHUB_SERVER_URL/$GITHUB_REPOSITORY"
            echo "capabilities: [__CAPABILITIES__]"
            echo "releases:"
            echo "  - version: ${GITHUB_REF_NAME#v}"
            echo "    sha256: $sha256"
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
"""


def _render_release_workflow(project: PluginProject) -> str:
    """Render the workflow publishing tagged wheels as GitHub releases."""
    return (
        _RELEASE_WORKFLOW.replace("__PLUGIN_ID__", project.plugin_id)
        .replace("__PLUGIN_NAME__", _toml_string(project.name))
        .replace("__CAPABILITIES__", ", ".join(project.capabilities))
    )


def _render_architecture_doc(project: PluginProject) -> str:
    """Render the generated dependency-boundary documentation."""
    return f"""# Architecture

`{project.package}` uses inward-facing dependencies:

```text
adapters -> workflow -> core
```

- `core` owns host-independent domain behavior and does not import DataLab.
- `workflow` may use DataLab's headless recipe contracts, but not GUI modules.
- `adapters/desktop.py` defines the DataLab plugin class: DataLab loads it from
  the project folder, and the installed entry point targets it.
- `adapters/web.py` records Web support explicitly and starts as unsupported.

The package root exposes identity metadata without importing a host adapter.
`tests/unit/test_architecture.py` checks these boundaries as the project grows.
"""


def _render_contributing() -> str:
    """Render concise development and architecture rules."""
    return """# Contributing

To try a change in DataLab, add the project folder in **Plugins > Configure
plugins... > Advanced** and reload the plugins: no installation is needed.

Install the project in editable mode and run its checks before submitting a
change:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m ruff check .
```

Keep domain algorithms in `core`, headless orchestration in `workflow`, and
host-specific behavior in `adapters`. Add focused tests at the same layer as
the behavior under test.
"""


def _render_changelog() -> str:
    """Render the initial project changelog."""
    return """# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Establish the independent DataLab plugin package and layered architecture.
"""


_BSD_3_CLAUSE = """BSD 3-Clause License

Copyright (c) Plugin authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

_GITIGNORE = """__pycache__/
.pytest_cache/
.ruff_cache/
build/
dist/
*.egg-info/
"""


def create_plugin_project(project: PluginProject) -> Path:
    """Create and return a layered plugin project directory."""
    destination = project.destination.resolve()
    if destination.exists():
        raise FileExistsError(_("Destination already exists: %s") % destination)
    files = {
        ".github/workflows/release.yml": _render_release_workflow(project),
        ".gitignore": _GITIGNORE,
        "CHANGELOG.md": _render_changelog(),
        "CONTRIBUTING.md": _render_contributing(),
        "LICENSE": _BSD_3_CLAUSE,
        "README.md": _render_readme(project),
        "doc/architecture.md": _render_architecture_doc(project),
        "pyproject.toml": _render_pyproject(project),
        f"src/{project.package}/__init__.py": _render_init(project),
        f"src/{project.package}/adapters/__init__.py": '"""Host adapters."""\n',
        f"src/{project.package}/adapters/desktop.py": _render_plugin(project),
        f"src/{project.package}/adapters/web.py": _render_web_adapter(),
        f"src/{project.package}/core/__init__.py": (
            '"""Host-independent domain code."""\n'
        ),
        f"src/{project.package}/workflow/__init__.py": _render_workflow_init(),
        f"src/{project.package}/workflow/recipes.py": _render_workflow_recipes(),
        "tests/integration/test_desktop_adapter.py": _render_test(project),
        "tests/unit/test_architecture.py": _render_architecture_test(project),
    }
    destination.mkdir(parents=True)
    for relative_path, content in files.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(content)
    return destination


def _default_package(name: str) -> str:
    """Derive a valid import package from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", _strip_accents(name).lower()).strip("_")
    if not slug.startswith("datalab_"):
        slug = f"datalab_{slug}"
    return slug


def _default_plugin_id(package: str, github_account: str = "") -> str:
    """Derive a reverse-domain plugin ID from an import package and its owner."""
    slug = package.removeprefix("datalab_").replace("_", "-")
    if github_account.lower() == OFFICIAL_GITHUB_ACCOUNT.lower():
        return f"org.datalab.{slug}"
    if github_account:
        return f"io.github.{github_account.lower()}.{slug}"
    return f"org.example.{slug}"


def _prompt(label: str, default: str | None = None) -> str:
    """Prompt for a value, displaying and accepting an optional default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(_("%s is required") % label)


def _project_from_args(args: argparse.Namespace) -> PluginProject:
    """Resolve explicit options and interactive defaults into a project.

    A single-file plugin only asks for its name and description: its module
    name, plugin ID and destination are derived, unless given as options.
    """
    name = args.name or _prompt(_("Plugin name"))
    github_account = args.github_account
    if not args.project:
        github_account = github_account or ""
        package = args.package or _default_package(name)
        plugin_id = args.plugin_id or _default_plugin_id(package, github_account)
    else:
        if github_account is None:
            github_account = (
                _prompt(_("GitHub account (optional)"), "")
                if args.plugin_id is None
                else ""
            )
        package = args.package or _prompt(_("Package name"), _default_package(name))
        plugin_id = args.plugin_id or _prompt(
            _("Plugin ID"), _default_plugin_id(package, github_account)
        )
    description = args.description or _prompt(
        _("Description"), _("%s plugin for DataLab") % name
    )
    destination = args.destination
    if destination is None:
        if args.project:
            destination = Path(_prompt(_("Destination"), package.replace("_", "-")))
        else:
            destination = get_default_plugin_folder()
    capabilities = tuple(dict.fromkeys(args.capabilities or ("processing",)))
    return PluginProject(
        destination=destination,
        name=name,
        package=package,
        plugin_id=plugin_id,
        description=description,
        capabilities=capabilities,
        object_kind=args.object_kind,
        license_id=args.license_id,
        github_account=github_account,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``datalab-plugin`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="datalab-plugin",
        description=_("Create and maintain DataLab plugins."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser(
        "create",
        help=_(
            "Create a single-file plugin, or a layered installable project "
            "with --project."
        ),
    )
    create_parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help=_(
            "Folder receiving the plugin file (default: the user plugin "
            "directory), or project folder with --project"
        ),
    )
    create_parser.add_argument(
        "--project",
        action="store_true",
        help=_("Create a layered installable project, to publish the plugin"),
    )
    create_parser.add_argument("--name", help=_("Plugin display name"))
    create_parser.add_argument("--package", help=_("Python import package name"))
    create_parser.add_argument("--plugin-id", help=_("Stable reverse-domain ID"))
    create_parser.add_argument("--description", help=_("One-line description"))
    create_parser.add_argument(
        "--github-account",
        help=_("GitHub user or organization hosting the project"),
    )
    create_parser.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        choices=_CAPABILITIES,
        help=_("Plugin capability; may be repeated"),
    )
    create_parser.add_argument(
        "--object-kind",
        choices=("signal", "image"),
        default="signal",
        help=_("Object type used by the sample processing"),
    )
    create_parser.add_argument(
        "--license",
        dest="license_id",
        choices=("BSD-3-Clause",),
        default="BSD-3-Clause",
        help=_("Project license (with --project)"),
    )
    catalog_parser = subparsers.add_parser(
        "catalog", help=_("Prepare, build or check a local plugin catalog folder.")
    )
    catalog_subparsers = catalog_parser.add_subparsers(
        dest="catalog_command", required=True
    )
    for name, text in (
        ("build", _("Write the catalog.json file listing the plugin files.")),
        ("check", _("Check the catalog.json file against the plugin files.")),
        (
            "prepare",
            _("Download dependency wheels for this host and build the local catalog."),
        ),
    ):
        subparser = catalog_subparsers.add_parser(name, help=text)
        subparser.add_argument(
            "directory", type=Path, help=_("Local plugin catalog folder")
        )
    return parser


def run_catalog_command(args: argparse.Namespace) -> int:
    """Build or check a local plugin catalog and print the outcome.

    Returns:
        Exit code: 0 on success, 1 if the catalog has problems
    """
    try:
        if args.catalog_command == "build":
            report = build_local_catalog(str(args.directory))
        elif args.catalog_command == "prepare":
            report = prepare_local_catalog(str(args.directory))
        else:
            report = check_local_catalog(str(args.directory))
    except (OSError, ValueError) as exc:
        print(_("Error: %s") % exc, file=sys.stderr)
        return 1
    for message in report.warnings:
        print(_("Warning: %s") % message)
    for message in report.errors:
        print(_("Error: %s") % message, file=sys.stderr)
    if report.errors:
        if args.catalog_command == "build":
            print(_("%s was not written.") % report.path, file=sys.stderr)
        return 1
    count = len(report.catalog["plugins"])
    if report.written:
        print(_("Wrote %s, listing %d plugins.") % (report.path, count))
    else:
        print(_("The local plugin catalog lists %d plugins.") % count)
    if report.dependency_wheels:
        print(_("Dependency artifacts for the public catalog:"))
        print("dependency_wheels:")
        for artifact in report.dependency_wheels:
            print(f"  - filename: {artifact['filename']}")
            print(f"    url: {artifact['url']}")
            print(f"    sha256: {artifact['sha256']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``datalab-plugin`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "catalog":
        return run_catalog_command(args)
    try:
        project = _project_from_args(args)
        if args.project:
            destination = create_plugin_project(project)
        else:
            destination = create_single_file_plugin(project)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    if args.project:
        print(_("Created DataLab plugin project at %s") % destination)
        return 0
    print(_("Created DataLab plugin %s at %s") % (project.plugin_id, destination))
    if is_default_plugin_folder(destination.parent):
        print(_("Select Plugins > Reload plugins in DataLab, or restart it."))
    else:
        print(
            _(
                "Add %s to the plugin directories (Plugins > Configure plugins... "
                "> Advanced), then reload the plugins."
            )
            % destination.parent
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
