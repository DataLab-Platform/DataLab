# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Generate a layered installable DataLab plugin project."""

from __future__ import annotations

import argparse
import dataclasses
import json
import keyword
import re
from collections.abc import Sequence
from pathlib import Path

from datalab.config import _

_PACKAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PLUGIN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)+$")
_CAPABILITIES = ("application", "processing", "io", "visualization")


@dataclasses.dataclass(frozen=True)
class PluginProject:
    """Configuration for a generated plugin project."""

    destination: Path
    name: str
    package: str
    plugin_id: str
    description: str
    capabilities: tuple[str, ...] = ("processing",)
    object_kind: str = "signal"
    license_id: str = "BSD-3-Clause"

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

    @property
    def distribution(self) -> str:
        """Return the Python distribution name."""
        return self.package.replace("_", "-")

    @property
    def class_name(self) -> str:
        """Return the generated plugin class name."""
        words = re.findall(r"[A-Za-z0-9]+", self.name)
        class_name = "".join(word[0].upper() + word[1:] for word in words)
        if not class_name:
            raise ValueError(_("Plugin name must contain letters or digits"))
        if class_name[0].isdigit():
            class_name = f"Generated{class_name}"
        if not class_name.endswith("Plugin"):
            class_name += "Plugin"
        return class_name


def _toml_string(value: str) -> str:
    """Return *value* as a TOML-compatible quoted string."""
    return json.dumps(value, ensure_ascii=True)


def _render_pyproject(project: PluginProject) -> str:
    """Render generated project metadata and plugin entry point."""
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
requires-python = ">=3.9"
dependencies = ["datalab-platform >= 1.3"]

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
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I"]
'''


def _render_plugin(project: PluginProject) -> str:
    """Render the Desktop adapter and optional owned sample processing."""
    if len(project.capabilities) == 1:
        capabilities = (
            "        capabilities=(PluginCapability."
            f"{project.capabilities[0].upper()},),"
        )
    else:
        capability_items = "\n".join(
            f"            PluginCapability.{capability.upper()},"
            for capability in project.capabilities
        )
        capabilities = f"""        capabilities=(
{capability_items}
        ),"""
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

PLUGIN_DESCRIPTION = {_toml_string(project.description)}
PLUGIN_ID = "{project.plugin_id}"
PLUGIN_NAME = {_toml_string(project.name)}
__version__ = "0.1.0"

__all__ = [
    "PLUGIN_DESCRIPTION",
    "PLUGIN_ID",
    "PLUGIN_NAME",
    "__version__",
]
'''


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

from datalab.recipes import RecipeDescriptor

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

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m ruff check .
```

Installing the project registers `{project.plugin_id}` through the
`datalab.plugins` entry-point group. Put host-independent algorithms in `core`,
compose them into headless recipes in `workflow`, and keep DataLab or browser
integration in `adapters`. The generated architecture test preserves these
dependency boundaries as the plugin grows.
"""


def _render_architecture_doc(project: PluginProject) -> str:
    """Render the generated dependency-boundary documentation."""
    return f"""# Architecture

`{project.package}` uses inward-facing dependencies:

```text
adapters -> workflow -> core
```

- `core` owns host-independent domain behavior and does not import DataLab.
- `workflow` may use DataLab's headless recipe contracts, but not GUI modules.
- `adapters/desktop.py` is the installed DataLab plugin entry point.
- `adapters/web.py` records Web support explicitly and starts as unsupported.

The package root exposes identity metadata without importing a host adapter.
`tests/unit/test_architecture.py` checks these boundaries as the project grows.
"""


def _render_contributing() -> str:
    """Render concise development and architecture rules."""
    return """# Contributing

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
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug.startswith("datalab_"):
        slug = f"datalab_{slug}"
    return slug


def _default_plugin_id(package: str) -> str:
    """Derive a reverse-domain plugin ID from an import package."""
    slug = package.removeprefix("datalab_").replace("_", "-")
    return f"org.datalab.{slug}"


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
    """Resolve explicit options and interactive defaults into a project."""
    name = args.name or _prompt(_("Plugin name"))
    package = args.package or _prompt(_("Package name"), _default_package(name))
    plugin_id = args.plugin_id or _prompt(_("Plugin ID"), _default_plugin_id(package))
    description = args.description or _prompt(
        _("Description"), _("%s plugin for DataLab") % name
    )
    destination = args.destination
    if destination is None:
        destination = Path(_prompt(_("Destination"), package.replace("_", "-")))
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
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``datalab-plugin`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="datalab-plugin",
        description=_("Create and maintain DataLab plugin projects."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser(
        "create", help=_("Create a layered installable plugin project.")
    )
    create_parser.add_argument("destination", nargs="?", type=Path)
    create_parser.add_argument("--name", help=_("Plugin display name"))
    create_parser.add_argument("--package", help=_("Python import package name"))
    create_parser.add_argument("--plugin-id", help=_("Stable reverse-domain ID"))
    create_parser.add_argument("--description", help=_("One-line description"))
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
        help=_("Project license"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plugin project generator command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project = _project_from_args(args)
        destination = create_plugin_project(project)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(_("Created DataLab plugin project at %s") % destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
