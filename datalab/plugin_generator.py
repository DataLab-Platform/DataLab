# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Generate a minimal installable DataLab plugin project."""

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
{project.package} = "{project.package}.plugin:{project.class_name}"

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
    """Render the plugin descriptor and optional owned sample processing."""
    capabilities = "\n".join(
        f"            PluginCapability.{capability.upper()},"
        for capability in project.capabilities
    )
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
    return f'''"""DataLab plugin integration."""

from __future__ import annotations

{processing_import}from datalab.plugins import PluginBase, PluginCapability, PluginInfo


class {project.class_name}(PluginBase):
    """DataLab plugin entry point."""

    PLUGIN_INFO = PluginInfo(
        id="{project.plugin_id}",
        name={_toml_string(project.name)},
        version="0.1.0",
        description={_toml_string(project.description)},
        capabilities=(
{capabilities}
        ),
    )
{processing_method}
    def create_actions(self) -> None:
        """Create plugin actions after the DataLab panels are ready."""
'''


def _render_init(project: PluginProject) -> str:
    """Render the package public API."""
    return f'''"""DataLab plugin package."""

from {project.package}.plugin import {project.class_name}

__all__ = ["{project.class_name}"]
'''


def _render_test(project: PluginProject) -> str:
    """Render a headless descriptor contract test."""
    expected_capabilities = "\n".join(
        f"            PluginCapability.{capability.upper()},"
        for capability in project.capabilities
    )
    return f'''"""Contract tests for the generated DataLab plugin."""

from datalab.plugins import PluginCapability

from {project.package} import {project.class_name}


def test_plugin_descriptor() -> None:
    """The generated entry point exposes stable SDK metadata."""
    assert {project.class_name}.get_plugin_id() == (
        "{project.plugin_id}"
    )
    assert {project.class_name}.PLUGIN_INFO.version == "0.1.0"
    assert {project.class_name}.PLUGIN_INFO.capabilities == frozenset(
        {{
{expected_capabilities}
        }}
    )
'''


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
`datalab.plugins` entry-point group. Keep scientific code independent from Qt;
use the generated plugin class only as the DataLab integration adapter.
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
    """Create and return a minimal plugin project directory."""
    destination = project.destination.resolve()
    if destination.exists():
        raise FileExistsError(_("Destination already exists: %s") % destination)
    files = {
        ".gitignore": _GITIGNORE,
        "LICENSE": _BSD_3_CLAUSE,
        "README.md": _render_readme(project),
        "pyproject.toml": _render_pyproject(project),
        f"src/{project.package}/__init__.py": _render_init(project),
        f"src/{project.package}/plugin.py": _render_plugin(project),
        "tests/test_plugin.py": _render_test(project),
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
        "create", help=_("Create a minimal installable plugin project.")
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
