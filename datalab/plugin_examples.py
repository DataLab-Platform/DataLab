# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Portable declarations for plugin-provided example workspaces."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path, PurePosixPath

__all__ = ["PluginExample"]


_LOCAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_PACKAGE_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$",
    flags=re.ASCII,
)


@dataclasses.dataclass(frozen=True)
class PluginExample:
    """Packaged workspace and optional recipe exposed by a plugin."""

    id: str
    title: str
    resource: str
    description: str = ""
    recipe_id: str | None = None
    expected_checks: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate identity, package resource, and optional workflow metadata."""
        if not isinstance(self.id, str) or not _LOCAL_ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                "Plugin example ID must contain lowercase letters, digits, '.', "
                "'_' or '-'"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Plugin example title must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("Plugin example description must be a string")
        if not isinstance(self.resource, str) or self.resource.count(":") != 1:
            raise ValueError("Plugin example resource must use 'package:path' syntax")
        if not _PACKAGE_PATTERN.fullmatch(self.package):
            raise ValueError("Plugin example resource package is invalid")
        path = PurePosixPath(self.resource_path)
        if (
            not self.resource_path
            or "\\" in self.resource_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("Plugin example resource path must be relative")
        if self.recipe_id is not None and (
            not isinstance(self.recipe_id, str) or self.recipe_id.count(":") != 1
        ):
            raise ValueError("Plugin example recipe ID must be namespaced")
        if isinstance(self.expected_checks, (str, bytes)) or not isinstance(
            self.expected_checks, Sequence
        ):
            raise TypeError("Plugin example expected checks must be a sequence")
        checks = tuple(self.expected_checks)
        if not all(isinstance(check, str) and check.strip() for check in checks):
            raise ValueError("Plugin example expected checks must be non-empty strings")
        if len(checks) != len(set(checks)):
            raise ValueError("Plugin example expected checks must be unique")
        object.__setattr__(self, "expected_checks", checks)

    @property
    def package(self) -> str:
        """Return the importable package containing the example resource."""
        return self.resource.split(":", maxsplit=1)[0]

    @property
    def resource_path(self) -> str:
        """Return the POSIX-style path relative to the resource package."""
        return self.resource.split(":", maxsplit=1)[1]

    def resolve(self) -> Traversable:
        """Resolve the declared resource without requiring a filesystem path."""
        resource = resources.files(self.package)
        for part in PurePosixPath(self.resource_path).parts:
            resource = resource.joinpath(part)
        if not resource.is_file():
            raise FileNotFoundError(
                f"Plugin example resource not found: {self.resource}"
            )
        return resource

    @contextmanager
    def as_file(self) -> Iterator[Path]:
        """Materialize the resource for APIs that require a filesystem path."""
        with resources.as_file(self.resolve()) as filename:
            yield Path(filename)
