# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Package resources declared by plugins with the ``package:path`` syntax."""

from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from importlib.resources.abc import Traversable
    else:
        from importlib.abc import Traversable

__all__ = ["LOCAL_ID_PATTERN", "resolve_package_resource", "split_package_resource"]


#: Plugin-local identifier syntax (examples, welcome page tiles)
LOCAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_PACKAGE_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$",
    flags=re.ASCII,
)


def split_package_resource(
    resource: str, label: str = "Plugin resource"
) -> tuple[str, str]:
    """Validate a ``package:path`` resource and return its two parts

    Args:
        resource: resource declaration, e.g. ``"my_plugin:icons/logo.svg"``
        label: resource description used in error messages

    Returns:
        Importable package name and POSIX path relative to this package

    Raises:
        ValueError: if the declaration is malformed
    """
    if not isinstance(resource, str) or resource.count(":") != 1:
        raise ValueError(f"{label} must use 'package:path' syntax")
    package, path = resource.split(":")
    if not _PACKAGE_PATTERN.fullmatch(package):
        raise ValueError(f"{label} package is invalid")
    posix_path = PurePosixPath(path)
    if not path or "\\" in path or posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"{label} path must be relative")
    return package, path


def resolve_package_resource(
    resource: str, label: str = "Plugin resource"
) -> Traversable:
    """Resolve a ``package:path`` resource without requiring a filesystem path

    Args:
        resource: resource declaration, e.g. ``"my_plugin:icons/logo.svg"``
        label: resource description used in error messages

    Returns:
        ``importlib.resources`` traversable, also valid for zipped packages

    Raises:
        ValueError: if the declaration is malformed
        FileNotFoundError: if the package does not contain the resource
    """
    package, path = split_package_resource(resource, label)
    traversable = resources.files(package)
    for part in PurePosixPath(path).parts:
        traversable = traversable.joinpath(part)
    # Python 3.9's zipfile.Path.is_file() is also True for missing entries
    missing = sys.version_info < (3, 10) and not traversable.exists()
    if missing or not traversable.is_file():
        raise FileNotFoundError(f"{label} not found: {resource}")
    return traversable
