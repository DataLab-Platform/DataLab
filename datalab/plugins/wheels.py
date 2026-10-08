# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin wheel inspection
-----------------------

Validate a plugin wheel without importing its Python code. DataLab-Web keeps a
verbatim copy of this module: host-specific values are passed as arguments.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import io
import pathlib
import re
import sys
import sysconfig
import zipfile
from collections.abc import Mapping
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

DESKTOP_ENTRY_POINT_GROUP = "datalab.plugins"
WEB_ENTRY_POINT_GROUP = "datalab.web_plugins"
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

_ENTRY_POINT_PATTERN = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<attribute>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib", ".wasm")
_RESERVED_TOP_LEVEL = frozenset(
    {
        "datalab",
        "guidata",
        "h5py",
        "js",
        "matplotlib",
        "micropip",
        "numpy",
        "packaging",
        "pandas",
        "PIL",
        "plotpy",
        "pyodide",
        "PyQt5",
        "PyQt6",
        "PySide6",
        "qtpy",
        "qwt",
        "scipy",
        "sigima",
        "sigimax",
        "skimage",
    }
)


class WheelInspectionError(ValueError):
    """Raised when an archive cannot be accepted as a plugin wheel."""


def _is_stdlib_module(name: str) -> bool:
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        return name in stdlib_names
    # Python 3.9 has no stdlib module list: locate the module instead.
    if name in sys.builtin_module_names:
        return True
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return False
    stdlib_dir = pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()
    origin = pathlib.Path(spec.origin).resolve()
    return stdlib_dir in origin.parents and "site-packages" not in origin.parts


def _read_bytes(path_or_bytes: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes, "rb") as file:
            data = file.read(MAX_WHEEL_BYTES + 1)
    elif isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        data = bytes(path_or_bytes)
    else:
        raise TypeError("Wheel input must be a path or bytes")
    if not data.startswith(b"PK"):
        raise WheelInspectionError("Plugin wheel is not a ZIP archive")
    if len(data) > MAX_WHEEL_BYTES:
        raise WheelInspectionError(
            f"Plugin wheel exceeds the {MAX_WHEEL_BYTES} byte size limit"
        )
    return data


def _safe_archive_entries(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    entries = tuple(archive.infolist())
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise WheelInspectionError("Plugin wheel contains too many archive entries")
    total_size = 0
    normalized_names: set[str] = set()
    for entry in entries:
        name = entry.filename.replace("\\", "/")
        path = pathlib.PurePosixPath(name)
        if not name or name.startswith("/") or ".." in path.parts:
            raise WheelInspectionError(f"Unsafe path in plugin wheel: {name!r}")
        normalized = str(path)
        if normalized in normalized_names:
            raise WheelInspectionError(
                f"Duplicate path in plugin wheel: {normalized!r}"
            )
        normalized_names.add(normalized)
        total_size += entry.file_size
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise WheelInspectionError(
                "Plugin wheel exceeds the uncompressed size limit"
            )
        if not entry.is_dir() and normalized.lower().endswith(_NATIVE_SUFFIXES):
            raise WheelInspectionError(
                f"Native payloads are not supported in plugin wheels: {normalized!r}"
            )
    return entries


def _single_dist_info(entries: tuple[zipfile.ZipInfo, ...]) -> str:
    directories = {
        entry.filename.replace("\\", "/").split("/", maxsplit=1)[0]
        for entry in entries
        if ".dist-info/" in entry.filename.replace("\\", "/")
    }
    directories = {name for name in directories if name.endswith(".dist-info")}
    if len(directories) != 1:
        raise WheelInspectionError(
            "Plugin wheel must contain exactly one .dist-info directory"
        )
    return directories.pop()


def _read_required_file(
    archive: zipfile.ZipFile, entries: tuple[zipfile.ZipInfo, ...], name: str
) -> bytes:
    names = {entry.filename.replace("\\", "/"): entry for entry in entries}
    try:
        entry = names[name]
    except KeyError as exc:
        raise WheelInspectionError(f"Plugin wheel is missing {name!r}") from exc
    return archive.read(entry)


def _parse_entry_points(data: bytes, group: str) -> list[dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string(data.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise WheelInspectionError("Invalid wheel entry_points.txt") from exc
    if not parser.has_section(group):
        raise WheelInspectionError(f"Plugin wheel has no {group!r} entry point")
    entry_points: list[dict[str, str]] = []
    for name, target in parser.items(group):
        normalized_name = name.strip()
        normalized_target = target.strip()
        if not normalized_name or not normalized_name.isidentifier():
            raise WheelInspectionError(f"Invalid plugin entry point name: {name!r}")
        match = _ENTRY_POINT_PATTERN.fullmatch(normalized_target)
        if match is None:
            raise WheelInspectionError(f"Invalid plugin entry point target: {target!r}")
        entry_points.append(
            {
                "name": normalized_name,
                "module": match.group("module"),
                "attribute": match.group("attribute"),
            }
        )
    return entry_points


def _top_level_packages(
    archive: zipfile.ZipFile,
    entries: tuple[zipfile.ZipInfo, ...],
    dist_info: str,
) -> list[str]:
    top_level_name = f"{dist_info}/top_level.txt"
    names = {entry.filename.replace("\\", "/") for entry in entries}
    if top_level_name in names:
        packages = {
            line.strip()
            for line in archive.read(top_level_name).decode("utf-8").splitlines()
            if line.strip()
        }
    else:
        packages = set()
        for entry in entries:
            if entry.is_dir():
                continue
            name = entry.filename.replace("\\", "/")
            if name.startswith(f"{dist_info}/"):
                continue
            top = name.split("/", maxsplit=1)[0]
            if "/" not in name:
                # Flat layout: a root-level module is itself a package.
                if top.endswith(".py"):
                    packages.add(top[: -len(".py")])
                continue
            if top.endswith(".data"):
                # Wheel payload directories are not importable packages.
                continue
            packages.add(top)
    for package in packages:
        if not package.isidentifier():
            raise WheelInspectionError(f"Invalid top-level package name: {package!r}")
        if package in _RESERVED_TOP_LEVEL or _is_stdlib_module(package):
            raise WheelInspectionError(
                f"Plugin wheel shadows a reserved package: {package!r}"
            )
    return sorted(packages)


def _validate_dependencies(
    requires_dist: list[str],
    available_distributions: Mapping[str, str],
    marker_environment: Mapping[str, str] | None,
    host_name: str,
) -> list[dict[str, Any]]:
    available: dict[str, Version] = {}
    for name, version in available_distributions.items():
        try:
            available[canonicalize_name(name)] = Version(version)
        except InvalidVersion:
            # Non-PEP 440 host versions cannot satisfy any specifier.
            continue
    environment = default_environment()
    environment.update(dict(marker_environment or {}))
    dependencies: list[dict[str, Any]] = []
    for raw_requirement in requires_dist:
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as exc:
            raise WheelInspectionError(
                f"Invalid wheel dependency: {raw_requirement!r}"
            ) from exc
        applies = requirement.marker is None or requirement.marker.evaluate(environment)
        key = canonicalize_name(requirement.name)
        installed = available.get(key)
        compatible = bool(
            not applies
            or (installed is not None and installed in requirement.specifier)
        )
        dependencies.append(
            {
                "requirement": str(requirement),
                "applies": applies,
                "installed_version": None if installed is None else str(installed),
                "compatible": compatible,
            }
        )
        if not compatible:
            if installed is None:
                detail = f"is not provided by {host_name}"
            else:
                detail = f"has incompatible installed version {installed}"
            raise WheelInspectionError(f"Plugin dependency {requirement!s} {detail}")
    return dependencies


def inspect_wheel(
    path_or_bytes: str | bytes | bytearray | memoryview,
    *,
    filename: str,
    entry_point_group: str,
    available_distributions: Mapping[str, str],
    host_name: str = "DataLab",
    python_version: str | None = None,
    marker_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly plugin wheel manifest without importing code.

    Args:
        path_or_bytes: Wheel file path or content
        filename: Wheel file name, which carries the distribution name and tags
        entry_point_group: Entry-point group the host loads plugins from
        available_distributions: Distribution versions provided by the host
        host_name: Host name used in dependency error messages
        python_version: Host Python version (default: running interpreter)
        marker_environment: Overrides of the environment markers

    Returns:
        Wheel manifest: identity, digest, entry points, packages and dependencies

    Raises:
        WheelInspectionError: The archive cannot be accepted as a plugin wheel
    """
    data = _read_bytes(path_or_bytes)
    if not filename.lower().endswith(".whl"):
        raise WheelInspectionError("Plugin archive filename must end with .whl")
    try:
        wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(filename)
    except (InvalidVersion, ValueError) as exc:
        raise WheelInspectionError(f"Invalid wheel filename: {filename!r}") from exc
    if any(tag.abi != "none" or tag.platform != "any" for tag in wheel_tags):
        raise WheelInspectionError("Only pure-Python *-none-any wheels are supported")

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise WheelInspectionError("Plugin wheel is not a valid ZIP archive") from exc
    with archive:
        entries = _safe_archive_entries(archive)
        dist_info = _single_dist_info(entries)
        wheel_metadata = BytesParser(policy=email_policy).parsebytes(
            _read_required_file(archive, entries, f"{dist_info}/WHEEL")
        )
        if wheel_metadata.get("Root-Is-Purelib", "").lower() != "true":
            raise WheelInspectionError(
                "Plugin wheel must declare Root-Is-Purelib: true"
            )
        declared_tags = wheel_metadata.get_all("Tag", [])
        if not declared_tags or any(
            not tag.endswith("-none-any") for tag in declared_tags
        ):
            raise WheelInspectionError("Plugin wheel declares a non-pure Python tag")

        package_metadata = BytesParser(policy=email_policy).parsebytes(
            _read_required_file(archive, entries, f"{dist_info}/METADATA")
        )
        distribution = package_metadata.get("Name")
        version = package_metadata.get("Version")
        if not distribution or canonicalize_name(distribution) != wheel_name:
            raise WheelInspectionError(
                "Wheel filename and METADATA distribution differ"
            )
        try:
            metadata_version = Version(version or "")
        except InvalidVersion as exc:
            raise WheelInspectionError(
                "Wheel METADATA contains an invalid version"
            ) from exc
        if metadata_version != wheel_version:
            raise WheelInspectionError("Wheel filename and METADATA version differ")

        required_python = package_metadata.get("Requires-Python")
        current_python = Version(
            python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        if required_python:
            try:
                specifier = SpecifierSet(required_python)
            except InvalidSpecifier as exc:
                raise WheelInspectionError(
                    "Wheel has invalid Requires-Python metadata"
                ) from exc
            if current_python not in specifier:
                raise WheelInspectionError(
                    f"Plugin requires Python {required_python}, "
                    f"host has {current_python}"
                )

        entry_points = _parse_entry_points(
            _read_required_file(archive, entries, f"{dist_info}/entry_points.txt"),
            entry_point_group,
        )
        packages = _top_level_packages(archive, entries, dist_info)
        dependencies = _validate_dependencies(
            package_metadata.get_all("Requires-Dist", []),
            available_distributions,
            marker_environment,
            host_name,
        )

    return {
        "filename": filename,
        "distribution": distribution,
        "version": str(metadata_version),
        "summary": package_metadata.get("Summary") or "",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "requires_python": required_python,
        "tags": sorted(str(tag) for tag in wheel_tags),
        "entry_points": entry_points,
        "top_level_packages": packages,
        "dependencies": dependencies,
    }


__all__ = [
    "DESKTOP_ENTRY_POINT_GROUP",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_UNCOMPRESSED_BYTES",
    "MAX_WHEEL_BYTES",
    "WEB_ENTRY_POINT_GROUP",
    "WheelInspectionError",
    "inspect_wheel",
]
