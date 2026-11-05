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
import io
import pathlib
import re
import sys
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from email.policy import default as email_policy
from importlib import metadata as importlib_metadata
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, compatible_tags, interpreter_name, parse_tag, sys_tags
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


#: Errors raised when reading a member of a damaged or unsupported ZIP archive
#: (corrupted headers or data, truncated data, encryption, unknown compression):
#: the central directory may be valid while a member is not.
ZIP_MEMBER_ERRORS = (
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    NotImplementedError,
    RuntimeError,
)


def _read_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo | str) -> bytes:
    try:
        return archive.read(member)
    except ZIP_MEMBER_ERRORS as exc:
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        raise WheelInspectionError(
            f"Plugin wheel member {name!r} cannot be read: {exc}"
        ) from exc


def _host_python(python_version: str | None) -> Version:
    return Version(
        python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    )


def _pure_python_tags(python: Version) -> frozenset[Tag]:
    """Return the pure-Python wheel tags a Python version installs, as pip does."""
    interpreter = f"{interpreter_name()}{python.major}{python.minor}"
    return frozenset(
        compatible_tags((python.major, python.minor), interpreter, ["any"])
    )


def is_compatible_wheel(
    filename: str, requires_python: str | None, python_version: str | None = None
) -> bool:
    """Return True if a Python version can install a pure-Python wheel.

    Args:
        filename: Wheel file name, which carries its tags
        requires_python: ``Requires-Python`` metadata of the wheel, if any
        python_version: Host Python version (default: running interpreter)
    """
    try:
        wheel_tags = parse_wheel_filename(filename)[3]
        python = _host_python(python_version)
        if requires_python and python not in SpecifierSet(requires_python):
            return False
    except (InvalidSpecifier, InvalidVersion, ValueError):
        return False
    return bool(wheel_tags & _pure_python_tags(python))


def _is_stdlib_module(name: str) -> bool:
    return name in sys.stdlib_module_names


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


def _safe_archive_entries(
    archive: zipfile.ZipFile, *, allow_native: bool = False
) -> tuple[zipfile.ZipInfo, ...]:
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
        if (
            not allow_native
            and not entry.is_dir()
            and normalized.lower().endswith(_NATIVE_SUFFIXES)
        ):
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
    return _read_member(archive, entry)


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
    *,
    allow_native: bool = False,
) -> list[str]:
    top_level_name = f"{dist_info}/top_level.txt"
    names = {entry.filename.replace("\\", "/") for entry in entries}
    if top_level_name in names:
        packages = {
            line.strip()
            for line in _read_member(archive, top_level_name)
            .decode("utf-8")
            .splitlines()
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
                elif allow_native and top.lower().endswith((".pyd", ".so")):
                    packages.add(top.split(".", maxsplit=1)[0])
                continue
            if top.endswith(".data"):
                if allow_native:
                    parts = name.split("/")
                    if len(parts) > 2 and parts[1] in ("purelib", "platlib"):
                        packages.add(parts[2].split(".", maxsplit=1)[0])
                continue
            if allow_native and top.endswith(".libs"):
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


def validate_dependencies(
    requires_dist: list[str],
    available_distributions: Mapping[str, str],
    marker_environment: Mapping[str, str] | None,
    host_name: str,
    *,
    allow_missing: bool = False,
    additional_distributions: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Check requirements without changing installed distribution versions."""
    provided_names = {canonicalize_name(name) for name in available_distributions}
    candidates = {
        canonicalize_name(name): tuple(Version(version) for version in versions)
        for name, versions in (additional_distributions or {}).items()
    }
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
        if applies and installed is None and key not in provided_names:
            compatible = any(
                version in requirement.specifier for version in candidates.get(key, ())
            )
        dependencies.append(
            {
                "requirement": str(requirement),
                "applies": applies,
                "installed_version": None if installed is None else str(installed),
                "compatible": compatible,
            }
        )
        if not compatible and not (allow_missing and installed is None):
            if installed is None:
                detail = f"is not provided by {host_name}"
            else:
                detail = f"has incompatible installed version {installed}"
            raise WheelInspectionError(f"Plugin dependency {requirement!s} {detail}")
    return dependencies


def _inspect_archive(
    path_or_bytes: str | bytes | bytearray | memoryview,
    *,
    filename: str,
    entry_point_group: str | None,
) -> dict[str, Any]:
    """Inspect archive structure and metadata, independently of the host."""
    plugin = entry_point_group is not None
    data = _read_bytes(path_or_bytes)
    if not filename.lower().endswith(".whl"):
        raise WheelInspectionError("Plugin archive filename must end with .whl")
    try:
        wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(filename)
    except (InvalidVersion, ValueError) as exc:
        raise WheelInspectionError(f"Invalid wheel filename: {filename!r}") from exc
    if plugin and any(tag.abi != "none" or tag.platform != "any" for tag in wheel_tags):
        raise WheelInspectionError("Only pure-Python *-none-any wheels are supported")

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise WheelInspectionError("Plugin wheel is not a valid ZIP archive") from exc
    with archive:
        entries = _safe_archive_entries(archive, allow_native=not plugin)
        dist_info = _single_dist_info(entries)
        wheel_metadata = BytesParser(policy=email_policy).parsebytes(
            _read_required_file(archive, entries, f"{dist_info}/WHEEL")
        )
        if plugin and wheel_metadata.get("Root-Is-Purelib", "").lower() != "true":
            raise WheelInspectionError(
                "Plugin wheel must declare Root-Is-Purelib: true"
            )
        declared_tags = wheel_metadata.get_all("Tag", [])
        if not declared_tags or (
            plugin and any(not tag.endswith("-none-any") for tag in declared_tags)
        ):
            raise WheelInspectionError("Plugin wheel declares a non-pure Python tag")
        if not plugin:
            if wheel_metadata.get("Root-Is-Purelib", "").lower() not in (
                "true",
                "false",
            ):
                raise WheelInspectionError("Wheel must declare Root-Is-Purelib")
            try:
                metadata_tags = {
                    tag
                    for declaration in declared_tags
                    for tag in parse_tag(declaration)
                }
            except ValueError as exc:
                raise WheelInspectionError("Wheel declares invalid tags") from exc
            if not wheel_tags.intersection(metadata_tags):
                raise WheelInspectionError("Wheel filename and WHEEL tags differ")

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
        if required_python:
            try:
                SpecifierSet(required_python)
            except InvalidSpecifier as exc:
                raise WheelInspectionError(
                    "Wheel has invalid Requires-Python metadata"
                ) from exc
        entry_points = (
            _parse_entry_points(
                _read_required_file(archive, entries, f"{dist_info}/entry_points.txt"),
                entry_point_group,
            )
            if entry_point_group is not None
            else []
        )
        packages = _top_level_packages(
            archive, entries, dist_info, allow_native=not plugin
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
        "requires_dist": package_metadata.get_all("Requires-Dist", []),
    }


def inspect_dependency_wheel(
    path_or_bytes: str | bytes | bytearray | memoryview, *, filename: str
) -> dict[str, Any]:
    """Inspect an ordinary dependency wheel, including native payloads.

    No code is imported. ``compatible`` describes the running interpreter;
    catalogs may also inspect wheels destined for other hosts.
    """
    manifest = _inspect_archive(
        path_or_bytes, filename=filename, entry_point_group=None
    )
    tags = parse_wheel_filename(filename)[3]
    required_python = manifest["requires_python"]
    manifest["compatible"] = bool(tags.intersection(sys_tags())) and (
        not required_python or _host_python(None) in SpecifierSet(required_python)
    )
    return manifest


def inspect_wheel(
    path_or_bytes: str | bytes | bytearray | memoryview,
    *,
    filename: str,
    entry_point_group: str,
    available_distributions: Mapping[str, str],
    host_name: str = "DataLab",
    python_version: str | None = None,
    marker_environment: Mapping[str, str] | None = None,
    allow_missing_dependencies: bool = False,
    additional_distributions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Inspect a pure-Python plugin wheel without importing its code.

    Args:
        path_or_bytes: Wheel file path or content
        filename: Wheel file name
        entry_point_group: Plugin entry-point group
        available_distributions: Distribution versions provided by the host
        host_name: Host name used in dependency error messages
        python_version: Host Python version (default: running interpreter)
        marker_environment: Overrides of the environment markers
        allow_missing_dependencies: Return missing requirements for planning;
         incompatible installed versions are still refused
        additional_distributions: Candidate versions in a catalog wheelhouse,
         used for metadata checks only; pip resolves the actual installation

    Returns:
        JSON-friendly plugin manifest
    """
    manifest = _inspect_archive(
        path_or_bytes, filename=filename, entry_point_group=entry_point_group
    )
    current_python = _host_python(python_version)
    required_python = manifest["requires_python"]
    if required_python and current_python not in SpecifierSet(required_python):
        raise WheelInspectionError(
            f"Plugin requires Python {required_python}, host has {current_python}"
        )
    if not is_compatible_wheel(filename, required_python, python_version):
        tags = parse_wheel_filename(filename)[3]
        interpreters = ".".join(sorted({tag.interpreter for tag in tags}))
        raise WheelInspectionError(
            f"Plugin wheel is built for {interpreters}, "
            f"host has Python {current_python}"
        )
    manifest["dependencies"] = validate_dependencies(
        manifest["requires_dist"],
        available_distributions,
        marker_environment,
        host_name,
        allow_missing=allow_missing_dependencies,
        additional_distributions=additional_distributions,
    )
    return manifest


def get_reference_distributions(root: str = "datalab-platform") -> dict[str, str]:
    """Snapshot the installed host's declared runtime dependency closure.

    Authoring tools must not mistake an already installed third-party plugin
    or its libraries for packages distributed with DataLab.
    """
    distributions: dict[str, importlib_metadata.Distribution] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            distributions.setdefault(canonicalize_name(name), distribution)
    pending = [(canonicalize_name(root), frozenset())]
    visited: dict[str, set[str]] = {}
    versions: dict[str, str] = {}
    while pending:
        name, extras = pending.pop()
        if name in visited and extras.issubset(visited[name]):
            continue
        if name not in distributions:
            raise WheelInspectionError(f"Host distribution is missing: {name}")
        distribution = distributions[name]
        enabled = visited.setdefault(name, set())
        enabled.update(extras)
        versions[name] = distribution.version
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or any(
                requirement.marker.evaluate({"extra": extra})
                for extra in {"", *enabled}
            ):
                pending.append(
                    (canonicalize_name(requirement.name), frozenset(requirement.extras))
                )
    return versions


__all__ = [
    "DESKTOP_ENTRY_POINT_GROUP",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_UNCOMPRESSED_BYTES",
    "MAX_WHEEL_BYTES",
    "WEB_ENTRY_POINT_GROUP",
    "WheelInspectionError",
    "get_reference_distributions",
    "inspect_dependency_wheel",
    "inspect_wheel",
    "is_compatible_wheel",
    "validate_dependencies",
]
