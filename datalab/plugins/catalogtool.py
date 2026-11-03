# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Local plugin catalog tool
-------------------------

Build and check the ``catalog.json`` file of a local plugin catalog: a folder
holding plugin artifacts, typically shared on the network. This module is Qt-free
and backs the ``datalab-plugin catalog`` command.
"""

from __future__ import annotations

import dataclasses
import json
import os
import os.path as osp
import tempfile
from collections.abc import Mapping, Sequence
from urllib.parse import quote

from packaging.utils import canonicalize_name

from datalab import __version__
from datalab.plugins.catalog import (
    CATALOG_FILENAME,
    MAX_CATALOG_BYTES,
    CatalogError,
    list_folder_modules,
    list_folder_wheels,
    load_catalog_json,
    read_folder_catalog,
    scan_wheel_folder,
)
from datalab.plugins.dependencies import DEPENDENCIES_DIRECTORY, download_dependencies
from datalab.plugins.modules import apply_catalog_metadata, inspect_module
from datalab.plugins.store import get_available_distributions
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    MAX_WHEEL_BYTES,
    WheelInspectionError,
    get_reference_distributions,
    inspect_dependency_wheel,
    inspect_wheel,
)

#: Python versions supported by DataLab: a plugin wheel must install on one of
#: them (update when DataLab supports a new Python version)
DESKTOP_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
#: Plugin fields edited in ``catalog.json`` and kept when building it again
CURATED_PLUGIN_FIELDS = (
    "id",
    "name",
    "tier",
    "status",
    "status_reason",
    "repository",
    "documentation",
    "capabilities",
    "keywords",
)


@dataclasses.dataclass
class CatalogReport:
    """Outcome of building or checking a local plugin catalog.

    Args:
        path: Path of the ``catalog.json`` file
        catalog: Catalog payload, built or read
        errors: Problems preventing DataLab from using the catalog as expected
        warnings: Other remarks
        written: Whether ``catalog.json`` was written
    """

    path: str
    catalog: dict = dataclasses.field(default_factory=dict)
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    written: bool = False
    dependency_wheels: list[dict] = dataclasses.field(default_factory=list)


def _get_root(directory: str) -> str:
    root = osp.normpath(osp.abspath(directory))
    if not osp.isdir(root):
        raise CatalogError(f"Plugin catalog folder not found: {root}")
    return root


def inspect_plugin_wheel(
    path: str,
    available: Mapping[str, str],
    additional_distributions: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Check a plugin wheel as DataLab does before installing it.

    The wheel must install on one of the Python versions supported by DataLab.

    Args:
        path: Wheel file path
        available: Distribution versions a plugin wheel may depend on

    Returns:
        Wheel manifest

    Raises:
        WheelInspectionError: DataLab would refuse to install the wheel
    """
    with open(path, "rb") as file:
        data = file.read(MAX_WHEEL_BYTES + 1)
    error = WheelInspectionError("No supported Python version")
    for python_version in DESKTOP_PYTHONS:
        try:
            return inspect_wheel(
                data,
                filename=osp.basename(path),
                entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
                available_distributions=available,
                additional_distributions=additional_distributions,
                python_version=python_version,
                marker_environment={
                    "python_version": python_version,
                    "python_full_version": f"{python_version}.0",
                },
            )
        except WheelInspectionError as exc:
            error = exc
    raise error


def _dependency_records(directory: str) -> tuple[list[dict], dict[str, list[str]]]:
    folder = osp.join(directory, DEPENDENCIES_DIRECTORY)
    if not osp.isdir(folder):
        return [], {}
    records: list[dict] = []
    versions: dict[str, list[str]] = {}
    for path in list_folder_wheels(folder):
        manifest = inspect_dependency_wheel(path, filename=osp.basename(path))
        records.append(
            {
                "filename": manifest["filename"],
                "url": f"{DEPENDENCIES_DIRECTORY}/{quote(manifest['filename'])}",
                "size": manifest["size_bytes"],
                "sha256": manifest["sha256"],
            }
        )
        name = canonicalize_name(manifest["distribution"])
        versions.setdefault(name, []).append(manifest["version"])
    return records, versions


def prepare_local_catalog(directory: str) -> CatalogReport:
    """Download missing dependency wheels for this host, then build the catalog."""
    root = _get_root(directory)
    report = CatalogReport(osp.join(root, CATALOG_FILENAME))
    available = get_reference_distributions()
    artifacts: dict[str, dict] = {}
    for path in list_folder_wheels(root):
        try:
            manifest = inspect_wheel(
                path,
                filename=osp.basename(path),
                entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
                available_distributions=available,
                allow_missing_dependencies=True,
            )
            missing = [
                item["requirement"]
                for item in manifest["dependencies"]
                if item["applies"] and not item["compatible"]
            ]
            if missing:
                for artifact in download_dependencies(
                    missing, osp.join(root, DEPENDENCIES_DIRECTORY), available
                ):
                    artifacts[artifact["sha256"]] = artifact
        except (OSError, ValueError) as exc:
            report.errors.append(f"{path}: {exc}")
    if report.errors:
        return report
    report = build_local_catalog(root)
    report.dependency_wheels = list(artifacts.values())
    return report


def _read_existing_plugins(path: str) -> list[dict]:
    """Return the plugins of an existing ``catalog.json``, if any."""
    if not osp.isfile(path):
        return []
    try:
        with open(path, "rb") as file:
            data = file.read(MAX_CATALOG_BYTES + 1)
    except OSError as exc:
        raise CatalogError(f"Cannot read {path}: {exc}") from exc
    plugins = load_catalog_json(data, path).get("plugins")
    if not isinstance(plugins, list) or not all(
        isinstance(plugin, dict) for plugin in plugins
    ):
        raise CatalogError(f"Invalid plugin catalog: {path}")
    return plugins


def _keep_curated_fields(plugin: dict, previous: dict) -> None:
    """Copy into a built plugin record the fields edited in the previous one."""
    for field in CURATED_PLUGIN_FIELDS:
        if field in previous:
            plugin[field] = previous[field]
    yanked = {
        str(release.get("version")): release["yanked"]
        for release in previous.get("releases", ())
        if isinstance(release, dict) and "yanked" in release
    }
    for release in plugin["releases"]:
        if release["version"] in yanked:
            release["yanked"] = yanked[release["version"]]
    if plugin.get("status") == "revoked":
        # As in the online catalog, revoked plugins cannot be installed
        plugin["releases"] = [
            {"version": release["version"], "sha256": release["sha256"]}
            for release in plugin["releases"]
        ]


def _write_atomically(path: str, catalog: dict) -> None:
    """Write a catalog file, which DataLab may read meanwhile."""
    directory = osp.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".catalog-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(catalog, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if osp.exists(temporary):
            os.remove(temporary)
        raise


def build_local_catalog(directory: str, *, write: bool = True) -> CatalogReport:
    """Build the ``catalog.json`` file listing plugin wheels and modules.

    Every wheel is checked as DataLab does before installing it. Fields edited
    in an existing ``catalog.json`` are kept: plugin ID, name, tier, status and
    its reason, repository, documentation, capabilities and keywords, and the
    reason why a release is yanked. Revoked plugins remain listed.

    The catalog is not written if a wheel is refused.

    Args:
        directory: Catalog folder
        write: Whether to write ``catalog.json``

    Raises:
        CatalogError: The folder or its existing catalog cannot be read
    """
    root = _get_root(directory)
    report = CatalogReport(osp.join(root, CATALOG_FILENAME))
    previous_plugins = _read_existing_plugins(report.path)
    payload, ignored = scan_wheel_folder(root)
    report.errors.extend(ignored)
    available = get_available_distributions({"datalab-platform": __version__})
    try:
        dependency_records, dependency_versions = _dependency_records(root)
        if dependency_records:
            available = get_reference_distributions()
    except (OSError, ValueError) as exc:
        report.errors.append(f"Dependency wheelhouse refused: {exc}")
        dependency_records = []
        dependency_versions = {}
    plugins: list[dict] = []
    for plugin in payload["plugins"]:
        releases = []
        previous = next(
            (
                item
                for item in previous_plugins
                if plugin.get("kind") == "module"
                and item.get("kind") == "module"
                and item.get("module") == plugin["module"]
            ),
            None,
        )
        for release in plugin["releases"]:
            path = osp.join(root, release["filename"])
            try:
                if plugin.get("kind") == "module":
                    manifest = inspect_module(path, filename=release["filename"])
                    if previous is not None:
                        old = previous.get("releases", [])
                        explicit_version = release["version"] or (
                            str(old[0].get("version", "")) if old else ""
                        )
                        manifest = apply_catalog_metadata(
                            manifest,
                            plugin_id=previous["id"],
                            name=previous["name"],
                            version=explicit_version,
                            summary=previous.get("summary", ""),
                        )
                        release["version"] = manifest["version"]
                        if old and old[0].get("requires_python"):
                            release["requires_python"] = old[0]["requires_python"]
                    report.warnings.extend(
                        f"{path}: {message}" for message in manifest["warnings"]
                    )
                    releases.append(release)
                    continue
                manifest = inspect_plugin_wheel(path, available, dependency_versions)
            except (OSError, ValueError) as exc:
                report.errors.append(f"{path} refused: {exc}")
                continue
            release["requires_dist"] = [
                item["requirement"] for item in manifest["dependencies"]
            ]
            if dependency_records:
                release["dependency_wheels"] = dependency_records
            releases.append(release)
        if releases:
            plugin["releases"] = releases
            if previous is not None:
                _keep_curated_fields(plugin, previous)
                for field in ("summary", "license"):
                    if field in previous:
                        plugin[field] = previous[field]
            plugins.append(plugin)

    for previous in previous_plugins:
        if previous.get("kind") == "module":
            matched = any(
                item.get("kind") == "module"
                and item.get("module") == previous.get("module")
                for item in plugins
            )
            if not matched and previous.get("status") == "revoked":
                plugins.append(previous)
            elif not matched:
                report.warnings.append(
                    f"{previous.get('name')}: no module left, removed from the catalog"
                )
            continue
        distribution = canonicalize_name(str(previous.get("distribution", "")))
        plugin = next(
            (item for item in plugins if distribution and item["id"] == distribution),
            None,
        )
        if plugin is not None:
            _keep_curated_fields(plugin, previous)
        elif previous.get("status") == "revoked":
            plugins.append(previous)
        else:
            report.warnings.append(
                f"{previous.get('name')}: no wheel left, removed from the catalog"
            )
    plugins.sort(key=lambda item: str(item.get("name", "")).casefold())
    report.catalog = {"schema_version": payload["schema_version"], "plugins": plugins}
    if write and not report.errors:
        try:
            _write_atomically(report.path, report.catalog)
        except OSError as exc:
            raise CatalogError(f"Cannot write {report.path}: {exc}") from exc
        report.written = True
    return report


def check_local_catalog(directory: str) -> CatalogReport:
    """Check that a local catalog folder may be used by DataLab.

    The wheels of the ``catalog.json`` file must exist, match their size and
    digest, and be accepted by DataLab; wheels of the folder missing from the
    catalog are reported. Without ``catalog.json``, DataLab lists the wheels of
    the folder, which are checked.

    Args:
        directory: Catalog folder

    Raises:
        CatalogError: The folder cannot be read
    """
    root = _get_root(directory)
    path = osp.join(root, CATALOG_FILENAME)
    if not osp.isfile(path):
        report = build_local_catalog(root, write=False)
        report.warnings = [
            f"No {CATALOG_FILENAME}: DataLab lists the plugin wheels of the folder"
        ]
        return report
    report = CatalogReport(path)
    try:
        plugins, _ignored = read_folder_catalog(root)
    except CatalogError as exc:
        report.errors.append(str(exc))
        return report
    with open(path, "rb") as file:
        report.catalog = load_catalog_json(file.read(MAX_CATALOG_BYTES + 1), path)
    available = get_available_distributions({"datalab-platform": __version__})
    listed: set[str] = set()
    for plugin in plugins:
        for release in plugin.releases:
            if not osp.isabs(release.url):
                continue
            label = f"{plugin.name} {release.version}"
            listed.add(osp.normcase(release.url))
            try:
                if plugin.kind == "module":
                    manifest = inspect_module(release.url, filename=release.filename)
                    apply_catalog_metadata(
                        manifest,
                        plugin_id=plugin.id,
                        name=plugin.name,
                        version=release.version,
                        summary=plugin.summary,
                    )
                    if (
                        manifest["size_bytes"] != release.size
                        or manifest["sha256"] != release.sha256
                    ):
                        raise ValueError(
                            f"{release.filename} does not match the size and digest "
                            "of the catalog"
                        )
                    continue
                release_available = dict(available)
                additional: dict[str, list[str]] = {}
                if release.dependency_wheels:
                    release_available = get_reference_distributions()
                for artifact in release.dependency_wheels:
                    if not osp.isabs(artifact.url):
                        report.warnings.append(
                            f"{label}: dependency is not stored locally: {artifact.url}"
                        )
                        continue
                    dependency = inspect_dependency_wheel(
                        artifact.url, filename=artifact.filename
                    )
                    if (
                        dependency["size_bytes"] != artifact.size
                        or dependency["sha256"] != artifact.sha256
                    ):
                        raise WheelInspectionError(
                            f"{artifact.filename} does not match the size and digest "
                            "of the catalog"
                        )
                    additional.setdefault(
                        canonicalize_name(dependency["distribution"]), []
                    ).append(dependency["version"])
                manifest = inspect_plugin_wheel(
                    release.url, release_available, additional
                )
            except OSError as exc:
                report.errors.append(f"{label}: cannot read {release.url}: {exc}")
                continue
            except ValueError as exc:
                report.errors.append(f"{label}: {release.url} refused: {exc}")
                continue
            if (
                manifest["size_bytes"] != release.size
                or manifest["sha256"] != release.sha256
            ):
                report.errors.append(
                    f"{label}: {release.url} does not match the size and digest "
                    "of the catalog"
                )
    report.warnings.extend(
        f"{wheel} is not listed in {CATALOG_FILENAME}"
        for wheel in [*list_folder_wheels(root), *list_folder_modules(root)]
        if osp.normcase(wheel) not in listed
    )
    return report


__all__ = [
    "DESKTOP_PYTHONS",
    "CatalogReport",
    "build_local_catalog",
    "check_local_catalog",
    "inspect_plugin_wheel",
    "prepare_local_catalog",
]
