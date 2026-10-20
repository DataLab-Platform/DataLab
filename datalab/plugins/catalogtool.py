# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Local plugin catalog tool
-------------------------

Build and check the ``catalog.json`` file of a local plugin catalog: a folder
holding plugin wheels, typically shared on the network. This module is Qt-free
and backs the ``datalab-plugin catalog`` command.
"""

from __future__ import annotations

import dataclasses
import json
import os
import os.path as osp
import tempfile
from collections.abc import Mapping

from packaging.utils import canonicalize_name

from datalab import __version__
from datalab.plugins.catalog import (
    CATALOG_FILENAME,
    MAX_CATALOG_BYTES,
    CatalogError,
    list_folder_wheels,
    load_catalog_json,
    read_folder_catalog,
    scan_wheel_folder,
)
from datalab.plugins.store import get_available_distributions
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    MAX_WHEEL_BYTES,
    WheelInspectionError,
    inspect_wheel,
)

#: Python versions supported by DataLab: a plugin wheel must install on one of
#: them (update when DataLab supports a new Python version)
DESKTOP_PYTHONS = ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14")
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


def _get_root(directory: str) -> str:
    root = osp.normpath(osp.abspath(directory))
    if not osp.isdir(root):
        raise CatalogError(f"Plugin catalog folder not found: {root}")
    return root


def inspect_plugin_wheel(path: str, available: Mapping[str, str]) -> dict:
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
                python_version=python_version,
                marker_environment={
                    "python_version": python_version,
                    "python_full_version": f"{python_version}.0",
                },
            )
        except WheelInspectionError as exc:
            error = exc
    raise error


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
    """Build the ``catalog.json`` file listing the plugin wheels of a folder.

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
    plugins: list[dict] = []
    for plugin in payload["plugins"]:
        releases = []
        for release in plugin["releases"]:
            path = osp.join(root, release["filename"])
            try:
                manifest = inspect_plugin_wheel(path, available)
            except (OSError, WheelInspectionError) as exc:
                report.errors.append(f"{path} refused: {exc}")
                continue
            release["requires_dist"] = [
                item["requirement"] for item in manifest["dependencies"]
            ]
            releases.append(release)
        if releases:
            plugin["releases"] = releases
            plugins.append(plugin)

    for previous in previous_plugins:
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
                manifest = inspect_plugin_wheel(release.url, available)
            except OSError as exc:
                report.errors.append(f"{label}: cannot read {release.url}: {exc}")
                continue
            except WheelInspectionError as exc:
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
        for wheel in list_folder_wheels(root)
        if osp.normcase(wheel) not in listed
    )
    return report


__all__ = [
    "DESKTOP_PYTHONS",
    "CatalogReport",
    "build_local_catalog",
    "check_local_catalog",
    "inspect_plugin_wheel",
]
