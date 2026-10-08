# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Installed plugin store
----------------------

Plugins installed from a file by DataLab itself: pure-Python wheels, kept as
ZIP archives on ``sys.path``, and single ``datalab_*.py`` modules. This module
is Qt-free: the plugin host provides the store root and its own distributions.

Layout of the store root::

    index.json          installed plugins (written atomically)
    wheels/<sha256>.whl
    modules/datalab_<name>.py
"""

from __future__ import annotations

import dataclasses
import datetime
import gc
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import os.path as osp
import re
import sys
import zipimport
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from importlib.machinery import PathFinder

from packaging.utils import canonicalize_name

from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    MAX_WHEEL_BYTES,
    inspect_wheel,
)

INDEX_SCHEMA_VERSION = 1
MAX_MODULE_BYTES = 4 * 1024 * 1024
WHEEL_KIND = "wheel"
MODULE_KIND = "module"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODULE_PREFIX = "datalab_"


class PluginInstallError(ValueError):
    """Raised when a plugin cannot be installed, removed or listed."""


def _normalized(path: str) -> str:
    return osp.normcase(osp.realpath(path))


@dataclasses.dataclass(frozen=True)
class InstalledPlugin:
    """Plugin file installed in the store.

    Args:
        kind: ``"wheel"`` or ``"module"``
        name: Distribution name (wheel) or module name (module)
        version: Distribution version, empty for a module
        filename: Name of the file the plugin was installed from
        sha256: SHA-256 digest of the installed file
        size_bytes: Size of the installed file
        packages: Top-level packages or modules provided by the plugin
        installed_at: Installation date (ISO 8601, UTC)
    """

    kind: str
    name: str
    version: str
    filename: str
    sha256: str
    size_bytes: int
    packages: tuple[str, ...]
    installed_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", tuple(self.packages))
        if self.kind not in (WHEEL_KIND, MODULE_KIND):
            raise ValueError(f"Unknown installed plugin kind: {self.kind!r}")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError(f"Invalid SHA-256 digest: {self.sha256!r}")
        if self.kind == MODULE_KIND and not _is_module_name(self.name):
            raise ValueError(f"Invalid plugin module name: {self.name!r}")
        if not all(
            isinstance(name, str) and name.isidentifier() for name in self.packages
        ):
            raise ValueError(f"Invalid plugin packages: {self.packages!r}")

    def is_imported(self) -> bool:
        """Return True if this process already imported one of its packages."""
        return any(name in sys.modules for name in self.packages)


@dataclasses.dataclass(frozen=True)
class InstallResult:
    """Outcome of a plugin installation.

    Args:
        plugin: Installed plugin
        replaced: Previously installed version of the same plugin, if any
        restart_required: The plugin packages are already imported: the new
         version is only loaded at the next start
    """

    plugin: InstalledPlugin
    replaced: InstalledPlugin | None
    restart_required: bool


def _is_module_name(name: str) -> bool:
    return (
        name.startswith(_MODULE_PREFIX)
        and len(name) > len(_MODULE_PREFIX)
        and name.isidentifier()
    )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _remove_from_sys_path(path: str) -> None:
    normalized = _normalized(path)
    for entry in [entry for entry in sys.path if entry]:
        if _normalized(entry) == normalized:
            sys.path.remove(entry)
            sys.path_importer_cache.pop(entry, None)
            # zipimport caches archive listings by path, even after removal
            getattr(zipimport, "_zip_directory_cache", {}).pop(entry, None)


def _release_archives() -> None:
    """Close wheel archives kept open by import and metadata caches."""
    importlib.invalidate_caches()
    # Not cleared by importlib.invalidate_caches() on Python 3.10 and 3.11
    finder = getattr(importlib_metadata, "MetadataPathFinder", None)
    if hasattr(finder, "invalidate_caches"):
        finder().invalidate_caches()
    gc.collect()


class InstalledPluginStore:
    """Store of the plugins installed from a file.

    Args:
        root: Store directory
        host_distributions: Distribution versions provided by the host that the
         environment metadata may lack (e.g. when running from sources)
    """

    def __init__(
        self, root: str, *, host_distributions: Mapping[str, str] | None = None
    ) -> None:
        self.root = osp.realpath(root)
        self.wheels_dir = osp.join(self.root, "wheels")
        self.modules_dir = osp.join(self.root, "modules")
        self.index_path = osp.join(self.root, "index.json")
        self._host_distributions = dict(host_distributions or {})

    def get_plugins(self) -> list[InstalledPlugin]:
        """Return installed plugins, in installation order."""
        try:
            with open(self.index_path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            raise PluginInstallError(
                f"Unreadable installed plugin index: {self.index_path}"
            ) from exc
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != INDEX_SCHEMA_VERSION
            or not isinstance(data.get("plugins"), list)
        ):
            raise PluginInstallError(
                f"Unsupported installed plugin index: {self.index_path}"
            )
        try:
            return [InstalledPlugin(**item) for item in data["plugins"]]
        except (TypeError, ValueError) as exc:
            raise PluginInstallError(
                f"Invalid installed plugin index: {self.index_path}"
            ) from exc

    def get_path(self, plugin: InstalledPlugin) -> str:
        """Return the file path of an installed plugin."""
        if plugin.kind == WHEEL_KIND:
            return osp.join(self.wheels_dir, f"{plugin.sha256}.whl")
        return osp.join(self.modules_dir, f"{plugin.name}.py")

    def find_plugin(self, filepath: str) -> InstalledPlugin | None:
        """Return the installed plugin providing a module file, if any.

        Args:
            filepath: Module file path, possibly inside a wheel archive
        """
        normalized = _normalized(filepath)
        for plugin in self.get_plugins():
            path = _normalized(self.get_path(plugin))
            if normalized == path or normalized.startswith(path + os.sep):
                return plugin
        return None

    def get_available_distributions(self) -> dict[str, str]:
        """Return distribution versions a plugin wheel may depend on."""
        distributions: dict[str, str] = {}
        for distribution in importlib_metadata.distributions():
            name = distribution.metadata["Name"]
            if name:
                distributions.setdefault(name, distribution.version)
        distributions.update(self._host_distributions)
        return distributions

    def _get_environment_distributions(self) -> set[str]:
        """Return canonical names of distributions installed outside the store."""
        root = _normalized(self.root) + os.sep
        names: set[str] = set()
        for distribution in importlib_metadata.distributions():
            name = distribution.metadata["Name"]
            if not name:
                continue
            try:
                location = _normalized(str(distribution.locate_file("")))
            except (NotImplementedError, OSError, TypeError):
                location = ""
            if not (location + os.sep).startswith(root):
                names.add(canonicalize_name(name))
        return names

    def inspect_wheel_file(self, path: str) -> dict:
        """Return the manifest of a plugin wheel file without importing it.

        Raises:
            WheelInspectionError: The file is not an installable plugin wheel
        """
        return inspect_wheel(
            path,
            filename=osp.basename(path),
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions=self.get_available_distributions(),
        )

    def install_wheel(
        self, path: str, *, expected_sha256: str | None = None
    ) -> InstallResult:
        """Install a plugin wheel file.

        Args:
            path: Wheel file path
            expected_sha256: Digest of the inspected file the user agreed to
             install: installation fails if the file changed since

        Raises:
            WheelInspectionError: The file is not an installable plugin wheel
            PluginInstallError: The plugin conflicts with installed packages
        """
        with open(path, "rb") as file:
            data = file.read(MAX_WHEEL_BYTES + 1)
        manifest = inspect_wheel(
            data,
            filename=osp.basename(path),
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions=self.get_available_distributions(),
        )
        self._check_expected_digest(manifest["sha256"], expected_sha256)
        name = manifest["distribution"]
        key = canonicalize_name(name)
        if key in self._get_environment_distributions():
            raise PluginInstallError(
                f"{name} is already installed in the Python environment"
            )
        plugin = InstalledPlugin(
            kind=WHEEL_KIND,
            name=name,
            version=manifest["version"],
            filename=osp.basename(path),
            sha256=manifest["sha256"],
            size_bytes=manifest["size_bytes"],
            packages=manifest["top_level_packages"],
            installed_at=_now(),
        )
        replaced = self._add(
            plugin,
            data,
            lambda other: (
                other.kind == WHEEL_KIND and canonicalize_name(other.name) == key
            ),
        )
        return InstallResult(plugin, replaced, plugin.is_imported())

    def inspect_module_file(self, path: str) -> dict:
        """Return the manifest of a ``datalab_*.py`` plugin module file.

        Raises:
            PluginInstallError: Invalid file name or conflicting module name
        """
        name, data = self._read_module(path)
        return {
            "filename": osp.basename(path),
            "name": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    def install_module(
        self, path: str, *, expected_sha256: str | None = None
    ) -> InstallResult:
        """Install a single-file ``datalab_*.py`` plugin module.

        Args:
            path: Module file path
            expected_sha256: Digest of the file the user agreed to install

        Raises:
            PluginInstallError: Invalid file name or conflicting module name
        """
        name, data = self._read_module(path)
        sha256 = hashlib.sha256(data).hexdigest()
        self._check_expected_digest(sha256, expected_sha256)
        plugin = InstalledPlugin(
            kind=MODULE_KIND,
            name=name,
            version="",
            filename=osp.basename(path),
            sha256=sha256,
            size_bytes=len(data),
            packages=(name,),
            installed_at=_now(),
        )
        replaced = self._add(
            plugin,
            data,
            lambda other: other.kind == MODULE_KIND and other.name == name,
        )
        # Same size and mtime second: the bytecode cache would look valid
        try:
            os.remove(importlib.util.cache_from_source(self.get_path(plugin)))
        except OSError:
            pass
        # Discovery reloads an imported module from the replaced file
        return InstallResult(plugin, replaced, restart_required=False)

    def uninstall(self, plugin: InstalledPlugin) -> None:
        """Uninstall a plugin: it is no longer discovered after a reload.

        Raises:
            PluginInstallError: The plugin is not installed
        """
        plugins = self.get_plugins()
        if plugin not in plugins:
            raise PluginInstallError(f"{plugin.name} is not installed")
        self._write_index([other for other in plugins if other != plugin])
        path = self.get_path(plugin)
        if plugin.kind == WHEEL_KIND:
            _remove_from_sys_path(path)
        _release_archives()
        try:
            os.remove(path)
        except OSError:
            # Removed at next activation, once no process uses it anymore
            pass

    def activate(self) -> list[str]:
        """Make installed plugins importable and delete leftover files.

        A wheel whose packages this process already imported (e.g. an older
        version) is only activated at the next start.

        Returns:
            Paths added to ``sys.path``
        """
        plugins = self.get_plugins()
        self._collect_garbage(plugins)
        sys_path = {_normalized(entry) for entry in sys.path if entry}
        added: list[str] = []
        for plugin in plugins:
            path = self.get_path(plugin)
            if not osp.isfile(path):
                logging.getLogger(__name__).warning(
                    "Installed plugin file is missing: %s", path
                )
                continue
            if plugin.kind == MODULE_KIND:
                path = self.modules_dir
            if _normalized(path) in sys_path:
                continue
            if plugin.kind == WHEEL_KIND and plugin.is_imported():
                continue
            sys_path.add(_normalized(path))
            added.append(path)
        sys.path.extend(added)
        importlib.invalidate_caches()
        return added

    @staticmethod
    def _check_expected_digest(sha256: str, expected_sha256: str | None) -> None:
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise PluginInstallError("The file changed since it was inspected")

    def _read_module(self, path: str) -> tuple[str, bytes]:
        """Return the name and content of a valid plugin module file."""
        name, extension = osp.splitext(osp.basename(path))
        if extension != ".py" or not _is_module_name(name):
            raise PluginInstallError(
                "A plugin module file name must look like datalab_<name>.py"
            )
        with open(path, "rb") as file:
            data = file.read(MAX_MODULE_BYTES + 1)
        if len(data) > MAX_MODULE_BYTES:
            raise PluginInstallError(
                f"Plugin module exceeds the {MAX_MODULE_BYTES} byte size limit"
            )
        modules_dir = _normalized(self.modules_dir)
        search_path = [
            entry for entry in sys.path if entry and _normalized(entry) != modules_dir
        ]
        spec = PathFinder.find_spec(name, search_path)
        if spec is not None:
            raise PluginInstallError(
                f"A module named {name!r} already exists: {spec.origin}"
            )
        return name, data

    def _add(
        self, plugin: InstalledPlugin, data: bytes, same_plugin
    ) -> InstalledPlugin | None:
        plugins = self.get_plugins()
        replaced = next((other for other in plugins if same_plugin(other)), None)
        if replaced is not None and replaced.sha256 == plugin.sha256:
            raise PluginInstallError(f"{plugin.name} is already installed")
        self._write_file(self.get_path(plugin), data)
        self._write_index([other for other in plugins if other != replaced] + [plugin])
        return replaced

    def _write_index(self, plugins: list[InstalledPlugin]) -> None:
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "plugins": [dataclasses.asdict(plugin) for plugin in plugins],
        }
        self._write_file(self.index_path, json.dumps(payload, indent=2).encode())

    @staticmethod
    def _write_file(path: str, data: bytes) -> None:
        os.makedirs(osp.dirname(path), exist_ok=True)
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)

    def _collect_garbage(self, plugins: list[InstalledPlugin]) -> None:
        """Delete files no longer installed, unless still on ``sys.path``."""
        kept = {_normalized(self.get_path(plugin)) for plugin in plugins}
        kept.update(_normalized(entry) for entry in sys.path if entry)
        leftovers = [
            entry.path
            for directory in (self.wheels_dir, self.modules_dir)
            if osp.isdir(directory)
            for entry in os.scandir(directory)
            if entry.is_file() and _normalized(entry.path) not in kept
        ]
        if leftovers:
            _release_archives()
        for path in leftovers:
            try:
                os.remove(path)
            except OSError:
                pass


__all__ = [
    "InstallResult",
    "InstalledPlugin",
    "InstalledPluginStore",
    "PluginInstallError",
]
