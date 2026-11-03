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
import importlib
import importlib.util
import json
import logging
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import zipimport
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from importlib.machinery import PathFinder

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name

from datalab.plugins.dependencies import (
    DEPENDENCIES_DIRECTORY,
    DependencyPlan,
    install_dependencies,
    installed_versions,
    plan_dependencies,
    validate_installed,
)
from datalab.plugins.modules import (
    MAX_MODULE_BYTES,
    ModuleInspectionError,
    apply_catalog_metadata,
    inspect_module,
    module_name,
)
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    MAX_WHEEL_BYTES,
    inspect_wheel,
)

INDEX_SCHEMA_VERSION = 1
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
        version: Distribution version, or known module version (otherwise empty)
        filename: Name of the file the plugin was installed from
        sha256: SHA-256 digest of the installed file
        size_bytes: Size of the installed file
        packages: Top-level packages or modules provided by the plugin
        installed_at: Installation date (ISO 8601, UTC)
        catalog_id: Catalog identity when installed from a catalog
    """

    kind: str
    name: str
    version: str
    filename: str
    sha256: str
    size_bytes: int
    packages: tuple[str, ...]
    installed_at: str
    catalog_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", tuple(self.packages))
        if self.kind not in (WHEEL_KIND, MODULE_KIND):
            raise ValueError(f"Unknown installed plugin kind: {self.kind!r}")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError(f"Invalid SHA-256 digest: {self.sha256!r}")
        if self.kind == MODULE_KIND:
            module_name(f"{self.name}.py")
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


def get_available_distributions(
    host_distributions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return distribution versions a plugin wheel may depend on.

    Args:
        host_distributions: Distribution versions provided by the host that the
         environment metadata may lack (e.g. when running from sources)
    """
    distributions: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            distributions.setdefault(name, distribution.version)
    distributions.update(host_distributions or {})
    return distributions


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
        self._blocked_dependency_paths: set[str] = set()
        self._deferred_plugin_digests: set[str] = set()

    def get_plugins(self) -> list[InstalledPlugin]:
        """Return installed plugins, in installation order."""
        data = self._read_index()
        try:
            return [InstalledPlugin(**item) for item in data["plugins"]]
        except (TypeError, ValueError) as exc:
            raise PluginInstallError(
                f"Invalid installed plugin index: {self.index_path}"
            ) from exc

    def _read_index(self) -> dict:
        try:
            with open(self.index_path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return {"schema_version": INDEX_SCHEMA_VERSION, "plugins": []}
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
        return data

    def get_dependency_paths(self) -> list[str]:
        """Return validated paths for this interpreter's shared dependencies."""
        roots = self._read_index().get("dependency_roots", [])
        if not isinstance(roots, list) or not all(
            isinstance(root, str) for root in roots
        ):
            raise PluginInstallError("Invalid installed dependency roots")
        directory = _normalized(osp.join(self.root, DEPENDENCIES_DIRECTORY))
        base = _normalized(self._dependency_parent())
        paths: list[str] = []
        for relative in roots:
            path = _normalized(osp.join(self.root, relative))
            if not path.startswith(directory + os.sep):
                raise PluginInstallError(
                    "Dependency directory is outside the installed plugin store: "
                    f"{path}"
                )
            if not path.startswith(base + os.sep):
                logging.getLogger(__name__).warning(
                    "Dependencies for another Python runtime are not activated: %s",
                    path,
                )
                continue
            paths.append(path)
        return paths

    def _dependency_parent(self) -> str:
        return osp.join(self.root, DEPENDENCIES_DIRECTORY, str(next(sys_tags())))

    def get_pending_enabled_names(self) -> list[str]:
        """Return newly installed distributions to enable after restarting."""
        names = self._read_index().get("pending_enabled", [])
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise PluginInstallError("Invalid pending plugin enablement")
        return names

    def clear_pending_enabled_names(self, names: list[str]) -> None:
        """Acknowledge distributions discovered after their installation."""
        self._write_index(
            self.get_plugins(),
            pending_enabled=[
                name for name in self.get_pending_enabled_names() if name not in names
            ],
        )

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
        return validate_installed(
            self.get_dependency_paths(),
            get_available_distributions(self._host_distributions),
        )

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

    def inspect_wheel_file(
        self, path: str, *, allow_missing_dependencies: bool = False
    ) -> dict:
        """Return the manifest of a plugin wheel file without importing it.

        Raises:
            WheelInspectionError: The file is not an installable plugin wheel
        """
        return inspect_wheel(
            path,
            filename=osp.basename(path),
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions=self.get_available_distributions(),
            allow_missing_dependencies=allow_missing_dependencies,
        )

    def prepare_wheel_file(
        self, path: str, wheelhouse: str | None = None
    ) -> tuple[dict, DependencyPlan | None]:
        """Inspect a plugin and plan missing dependencies from local wheels."""
        manifest = self.inspect_wheel_file(path, allow_missing_dependencies=True)
        missing = [
            dependency["requirement"]
            for dependency in manifest["dependencies"]
            if dependency["applies"] and not dependency["compatible"]
        ]
        if not missing:
            return manifest, None
        source = wheelhouse or osp.join(osp.dirname(path), DEPENDENCIES_DIRECTORY)
        plan = plan_dependencies(
            missing,
            source,
            self.get_available_distributions(),
            self.get_dependency_paths(),
        )
        manifest["additional_dependencies"] = [
            dataclasses.asdict(wheel) for wheel in plan.wheels
        ]
        manifest["dependency_source"] = source
        return manifest, plan

    def install_wheel(
        self,
        path: str,
        *,
        expected_sha256: str | None = None,
        dependency_plan: DependencyPlan | None = None,
        catalog_id: str = "",
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
        available = self.get_available_distributions()
        if dependency_plan is not None:
            if available != dependency_plan.protected_versions:
                raise PluginInstallError("Installed packages changed since planning")
            for wheel in dependency_plan.wheels:
                key = canonicalize_name(wheel.distribution)
                if key in available:
                    raise PluginInstallError(
                        f"Refusing to replace {wheel.distribution}"
                    )
                available[key] = wheel.version
        manifest = inspect_wheel(
            data,
            filename=osp.basename(path),
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions=available,
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
            catalog_id=catalog_id,
        )
        for other in self.get_plugins():
            if other.kind == MODULE_KIND and (
                set(plugin.packages).intersection(other.packages)
                or (catalog_id and other.catalog_id == catalog_id)
            ):
                raise PluginInstallError(
                    "The plugin conflicts with an installed module"
                )
        installed_directory: str | None = None
        success = False
        try:
            roots = None
            pending = None
            if dependency_plan is not None and dependency_plan.wheels:
                packages = set(plugin.packages)
                for other in self.get_plugins():
                    packages.update(other.packages)
                for wheel in dependency_plan.wheels:
                    for package in wheel.packages:
                        if (
                            package in packages
                            or PathFinder.find_spec(package) is not None
                        ):
                            raise PluginInstallError(
                                "Dependency package conflicts with an existing "
                                f"module: {package}"
                            )
                        packages.add(package)
                parent = self._dependency_parent()
                os.makedirs(parent, exist_ok=True)
                # Only the index publishes this generation; keep DLL paths short.
                installed_directory = tempfile.mkdtemp(prefix="", dir=parent)
                install_dependencies(dependency_plan, installed_directory)
                if (
                    self.get_available_distributions()
                    != dependency_plan.protected_versions
                ):
                    raise PluginInstallError(
                        "Installed packages changed during dependency installation"
                    )
                roots = [
                    *self._read_index().get("dependency_roots", []),
                    osp.relpath(installed_directory, self.root),
                ]
            blocked = set(self._blocked_dependency_paths)
            if installed_directory is not None:
                blocked.add(_normalized(installed_directory))
            pending_dependencies = installed_versions(list(blocked))
            restart = (
                plugin.is_imported()
                or installed_directory is not None
                or any(
                    dependency["applies"]
                    and canonicalize_name(Requirement(dependency["requirement"]).name)
                    in pending_dependencies
                    for dependency in manifest["dependencies"]
                )
            )
            if restart and not any(
                other.kind == WHEEL_KIND and canonicalize_name(other.name) == key
                for other in self.get_plugins()
            ):
                pending = [*self.get_pending_enabled_names(), key]
            replaced = self._add(
                plugin,
                data,
                lambda other: (
                    other.kind == WHEEL_KIND and canonicalize_name(other.name) == key
                ),
                dependency_roots=roots,
                pending_enabled=pending,
                repair=installed_directory is not None,
            )
            if installed_directory is not None:
                self._blocked_dependency_paths.add(_normalized(installed_directory))
            if restart:
                self._deferred_plugin_digests.add(plugin.sha256)
            success = True
            return InstallResult(plugin, replaced, restart)
        finally:
            if not success and installed_directory is not None:
                shutil.rmtree(installed_directory)

    def inspect_module_file(self, path: str) -> dict:
        """Return the manifest of a ``datalab_*.py`` plugin module file.

        Raises:
            PluginInstallError: Invalid file name or conflicting module name
        """
        name, data = self._read_module(path)
        return inspect_module(data, filename=f"{name}.py")

    def install_module(
        self,
        path: str,
        *,
        expected_sha256: str | None = None,
        catalog_id: str = "",
        catalog_name: str = "",
        catalog_version: str = "",
    ) -> InstallResult:
        """Install a single-file ``datalab_*.py`` plugin module.

        Args:
            path: Module file path
            expected_sha256: Digest of the file the user agreed to install

        Raises:
            PluginInstallError: Invalid file name or conflicting module name
        """
        name, data = self._read_module(path)
        manifest = inspect_module(data, filename=f"{name}.py")
        if catalog_id:
            manifest = apply_catalog_metadata(
                manifest,
                plugin_id=catalog_id,
                name=catalog_name,
                version=catalog_version,
            )
        sha256 = manifest["sha256"]
        self._check_expected_digest(sha256, expected_sha256)
        plugin = InstalledPlugin(
            kind=MODULE_KIND,
            name=name,
            version=manifest["version"],
            filename=osp.basename(path),
            sha256=sha256,
            size_bytes=len(data),
            packages=(name,),
            installed_at=_now(),
            catalog_id=catalog_id,
        )
        for other in self.get_plugins():
            if other.kind == MODULE_KIND and _normalized(
                self.get_path(other)
            ) == _normalized(self.get_path(plugin)):
                if other.name != name:
                    raise PluginInstallError(
                        "Module file name conflicts with an installed module"
                    )
                if other.catalog_id and catalog_id and other.catalog_id != catalog_id:
                    raise PluginInstallError(
                        f"Module {name} belongs to another catalog plugin"
                    )
            elif other.kind == WHEEL_KIND and name in other.packages:
                raise PluginInstallError("Module conflicts with an installed wheel")
            elif catalog_id and other.catalog_id == catalog_id:
                raise PluginInstallError(
                    "Catalog plugin uses another installed artifact"
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
        self._write_index(
            [other for other in plugins if other != plugin],
            pending_enabled=[
                name
                for name in self.get_pending_enabled_names()
                if name != canonicalize_name(plugin.name)
            ],
        )
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
        paths = self.get_dependency_paths()
        validate_installed(paths, get_available_distributions(self._host_distributions))
        added: list[str] = [
            path
            for path in paths
            if _normalized(path) not in sys_path
            and path not in self._blocked_dependency_paths
        ]
        sys_path.update(_normalized(path) for path in added)
        for plugin in plugins:
            if plugin.sha256 in self._deferred_plugin_digests:
                continue
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
        try:
            name = module_name(osp.basename(path))
        except ModuleInspectionError as exc:
            raise PluginInstallError(str(exc)) from exc
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
        self,
        plugin: InstalledPlugin,
        data: bytes,
        same_plugin,
        *,
        dependency_roots: list[str] | None = None,
        pending_enabled: list[str] | None = None,
        repair: bool = False,
    ) -> InstalledPlugin | None:
        plugins = self.get_plugins()
        replaced = next((other for other in plugins if same_plugin(other)), None)
        if replaced is not None and replaced.sha256 == plugin.sha256 and not repair:
            raise PluginInstallError(f"{plugin.name} is already installed")
        path = self.get_path(plugin)
        previous_data = None
        if plugin.kind == MODULE_KIND and osp.isfile(path):
            with open(path, "rb") as stream:
                previous_data = stream.read(MAX_MODULE_BYTES + 1)
        self._write_file(path, data)
        try:
            self._write_index(
                [other for other in plugins if other != replaced] + [plugin],
                dependency_roots=dependency_roots,
                pending_enabled=pending_enabled,
            )
        except (OSError, ValueError):
            if plugin.kind == MODULE_KIND:
                if previous_data is None:
                    os.remove(path)
                else:
                    self._write_file(path, previous_data)
            raise
        return replaced

    def _write_index(
        self,
        plugins: list[InstalledPlugin],
        *,
        dependency_roots: list[str] | None = None,
        pending_enabled: list[str] | None = None,
    ) -> None:
        payload = {
            **self._read_index(),
            "schema_version": INDEX_SCHEMA_VERSION,
            "plugins": [dataclasses.asdict(plugin) for plugin in plugins],
        }
        if dependency_roots is not None:
            payload["dependency_roots"] = dependency_roots
        if pending_enabled is not None:
            payload["pending_enabled"] = pending_enabled
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
    "get_available_distributions",
]
