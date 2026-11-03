# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Offline installation of additional plugin dependencies.

pip runs only in a child process. Resolution sees the host's installed metadata;
installation writes an approved, hashed graph into a new directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import os.path as osp
import runpy
import shutil
import subprocess
import sys
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from datalab.plugins.wheels import (
    MAX_WHEEL_BYTES,
    inspect_dependency_wheel,
    validate_dependencies,
)

DEPENDENCIES_DIRECTORY = "dependencies"
PIP_WORKER_ARGUMENT = "--internal-plugin-pip"
PIP_TIMEOUT = 180
MAX_DEPENDENCY_WHEELS = 1000
MAX_NATIVE_PATH = 260 if sys.platform == "win32" else None


class DependencyError(ValueError):
    """A dependency cannot be resolved or installed without changing the host."""


@dataclasses.dataclass(frozen=True)
class DependencyWheel:
    """An inspected wheel selected by pip."""

    path: str
    distribution: str
    version: str
    sha256: str
    size_bytes: int
    packages: tuple[str, ...]
    requires_dist: tuple[str, ...]


@dataclasses.dataclass
class DependencyPlan:
    """A resolved graph and its private, snapshotted wheelhouse."""

    directory: str
    wheels: tuple[DependencyWheel, ...]
    protected_versions: dict[str, str]
    installed_paths: tuple[str, ...]
    _cleanup: weakref.finalize = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cleanup = weakref.finalize(self, shutil.rmtree, self.directory)

    def close(self) -> None:
        """Remove the temporary files owned by this plan."""
        self._cleanup()


def installed_versions(paths: Sequence[str]) -> dict[str, str]:
    """Read distribution metadata from installed dependency directories."""
    versions: dict[str, str] = {}
    for path in paths:
        if not osp.isdir(path):
            raise DependencyError(f"Installed dependency directory is missing: {path}")
        for distribution in metadata.distributions(path=[path]):
            name = distribution.metadata["Name"]
            if not name:
                raise DependencyError(f"Dependency metadata has no Name: {path}")
            key = canonicalize_name(name)
            if key in versions and versions[key] != distribution.version:
                raise DependencyError(f"Conflicting installed dependency: {name}")
            versions[key] = distribution.version
    return versions


def validate_installed(paths: Sequence[str], host: Mapping[str, str]) -> dict[str, str]:
    """Validate shared dependencies against the current, immutable host."""
    available = {canonicalize_name(name): version for name, version in host.items()}
    for name, version in installed_versions(paths).items():
        if name in available and Version(available[name]) != Version(version):
            raise DependencyError(
                f"Installed dependency {name} {version} conflicts with "
                f"the host version {available[name]}"
            )
        available.setdefault(name, version)
    for path in paths:
        for distribution in metadata.distributions(path=[path]):
            validate_dependencies(
                distribution.requires or [], available, None, "DataLab"
            )
    return available


def _requirements(values: Sequence[str]) -> list[str]:
    requirements: list[str] = []
    for value in values:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise DependencyError(f"Invalid dependency: {value!r}") from exc
        if requirement.url:
            raise DependencyError(
                f"Direct dependency URLs are not supported offline: {requirement}"
            )
        requirements.append(str(requirement))
    return requirements


def _pip(arguments: Sequence[str], directory: str, paths: Sequence[str] = ()) -> None:
    request = osp.join(directory, "pip-request.json")
    log = osp.join(directory, "pip.log")
    host_paths = [osp.abspath(path) for path in (*sys.path, *paths)]
    with open(request, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "args": [
                    "--isolated",
                    "--disable-pip-version-check",
                    "--no-input",
                    *arguments,
                ],
                "paths": host_paths,
                "log": log,
            },
            stream,
        )
    if getattr(sys, "frozen", False):
        command = [sys.executable, PIP_WORKER_ARGUMENT, request]
    else:
        bootstrap = (
            "import json,runpy,sys\n"
            "with open(sys.argv[1], encoding='utf-8') as stream:\n"
            "    sys.path[:0] = json.load(stream)['paths']\n"
            "runpy.run_module('datalab.plugins.dependencies', run_name='__main__')"
        )
        command = [sys.executable, "-I", "-c", bootstrap, request]
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=PIP_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DependencyError(f"Cannot run the dependency installer: {exc}") from exc
    if result.returncode:
        try:
            with open(log, encoding="utf-8") as stream:
                detail = stream.read()
        except FileNotFoundError:
            detail = result.stdout.decode("utf-8", errors="replace")
        raise DependencyError(
            f"Dependency installer failed ({result.returncode}):\n{detail}"
        )


def run_pip_request(request_path: str) -> int:
    """Execute pip's CLI in the dedicated normal or frozen child process."""
    with open(request_path, encoding="utf-8") as stream:
        request = json.load(stream)
    sys.path[:0] = request["paths"]
    original_streams = sys.stdin, sys.stdout, sys.stderr
    try:
        with open(request["log"], "w", encoding="utf-8") as log:
            sys.stdout = sys.stderr = log
            with open(os.devnull, encoding="utf-8") as stdin:
                sys.stdin = stdin
                sys.argv = ["pip", *request["args"]]
                try:
                    runpy.run_module("pip", run_name="__main__", alter_sys=True)
                except SystemExit as exc:
                    if exc.code is None:
                        return 0
                    return exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdin, sys.stdout, sys.stderr = original_streams
    return 0


def _constraints(directory: str, available: Mapping[str, str]) -> str:
    path = osp.join(directory, "constraints.txt")
    lines: list[str] = []
    for name, version in sorted(available.items()):
        try:
            Version(version)
        except InvalidVersion as exc:
            raise DependencyError(
                f"Cannot protect {name}: invalid installed version {version!r}"
            ) from exc
        lines.append(f"{canonicalize_name(name)}=={version}\n")
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    return path


def _copy_wheels(
    source: str, destination: str, available: Mapping[str, str]
) -> dict[str, DependencyWheel]:
    if not osp.isdir(source):
        raise DependencyError(f"Dependency wheelhouse not found: {source}")
    paths = sorted(Path(source).glob("*.whl"))
    if len(paths) > MAX_DEPENDENCY_WHEELS:
        raise DependencyError(f"Too many dependency wheels in {source}")
    wheels: dict[str, DependencyWheel] = {}
    protected = {canonicalize_name(name) for name in available}
    for path in paths:
        try:
            name = parse_wheel_filename(path.name)[0]
        except ValueError as exc:
            raise DependencyError(f"Invalid dependency wheel: {path}") from exc
        if name in protected:
            continue
        with path.open("rb") as stream:
            data = stream.read(MAX_WHEEL_BYTES + 1)
        manifest = inspect_dependency_wheel(data, filename=path.name)
        if not manifest["compatible"]:
            continue
        _requirements(manifest["requires_dist"])
        target = osp.join(destination, path.name)
        with open(target, "wb") as stream:
            stream.write(data)
        wheels[path.name] = DependencyWheel(
            target,
            manifest["distribution"],
            manifest["version"],
            manifest["sha256"],
            manifest["size_bytes"],
            tuple(manifest["top_level_packages"]),
            tuple(manifest["requires_dist"]),
        )
    return wheels


def plan_dependencies(
    requirements: Sequence[str],
    wheelhouse: str,
    available: Mapping[str, str],
    installed_paths: Sequence[str] = (),
) -> DependencyPlan:
    """Resolve missing requirements from local wheels, without installing them.

    The returned plan owns temporary files and must be closed by its caller.
    """
    values = _requirements(requirements)
    directory = tempfile.mkdtemp(prefix="datalab-dependencies-")
    success = False
    try:
        copied = osp.join(directory, "wheels")
        os.mkdir(copied)
        wheels = _copy_wheels(wheelhouse, copied, available)
        report = osp.join(directory, "report.json")
        constraints = _constraints(directory, available)
        _pip(
            [
                "install",
                "--dry-run",
                "--report",
                report,
                "--no-index",
                "--find-links",
                copied,
                "--only-binary=:all:",
                "--no-cache-dir",
                "--constraint",
                constraints,
                *values,
            ],
            directory,
            installed_paths,
        )
        with open(report, encoding="utf-8") as stream:
            payload = json.load(stream)
        selected: list[DependencyWheel] = []
        protected = {canonicalize_name(name) for name in available}
        if payload.get("version") != "1":
            raise DependencyError("Unsupported pip installation report")
        for item in payload["install"]:
            name = canonicalize_name(item["metadata"]["name"])
            if name in protected:
                raise DependencyError(f"Refusing to replace installed package: {name}")
            digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
            wheel = next(
                (wheel for wheel in wheels.values() if wheel.sha256 == digest), None
            )
            if (
                wheel is None
                or canonicalize_name(wheel.distribution) != name
                or Version(wheel.version) != Version(item["metadata"]["version"])
            ):
                raise DependencyError(f"pip selected an unapproved dependency: {name}")
            selected.append(wheel)
        plan = DependencyPlan(
            directory, tuple(selected), dict(available), tuple(installed_paths)
        )
        success = True
        return plan
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DependencyError(f"Invalid pip installation report: {exc}") from exc
    finally:
        if not success:
            shutil.rmtree(directory)


def install_dependencies(plan: DependencyPlan, target: str) -> None:
    """Install exactly the approved wheels into a new, inactive directory."""
    requirements = osp.join(plan.directory, "approved.txt")
    lines: list[str] = []
    for wheel in plan.wheels:
        with open(wheel.path, "rb") as stream:
            digest = hashlib.sha256(stream.read(MAX_WHEEL_BYTES + 1)).hexdigest()
        if digest != wheel.sha256:
            raise DependencyError(f"Dependency changed after consent: {wheel.path}")
        lines.append(
            f"{wheel.distribution} @ {Path(wheel.path).as_uri()} "
            f"--hash=sha256:{wheel.sha256}\n"
        )
    with open(requirements, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    _pip(
        [
            "install",
            "--no-index",
            "--only-binary=:all:",
            "--no-cache-dir",
            "--no-deps",
            "--require-hashes",
            "--target",
            target,
            "--requirement",
            requirements,
        ],
        plan.directory,
        plan.installed_paths,
    )
    if MAX_NATIVE_PATH is not None:
        for path in Path(target).rglob("*"):
            if (
                path.suffix.lower() in (".dll", ".pyd")
                and len(str(path)) >= MAX_NATIVE_PATH
            ):
                raise DependencyError(
                    "Native library path exceeds the Windows DLL path limit; "
                    f"use a shorter configuration directory: {path}"
                )
    versions = installed_versions([target])
    expected = {
        canonicalize_name(wheel.distribution): wheel.version for wheel in plan.wheels
    }
    if versions != expected:
        raise DependencyError("Installed dependencies differ from the approved plan")
    validate_installed([*plan.installed_paths, target], plan.protected_versions)


def download_dependencies(
    requirements: Sequence[str], destination: str, available: Mapping[str, str]
) -> list[dict]:
    """Prepare a wheelhouse online without installing or replacing packages.

    This is an explicit authoring operation, never an installer fallback.
    """
    values = _requirements(requirements)
    with tempfile.TemporaryDirectory(prefix="datalab-wheelhouse-") as directory:
        report = osp.join(directory, "report.json")
        _pip(
            [
                "install",
                "--dry-run",
                "--ignore-installed",
                "--report",
                report,
                "--only-binary=:all:",
                "--no-cache-dir",
                "--constraint",
                _constraints(directory, available),
                *values,
            ],
            directory,
        )
        with open(report, encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("version") != "1":
            raise DependencyError("Unsupported pip installation report")
        records: list[dict] = []
        download = osp.join(directory, "download")
        os.mkdir(download)
        sources: dict[str, str] = {}
        protected = {canonicalize_name(name) for name in available}
        arguments = [
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--require-hashes",
            "--dest",
            download,
        ]
        for item in payload["install"]:
            name = canonicalize_name(item["metadata"]["name"])
            if name in protected:
                if Version(item["metadata"]["version"]) != Version(available[name]):
                    raise DependencyError(
                        f"Refusing to replace installed package: {name}"
                    )
                continue
            source = item["download_info"]["url"]
            digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
            if urlparse(source).scheme != "https":
                raise DependencyError(f"Refusing dependency source: {source}")
            arguments.append(f"{source}#sha256={digest}")
            sources[digest] = source
        if not sources:
            return []
        _pip(arguments, directory)
        os.makedirs(destination, exist_ok=True)
        for path in sorted(Path(download).glob("*.whl")):
            manifest = inspect_dependency_wheel(str(path), filename=path.name)
            _requirements(manifest["requires_dist"])
            digest = manifest["sha256"]
            if digest not in sources:
                raise DependencyError(f"Unapproved downloaded dependency: {path.name}")
            target = Path(destination) / path.name
            if target.exists():
                with target.open("rb") as stream:
                    existing = hashlib.sha256(
                        stream.read(MAX_WHEEL_BYTES + 1)
                    ).hexdigest()
                if existing != digest:
                    raise DependencyError(f"A different wheel already exists: {target}")
            else:
                shutil.copyfile(path, target)
            records.append(
                {
                    "filename": path.name,
                    "url": sources[digest],
                    "sha256": digest,
                }
            )
        return records


if __name__ == "__main__":
    sys.exit(run_pip_request(sys.argv[1]))
