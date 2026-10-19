# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin catalog client
---------------------

Read the DataLab plugin catalog (``catalog.json``) and download release wheels
whose size and SHA-256 digest match the catalog. This module is Qt-free: the
plugin dialog calls it from a worker thread.

Besides the online catalog, local catalogs are folders, typically shared on the
network, read as files: their ``catalog.json`` if any, otherwise the plugin
wheels they hold.
"""

from __future__ import annotations

import configparser
import dataclasses
import hashlib
import io
import json
import os
import os.path as osp
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from email.message import Message
from email.parser import BytesParser
from email.policy import default as email_policy

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from datalab import __version__
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    MAX_WHEEL_BYTES,
    is_compatible_wheel,
)

DEFAULT_CATALOG_URL = "https://datalab-platform.com/plugins/catalog.json"
CATALOG_FILENAME = "catalog.json"
CATALOG_SCHEMA_VERSION = 1
LOCAL_TIER = "local"
MAX_CATALOG_BYTES = 8 * 1024 * 1024
MAX_FOLDER_WHEELS = 1000
DOWNLOAD_TIMEOUT = 60

# file: URLs serve catalogs mirrored on a local folder
_ALLOWED_SCHEMES = ("https", "file")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# Normalized Project-URL labels (PEP 753) of the source repository, by priority
_REPOSITORY_LABELS = ("source", "sourcecode", "repository", "code", "homepage")


class CatalogError(ValueError):
    """Raised when the catalog or a release cannot be read or verified."""


@dataclasses.dataclass(frozen=True)
class CatalogRelease:
    """Release of a catalog plugin.

    Args:
        version: Plugin version
        sha256: SHA-256 digest of the wheel
        filename: Wheel file name (empty for revoked plugins)
        url: Absolute wheel URL, or file path for local catalogs (empty for
         revoked plugins)
        size: Wheel size in bytes
        targets: Hosts supported by the wheel (``desktop``, ``web``)
        yanked: Reason why the release should not be installed, if any
        requires_python: ``Requires-Python`` metadata of the wheel, if any
    """

    version: str
    sha256: str
    filename: str = ""
    url: str = ""
    size: int = 0
    targets: tuple[str, ...] = ()
    yanked: str = ""
    requires_python: str | None = None


@dataclasses.dataclass(frozen=True)
class CatalogPlugin:
    """Plugin listed in the catalog, with its releases from the newest.

    ``source`` is the folder of the local catalog listing the plugin, empty for
    the online catalog.
    """

    id: str
    name: str
    tier: str
    status: str
    repository: str
    releases: tuple[CatalogRelease, ...]
    status_reason: str = ""
    distribution: str = ""
    summary: str = ""
    license: str = ""
    keywords: tuple[str, ...] = ()
    source: str = ""

    def get_installable_release(
        self, target: str = "desktop", python_version: str | None = None
    ) -> CatalogRelease | None:
        """Return the newest release that may be installed on a host.

        Args:
            target: Host (``desktop`` or ``web``)
            python_version: Host Python version, ignored if empty (default:
             running interpreter)
        """
        if self.status == "revoked":
            return None
        return next(
            (
                release
                for release in self.releases
                if release.url
                and not release.yanked
                and target in release.targets
                and (
                    python_version == ""
                    or is_compatible_wheel(
                        release.filename, release.requires_python, python_version
                    )
                )
            ),
            None,
        )

    def matches(self, text: str) -> bool:
        """Return True if the plugin matches a search text."""
        words = text.casefold().split()
        haystack = " ".join(
            (self.name, self.id, self.summary, " ".join(self.keywords))
        ).casefold()
        return all(word in haystack for word in words)


def _read(location: str, limit: int) -> bytes:
    """Return the content of a file path, or of an HTTPS or file URL.

    Args:
        location: Absolute file path (local catalogs) or URL
        limit: Maximum size in bytes
    """
    # Testing the path first: urlparse("C:\\x").scheme is "c"
    if osp.isabs(location):
        try:
            with open(location, "rb") as file:
                data = file.read(limit + 1)
        except OSError as exc:
            raise CatalogError(f"Cannot read {location}: {exc}") from exc
    else:
        if urllib.parse.urlparse(location).scheme not in _ALLOWED_SCHEMES:
            raise CatalogError(f"Unsupported URL scheme: {location}")
        request = urllib.request.Request(
            location, headers={"User-Agent": f"DataLab/{__version__}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                data = response.read(limit + 1)
        except (urllib.error.URLError, OSError) as exc:
            raise CatalogError(f"Cannot download {location}: {exc}") from exc
    if len(data) > limit:
        raise CatalogError(f"{location} exceeds the {limit} byte size limit")
    return data


def _folder_resolver(root: str) -> Callable[[str], str]:
    """Return a function resolving release URLs relative to a catalog folder.

    Relative URLs become file paths, which must stay inside the folder; absolute
    URLs are kept.
    """

    def resolve(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.netloc:
            return url
        relative = urllib.parse.unquote(parsed.path).split("/")
        path = osp.normpath(osp.join(root, *relative))
        try:
            inside = osp.commonpath([root, path]) == root and path != root
        except ValueError:  # Paths on different drives
            inside = False
        if not inside:
            raise ValueError(f"Release outside the catalog folder: {url!r}")
        return path

    return resolve


def _parse_release(item: dict, resolve: Callable[[str], str]) -> CatalogRelease:
    if not _SHA256_PATTERN.fullmatch(item["sha256"]):
        raise ValueError(f"Invalid SHA-256 digest: {item['sha256']!r}")
    url = item.get("url", "")
    return CatalogRelease(
        version=str(item["version"]),
        sha256=item["sha256"],
        filename=osp.basename(item.get("filename", "")),
        url=resolve(url) if url else "",
        size=int(item.get("size", 0)),
        targets=tuple(item.get("targets", ())),
        yanked=item.get("yanked", ""),
        requires_python=item.get("requires_python"),
    )


def _parse_plugin(
    item: dict, resolve: Callable[[str], str], source: str
) -> CatalogPlugin:
    return CatalogPlugin(
        id=item["id"],
        name=item["name"],
        tier=item["tier"],
        status=item["status"],
        repository=item["repository"],
        releases=tuple(
            _parse_release(release, resolve) for release in item["releases"]
        ),
        status_reason=item.get("status_reason", ""),
        distribution=item.get("distribution", ""),
        summary=item.get("summary", ""),
        license=item.get("license", ""),
        keywords=tuple(item.get("keywords", ())),
        source=source,
    )


def load_catalog_json(data: bytes, location: str) -> dict:
    """Return the payload of a ``catalog.json`` file, checking its format.

    Raises:
        CatalogError: The content is not a supported catalog
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CatalogError(f"Invalid plugin catalog: {location}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CATALOG_SCHEMA_VERSION
    ):
        raise CatalogError(f"Unsupported plugin catalog format: {location}")
    return payload


def _parse_catalog(
    payload: dict, location: str, resolve: Callable[[str], str], source: str = ""
) -> list[CatalogPlugin]:
    try:
        return [_parse_plugin(item, resolve, source) for item in payload["plugins"]]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CatalogError(f"Invalid plugin catalog: {location}") from exc


def fetch_catalog(url: str = DEFAULT_CATALOG_URL) -> list[CatalogPlugin]:
    """Download and parse the plugin catalog.

    Raises:
        CatalogError: The catalog cannot be downloaded or is invalid
    """
    payload = load_catalog_json(_read(url, MAX_CATALOG_BYTES), url)
    return _parse_catalog(
        payload, url, lambda release_url: urllib.parse.urljoin(url, release_url)
    )


def _dist_info_file(archive: zipfile.ZipFile, name: str) -> bytes | None:
    paths = [
        path
        for path in archive.namelist()
        if path.count("/") == 1 and path.endswith(f".dist-info/{name}")
    ]
    return archive.read(paths[0]) if len(paths) == 1 else None


def _get_repository(metadata: Message) -> str:
    """Return the source repository URL declared in wheel metadata, if any."""
    urls: dict[str, str] = {}
    for value in metadata.get_all("Project-URL", []):
        label, _sep, url = str(value).partition(",")
        urls.setdefault(re.sub(r"[^a-z]", "", label.lower()), url.strip())
    candidates = [urls.get(label, "") for label in _REPOSITORY_LABELS]
    candidates.append(str(metadata.get("Home-page") or "").strip())
    return next(
        (url for url in candidates if url.startswith(("https://", "http://"))), ""
    )


def _get_license(metadata: Message) -> str:
    """Return a short license description from wheel metadata."""
    expression = str(metadata.get("License-Expression") or "").strip()
    if expression:
        return expression
    # The License field may hold the full license text
    text = str(metadata.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 80:
        return text
    for classifier in metadata.get_all("Classifier", []):
        if str(classifier).startswith("License :: "):
            return str(classifier).rsplit(" :: ", 1)[-1]
    return ""


def _get_keywords(metadata: Message) -> list[str]:
    text = str(metadata.get("Keywords") or "")
    words = text.split(",") if "," in text else text.split()
    return [word.strip() for word in words if word.strip()]


def read_wheel_entry(path: str) -> tuple[dict, dict]:
    """Return the catalog records describing a plugin wheel file.

    The wheel is only read, not validated as thoroughly as when installing it.

    Args:
        path: Wheel file path

    Returns:
        Plugin record without releases, and release record whose URL is the
        file name, both as written in ``catalog.json``

    Raises:
        CatalogError: The file is not a pure-Python DataLab desktop plugin wheel
    """
    filename = osp.basename(path)
    try:
        name, version, _build, tags = parse_wheel_filename(filename)
    except (InvalidVersion, ValueError) as exc:
        raise CatalogError("invalid wheel file name") from exc
    if any(tag.abi != "none" or tag.platform != "any" for tag in tags):
        raise CatalogError("not a pure-Python wheel")
    try:
        if not 0 < osp.getsize(path) <= MAX_WHEEL_BYTES:
            raise CatalogError(f"size not between 1 and {MAX_WHEEL_BYTES} bytes")
        with open(path, "rb") as file:
            data = file.read(MAX_WHEEL_BYTES + 1)
    except OSError as exc:
        raise CatalogError(f"cannot be read: {exc}") from exc
    if not 0 < len(data) <= MAX_WHEEL_BYTES:
        raise CatalogError(f"size not between 1 and {MAX_WHEEL_BYTES} bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            metadata_data = _dist_info_file(archive, "METADATA")
            entry_points = _dist_info_file(archive, "entry_points.txt") or b""
    except zipfile.BadZipFile as exc:
        raise CatalogError("not a valid ZIP archive") from exc
    if metadata_data is None:
        raise CatalogError("no METADATA file")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string(entry_points.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise CatalogError("invalid entry_points.txt") from exc
    if not parser.has_section(DESKTOP_ENTRY_POINT_GROUP):
        raise CatalogError(
            f"not a DataLab desktop plugin (no {DESKTOP_ENTRY_POINT_GROUP} entry point)"
        )
    metadata = BytesParser(policy=email_policy).parsebytes(metadata_data)
    distribution = str(metadata.get("Name") or "")
    try:
        same_version = Version(str(metadata.get("Version") or "")) == version
    except InvalidVersion:
        same_version = False
    if canonicalize_name(distribution) != name or not same_version:
        raise CatalogError("file name and METADATA differ")
    plugin = {
        "id": name,
        "name": distribution,
        "tier": LOCAL_TIER,
        "status": "active",
        "repository": _get_repository(metadata),
        "distribution": distribution,
        "summary": str(metadata.get("Summary") or ""),
        "license": _get_license(metadata),
        "keywords": _get_keywords(metadata),
    }
    release = {
        "version": str(version),
        "filename": filename,
        "url": urllib.parse.quote(filename),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "targets": ["desktop"],
    }
    requires_python = str(metadata.get("Requires-Python") or "")
    if requires_python:
        release["requires_python"] = requires_python
    return plugin, release


def list_folder_wheels(directory: str) -> list[str]:
    """Return the paths of the wheel files of a folder, sorted by name.

    Raises:
        CatalogError: The folder cannot be read or holds too many wheels
    """
    try:
        names = sorted(
            name
            for name in os.listdir(directory)
            if name.lower().endswith(".whl") and osp.isfile(osp.join(directory, name))
        )
    except OSError as exc:
        raise CatalogError(f"Cannot read the plugin catalog folder: {exc}") from exc
    if len(names) > MAX_FOLDER_WHEELS:
        raise CatalogError(
            f"{directory} holds more than {MAX_FOLDER_WHEELS} wheels: "
            f"list the plugins in a {CATALOG_FILENAME} file"
        )
    return [osp.join(directory, name) for name in names]


def scan_wheel_folder(directory: str) -> tuple[dict, list[str]]:
    """Return the catalog describing the plugin wheels of a folder.

    Releases of a plugin are grouped by distribution, from the newest.

    Returns:
        Catalog payload, as written in ``catalog.json``, and messages about the
        ignored files

    Raises:
        CatalogError: The folder cannot be read
    """
    warnings: list[str] = []
    plugins: dict[str, dict] = {}
    releases: dict[str, dict[Version, dict]] = {}
    for path in list_folder_wheels(directory):
        try:
            plugin, release = read_wheel_entry(path)
        except CatalogError as exc:
            warnings.append(f"{path} ignored: {exc}")
            continue
        versions = releases.setdefault(plugin["id"], {})
        version = Version(release["version"])
        if version in versions:
            duplicate = versions[version]["filename"]
            warnings.append(f"{path} ignored: same release as {duplicate}")
            continue
        versions[version] = release
        if version == max(versions):
            plugins[plugin["id"]] = plugin
    records = []
    for key, plugin in sorted(
        plugins.items(), key=lambda item: item[1]["name"].casefold()
    ):
        versions = releases[key]
        plugin["releases"] = [versions[version] for version in sorted(versions)[::-1]]
        records.append(plugin)
    return {"schema_version": CATALOG_SCHEMA_VERSION, "plugins": records}, warnings


def read_folder_catalog(directory: str) -> tuple[list[CatalogPlugin], list[str]]:
    """Read a local catalog folder.

    The folder ``catalog.json`` lists the plugins if it exists, with release URLs
    relative to the folder; otherwise, the plugin wheels of the folder are listed.

    Args:
        directory: Catalog folder, possibly shared on the network

    Returns:
        Plugins, and messages about the ignored files

    Raises:
        CatalogError: The folder or its catalog cannot be read
    """
    root = osp.normpath(osp.abspath(directory))
    if not osp.isdir(root):
        raise CatalogError(f"Plugin catalog folder not found: {root}")
    path = osp.join(root, CATALOG_FILENAME)
    if osp.isfile(path):
        payload = load_catalog_json(_read(path, MAX_CATALOG_BYTES), path)
        warnings: list[str] = []
        location = path
    else:
        payload, warnings = scan_wheel_folder(root)
        location = root
    return _parse_catalog(payload, location, _folder_resolver(root), root), warnings


@dataclasses.dataclass(frozen=True)
class CatalogsContent:
    """Plugins of the configured catalogs.

    Args:
        plugins: Plugins, a plugin listed by several catalogs appearing once
        errors: Messages about the catalogs that cannot be read, the ignored
         files and the plugins withdrawn from a masked catalog
    """

    plugins: tuple[CatalogPlugin, ...]
    errors: tuple[str, ...]


def load_catalogs(url: str, directories: Sequence[str] = ()) -> CatalogsContent:
    """Read the local catalogs, then download the online catalog.

    A plugin listed in several catalogs is taken from the first one: local
    catalogs take precedence over the online catalog, so that an organization
    may provide its own releases. A catalog that cannot be read does not prevent
    the others from being used.

    Args:
        url: Online catalog URL (empty to ignore it)
        directories: Local catalog folders, by priority
    """
    plugins: list[CatalogPlugin] = []
    errors: list[str] = []
    ids: set[str] = set()
    distributions: set[str] = set()
    readers: list[Callable[[], tuple[list[CatalogPlugin], list[str]]]] = [
        lambda directory=directory: read_folder_catalog(directory)
        for directory in directories
    ]
    if url:
        readers.append(lambda: (fetch_catalog(url), []))
    for read in readers:
        try:
            items, warnings = read()
        except CatalogError as exc:
            errors.append(str(exc))
            continue
        errors.extend(warnings)
        for plugin in items:
            distribution = canonicalize_name(plugin.distribution)
            if plugin.id in ids or (distribution and distribution in distributions):
                if plugin.status == "revoked":
                    errors.append(
                        f"{plugin.name} ({plugin.id}) is withdrawn from "
                        f"{plugin.source or url}: {plugin.status_reason}"
                    )
                continue
            ids.add(plugin.id)
            if distribution:
                distributions.add(distribution)
            plugins.append(plugin)
    return CatalogsContent(tuple(plugins), tuple(errors))


def download_release(release: CatalogRelease, directory: str) -> str:
    """Download a release wheel into a directory after checking it.

    Returns:
        Path of the downloaded wheel, named as in the catalog

    Raises:
        CatalogError: The download failed or does not match the catalog
    """
    if not release.url or not release.filename.endswith(".whl"):
        raise CatalogError("This release cannot be downloaded")
    if not 0 < release.size <= MAX_WHEEL_BYTES:
        raise CatalogError(f"Invalid wheel size in the catalog: {release.size}")
    data = _read(release.url, release.size)
    if len(data) != release.size or hashlib.sha256(data).hexdigest() != release.sha256:
        raise CatalogError(
            f"{release.filename} does not match the size and digest of the catalog"
        )
    path = osp.join(directory, release.filename)
    with open(path, "wb") as file:
        file.write(data)
    return path


__all__ = [
    "CATALOG_FILENAME",
    "DEFAULT_CATALOG_URL",
    "LOCAL_TIER",
    "CatalogError",
    "CatalogPlugin",
    "CatalogRelease",
    "CatalogsContent",
    "download_release",
    "fetch_catalog",
    "list_folder_wheels",
    "load_catalog_json",
    "load_catalogs",
    "read_folder_catalog",
    "read_wheel_entry",
    "scan_wheel_folder",
]
