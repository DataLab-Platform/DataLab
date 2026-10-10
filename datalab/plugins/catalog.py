# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin catalog client
---------------------

Read the DataLab plugin catalog (``catalog.json``) and download release wheels
whose size and SHA-256 digest match the catalog. This module is Qt-free: the
plugin dialog calls it from a worker thread.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os.path as osp
import re
import urllib.error
import urllib.parse
import urllib.request

from datalab import __version__
from datalab.plugins.wheels import MAX_WHEEL_BYTES, is_compatible_wheel

DEFAULT_CATALOG_URL = "https://datalab-platform.com/plugins/catalog.json"
CATALOG_SCHEMA_VERSION = 1
MAX_CATALOG_BYTES = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT = 60

# file: URLs serve catalogs mirrored on a local or network folder
_ALLOWED_SCHEMES = ("https", "file")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CatalogError(ValueError):
    """Raised when the catalog or a release cannot be read or verified."""


@dataclasses.dataclass(frozen=True)
class CatalogRelease:
    """Release of a catalog plugin.

    Args:
        version: Plugin version
        sha256: SHA-256 digest of the wheel
        filename: Wheel file name (empty for revoked plugins)
        url: Absolute wheel URL (empty for revoked plugins)
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
    """Plugin listed in the catalog, with its releases from the newest."""

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


def _read(url: str, limit: int) -> bytes:
    """Return the content of an HTTPS or file URL, up to a size limit."""
    if urllib.parse.urlparse(url).scheme not in _ALLOWED_SCHEMES:
        raise CatalogError(f"Unsupported URL scheme: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": f"DataLab/{__version__}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            data = response.read(limit + 1)
    except (urllib.error.URLError, OSError) as exc:
        raise CatalogError(f"Cannot download {url}: {exc}") from exc
    if len(data) > limit:
        raise CatalogError(f"{url} exceeds the {limit} byte size limit")
    return data


def _parse_release(item: dict, catalog_url: str) -> CatalogRelease:
    if not _SHA256_PATTERN.fullmatch(item["sha256"]):
        raise ValueError(f"Invalid SHA-256 digest: {item['sha256']!r}")
    url = item.get("url", "")
    return CatalogRelease(
        version=str(item["version"]),
        sha256=item["sha256"],
        filename=osp.basename(item.get("filename", "")),
        url=urllib.parse.urljoin(catalog_url, url) if url else "",
        size=int(item.get("size", 0)),
        targets=tuple(item.get("targets", ())),
        yanked=item.get("yanked", ""),
        requires_python=item.get("requires_python"),
    )


def _parse_plugin(item: dict, catalog_url: str) -> CatalogPlugin:
    return CatalogPlugin(
        id=item["id"],
        name=item["name"],
        tier=item["tier"],
        status=item["status"],
        repository=item["repository"],
        releases=tuple(
            _parse_release(release, catalog_url) for release in item["releases"]
        ),
        status_reason=item.get("status_reason", ""),
        distribution=item.get("distribution", ""),
        summary=item.get("summary", ""),
        license=item.get("license", ""),
        keywords=tuple(item.get("keywords", ())),
    )


def fetch_catalog(url: str = DEFAULT_CATALOG_URL) -> list[CatalogPlugin]:
    """Download and parse the plugin catalog.

    Raises:
        CatalogError: The catalog cannot be downloaded or is invalid
    """
    data = _read(url, MAX_CATALOG_BYTES)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CatalogError(f"Invalid plugin catalog: {url}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CATALOG_SCHEMA_VERSION
    ):
        raise CatalogError(f"Unsupported plugin catalog format: {url}")
    try:
        return [_parse_plugin(item, url) for item in payload["plugins"]]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CatalogError(f"Invalid plugin catalog: {url}") from exc


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
    "DEFAULT_CATALOG_URL",
    "CatalogError",
    "CatalogPlugin",
    "CatalogRelease",
    "download_release",
    "fetch_catalog",
]
