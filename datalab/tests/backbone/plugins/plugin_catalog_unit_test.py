# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the plugin catalog client, with catalogs served from files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from datalab.plugins.catalog import CatalogError, download_release, fetch_catalog

FILENAME = "example_plugin-1.0.0-py3-none-any.whl"


def write_catalog(directory: Path, plugins: list[dict], schema_version=1) -> str:
    """Write a catalog file and return its URL."""
    path = directory / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "plugins": plugins}),
        encoding="utf-8",
    )
    return path.as_uri()


def publish(directory: Path, data: bytes, version: str = "1.0.0", **kwargs) -> dict:
    """Write a wheel in the catalog folder and return its release record."""
    digest = hashlib.sha256(data).hexdigest()
    filename = FILENAME.replace("1.0.0", version)
    path = directory / "wheels" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    release = {
        "version": version,
        "filename": filename,
        "url": f"wheels/{digest}/{filename}",
        "sha256": digest,
        "size": len(data),
        "targets": ["desktop", "web"],
    }
    release.update(kwargs)
    return release


def plugin_record(releases: list[dict], **kwargs) -> dict:
    """Return a catalog plugin record."""
    record = {
        "id": "io.github.someone.example",
        "name": "Example Plugin",
        "tier": "community",
        "status": "active",
        "repository": "https://github.com/someone/example-plugin",
        "distribution": "example-plugin",
        "summary": "Spectral baseline tools",
        "license": "MIT",
        "keywords": ["spectroscopy"],
        "releases": releases,
    }
    record.update(kwargs)
    return record


def test_release_is_downloaded_from_a_relative_url(tmp_path: Path) -> None:
    """Release URLs are relative to the catalog and verified once downloaded."""
    release = publish(tmp_path, b"wheel content")
    url = write_catalog(tmp_path, [plugin_record([release])])

    (plugin,) = fetch_catalog(url)
    latest = plugin.get_installable_release()
    destination = tmp_path / "download"
    destination.mkdir()
    path = download_release(latest, str(destination))

    assert plugin.name == "Example Plugin" and plugin.tier == "community"
    assert latest.url == (tmp_path / release["url"]).as_uri()
    assert Path(path) == destination / FILENAME
    assert Path(path).read_bytes() == b"wheel content"


def test_installable_release_skips_yanked_web_only_and_revoked(tmp_path: Path) -> None:
    """Only the newest desktop release that is not withdrawn is offered."""
    releases = [
        publish(tmp_path, b"3", "3.0.0", yanked="Corrupted results"),
        publish(tmp_path, b"2", "2.0.0", targets=["web"]),
        publish(tmp_path, b"1", "1.0.0"),
    ]
    url = write_catalog(
        tmp_path,
        [
            plugin_record(releases),
            plugin_record(
                [{"version": "1.0.0", "sha256": "a" * 64}],
                id="io.github.someone.revoked",
                status="revoked",
                status_reason="Malicious code",
            ),
        ],
    )

    plugin, revoked = fetch_catalog(url)

    assert plugin.get_installable_release().version == "1.0.0"
    assert plugin.get_installable_release("web").version == "2.0.0"
    assert revoked.get_installable_release() is None
    assert plugin.matches("SPECTRO baseline") and not plugin.matches("camera")


def test_downloads_must_match_the_catalog(tmp_path: Path) -> None:
    """A wheel changed after publication is refused."""
    release = publish(tmp_path, b"original")
    (tmp_path / release["url"]).write_bytes(b"tampered")
    url = write_catalog(tmp_path, [plugin_record([release])])
    (plugin,) = fetch_catalog(url)

    with pytest.raises(CatalogError, match="does not match"):
        download_release(plugin.get_installable_release(), str(tmp_path))

    (tmp_path / release["url"]).write_bytes(b"original and more")
    with pytest.raises(CatalogError, match="size limit"):
        download_release(plugin.get_installable_release(), str(tmp_path))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"schema_version": 2, "plugins": []}', "Unsupported plugin catalog"),
        ("not json", "Invalid plugin catalog"),
        ('{"schema_version": 1, "plugins": [{"id": "x"}]}', "Invalid plugin catalog"),
    ],
)
def test_invalid_catalogs_are_refused(
    tmp_path: Path, content: str, message: str
) -> None:
    """Unknown formats and incomplete records are reported."""
    path = tmp_path / "catalog.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CatalogError, match=message):
        fetch_catalog(path.as_uri())


def test_only_https_and_file_urls_are_used() -> None:
    """Plain HTTP could be tampered with on the way."""
    with pytest.raises(CatalogError, match="Unsupported URL scheme"):
        fetch_catalog("http://example.org/catalog.json")


if __name__ == "__main__":
    pytest.main([__file__])
