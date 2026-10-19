# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for local plugin catalogs: folders read as files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from datalab.config.appinfo import (
    DATALAB_PLUGINS_CATALOG_ENV_VAR,
    get_env_plugin_catalog_paths,
)
from datalab.plugins.catalog import (
    CatalogError,
    download_release,
    load_catalogs,
    read_folder_catalog,
)
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

DISTRIBUTION = "datalab-spectro-tools"


def add_wheel(
    folder: Path, version: str, distribution: str = DISTRIBUTION, **kwargs
) -> Path:
    """Write a plugin wheel in a folder and return its path."""
    kwargs.setdefault("requires_dist", ("datalab-platform>=1.0",))
    path = folder / wheel_filename(distribution, version)
    path.write_bytes(
        make_plugin_wheel(
            distribution=distribution,
            version=version,
            package=distribution.replace("-", "_"),
            **kwargs,
        )
    )
    return path


def online_catalog(folder: Path, plugins: list[dict]) -> str:
    """Write an online-like catalog and return its URL."""
    path = folder / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 1, "plugins": plugins}), encoding="utf-8"
    )
    return path.as_uri()


def online_record(plugin_id: str, distribution: str, **kwargs) -> dict:
    """Return an online catalog plugin record with one fake release."""
    record = {
        "id": plugin_id,
        "name": f"Online {distribution}",
        "tier": "community",
        "status": "active",
        "repository": "https://github.com/someone/plugin",
        "distribution": distribution,
        "releases": [
            {
                "version": "9.0.0",
                "filename": wheel_filename(distribution, "9.0.0"),
                "url": f"wheels/{wheel_filename(distribution, '9.0.0')}",
                "sha256": "b" * 64,
                "size": 10,
                "targets": ["desktop"],
            }
        ],
    }
    record.update(kwargs)
    return record


def test_folder_wheels_are_listed_from_the_newest(tmp_path: Path) -> None:
    """Without catalog.json, the plugin wheels of a folder are the catalog."""
    metadata = (
        "License-Expression: MIT",
        "Keywords: spectroscopy,baseline",
        "Project-URL: Source Code, https://git.example.org/spectro",
    )
    add_wheel(tmp_path, "1.0.0")
    newest = add_wheel(tmp_path, "1.1.0", extra_metadata=metadata)
    add_wheel(
        tmp_path, "1.0.0", distribution="other-tool", entry_points="[console]\nx=a:b\n"
    )
    (tmp_path / "broken-1.0-py3-none-any.whl").write_bytes(b"not a zip")
    (tmp_path / "native-1.0-cp311-cp311-win_amd64.whl").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("not a wheel", encoding="utf-8")

    (plugin,), warnings = read_folder_catalog(str(tmp_path))

    assert plugin.id == DISTRIBUTION and plugin.name == DISTRIBUTION
    assert plugin.tier == "local" and plugin.source == str(tmp_path)
    assert plugin.license == "MIT" and plugin.keywords == ("spectroscopy", "baseline")
    assert plugin.repository == "https://git.example.org/spectro"
    assert plugin.summary == "Example DataLab plugin"
    assert [release.version for release in plugin.releases] == ["1.1.0", "1.0.0"]
    release = plugin.get_installable_release()
    assert release.url == str(newest) and release.requires_python == ">=3.9"
    assert release.sha256 == hashlib.sha256(newest.read_bytes()).hexdigest()
    assert len(warnings) == 3
    assert any("broken" in message and "ZIP" in message for message in warnings)
    assert any("pure-Python" in message for message in warnings)
    assert any("no datalab.plugins entry point" in message for message in warnings)

    destination = tmp_path / "download"
    destination.mkdir()
    path = download_release(release, str(destination))
    assert Path(path).read_bytes() == newest.read_bytes()

    newest.write_bytes(newest.read_bytes()[:-1] + b"!")
    with pytest.raises(CatalogError, match="does not match"):
        download_release(release, str(destination))


def test_folder_catalog_json_lists_the_plugins(tmp_path: Path) -> None:
    """A catalog.json file lists the plugins, with paths relative to the folder."""
    (tmp_path / "wheels").mkdir()
    listed = add_wheel(tmp_path / "wheels", "2.0.0")
    add_wheel(tmp_path, "3.0.0")
    data = listed.read_bytes()
    release = {
        "version": "2.0.0",
        "filename": listed.name,
        "url": f"wheels/{listed.name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "targets": ["desktop"],
    }
    record = online_record("org.example.spectro", DISTRIBUTION, releases=[release])
    online_catalog(tmp_path, [record])

    (plugin,), warnings = read_folder_catalog(str(tmp_path))

    assert not warnings
    assert plugin.id == "org.example.spectro" and plugin.source == str(tmp_path)
    assert [release.version for release in plugin.releases] == ["2.0.0"]
    assert plugin.releases[0].url == str(listed)

    release["url"] = "../outside.whl"
    online_catalog(tmp_path, [record])
    with pytest.raises(CatalogError, match="Invalid plugin catalog"):
        read_folder_catalog(str(tmp_path))


def test_local_catalogs_take_precedence(tmp_path: Path) -> None:
    """Local catalogs hide the same plugins in the next catalogs."""
    first, second, site = tmp_path / "first", tmp_path / "second", tmp_path / "site"
    for folder in (first, second, site):
        folder.mkdir()
    add_wheel(first, "1.0.0")
    add_wheel(second, "2.0.0")
    add_wheel(second, "1.0.0", distribution="datalab-camera-tools")
    url = online_catalog(
        site,
        [
            online_record("io.github.someone.spectro", DISTRIBUTION),
            online_record(
                "io.github.someone.camera",
                "datalab-camera-tools",
                status="revoked",
                status_reason="Malicious code",
            ),
            online_record("io.github.someone.pulse", "datalab-pulse-tools"),
        ],
    )

    content = load_catalogs(url, [str(tmp_path / "missing"), str(first), str(second)])

    assert [(plugin.distribution, plugin.source) for plugin in content.plugins] == [
        (DISTRIBUTION, str(first)),
        ("datalab-camera-tools", str(second)),
        ("datalab-pulse-tools", ""),
    ]
    assert content.plugins[0].releases[0].version == "1.0.0"
    missing, withdrawn = content.errors
    assert missing.startswith("Plugin catalog folder not found")
    assert "withdrawn" in withdrawn and "Malicious code" in withdrawn

    content = load_catalogs((site / "missing.json").as_uri(), [str(first)])
    assert [plugin.source for plugin in content.plugins] == [str(first)]
    assert content.errors[0].startswith("Cannot download")


def test_catalog_folders_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Folders of the environment variable are kept even when unavailable."""
    existing = tmp_path / "existing"
    existing.mkdir()
    missing = tmp_path / "missing"
    value = os.pathsep.join((f" {existing} ", "", str(missing), str(existing)))
    monkeypatch.setenv(DATALAB_PLUGINS_CATALOG_ENV_VAR, value)

    assert get_env_plugin_catalog_paths() == [str(existing), str(missing)]

    monkeypatch.delenv(DATALAB_PLUGINS_CATALOG_ENV_VAR)
    assert not get_env_plugin_catalog_paths()


if __name__ == "__main__":
    pytest.main([__file__])
