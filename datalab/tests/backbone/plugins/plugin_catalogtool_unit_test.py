# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the ``datalab-plugin catalog`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalab.plugins.catalog import CATALOG_FILENAME, read_folder_catalog
from datalab.plugins.catalogtool import build_local_catalog
from datalab.plugins.generator import main
from datalab.tests.backbone.plugins.plugin_local_catalog_unit_test import (
    DISTRIBUTION,
    add_wheel,
)


def read_catalog(folder: Path) -> dict:
    """Return the content of the catalog.json file of a folder."""
    return json.loads((folder / CATALOG_FILENAME).read_text(encoding="utf-8"))


def write_catalog(folder: Path, catalog: dict) -> None:
    """Write the catalog.json file of a folder."""
    (folder / CATALOG_FILENAME).write_text(json.dumps(catalog), encoding="utf-8")


def test_build_lists_the_wheels_and_keeps_curated_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A rebuilt catalog keeps what the administrator edited."""
    add_wheel(tmp_path, "1.0.0")
    assert main(["catalog", "build", str(tmp_path)]) == 0
    assert "listing 1 plugins" in capsys.readouterr().out

    catalog = read_catalog(tmp_path)
    (plugin,) = catalog["plugins"]
    assert plugin["id"] == DISTRIBUTION and plugin["tier"] == "local"
    (release,) = plugin["releases"]
    assert release["url"] == release["filename"]
    assert release["requires_dist"] == ["datalab-platform>=1.0"]
    (listed,), warnings = read_folder_catalog(str(tmp_path))
    assert listed.get_installable_release().version == "1.0.0" and not warnings

    plugin.update(
        id="org.example.spectro",
        name="Spectro tools",
        status="deprecated",
        status_reason="Use Spectro tools 2",
    )
    release["yanked"] = "Wrong baseline"
    catalog["plugins"].append(
        {
            "id": "org.example.gone",
            "name": "Gone",
            "tier": "local",
            "status": "active",
            "repository": "",
            "releases": [],
        }
    )
    catalog["plugins"].append(
        {
            "id": "org.example.revoked",
            "name": "Revoked",
            "tier": "local",
            "status": "revoked",
            "status_reason": "Malicious code",
            "repository": "",
            "releases": [{"version": "1.0.0", "sha256": "c" * 64}],
        }
    )
    write_catalog(tmp_path, catalog)
    add_wheel(tmp_path, "1.1.0")

    report = build_local_catalog(str(tmp_path))

    assert report.written and not report.errors
    assert report.warnings == ["Gone: no wheel left, removed from the catalog"]
    revoked, plugin = read_catalog(tmp_path)["plugins"]
    assert revoked["status_reason"] == "Malicious code"
    assert plugin["id"] == "org.example.spectro" and plugin["status"] == "deprecated"
    assert [item["version"] for item in plugin["releases"]] == ["1.1.0", "1.0.0"]
    assert plugin["releases"][1]["yanked"] == "Wrong baseline"
    assert "yanked" not in plugin["releases"][0]


def test_build_refuses_wheels_that_datalab_would_not_install(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A wheel requiring a package that DataLab lacks prevents the build."""
    add_wheel(tmp_path, "1.0.0", requires_dist=("unknown-package-xyz>=1",))

    assert main(["catalog", "build", str(tmp_path)]) == 1

    error = capsys.readouterr().err
    assert "unknown-package-xyz" in error and "was not written" in error
    assert not (tmp_path / CATALOG_FILENAME).exists()
    assert main(["catalog", "build", str(tmp_path / "missing")]) == 1


def test_check_compares_the_catalog_with_the_wheels(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The check reports changed wheels and wheels missing from the catalog."""
    wheel = add_wheel(tmp_path, "1.0.0")
    assert main(["catalog", "check", str(tmp_path)]) == 0
    assert f"No {CATALOG_FILENAME}" in capsys.readouterr().out

    assert main(["catalog", "build", str(tmp_path)]) == 0
    assert main(["catalog", "check", str(tmp_path)]) == 0
    unlisted = add_wheel(tmp_path, "1.1.0")
    capsys.readouterr()
    assert main(["catalog", "check", str(tmp_path)]) == 0
    assert f"{unlisted} is not listed" in capsys.readouterr().out

    other = tmp_path / "other"
    other.mkdir()
    wheel.write_bytes(add_wheel(other, "1.0.0", distribution="other-tool").read_bytes())
    assert main(["catalog", "check", str(tmp_path)]) == 1
    assert "refused" in capsys.readouterr().err

    wheel.unlink()
    assert main(["catalog", "check", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err


if __name__ == "__main__":
    pytest.main([__file__])
