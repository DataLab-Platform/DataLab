# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Installation check test

Checking the installation check run by the ``--smoke-test`` command line option.
"""

# guitest: skip

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

import datalab
from datalab import app
from datalab.env import DLExecEnv, execenv
from datalab.utils import smoketest
from datalab.utils.smoketest import DEFAULT_REPORT, Check, run_smoke_test


def _passing() -> str:
    return "fine"


def _failing() -> str:
    raise ValueError("broken")


def _exiting() -> str:
    sys.exit(3)


def _read_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _statuses(report: dict) -> dict[str, str]:
    return {check["name"]: check["status"] for check in report["checks"]}


def test_report_and_failure(tmp_path: Path) -> None:
    """A failed check is reported with its traceback, without stopping the next
    checks, and makes the exit code non-zero."""
    path = tmp_path / "subdir" / "report.json"
    checks = (
        Check("first", _failing),
        Check("second", _exiting),
        Check("third", _passing),
        Check("frozen", _passing, frozen_only=True),
    )
    assert run_smoke_test(str(path), checks) == 1
    report = _read_report(path)
    assert _statuses(report) == {
        "first": "failed",
        "second": "failed",
        "third": "passed",
        "frozen": "skipped",
    }
    failed = report["checks"][0]
    assert failed["detail"] == "ValueError: broken"
    assert "Traceback" in failed["traceback"]
    assert report["checks"][2]["detail"] == "fine"
    assert report["summary"] == {"passed": 1, "failed": 2, "skipped": 1}
    assert report["datalab_version"] == datalab.__version__
    assert report["frozen"] is False


def test_success(tmp_path: Path) -> None:
    """The exit code is 0 when no check failed (skipped checks do not fail)."""
    checks = (Check("ok", _passing), Check("frozen", _failing, frozen_only=True))
    assert run_smoke_test(str(tmp_path / "report.json"), checks) == 0


def test_frozen_only_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks of the standalone version run when the application is frozen."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    path = tmp_path / "report.json"
    assert run_smoke_test(str(path), (Check("frozen", _passing, True),)) == 0
    report = _read_report(path)
    assert report["frozen"] is True
    assert _statuses(report) == {"frozen": "passed"}


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m datalab.utils.smoketest REPORT`` runs the default checks."""
    monkeypatch.setattr(smoketest, "CHECKS", (Check("ok", _passing),))
    path = tmp_path / "report.json"
    assert smoketest.main([str(path)]) == 0
    assert _statuses(_read_report(path)) == {"ok": "passed"}


@pytest.mark.parametrize(
    "func",
    (
        smoketest.check_packages,
        smoketest.check_macro_templates,
        smoketest.check_images,
        smoketest.check_processing_pool,
    ),
)
def test_installation_checks(func) -> None:
    """Checks which do not depend on build artifacts pass in this environment."""
    assert func()


def test_build_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The build manifest is read next to the executable."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "DataLab.exe"))
    with pytest.raises(FileNotFoundError):
        smoketest.check_build_manifest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"packages": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="No package"):
        smoketest.check_build_manifest()
    content = {"python_version": "3.11.9", "packages": {"datalab-platform": "1.4.0"}}
    manifest.write_text(json.dumps(content), encoding="utf-8")
    assert smoketest.check_build_manifest() == "built with Python 3.11.9, 1 packages"


def test_pdf_documentation(tmp_path: Path) -> None:
    """Both PDF files must exist and be PDF files."""
    with pytest.raises(FileNotFoundError, match="DataLab_en.pdf"):
        smoketest.check_pdf_documentation(str(tmp_path))
    docdir = tmp_path / "doc"
    docdir.mkdir()
    for lang in ("en", "fr"):
        (docdir / f"DataLab_{lang}.pdf").write_bytes(b"%PDF-1.7\n")
    assert smoketest.check_pdf_documentation(str(tmp_path)).startswith("en: ")
    (docdir / "DataLab_fr.pdf").write_bytes(b"<html>")
    with pytest.raises(ValueError, match="Not a PDF file"):
        smoketest.check_pdf_documentation(str(tmp_path))


def test_translations(tmp_path: Path) -> None:
    """French catalogs must be found, without falling back to English."""
    localedirs = {"smoketest": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="smoketest"):
        smoketest.check_translations(localedirs)
    catalog = tmp_path / "fr" / "LC_MESSAGES" / "smoketest.mo"
    catalog.parent.mkdir(parents=True)
    # GNU gettext catalog without any message: magic number, revision, number of
    # messages, offsets of the original and translated message tables, hash table
    catalog.write_bytes(struct.pack("<7I", 0x950412DE, 0, 0, 28, 28, 0, 0))
    assert smoketest.check_translations(localedirs) == "smoketest"


@pytest.mark.parametrize(
    "args, report, h5files",
    (
        ([], None, None),
        (["--smoke-test"], DEFAULT_REPORT, None),
        (["--smoke-test", "report.json"], "report.json", None),
        (["--smoke-test", "report.json", "data.h5"], "report.json", ["data.h5"]),
        (["data.h5", "--smoke-test"], DEFAULT_REPORT, ["data.h5"]),
    ),
)
def test_command_line(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    report: str | None,
    h5files: list[str] | None,
) -> None:
    """The ``--smoke-test`` option takes an optional report path."""
    monkeypatch.setattr(sys, "argv", ["DataLab", *args])
    env = DLExecEnv()
    assert env.smoke_test_report == report
    assert env.h5files == h5files


def test_run_exits_with_check_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``--smoke-test``, DataLab checks its installation and exits with the
    result, without initializing the application."""
    reports = []

    def fake_run_smoke_test(report: str) -> int:
        reports.append(report)
        return 1

    def unexpected_initialization(**_kwargs) -> None:
        pytest.fail("The application must not be initialized")

    monkeypatch.setattr(app, "run_smoke_test", fake_run_smoke_test)
    monkeypatch.setattr(app, "ensure_initialized", unexpected_initialization)
    monkeypatch.setattr(execenv, "smoke_test_report", "report.json")
    with pytest.raises(SystemExit) as info:
        app.run()
    assert info.value.code == 1
    assert reports == ["report.json"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
