# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Regression tests for the documentation screenshot pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.tests.scenarios.tutorial_screenshots import save_screenshot
from datalab.utils import qthelpers as qth


def test_screenshot_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Localized captures must honor the requested directory, not modify doc assets."""
    with qt_app_context():
        widget = QW.QLabel("Screenshot output")
        try:
            monkeypatch.setenv("LANG", "fr")
            monkeypatch.setattr(execenv, "screenshot_path", str(tmp_path))
            monkeypatch.setattr(qth, "SHOTPATH", str(tmp_path / "legacy"))
            qth.grab_save_window(widget, "settings_example", add_timestamp=False)
            assert not QG.QImage(str(tmp_path / "settings_example.fr.png")).isNull()
        finally:
            widget.close()


def test_screenshot_save_failure(tmp_path: Path) -> None:
    """An unwritable destination must not be reported as a successful capture."""
    with qt_app_context():
        widget = QW.QLabel("Screenshot failure")
        try:
            with pytest.raises(OSError, match="Could not save screenshot"):
                save_screenshot(widget, tmp_path / "invalid.extension")
        finally:
            widget.close()


def test_tutorial_screenshot_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render the actual tutorial workflows without touching an existing user INI."""
    root = Path(__file__).resolve().parents[3]
    script = root / "doc" / "update_screenshots.py"
    if not script.is_file():
        pytest.skip("Screenshot generator is available in a source checkout only")
    home = tmp_path / "personal-home"
    config = home / ".DataLab_v1" / "DataLab_v1.ini"
    config.parent.mkdir(parents=True)
    original = b"[main]\nplugins_enabled = False\n"
    config.write_bytes(original)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    output = tmp_path / "images"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    expected = {
        "fabry_perot": ("13", "16", "24"),
        "spectrum": ("06", "09", "20", "21"),
        "blobs": ("01", "02", "05", "11"),
    }
    for tutorial, names in expected.items():
        for name in names:
            image = QG.QImage(str(output / "tutorials" / tutorial / f"{name}.png"))
            assert not image.isNull(), (tutorial, name)
            assert image.width() >= 250 and image.height() >= 500
    for language in ("en", "fr"):
        for name in (
            "s_app_at_startup",
            "settings_main",
            "history_panel",
            "s_beautiful",
            "i_beautiful",
            "i_blob_detection_flower",
        ):
            image = QG.QImage(str(output / "shots" / f"{name}.{language}.png"))
            assert not image.isNull(), (name, language)
            if name == "settings_main":
                assert (image.width(), image.height()) == (900, 700)
    assert config.read_bytes() == original
    assert set(home.rglob("*")) == {config.parent, config}
