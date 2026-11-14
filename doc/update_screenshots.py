# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Render documentation screenshots in language-specific, isolated Qt processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("interface", "gallery", "fabry-perot", "spectrum", "blobs")


def generate_screenshots(groups: list[str], output_dir: Path, language: str) -> None:
    """Render the requested scenes without loading or resetting personal settings."""
    from datalab.config import Conf, initialize, reset

    initialize(load_user_config=False)

    import guidata
    from qtpy import QtGui as QG

    from datalab.env import execenv
    from datalab.tests import datalab_test_app_context, destroy_closed_test_windows
    from datalab.tests.features.common import history_panel_app_test
    from datalab.tests.features.utilities import settings_unit_test
    from datalab.tests.scenarios import beautiful_app
    from datalab.tests.scenarios import tutorial_screenshots as scenes

    qapp = guidata.qapplication()
    qapp.setFont(QG.QFont("Segoe UI", 9))
    execenv.unattended = True
    execenv.screenshot_path = str(output_dir / "shots")
    for group in groups:
        if language != "en" and group not in ("interface", "gallery"):
            continue
        reset()
        Conf.rpc_server_enabled.set(False)
        Conf.faulthandler_enabled.set(False)
        Conf.process_isolation_enabled.set(False)
        Conf.show_result_dialog.set(False)
        print(f"Rendering {group} ({language})...", flush=True)
        if group == "interface":
            with datalab_test_app_context(
                size=(1400, 900), console=False, exec_loop=False
            ) as win:
                win.statusBar().hide()
                win.memorystatus.set_demo_mode(True)
                scenes.save_screenshot(
                    win, output_dir / "shots" / f"s_app_at_startup.{language}.png"
                )
                Conf.welcome_on_startup.set(False)
            destroy_closed_test_windows()
            reset()
            settings_unit_test.capture_settings_screenshots()
            history_panel_app_test.test_history_panel(screenshots=True)
        elif group == "gallery":
            beautiful_app.run_beautiful_scenario(screenshots=True)
            beautiful_app.run_blob_detection_on_flower_image(screenshots=True)
        elif language == "en":
            capture = {
                "fabry-perot": scenes.capture_fabry_perot,
                "spectrum": scenes.capture_spectrum,
                "blobs": scenes.capture_blobs,
            }[group]
            capture(output_dir / "tutorials" / group.replace("-", "_"))


def main() -> None:
    """Run each language in a separate process with its own disposable home."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument(
        "--language", nargs="+", choices=("en", "fr"), default=["en", "fr"]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "images")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # DataLab also parses sys.argv; do not pass it the screenshot runner options.
        sys.argv = sys.argv[:1]
        generate_screenshots(args.group, args.output_dir.resolve(), args.language[0])
        return
    for language in args.language:
        with TemporaryDirectory(prefix="datalab-screenshots-") as home:
            env = os.environ.copy()
            env.update(
                USERPROFILE=home,
                HOME=home,
                XDG_CONFIG_HOME=home,
                LANG=language,
                QT_COLOR_MODE="light",
                QT_QPA_PLATFORM="offscreen",
                QT_SCALE_FACTOR="1",
                QT_AUTO_SCREEN_SCALE_FACTOR="0",
                PYTHONIOENCODING="utf-8",
            )
            env.pop("DATALAB_PLUGINS", None)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--language",
                    language,
                    "--output-dir",
                    str(args.output_dir.resolve()),
                    "--group",
                    *args.group,
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
