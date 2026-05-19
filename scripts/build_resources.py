# -*- coding: utf-8 -*-
"""
Regenerate committed binary graphics resources from their SVG sources.

This script produces three files that are checked into git so that the
release pipeline does not need Inkscape / ImageMagick:

    * ``resources/DataLab.ico``   - multi-size Windows icon used by the EXE
    * ``wix/dialog.bmp``          - 493x312 background of the WiX UI dialog
    * ``wix/banner.bmp``          - 493x58 banner used by the WiX UI

Run this script only when the corresponding SVG sources change
(``resources/DataLab.svg``, ``resources/WixUIDialog.svg``,
``resources/WixUIBanner.svg``).

Requirements
------------

* Inkscape (``inkscape`` on PATH, or ``INKSCAPE`` environment variable
  pointing to ``inkscape.exe``). The Windows default
  ``C:\\Program Files\\Inkscape\\bin\\inkscape.exe`` is tried as a fallback.
* ImageMagick (``magick`` on PATH).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = REPO_ROOT / "resources"
WIX = REPO_ROOT / "wix"

ICO_SIZES = (16, 24, 32, 48, 128, 256)
DIALOG_SIZE = (493, 312)
BANNER_SIZE = (493, 58)


def _find_inkscape() -> str:
    """Locate the Inkscape executable."""
    env = os.environ.get("INKSCAPE")
    if env and Path(env).is_file():
        return env
    found = shutil.which("inkscape")
    if found:
        return found
    fallback = Path(r"C:\Program Files\Inkscape\bin\inkscape.exe")
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError(
        "Inkscape not found. Install Inkscape and add it to PATH, or set "
        "the INKSCAPE environment variable."
    )


def _find_magick() -> str:
    """Locate the ImageMagick `magick` executable."""
    found = shutil.which("magick")
    if found:
        return found
    raise RuntimeError(
        "ImageMagick `magick` not found. Install ImageMagick and add it to PATH."
    )


def _run(cmd: list[str]) -> None:
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def build_ico(inkscape: str, magick: str) -> None:
    """Generate ``resources/DataLab.ico`` from ``resources/DataLab.svg``."""
    src = RESOURCES / "DataLab.svg"
    dst = RESOURCES / "DataLab.ico"
    if not src.is_file():
        raise FileNotFoundError(src)
    with tempfile.TemporaryDirectory() as tmp:
        png_paths: list[str] = []
        for size in ICO_SIZES:
            png = Path(tmp) / f"tmp-{size}.png"
            _run([inkscape, str(src), "-o", str(png), "-w", str(size), "-h", str(size)])
            png_paths.append(str(png))
        _run([magick, *png_paths, str(dst)])
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def _svg_to_bmp(
    inkscape: str, magick: str, src: Path, dst: Path, width: int, height: int
) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "tmp.png"
        _run([inkscape, str(src), "-o", str(png), "-w", str(width), "-h", str(height)])
        # bmp3: format is required by WiX UI (legacy BMP without alpha).
        _run([magick, str(png), f"bmp3:{dst}"])
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def build_wix_bitmaps(inkscape: str, magick: str) -> None:
    """Generate ``wix/dialog.bmp`` and ``wix/banner.bmp`` from their SVG sources."""
    _svg_to_bmp(
        inkscape,
        magick,
        RESOURCES / "WixUIDialog.svg",
        WIX / "dialog.bmp",
        *DIALOG_SIZE,
    )
    _svg_to_bmp(
        inkscape,
        magick,
        RESOURCES / "WixUIBanner.svg",
        WIX / "banner.bmp",
        *BANNER_SIZE,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "targets",
        nargs="*",
        choices=("ico", "wix", "all"),
        default=["all"],
        help="Subset of resources to regenerate (default: all).",
    )
    args = parser.parse_args(argv)
    targets = set(args.targets)
    if "all" in targets:
        targets = {"ico", "wix"}

    inkscape = _find_inkscape()
    magick = _find_magick()
    print(f"Using Inkscape: {inkscape}")
    print(f"Using ImageMagick: {magick}")

    if "ico" in targets:
        build_ico(inkscape, magick)
    if "wix" in targets:
        build_wix_bitmaps(inkscape, magick)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
