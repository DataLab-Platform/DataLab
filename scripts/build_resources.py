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

Pillow handles all image encoding (ICO assembly with PNG-compressed
256x256 sub-image, and 24 bpp BMP v3 for the WiX UI assets). It is
already pulled transitively by scikit-image and PlotPy, so no extra
install step is needed.
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


def _run(cmd: list[str]) -> None:
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def build_ico(inkscape: str) -> None:
    """Generate ``resources/DataLab.ico`` from ``resources/DataLab.svg``.

    Each size is rasterised individually by Inkscape (best quality), then
    assembled into a multi-image ``.ico``. The 256x256 sub-image is stored
    PNG-compressed and the smaller sizes as 32-bit BMP (DIB), matching the
    byte layout of the icon originally committed in 2023 and keeping the
    file ~100 KB instead of ~350 KB.
    """
    from PIL import Image

    src = RESOURCES / "DataLab.svg"
    dst = RESOURCES / "DataLab.ico"
    if not src.is_file():
        raise FileNotFoundError(src)

    import io
    import struct

    entries: list[tuple[int, bytes]] = []  # (size, payload bytes as embedded)
    with tempfile.TemporaryDirectory() as tmp:
        for size in ICO_SIZES:
            png = Path(tmp) / f"tmp-{size}.png"
            _run([inkscape, str(src), "-o", str(png), "-w", str(size), "-h", str(size)])
            img = Image.open(png).convert("RGBA")
            if size >= 256:
                # Store as PNG.
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                entries.append((size, buf.getvalue()))
            else:
                # Store as 32bpp BMP/DIB: BITMAPINFOHEADER + BGRA pixels
                # (bottom-up) + AND mask (all zero, RGBA already has alpha).
                bgra = bytearray(size * size * 4)
                pixels = img.tobytes()  # RGBA, top-down
                row = size * 4
                for y in range(size):
                    src_off = (size - 1 - y) * row
                    dst_off = y * row
                    for x in range(size):
                        r, g, b, a = pixels[src_off + x * 4 : src_off + x * 4 + 4]
                        bgra[dst_off + x * 4 : dst_off + x * 4 + 4] = bytes(
                            (b, g, r, a)
                        )
                # AND mask: 1 bpp, rows padded to 4 bytes, height = size.
                mask_row = ((size + 31) // 32) * 4
                and_mask = b"\x00" * (mask_row * size)
                # BITMAPINFOHEADER: height is doubled (XOR + AND mask).
                header = struct.pack(
                    "<IiiHHIIiiII",
                    40,  # biSize
                    size,  # biWidth
                    size * 2,  # biHeight (xor + and)
                    1,  # biPlanes
                    32,  # biBitCount
                    0,  # biCompression = BI_RGB
                    len(bgra),  # biSizeImage
                    0,
                    0,
                    0,
                    0,
                )
                entries.append((size, header + bytes(bgra) + and_mask))

    # ICONDIR + ICONDIRENTRY[] + image data.
    n = len(entries)
    out = bytearray()
    out += struct.pack("<HHH", 0, 1, n)  # reserved, type=1 (icon), count
    data_offset = 6 + 16 * n
    image_blob = bytearray()
    for size, payload in entries:
        w = 0 if size >= 256 else size
        h = 0 if size >= 256 else size
        out += struct.pack(
            "<BBBBHHII",
            w,
            h,
            0,  # color count (0 for >= 256 colors / 32bpp)
            0,  # reserved
            1,  # planes
            32,  # bit count
            len(payload),
            data_offset,
        )
        data_offset += len(payload)
        image_blob += payload
    out += image_blob
    dst.write_bytes(bytes(out))
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def _svg_to_bmp(inkscape: str, src: Path, dst: Path, width: int, height: int) -> None:
    """Rasterise ``src`` (SVG) and save it as a 24 bpp BMP v3 (no alpha).

    WiX UI requires the legacy BMP format (BITMAPINFOHEADER, uncompressed,
    no alpha channel) - which is exactly what Pillow writes by default when
    the input image is in RGB mode.
    """
    from PIL import Image

    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "tmp.png"
        _run([inkscape, str(src), "-o", str(png), "-w", str(width), "-h", str(height)])
        # Convert RGBA -> RGB (flatten on white) to strip the alpha channel,
        # then save as BMP v3 (24 bpp, uncompressed).
        img = Image.open(png).convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(dst, format="BMP")
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def build_wix_bitmaps(inkscape: str) -> None:
    """Generate ``wix/dialog.bmp`` and ``wix/banner.bmp`` from their SVG sources."""
    _svg_to_bmp(
        inkscape,
        RESOURCES / "WixUIDialog.svg",
        WIX / "dialog.bmp",
        *DIALOG_SIZE,
    )
    _svg_to_bmp(
        inkscape,
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
    print(f"Using Inkscape: {inkscape}")

    if "ico" in targets:
        build_ico(inkscape)
    if "wix" in targets:
        build_wix_bitmaps(inkscape)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
