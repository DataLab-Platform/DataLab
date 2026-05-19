# -*- coding: utf-8 -*-
"""
Regenerate committed binary graphics resources from their SVG sources.

This script produces three files that are checked into git so that the
release pipeline does not need any external rasteriser:

    * ``resources/DataLab.ico``   - multi-size Windows icon used by the EXE
    * ``wix/dialog.bmp``          - 493x312 background of the WiX UI dialog
    * ``wix/banner.bmp``          - 493x58 banner used by the WiX UI

Run this script only when the corresponding SVG sources change
(``resources/DataLab.svg``, ``resources/WixUIDialog.svg``,
``resources/WixUIBanner.svg``).

Requirements
------------

* Qt (via ``qtpy`` / PyQt5) - already a hard dependency of DataLab, used
  here through ``QSvgRenderer`` for SVG -> raster conversion.
* Pillow - already pulled transitively by scikit-image and PlotPy.
  Handles ICO assembly (with PNG-compressed 256x256 sub-image) and
  24 bpp BMP v3 encoding for the WiX UI assets.

No external tool (Inkscape, ImageMagick, ...) is required.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = REPO_ROOT / "resources"
WIX = REPO_ROOT / "wix"

ICO_SIZES = (16, 24, 32, 48, 128, 256)
DIALOG_SIZE = (493, 312)
BANNER_SIZE = (493, 58)


def _render_svg(src: Path, width: int, height: int):
    """Rasterise ``src`` (SVG) at the requested size and return a PIL RGBA image."""
    from PIL import Image
    from qtpy.QtCore import QSize
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtSvg import QSvgRenderer

    if not src.is_file():
        raise FileNotFoundError(src)
    renderer = QSvgRenderer(str(src))
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {src}")
    img = QImage(QSize(width, height), QImage.Format_ARGB32)
    img.fill(0x00000000)  # transparent
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.setRenderHint(QPainter.TextAntialiasing)
    renderer.render(painter)
    painter.end()
    # Convert QImage -> PIL Image (RGBA, top-down).
    img = img.convertToFormat(QImage.Format_RGBA8888)
    ptr = img.constBits()
    try:
        ptr.setsize(img.sizeInBytes())
    except AttributeError:
        # PySide returns a memoryview-like object that already has the right size.
        pass
    return Image.frombytes("RGBA", (width, height), bytes(ptr))


def build_ico() -> None:
    """Generate ``resources/DataLab.ico`` from ``resources/DataLab.svg``.

    Each size is rasterised individually via Qt, then assembled into a
    multi-image ``.ico``. The 256x256 sub-image is stored PNG-compressed
    and the smaller sizes as 32-bit BMP (DIB), matching the byte layout
    of the icon originally committed in 2023 and keeping the file
    ~100 KB instead of ~350 KB.
    """
    src = RESOURCES / "DataLab.svg"
    dst = RESOURCES / "DataLab.ico"

    entries: list[tuple[int, bytes]] = []  # (size, payload bytes as embedded)
    for size in ICO_SIZES:
        img = _render_svg(src, size, size)
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
                    bgra[dst_off + x * 4 : dst_off + x * 4 + 4] = bytes((b, g, r, a))
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


def _svg_to_bmp(src: Path, dst: Path, width: int, height: int) -> None:
    """Rasterise ``src`` (SVG) and save it as a 24 bpp BMP v3 (no alpha).

    WiX UI requires the legacy BMP format (BITMAPINFOHEADER, uncompressed,
    no alpha channel) - which is exactly what Pillow writes by default when
    the input image is in RGB mode.
    """
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = _render_svg(src, width, height)
    # Convert RGBA -> RGB (flatten on white) to strip the alpha channel,
    # then save as BMP v3 (24 bpp, uncompressed).
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(dst, format="BMP")
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def build_wix_bitmaps() -> None:
    """Generate ``wix/dialog.bmp`` and ``wix/banner.bmp`` from their SVG sources."""
    _svg_to_bmp(RESOURCES / "WixUIDialog.svg", WIX / "dialog.bmp", *DIALOG_SIZE)
    _svg_to_bmp(RESOURCES / "WixUIBanner.svg", WIX / "banner.bmp", *BANNER_SIZE)


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

    # A real QApplication is required so QSvgRenderer can access system fonts
    # (Century Gothic, etc.). Do NOT use the "offscreen" platform plugin: it
    # ships without system font integration on Windows.
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    print(f"Using Qt SVG renderer ({type(app).__module__})")

    if "ico" in targets:
        build_ico()
    if "wix" in targets:
        build_wix_bitmaps()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
