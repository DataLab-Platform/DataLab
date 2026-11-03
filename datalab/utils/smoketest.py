# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Installation check (``--smoke-test`` command line option)
=========================================================

Checks that a DataLab installation is complete, without opening the main window:
packages, bundled resources and processing in a separate process. Run it with::

    DataLab.exe --smoke-test report.json
    datalab --smoke-test report.json
    python -m datalab.utils.smoketest report.json

The checks are deliberately limited to what packaging may break, in particular the
resources of the standalone version that are only read on demand (e.g. macro
templates, PDF documentation). A JSON report is written, and the exit code is 0
when no check failed, 1 otherwise.

Only standard library modules are imported at module level: each check imports
what it needs, so that a missing module makes a check fail instead of preventing
the report from being written.
"""

from __future__ import annotations

import gettext
import importlib
import json
import os
import os.path as osp
import pkgutil
import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

#: Default report file name, when ``--smoke-test`` is passed without a path
DEFAULT_REPORT = "datalab_smoke_test.json"

#: Packages that the application imports, with their import name
PACKAGES = ("datalab", "sigima", "sigimax", "plotpy", "guidata", "datalab_capsule")

#: Gettext domains whose French catalog must be available
TRANSLATION_DOMAINS = ("datalab", "sigimax", "sigima", "guidata", "plotpy")

#: Images that must be found, designated with the ``module:filename`` syntax
IMAGES = (
    "datalab:DataLab.svg",
    "datalab:history.svg",
    "datalab:DataLab-Splash.png",
    "sigimax:help_pdf.svg",
    "plotpy:autoscale.svg",
    "guidata:apply.svg",
)

#: Maximum time (s) to wait for the processing done in a separate process
POOL_TIMEOUT = 120.0

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"


@dataclass(frozen=True)
class Check:
    """Installation check.

    Attributes:
        name: Stable identifier, used in the report.
        func: Function running the check: it returns a short description of what
         was checked, and raises an exception when the check fails.
        frozen_only: True if the check is only relevant to the standalone version.
    """

    name: str
    func: Callable[[], str]
    frozen_only: bool = False


def is_frozen() -> bool:
    """Return True if running the standalone (frozen) version of DataLab."""
    return bool(getattr(sys, "frozen", False))


def check_packages() -> str:
    """Import the packages of the application and of its Qt binding.

    Returns:
        Package versions.
    """
    versions, errors = [], []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            versions.append(f"{name} {getattr(module, '__version__', '?')}")
    try:
        import qtpy  # pylint: disable=import-outside-toplevel

        versions.append(f"{qtpy.API_NAME} {qtpy.PYQT_VERSION or qtpy.PYSIDE_VERSION}")
        versions.append(f"Qt {qtpy.QT_VERSION}")
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"qtpy: {type(exc).__name__}: {exc}")
    if errors:
        raise ImportError("; ".join(errors))
    return ", ".join(versions)


def check_build_manifest() -> str:
    """Check the build manifest of the standalone version.

    Returns:
        Python version and number of packages of the build.
    """
    path = osp.join(osp.dirname(sys.executable), "manifest.json")
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    packages = manifest.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError(f"No package listed in {path}")
    return (
        f"built with Python {manifest.get('python_version', '?')}, "
        f"{len(packages)} packages"
    )


def check_macro_templates() -> str:
    """Check that the macro templates provided with DataLab can be read.

    Returns:
        Names of the templates.
    """
    # pylint: disable=import-outside-toplevel
    from datalab.gui import macros_templates
    from datalab.gui.macros_templates import _TEMPLATE_FILES

    for filename in _TEMPLATE_FILES:
        data = pkgutil.get_data(macros_templates.__name__, filename)
        if not data or not data.strip():
            raise ValueError(f"Macro template {filename} is empty or missing")
    return ", ".join(_TEMPLATE_FILES)


def check_pdf_documentation(datapath: str | None = None) -> str:
    """Check the PDF documentation provided with DataLab.

    Args:
        datapath: DataLab data directory (default: the one of the installation).

    Returns:
        Size of the PDF files.
    """
    if datapath is None:
        # pylint: disable=import-outside-toplevel
        from datalab.config import DATAPATH

        datapath = DATAPATH
    sizes = []
    for lang in ("en", "fr"):
        path = osp.join(datapath, "doc", f"DataLab_{lang}.pdf")
        if not osp.isfile(path):
            raise FileNotFoundError(f"PDF documentation not found: {path}")
        with open(path, "rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError(f"Not a PDF file: {path}")
        sizes.append(f"{lang}: {osp.getsize(path) / 2**20:.1f} MB")
    return ", ".join(sizes)


def check_translations(localedirs: dict[str, str] | None = None) -> str:
    """Check that French translation catalogs can be loaded.

    Args:
        localedirs: Locale directory of each gettext domain (default: the one of
         each package of :data:`TRANSLATION_DOMAINS`).

    Returns:
        Checked gettext domains.
    """
    if localedirs is None:
        # pylint: disable=import-outside-toplevel
        from guidata.configtools import get_module_locale_path

        localedirs = {}
        for domain in TRANSLATION_DOMAINS:
            importlib.import_module(domain)
            localedirs[domain] = get_module_locale_path(domain)
    errors = []
    for domain, localedir in localedirs.items():
        try:
            gettext.translation(domain, localedir, languages=["fr"])
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{domain} ({localedir}): {type(exc).__name__}: {exc}")
    if errors:
        raise FileNotFoundError("; ".join(errors))
    return ", ".join(localedirs)


def check_images() -> str:
    """Check that images of DataLab and of its libraries are found.

    Returns:
        Number of images found.
    """
    # pylint: disable=import-outside-toplevel,unused-import
    import guidata.config  # noqa: F401
    import plotpy.config  # noqa: F401
    import sigimax.config  # noqa: F401
    from guidata.configtools import get_image_file_path

    import datalab.config  # noqa: F401

    errors = []
    for name in IMAGES:
        try:
            path = get_image_file_path(name, None)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{name}: {exc}")
            continue
        if not osp.isfile(path) or osp.getsize(path) == 0:
            errors.append(f"{name}: empty or missing file {path}")
    if errors:
        raise FileNotFoundError("; ".join(errors))
    return f"{len(IMAGES)} images"


def check_processing_pool(timeout: float = POOL_TIMEOUT) -> str:
    """Flip an image in the separate process used for processing.

    In the standalone version, the separate process is a new instance of the
    executable, which must import the processing function (see issue #304).

    Args:
        timeout: Maximum time (s) to wait for the result.

    Returns:
        Duration of the processing.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import sigima.proc.image as sipi
    from sigima.objects import create_image

    from datalab.gui.processor import base
    from datalab.gui.processor.base import Worker
    from datalab.gui.processor.geometry_postprocess import GeometricTransformWrapper

    src = create_image("smoke test", np.arange(12.0).reshape(3, 4))
    func = GeometricTransformWrapper(sipi.fliph, "fliph")
    start = time.monotonic()
    created = base.POOL is None
    if created:
        Worker.create_pool()
    try:
        worker = Worker()
        worker.run(func, (src,))
        while not worker.is_computation_finished():
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"No result after {timeout:g} s")
            time.sleep(0.05)
        output = worker.get_result()
    finally:
        if created:
            Worker.terminate_pool(wait=False)
    if output.error_msg:
        raise RuntimeError(output.error_msg)
    np.testing.assert_array_equal(output.result.data, sipi.fliph(src).data)
    return f"image flipped in {time.monotonic() - start:.1f} s"


CHECKS: tuple[Check, ...] = (
    Check("packages", check_packages),
    Check("build_manifest", check_build_manifest, frozen_only=True),
    Check("macro_templates", check_macro_templates),
    Check("pdf_documentation", check_pdf_documentation),
    Check("translations", check_translations),
    Check("images", check_images),
    Check("processing_pool", check_processing_pool),
)


def run_check(check: Check, frozen: bool) -> dict:
    """Run a check and return its result.

    Args:
        check: Check to run.
        frozen: True if running the standalone version.

    Returns:
        Result of the check (name, status, detail, duration, traceback if failed).
    """
    result = {"name": check.name}
    start = time.perf_counter()
    if check.frozen_only and not frozen:
        result.update(status=SKIPPED, detail="only for the standalone version")
    else:
        try:
            result.update(status=PASSED, detail=check.func())
        except KeyboardInterrupt:
            raise
        # A check calling sys.exit must not prevent the report from being written
        except BaseException as exc:  # pylint: disable=broad-except
            result.update(
                status=FAILED,
                detail=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    result["duration"] = round(time.perf_counter() - start, 3)
    return result


def run_smoke_test(
    report_path: str = DEFAULT_REPORT, checks: tuple[Check, ...] | None = None
) -> int:
    """Run the installation checks and write a JSON report.

    A failed check does not prevent the next ones from running.

    Args:
        report_path: Path of the JSON report to write.
        checks: Checks to run (default: :data:`CHECKS`).

    Returns:
        Exit code: 0 if no check failed, 1 otherwise.
    """
    frozen = is_frozen()
    checks = CHECKS if checks is None else checks
    results = [run_check(check, frozen) for check in checks]
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in (PASSED, FAILED, SKIPPED)
    }
    try:
        # pylint: disable=import-outside-toplevel
        from datalab import __version__ as version
    except Exception:  # pylint: disable=broad-except
        version = None
    report = {
        "datalab_version": version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "frozen": frozen,
        "summary": counts,
        "checks": results,
    }
    report_path = osp.abspath(report_path)
    os.makedirs(osp.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for result in results:
        print(f"[{result['status']}] {result['name']}: {result['detail']}")
    print(
        f"DataLab installation check: {counts[PASSED]} passed, "
        f"{counts[FAILED]} failed, {counts[SKIPPED]} skipped (report: {report_path})"
    )
    return 1 if counts[FAILED] else 0


def main(argv: list[str] | None = None) -> int:
    """Run the installation checks from the command line.

    Args:
        argv: Command line arguments (default: ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    # pylint: disable=import-outside-toplevel
    import argparse

    parser = argparse.ArgumentParser(description="Check a DataLab installation")
    parser.add_argument(
        "report",
        nargs="?",
        default=DEFAULT_REPORT,
        help=f"path of the JSON report (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args(argv)
    return run_smoke_test(args.report)


if __name__ == "__main__":
    sys.exit(main())
