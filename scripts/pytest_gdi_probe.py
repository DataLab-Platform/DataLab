# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Standalone pytest plugin: GDI object probe for DataLab tests (Windows-only).
===========================================================================

Traces the number of Windows GDI objects during a pytest run, at two levels:

* a global time series (background sampler thread) plus the process peak;
* a per-test delta (GDI objects retained by each test), to pinpoint leaks.

It is a **self-contained, injected** plugin: it lives in DataLab's tooling
(``scripts/``) and is loaded on the fly with ``-p pytest_gdi_probe`` -- it is
*never* installed into the test venv. The very same working-copy file therefore
instruments every cell of the comparison matrix (PyPI / main / develop)
identically, without touching DataLab's shipped code or its conftest, and
without having to commit the probe to any DataLab branch.

Usage (the directory holding this file must be on ``PYTHONPATH``)::

    pytest datalab -p pytest_gdi_probe --gdi-probe \
        --gdi-csv-prefix C:\\path\\to\\run

Outputs (when ``--gdi-probe`` is given and the platform is Windows):

* ``<prefix>.gdi-pertest.csv`` -- nodeid, before, after, delta, duration_s
* ``<prefix>.gdi-timeline.csv`` -- elapsed_s, gdi_count, nodeid
* a terminal summary (start / end / net / peak + top-N growing tests)

On non-Windows platforms the plugin is inert: the run proceeds normally and the
terminal summary reports that the probe is unavailable.

The module is also runnable directly for a quick self-test::

    python pytest_gdi_probe.py --selftest
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import threading
import time

try:  # pytest is present when loaded as a plugin, but not for --selftest.
    import pytest

    _hookimpl = pytest.hookimpl
except ImportError:  # pragma: no cover - self-test path without pytest
    pytest = None

    def _hookimpl(**_kwargs):
        """No-op stand-in for ``pytest.hookimpl`` when pytest is absent."""

        def _decorator(func):
            return func

        return _decorator

# --- Low-level GDI access (Windows-only) -------------------------------------

# GetGuiResources(hProcess, uiFlags) flags (winuser.h):
GR_GDIOBJECTS = 0  # count of GDI objects currently held by the process
GR_GDIOBJECTS_PEAK = 2  # peak count of GDI objects (Windows 7+)

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32.GetGuiResources.restype = wintypes.DWORD
    _user32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []


def is_available() -> bool:
    """Return whether GDI probing is supported on this platform."""
    return _IS_WINDOWS


def gdi_count() -> int | None:
    """Return the number of GDI objects held by the current process.

    Returns:
        The GDI object count, or ``None`` if not running on Windows.
    """
    if not _IS_WINDOWS:
        return None
    return int(_user32.GetGuiResources(_kernel32.GetCurrentProcess(), GR_GDIOBJECTS))


def gdi_peak() -> int | None:
    """Return the peak number of GDI objects held by the current process.

    Uses ``GR_GDIOBJECTS_PEAK`` so the OS reports the peak directly, without
    relying on the sampling thread.

    Returns:
        The peak GDI object count, or ``None`` if not running on Windows.
    """
    if not _IS_WINDOWS:
        return None
    return int(
        _user32.GetGuiResources(_kernel32.GetCurrentProcess(), GR_GDIOBJECTS_PEAK)
    )


def settle() -> None:
    """Flush deferred deletions so the GDI count reflects *retained* objects.

    Qt frees GDI-backed resources lazily (via ``deleteLater``/cyclic GC), so a
    raw post-test count is noisy. We run a garbage collection and dispatch any
    pending ``DeferredDelete`` events, *without* a full ``processEvents()``
    (which could have side effects on the test under measure).
    """
    gc.collect()
    try:  # Qt may not be importable in every context; degrade gracefully.
        from qtpy.QtCore import QCoreApplication, QEvent  # noqa: PLC0415

        app = QCoreApplication.instance()
        if app is not None:
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    except Exception:  # noqa: BLE001 - best-effort settling, never fatal
        pass


# --- Background time-series sampler -------------------------------------------


class GdiSampler:
    """Background thread sampling the GDI count at a fixed interval.

    Samples are stored as ``(elapsed_s, count, nodeid)`` tuples, where
    ``nodeid`` is the test running when the sample was taken (best-effort).
    """

    def __init__(self, interval_ms: int = 250) -> None:
        self._interval = max(10, int(interval_ms)) / 1000.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0 = 0.0
        self.samples: list[tuple[float, int, str]] = []
        self.current_nodeid: str = ""

    def start(self) -> None:
        """Start sampling in a daemon thread (no-op if unavailable)."""
        if not _IS_WINDOWS or self._thread is not None:
            return
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gdi-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            count = gdi_count()
            if count is not None:
                self.samples.append(
                    (time.perf_counter() - self._t0, count, self.current_nodeid)
                )
            self._stop.wait(self._interval)

    def stop(self) -> None:
        """Stop sampling and join the thread."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    @property
    def peak(self) -> int:
        """Highest GDI count observed by the sampler (0 if no samples)."""
        return max((c for _, c, _ in self.samples), default=0)


# --- pytest plugin state ------------------------------------------------------


class _ProbeState:
    """Holds the live probe state for the session."""

    def __init__(self, csv_prefix: str, interval_ms: int) -> None:
        self.csv_prefix = csv_prefix
        self.sampler = GdiSampler(interval_ms)
        self.start_count: int | None = None
        self.end_count: int | None = None
        # Per-test rows: (nodeid, before, after, delta, duration_s)
        self.pertest: list[tuple[str, int, int, int, float]] = []


# Single-session module-global state (a -p plugin is a singleton per run).
_STATE: _ProbeState | None = None


# --- pytest hooks -------------------------------------------------------------


def pytest_addoption(parser):
    """Register the probe's command-line options."""
    group = parser.getgroup("gdi-probe", "GDI object probe (Windows-only)")
    group.addoption(
        "--gdi-probe",
        action="store_true",
        default=False,
        help="Trace the number of Windows GDI objects during the test run.",
    )
    group.addoption(
        "--gdi-probe-interval",
        action="store",
        type=int,
        default=250,
        metavar="MS",
        help="Sampling interval for the GDI time series, in ms (default: 250).",
    )
    group.addoption(
        "--gdi-csv-prefix",
        action="store",
        default="",
        metavar="PREFIX",
        help="Path prefix for the CSV outputs "
        "(<prefix>.gdi-pertest.csv / <prefix>.gdi-timeline.csv). "
        "Defaults to './gdi_probe'.",
    )


def pytest_configure(config):
    """Start the sampler and record the baseline GDI count."""
    global _STATE  # noqa: PLW0603 - single-session plugin state
    if not config.getoption("--gdi-probe"):
        return
    prefix = config.getoption("--gdi-csv-prefix") or "gdi_probe"
    interval = config.getoption("--gdi-probe-interval")
    _STATE = _ProbeState(prefix, interval)
    if is_available():
        _STATE.start_count = gdi_count()
        _STATE.sampler.start()


@_hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: ARG001 - pytest hook signature
    """Bracket each test with a before/after GDI measurement."""
    if _STATE is None or not is_available():
        yield
        return
    _STATE.sampler.current_nodeid = item.nodeid
    before = gdi_count() or 0
    t0 = time.perf_counter()
    yield  # run the full per-item protocol (setup + call + teardown)
    settle()
    after = gdi_count() or 0
    _STATE.pertest.append(
        (item.nodeid, before, after, after - before, time.perf_counter() - t0)
    )


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest hook signature
    """Stop the sampler and write the CSV outputs."""
    if _STATE is None or not is_available():
        return
    _STATE.sampler.stop()
    _STATE.end_count = gdi_count()
    # Per-test deltas.
    pertest_path = f"{_STATE.csv_prefix}.gdi-pertest.csv"
    try:
        with open(pertest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["nodeid", "gdi_before", "gdi_after", "gdi_delta", "duration_s"]
            )
            for nodeid, before, after, delta, dur in _STATE.pertest:
                writer.writerow([nodeid, before, after, delta, f"{dur:.3f}"])
    except OSError:
        pass
    # Time series.
    timeline_path = f"{_STATE.csv_prefix}.gdi-timeline.csv"
    try:
        with open(timeline_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["elapsed_s", "gdi_count", "nodeid"])
            for elapsed, count, nodeid in _STATE.sampler.samples:
                writer.writerow([f"{elapsed:.3f}", count, nodeid])
    except OSError:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    """Print the GDI probe summary and the top-N growing tests."""
    if not config.getoption("--gdi-probe"):
        return
    tr = terminalreporter
    tr.write_sep("=", "GDI object probe")
    if not is_available():
        tr.write_line("GDI probe: unavailable (non-Windows platform).")
        return
    if _STATE is None:
        return
    start = _STATE.start_count
    end = _STATE.end_count
    peak = max(_STATE.sampler.peak, gdi_peak() or 0)
    net = (end - start) if (start is not None and end is not None) else None
    tr.write_line(
        f"GDI objects  start={start}  end={end}  "
        f"net={'+' if (net or 0) >= 0 else ''}{net}  peak={peak}"
    )
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        tr.write_line(
            "Note: the offscreen platform exercises almost no GDI (no native "
            "windows), so near-zero GDI is expected here -- a real leak shows up "
            "as growing RAM/working-set, not GDI. Run with native windows "
            "(--show-windows) for a meaningful GDI signal."
        )
    samples = len(_STATE.sampler.samples)
    tr.write_line(
        f"Time series: {samples} samples -> {_STATE.csv_prefix}.gdi-timeline.csv"
    )
    tr.write_line(f"Per-test deltas -> {_STATE.csv_prefix}.gdi-pertest.csv")
    top = sorted(_STATE.pertest, key=lambda r: r[3], reverse=True)[:20]
    growing = [r for r in top if r[3] > 0]
    if growing:
        tr.write_line("Top tests by GDI growth (delta objects):")
        for nodeid, before, after, delta, _dur in growing:
            tr.write_line(f"  +{delta:>5}  ({before} -> {after})  {nodeid}")
    else:
        tr.write_line("No net GDI growth attributed to individual tests.")


# --- Stand-alone self-test ----------------------------------------------------


def _selftest() -> int:
    """Quick sanity check of the GDI primitives, runnable without pytest."""
    if not is_available():
        print("GDI probe self-test: SKIPPED (non-Windows platform).")
        return 0
    count = gdi_count()
    peak = gdi_peak()
    print(f"gdi_count() = {count}")
    print(f"gdi_peak()  = {peak}")
    assert isinstance(count, int) and count >= 0, "gdi_count must be a non-negative int"
    assert isinstance(peak, int) and peak >= count, "gdi_peak must be >= gdi_count"
    sampler = GdiSampler(interval_ms=20)
    sampler.start()
    time.sleep(0.2)
    sampler.stop()
    print(f"sampler collected {len(sampler.samples)} samples, peak={sampler.peak}")
    assert sampler.samples, "sampler should collect at least one sample"
    print("GDI probe self-test: OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    raise SystemExit(0)
