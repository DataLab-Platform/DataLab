# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Measure the cost of provenance and capsules in DataLab Desktop.

Usage (from the repository root)::

    python scripts/run_with_env.py python scripts/benchmark_provenance.py
        [--profile quick|representative|stress] [--repeat N] [--output PATH]
        [--case PREFIX]... [--no-isolation] [--sample-ms MS]

Writes a report in the format of DataLab-Capsule's benchmarks (version 1, see
``doc/performance.md`` of DataLab-Capsule). Sections (``--case`` keeps the
cases whose id starts with one of the given prefixes):

- ``capture``: a light operation (constant added to a signal), a costly one
  (Gaussian filter of an image) and an aggregation (average of images), run
  through the ordinary processing path with provenance capture, then with
  capture neutralised in this script only (no setting, no change of DataLab);
  results must be equal. The difference and, when the duration without capture
  is significant, the ratio are reported;
- ``workspace``: saving and reopening the HDF5 workspace of a processing chain;
- ``consult``: describing the provenance of a result, checking its data,
  verifying a step by recomputing it, and verifying a step whose input state is
  no longer available (expected verdict: not verified);
- ``capsule``: the workspace bytes, export and opening through
  :class:`~datalab.gui.capsuleio.CapsuleInputOutput`, then inspection and
  opening through the capsule dialog, with the time during which the Qt
  interface could not process events (``ui_blocked_ms``, measured with a 10 ms
  timer);
- ``ledger``: a ledger of 1000 activities (representative and stress
  profiles): saving, manifest size, consultation of one result, listing all
  activities in the Local replay tab, reopening the dialog, refreshing the
  revisions.

Memory is the resident memory of the whole process tree (DataLab and its
computation processes), summed at the same instant by an observer process.
Local revisions are kept in a temporary folder, never in the user's
configuration. This is a developer script: it needs the test helpers of
DataLab (and pytest).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import importlib.metadata
import json
import os
import os.path as osp
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from datalab_capsule.archive import MANIFEST_NAME
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import create_image, create_signal

from datalab.capsulestore import RevisionStore
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.provenance_unit_test import app
from datalab.widgets.capsule import CapsuleDialog

REPORT_VERSION = 1
PROFILES = ("quick", "representative", "stress")
PACKAGES = (
    "datalab-platform",
    "DataLab-Capsule",
    "Sigima",
    "guidata",
    "PlotPy",
    "numpy",
    "h5py",
    "QtPy",
    "PyQt5",
    "PyQt6",
    "psutil",
)
MIB = 1024**2
#: Sizes by profile: signal points, image side, images averaged, chain images,
#: ledger activities.
SIZES = {
    "quick": {"points": 1000, "image": 512, "average": 3, "chain": 512, "ledger": 0},
    "representative": {
        "points": 100_000,
        "image": 2048,
        "average": 5,
        "chain": 1024,
        "ledger": 1000,
    },
    "stress": {
        "points": 1_000_000,
        "image": 4096,
        "average": 5,
        "chain": 4096,
        "ledger": 1000,
    },
}
OBSERVER = r"""
import json, sys, threading, time
import psutil
root = psutil.Process(int(sys.argv[1]))
period = float(sys.argv[2]) / 1000
state = {"peak": 0, "samples": 0}
lock = threading.Lock()
def rss():
    total = 0
    for proc in [root] + root.children(recursive=True):
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total
def sample():
    while True:
        value = rss()
        with lock:
            state["peak"] = max(state["peak"], value)
            state["samples"] += 1
        time.sleep(period)
threading.Thread(target=sample, daemon=True).start()
for line in sys.stdin:
    current = rss()
    with lock:
        reply = {"current": current, "peak": max(state["peak"], current),
                 "samples": state["samples"]}
        if line.strip() == "reset":
            state["peak"], state["samples"] = current, 0
    print(json.dumps(reply), flush=True)
"""


class TreeObserver:
    """Resident memory of this process and its children, sampled by another one."""

    def __init__(self, period_ms: float) -> None:
        self.period_ms = period_ms
        self.process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", OBSERVER, str(os.getpid()), str(period_ms)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def __command(self, command: str) -> dict[str, int]:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def reset(self) -> int:
        """Start a new peak; return the current resident memory."""
        return self.__command("reset")["current"]

    def read(self) -> dict[str, int]:
        """Return the current memory, the peak and the samples since reset."""
        return self.__command("read")

    def close(self) -> None:
        """Stop the observer."""
        self.process.stdin.close()
        self.process.wait(10)


def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    for name, path in (("commit", root), ("capsule_commit", None)):
        if path is None:
            import datalab_capsule  # pylint: disable=import-outside-toplevel

            path = osp.dirname(datalab_capsule.__file__)
        try:
            versions[name] = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            versions[name] = None
    return versions


def _environment(isolation: bool) -> dict[str, Any]:
    import psutil  # pylint: disable=import-outside-toplevel

    freq = psutil.cpu_freq()
    return {
        "runtime_kind": "desktop",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "cpu_max_mhz": None if freq is None else freq.max,
        "qt": QC.qVersion(),
        "process_isolation": isolation,
    }


def _process_events(duration_s: float) -> None:
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        QW.QApplication.processEvents()
        time.sleep(0.001)


def blocked(task: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run *task* from the event loop; return its result, duration and blocking.

    A 10 ms timer ticks while the event loop runs: the longest gap between two
    ticks around the task is the time during which the interface could not
    process events.
    """
    ticks: list[float] = []
    timer = QC.QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start()
    _process_events(0.1)
    start = time.perf_counter()
    result = task()
    duration = time.perf_counter() - start
    _process_events(0.1)
    timer.stop()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    return result, duration * 1000, (max(gaps) if gaps else float("nan")) * 1000


class Report:
    """Cases of a report, with the common aggregates."""

    def __init__(self, args: argparse.Namespace, observer: TreeObserver | None):
        self.args = args
        self.observer = observer
        self.cases: list[dict[str, Any]] = []

    def wanted(self, case_id: str) -> bool:
        """Return True when *case_id* is selected by ``--case``."""
        return not self.args.case or any(case_id.startswith(p) for p in self.args.case)

    def add(
        self,
        case_id: str,
        params: dict[str, Any],
        durations: list[float] | None = None,
        sizes: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Add a case and print its summary."""
        entry: dict[str, Any] = {
            "id": case_id,
            "params": params,
            "mode": {
                "execution": "desktop",
                "process_isolation": not self.args.no_isolation,
            },
            "status": "ok",
            "sizes": sizes or {},
        }
        if durations:
            entry["durations_ms"] = [round(d, 3) for d in durations]
            entry["median_ms"] = round(statistics.median(durations), 3)
            entry["range_ms"] = [round(min(durations), 3), round(max(durations), 3)]
            entry["min_ms"] = round(min(durations), 3)
        entry["memory"] = memory or {}
        entry.update(extra)
        self.cases.append(entry)
        summary = f"{case_id}: {entry['status']}"
        if "median_ms" in entry:
            summary += f", median {entry['median_ms']:.1f} ms"
        if "ui_blocked_ms" in entry:
            summary += f", UI blocked {max(entry['ui_blocked_ms']):.0f} ms"
        rss = entry["memory"].get("rss_tree") or {}
        if rss.get("peak_delta_bytes") is not None:
            summary += f", RSS +{rss['peak_delta_bytes'] / MIB:.1f} MiB"
        print(summary, file=sys.stderr)
        return entry

    def error(self, case_id: str, params: dict[str, Any], exc: Exception) -> None:
        """Add a case that failed."""
        self.cases.append(
            {
                "id": case_id,
                "params": params,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"{case_id}: error, {type(exc).__name__}: {exc}", file=sys.stderr)

    def memory(self, task: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
        """Run *task* once and return its result and the process tree memory."""
        if self.observer is None:
            return task(), {"rss_tree": {"value": None, "reason": "disabled"}}
        gc.collect()
        initial = self.observer.reset()
        result = task()
        read = self.observer.read()
        return result, {
            "rss_tree": {
                "initial_bytes": initial,
                "peak_bytes": read["peak"],
                "final_bytes": read["current"],
                "peak_delta_bytes": read["peak"] - initial,
                "samples": read["samples"],
                "period_ms": self.observer.period_ms,
                "method": "sum of the resident memory of the process tree at each "
                "sample, observed from another process",
            }
        }


def _timed(task: Callable[[], Any], repeat: int) -> tuple[list[float], Any]:
    durations, result = [], None
    for _index in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = task()
        durations.append((time.perf_counter() - start) * 1000)
    return durations, result


@contextlib.contextmanager
def capture_disabled(win) -> Any:
    """Neutralise provenance capture in *win*, for this benchmark only.

    ``Recorder.begin`` returns None when an execution is not captured, and the
    processing path then records nothing: the rest of the path is unchanged.
    """
    provenance = win.provenance
    provenance.begin = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        del provenance.begin


def _image(title: str, size: int, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    data = 2000 + 1500 * np.sin(6 * xx) * np.cos(4 * yy) + rng.normal(0, 40, xx.shape)
    return create_image(title, data.astype(np.float64), units=("mm", "mm", "counts"))


def _new_objects(panel, before: set[str]) -> list[Any]:
    return [
        panel.objmodel[u] for u in panel.objmodel.get_object_ids() if u not in before
    ]


def _run(panel, compute: Callable[[], None]) -> Any:
    before = set(panel.objmodel.get_object_ids())
    compute()
    (obj,) = _new_objects(panel, before)
    return obj


def _remove(panel, obj) -> None:
    panel.objview.select_objects([get_uuid(obj)])
    panel.remove_object(force=True)


def bench_capture(win, report: Report, sizes: dict[str, int], repeat: int) -> None:
    """Capture overhead on the ordinary processing path."""
    spanel, ipanel = win.signalpanel, win.imagepanel
    points = sizes["points"]
    x = np.linspace(0.0, 1.0, points)
    signal = create_signal("S", x, np.sin(20 * x), units=("s", "V"))
    spanel.add_object(signal)
    image = _image("I", sizes["image"], 0)
    ipanel.add_object(image)
    averaged = [_image(f"A{i}", sizes["chain"], i + 1) for i in range(sizes["average"])]
    for obj in averaged:
        ipanel.add_object(obj)
    operations = (
        (
            "capture.light",
            {"operation": "addition_constant", "points": points},
            spanel,
            [signal],
            lambda: spanel.processor.compute_1_to_1(
                sips.addition_constant,
                param=sigima.params.ConstantParam.create(value=1.0),
                edit=False,
            ),
        ),
        (
            "capture.costly",
            {"operation": "gaussian_filter", "shape": [sizes["image"]] * 2},
            ipanel,
            [image],
            lambda: ipanel.processor.compute_1_to_1(
                sipi.gaussian_filter,
                param=sigima.params.GaussianParam.create(sigma=2.0),
                edit=False,
            ),
        ),
        (
            "capture.aggregation",
            {
                "operation": "average",
                "inputs": sizes["average"],
                "shape": [sizes["chain"]] * 2,
            },
            ipanel,
            averaged,
            lambda: ipanel.processor.compute_n_to_1(sipi.average, edit=False),
        ),
    )
    for case_id, params, panel, inputs, compute in operations:
        if not report.wanted(case_id):
            continue
        try:
            panel.objview.select_objects([get_uuid(o) for o in inputs])
            results: dict[bool, Any] = {}
            durations: dict[bool, list[float]] = {True: [], False: []}
            activities = len(win.provenance.ledger.activities)
            for _index in range(repeat + 1):  # one warm-up run of each
                for captured in (True, False):
                    panel.objview.select_objects([get_uuid(o) for o in inputs])
                    gc.collect()
                    ctx = (
                        contextlib.nullcontext() if captured else capture_disabled(win)
                    )
                    with ctx:
                        start = time.perf_counter()
                        obj = _run(panel, compute)
                        elapsed = (time.perf_counter() - start) * 1000
                    if _index:
                        durations[captured].append(elapsed)
                    data = getattr(obj, "data", None)
                    results[captured] = np.array(obj.y if data is None else data)
                    _remove(panel, obj)
            assert np.array_equal(results[True], results[False]), "results differ"
            recorded = len(win.provenance.ledger.activities) - activities
            assert recorded == repeat + 1, f"{recorded} activities recorded"
            panel.objview.select_objects([get_uuid(o) for o in inputs])
            _obj, memory = report.memory(lambda: _run(panel, compute))
            _remove(panel, _obj)
            on, off = durations[True], durations[False]
            report.add(f"{case_id}.with-capture", params, on, memory=memory)
            report.add(f"{case_id}.without-capture", params, off)
            median_on, median_off = statistics.median(on), statistics.median(off)
            report.add(
                f"{case_id}.overhead",
                params,
                overhead_ms=round(median_on - median_off, 3),
                ratio=round(median_on / median_off, 3) if median_off >= 1.0 else None,
                ratio_reason=None
                if median_off >= 1.0
                else "duration without capture below 1 ms",
                results_equal=True,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error(case_id, params, exc)


def build_chain(win, size: int) -> list[Any]:
    """Build an image chain: source, two filters, an average; return objects."""
    panel = win.imagepanel
    source = _image("Source", size, 10)
    panel.add_object(source)
    objects = [source]
    for sigma in (1.0, 2.0):
        panel.objview.select_objects([get_uuid(objects[-1])])
        objects.append(
            _run(
                panel,
                lambda s=sigma: panel.processor.compute_1_to_1(
                    sipi.gaussian_filter,
                    param=sigima.params.GaussianParam.create(sigma=s),
                    edit=False,
                ),
            )
        )
    panel.objview.select_objects([get_uuid(o) for o in objects])
    objects.append(
        _run(panel, lambda: panel.processor.compute_n_to_1(sipi.average, edit=False))
    )
    return objects


def bench_workspace(
    win, report: Report, sizes: dict[str, int], repeat: int, tmpdir: str
) -> str:
    """Save and reopen the chain workspace; return its file name."""
    path = osp.join(tmpdir, "chain.h5")
    objects = build_chain(win, sizes["chain"])
    params = {"objects": len(objects), "shape": [sizes["chain"]] * 2}
    arrays = sum(int(o.data.nbytes) for o in objects)
    if report.wanted("workspace.save"):
        try:
            durations, _ = _timed(lambda: win.save_h5_workspace(path), repeat)
            _none, memory = report.memory(lambda: win.save_h5_workspace(path))
            report.add(
                "workspace.save",
                params,
                durations,
                sizes={"array_bytes": arrays, "hdf5_bytes": osp.getsize(path)},
                memory=memory,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error("workspace.save", params, exc)
    else:
        win.save_h5_workspace(path)
    if report.wanted("workspace.reopen"):
        try:
            durations, _ = _timed(
                lambda: win.load_h5_workspace([path], reset_all=True), repeat
            )
            _none, memory = report.memory(
                lambda: win.load_h5_workspace([path], reset_all=True)
            )
            report.add(
                "workspace.reopen",
                params,
                durations,
                sizes={"hdf5_bytes": osp.getsize(path)},
                memory=memory,
                file_status=win.provenance.file_status,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error("workspace.reopen", params, exc)
    return path


def bench_consult(win, report: Report, repeat: int, path: str) -> None:
    """Consult, check and verify the provenance of the chain result."""
    win.load_h5_workspace([path], reset_all=True)
    panel = win.imagepanel
    result = panel.objmodel[panel.objmodel.get_object_ids()[-1]]
    uuid_ = get_uuid(result)
    activity_id = win.provenance.ledger.activities[-1]["activity_id"]
    params = {
        "object": result.title,
        "activities": len(win.provenance.ledger.activities),
    }
    execution = win.signalpanel.processor.execution
    for case_id, task in (
        ("consult.describe", lambda: win.provenance.describe_object(uuid_)),
        (
            "consult.check-data",
            lambda: win.provenance.describe_object(uuid_, check_data=True),
        ),
        ("consult.verify", lambda: win.verify_provenance_activity(activity_id)),
    ):
        if not report.wanted(case_id):
            continue
        try:
            before = execution.computations
            durations, last = _timed(task, repeat)
            computations = execution.computations - before
            if case_id != "consult.verify":
                assert computations == 0, f"{computations} computations"
            extra = {"computations": computations}
            if case_id == "consult.verify":
                extra["verdict"] = last["verdict"]
                assert last["verdict"] in ("exact", "within_tolerance"), last
            report.add(case_id, params, durations, **extra)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error(case_id, params, exc)
    case_id = "consult.verify-unavailable-input"
    if report.wanted(case_id):
        try:
            # The input of the first filter is modified after the computation:
            # its recorded state is no longer held by any object.
            source = panel.objmodel[panel.objmodel.get_object_ids()[0]]
            source.data = source.data + 1.0
            first = win.provenance.ledger.activities[0]["activity_id"]
            durations, last = _timed(
                lambda: win.verify_provenance_activity(first), repeat
            )
            assert last["verdict"] == "not_verified", last["verdict"]
            report.add(case_id, params, durations, verdict=last["verdict"])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error(case_id, params, exc)


def bench_capsule(win, report: Report, repeat: int, path: str, tmpdir: str) -> None:
    """Capsule paths of the Desktop adapter and of the dialog."""
    win.load_h5_workspace([path], reset_all=True)
    capsuleio = win.capsuleio
    capsule = osp.join(tmpdir, "chain.dlcapsule")
    params = {"workspace": osp.basename(path)}
    sizes = {"hdf5_bytes": osp.getsize(path)}

    def _export() -> str:
        return capsuleio.export(capsule, name="benchmark")

    cases = (
        ("capsule.workspace-bytes", capsuleio.workspace_bytes),
        ("capsule.export", _export),
        ("capsule.open", lambda: capsuleio.open(capsule)),
    )
    for case_id, task in cases:
        if not report.wanted(case_id) and case_id != "capsule.export":
            continue
        try:
            durations, _ = _timed(task, repeat)
            _none, memory = report.memory(task)
            sizes["archive_bytes"] = (
                osp.getsize(capsule) if osp.exists(capsule) else None
            )
            if report.wanted(case_id):
                report.add(case_id, params, durations, sizes=dict(sizes), memory=memory)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error(case_id, params, exc)
    dialog = CapsuleDialog(win)
    try:
        for case_id, task in (
            ("capsule.dialog-inspect", lambda: dialog.inspect_tab.choose(capsule)),
            ("capsule.dialog-open", lambda: dialog.inspect_tab.open(confirm=False)),
            ("capsule.dialog-replay-list", dialog.replay_tab.refresh),
        ):
            if not report.wanted(case_id):
                continue
            try:
                durations, ui_blocked = [], []
                for _index in range(repeat):
                    done, duration, gap = blocked(task)
                    assert done is not False, dialog.error
                    durations.append(duration)
                    ui_blocked.append(round(gap, 3))
                _none, memory = report.memory(task)
                report.add(
                    case_id,
                    params,
                    durations,
                    sizes=dict(sizes),
                    memory=memory,
                    ui_blocked_ms=ui_blocked,
                    ui_blocked_method="longest gap between the ticks of a 10 ms "
                    "Qt timer around the task",
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                report.error(case_id, params, exc)
    finally:
        dialog.close()


def bench_ledger(win, report: Report, count: int, repeat: int, tmpdir: str) -> None:
    """A long ledger: small signals processed *count* times."""
    panel = win.signalpanel
    x = np.linspace(0.0, 1.0, 100)
    obj = create_signal("L0", x, np.sin(10 * x), units=("s", "V"))
    panel.add_object(obj)
    first = obj
    start = time.perf_counter()
    for _index in range(count):
        panel.objview.select_objects([get_uuid(obj)])
        obj = _run(
            panel,
            lambda: panel.processor.compute_1_to_1(
                sips.addition_constant,
                param=sigima.params.ConstantParam.create(value=1.0),
                edit=False,
            ),
        )
    build_s = time.perf_counter() - start
    activities = len(win.provenance.ledger.activities)
    params = {"activities": activities, "points": 100}
    path = osp.join(tmpdir, "ledger.h5")
    capsule = osp.join(tmpdir, "ledger.dlcapsule")
    sizes: dict[str, Any] = {"objects": len(panel), "activities": activities}
    report.add("ledger.build", params, [build_s * 1000], sizes=dict(sizes))

    def _manifest_size() -> int:
        with zipfile.ZipFile(capsule) as archive:
            return archive.getinfo(MANIFEST_NAME).file_size

    steps = (
        ("ledger.save", lambda: win.save_h5_workspace(path)),
        ("ledger.export", lambda: win.capsuleio.export(capsule, name="ledger")),
        ("ledger.describe-last", lambda: win.provenance.describe_object(get_uuid(obj))),
        (
            "ledger.describe-first",
            lambda: win.provenance.describe_object(get_uuid(first)),
        ),
    )
    for case_id, task in steps:
        if not report.wanted(case_id):
            continue
        try:
            durations, _ = _timed(task, repeat)
            if osp.exists(path):
                sizes["hdf5_bytes"] = osp.getsize(path)
            if osp.exists(capsule):
                sizes["archive_bytes"] = osp.getsize(capsule)
                sizes["manifest_bytes"] = _manifest_size()
            report.add(case_id, params, durations, sizes=dict(sizes))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            report.error(case_id, params, exc)
    dialog = None
    try:
        for case_id, task in (
            ("ledger.dialog-open", lambda: CapsuleDialog(win)),
            ("ledger.replay-list", lambda: dialog.replay_tab.refresh()),
            ("ledger.revisions-refresh", lambda: dialog.revisions_tab.refresh()),
        ):
            if case_id == "ledger.dialog-open":
                durations, ui_blocked = [], []
                for _index in range(repeat):
                    if dialog is not None:
                        dialog.close()
                    dialog, duration, gap = blocked(task)
                    durations.append(duration)
                    ui_blocked.append(round(gap, 3))
            elif report.wanted(case_id):
                durations, ui_blocked = [], []
                for _index in range(repeat):
                    _done, duration, gap = blocked(task)
                    durations.append(duration)
                    ui_blocked.append(round(gap, 3))
            else:
                continue
            if report.wanted(case_id):
                extra = {"ui_blocked_ms": ui_blocked}
                if case_id == "ledger.replay-list":
                    extra["rows"] = len(dialog.replay_tab.activity_ids())
                report.add(case_id, params, durations, sizes=dict(sizes), **extra)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        report.error("ledger.dialog", params, exc)
    finally:
        if dialog is not None:
            dialog.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", metavar="PATH")
    parser.add_argument("--case", action="append", default=[], metavar="PREFIX")
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--no-rss", action="store_true")
    parser.add_argument("--sample-ms", type=float, default=5.0)
    parser.add_argument(
        "--ledger",
        type=int,
        metavar="N",
        help="activities of the long ledger (default: by profile, none in quick)",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the benchmark and write its report."""
    args = parse_args(argv)
    execenv.unattended = True
    sizes = dict(SIZES[args.profile])
    if args.ledger is not None:
        sizes["ledger"] = args.ledger
    observer = None if args.no_rss else TreeObserver(args.sample_ms)
    report = Report(args, observer)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    isolation = not args.no_isolation
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with app(isolation=isolation) as win:
                win.capsuleio.revisions = RevisionStore(osp.join(tmpdir, "revisions"))
                bench_capture(win, report, sizes, args.repeat)
                win.reset_all()
                path = bench_workspace(win, report, sizes, args.repeat, tmpdir)
                bench_consult(win, report, args.repeat, path)
                bench_capsule(win, report, args.repeat, path, tmpdir)
                if sizes["ledger"] and report.wanted("ledger"):
                    win.reset_all()
                    bench_ledger(win, report, sizes["ledger"], args.repeat, tmpdir)
                environment = _environment(isolation)
    finally:
        if observer is not None:
            observer.close()
    result = {
        "report_version": REPORT_VERSION,
        "run_id": str(uuid.uuid4()),
        "launcher": osp.basename(__file__),
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": args.profile,
        "repeat": args.repeat,
        "phase": "all",
        "versions": _versions(),
        "environment": environment,
        "memory_methods": {
            "rss_tree": f"resident memory of DataLab and its child processes, "
            f"summed at each sample every {args.sample_ms} ms by an observer "
            "process, during one run measured apart from the durations",
        },
        "cases": report.cases,
    }
    text = json.dumps(result, indent=1)
    if args.output:
        os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(text)
    else:
        print(text)
    return result


if __name__ == "__main__":
    main()
