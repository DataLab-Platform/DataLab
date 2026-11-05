# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Measure how the duration of a processing grows with the number of objects.

Usage (from the repository root)::

    python scripts/run_with_env.py python scripts/benchmark_object_growth.py
        [--iterations N] [--panel signal|image|both] [--profile-at K]...
        [--no-isolation] [--output PATH]

Applies the same light processing (a constant added to the last object) N
times in a row, through the ordinary processing path, so that the workspace
grows by one object per iteration, and records the duration of every
iteration, including the processing of pending Qt events (tree and plot
refresh). With ``--profile-at K``, the iteration K is also profiled with
cProfile and its most expensive functions are printed.

The first iteration starts the computation process and is reported apart.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import os.path as osp
import pstats
import statistics
import sys
import time

import numpy as np
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from qtpy import QtWidgets as QW
from sigima.objects import create_image, create_signal

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.provenance_unit_test import app


def _setup(win, panel_name: str):
    if panel_name == "signal":
        panel = win.signalpanel
        x = np.linspace(0.0, 1.0, 1000)
        panel.add_object(create_signal("S0", x, np.sin(10 * x)))
        func = sips.addition_constant
    else:
        panel = win.imagepanel
        win.set_current_panel("image")
        data = np.random.default_rng(0).random((64, 64))
        panel.add_object(create_image("I0", data))
        func = sipi.addition_constant
    return panel, func


def _step(panel, func, state: dict) -> None:
    panel.objview.select_objects([get_uuid(state["obj"])])
    before = set(panel.objmodel.get_object_ids())
    panel.processor.compute_1_to_1(
        func, param=sigima.params.ConstantParam.create(value=1.0), edit=False
    )
    (uid,) = [u for u in panel.objmodel.get_object_ids() if u not in before]
    state["obj"] = panel.objmodel[uid]
    QW.QApplication.processEvents()


def run_panel(win, panel_name: str, iterations: int, profile_at: list[int]) -> dict:
    """Run the iterations on one panel and return their durations (ms)."""
    panel, func = _setup(win, panel_name)
    state = {"obj": panel.objmodel[panel.objmodel.get_object_ids()[0]]}
    durations: list[float] = []
    profiles: dict[int, str] = {}
    for index in range(1, iterations + 1):
        profiler = cProfile.Profile() if index in profile_at else None
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        _step(panel, func, state)
        if profiler is not None:
            profiler.disable()
        durations.append((time.perf_counter() - start) * 1000)
        if profiler is not None:
            text = io.StringIO()
            stats = pstats.Stats(profiler, stream=text)
            stats.sort_stats("cumulative").print_stats(30)
            profiles[index] = text.getvalue()
            if os.environ.get("DATALAB_GROWTH_PROFILE_DIR"):
                stats.dump_stats(
                    osp.join(
                        os.environ["DATALAB_GROWTH_PROFILE_DIR"],
                        f"{panel_name}-{index}.prof",
                    )
                )
        if index in (1, 2, 10) or index % 50 == 0:
            print(
                f"{panel_name} #{index}: {durations[-1]:.0f} ms "
                f"({len(panel.objmodel)} objects)",
                file=sys.stderr,
                flush=True,
            )
    return {"durations_ms": durations, "profiles": profiles}


def summary(durations: list[float]) -> dict:
    """Median duration by window of 50 iterations (first one excluded)."""
    windows = {}
    for start in range(1, len(durations), 50):
        chunk = durations[start : start + 50]
        windows[f"{start + 1}-{start + len(chunk)}"] = round(
            statistics.median(chunk), 1
        )
    return {"first_ms": round(durations[0], 1), "median_by_window_ms": windows}


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--panel", choices=("signal", "image", "both"), default="both")
    parser.add_argument("--profile-at", type=int, action="append", default=[])
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--output", metavar="PATH")
    args = parser.parse_args()
    execenv.unattended = True
    panels = ("signal", "image") if args.panel == "both" else (args.panel,)
    results = {}
    for panel_name in panels:
        with app(isolation=not args.no_isolation) as win:
            results[panel_name] = run_panel(
                win, panel_name, args.iterations, args.profile_at
            )
    report = {
        panel_name: summary(result["durations_ms"])
        for panel_name, result in results.items()
    }
    print(json.dumps(report, indent=1))
    for panel_name, result in results.items():
        for index, text in result["profiles"].items():
            print(f"\n=== {panel_name}, iteration {index}\n{text[:7000]}")
    if args.output:
        os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file)


if __name__ == "__main__":
    main()
