# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase mode
=============

The :mod:`datalab.gui.showcase` package implements the showcase mode of DataLab:
live and unattended demos of DataLab features, for exhibitions and booths.

The showcase mode is started from the **?** menu (**Showcase mode...**), with the
``--showcase`` command-line options of DataLab, or with the ``datalab-showcase``
command. It plays the native scenarios (:mod:`datalab.gui.showcase.scenarios`)
and the scenarios generated for the active application plugins
(:mod:`datalab.gui.showcase.applications`), and optionally ends each loop with a
DataLab Web interlude, for booths with a single screen
(:mod:`datalab.gui.showcase.scenarios.web_interlude`).

.. note::

    This package is part of the application: it must not import the test suite
    (:mod:`datalab.tests`), whose dependencies (e.g., ``pytest``) are not
    available in the standalone application.
"""

from __future__ import annotations

from datalab.env import execenv
from datalab.gui.showcase.dialog import ShowcaseDialog, show_showcase_dialog
from datalab.gui.showcase.engine import (
    ScenarioSkipped,
    ShowcaseContext,
    ShowcaseScenario,
)
from datalab.gui.showcase.player import (
    ScenarioResult,
    ShowcaseOptions,
    ShowcasePlayer,
    get_showcase_scenarios,
    is_showcase_playing,
    play_from_command_line,
    play_showcase,
    quiet_startup,
)

__all__ = [
    "ScenarioResult",
    "ScenarioSkipped",
    "ShowcaseContext",
    "ShowcaseDialog",
    "ShowcaseOptions",
    "ShowcasePlayer",
    "ShowcaseScenario",
    "get_showcase_scenarios",
    "is_showcase_playing",
    "play_from_command_line",
    "play_showcase",
    "quiet_startup",
    "run",
    "show_showcase_dialog",
]


def run() -> None:
    """Start DataLab, then play the showcase in a loop, in full screen

    This is the ``datalab-showcase`` command, equivalent to
    ``datalab --showcase --showcase-loop --showcase-fullscreen``. It accepts the
    command-line options of DataLab (e.g. ``--showcase d1,d3`` to play only some
    scenarios, ``--showcase-screen 1``).
    """
    # pylint: disable=import-outside-toplevel
    from datalab.app import run as run_datalab

    if execenv.showcase is None:
        execenv.showcase = "all"
    execenv.showcase_loop = True
    execenv.showcase_fullscreen = True
    run_datalab()
