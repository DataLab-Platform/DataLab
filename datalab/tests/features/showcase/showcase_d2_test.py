# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D2 tests (provenance capsules)

The image chain of D2 also generates the showcase capsule of DataLab Web: it must
stay reproducible. D2 is played twice in a row in check mode: it leaves no
temporary file, and gives back the History record mode, the revision store and
the History panel as they were.
"""

from __future__ import annotations

import glob
import os.path as osp
import tempfile

from datalab.config import Conf
from datalab.gui.showcase import ShowcaseOptions, ShowcasePlayer
from datalab.gui.showcase.scenarios import d2_capsule
from datalab.tests import datalab_test_app_context


def test_cells_image_is_reproducible() -> None:
    """The raw image of the chain is the same at each call"""
    first, second = d2_capsule.create_cells_image(), d2_capsule.create_cells_image()
    assert first.data.dtype.name == "uint8"
    assert first.data.shape == (d2_capsule.CELLS_SIZE, d2_capsule.CELLS_SIZE)
    assert d2_capsule.checksum(first) == d2_capsule.checksum(second)


def test_d2_plays_twice_without_leftovers() -> None:
    """D2 plays twice in a row, then gives back the state it changed"""
    pattern = osp.join(tempfile.gettempdir(), "datalab-showcase-d2-*")
    before = set(glob.glob(pattern))
    with Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context(console=False, history=True) as win:
            history = win.historypanel
            dock = win.docks[history]
            dock.hide()
            revisions = win.capsuleio.revisions
            known = len(revisions.revisions())
            scenario = d2_capsule.SCENARIO
            options = ShowcaseOptions(speed=0.0)
            results = ShowcasePlayer(win, [scenario, scenario], options).play()
            assert [r.status for r in results] == ["done", "done"], [
                r.message for r in results
            ]
            assert not history.ui.actions["record"].isChecked()
            assert not history.record_mode_enabled
            # Revisions were kept in a temporary store, not in the user's one
            assert win.capsuleio.revisions is revisions
            assert len(revisions.revisions()) == known
            assert not dock.toggleViewAction().isChecked()
            # The temporary capsules are gone: no revision of them may be exported
            assert win.provenance.capsule_origin is None
            # As after any scenario, the workspace of the last play is kept
            assert len(win.imagepanel) == 3
    assert set(glob.glob(pattern)) == before
