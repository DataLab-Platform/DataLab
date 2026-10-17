# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Check provenance and capsules with an installed DataLab (no test helpers).

Usage::

    python capsule_smoke_check.py OUTDIR

Runs, with the DataLab installed in the current environment (no checkout, no
pytest), in a temporary user profile so that the user's configuration is never
touched:

1. a processing chain (constant added to a signal, then normalization), saved
   to ``OUTDIR/chain.h5``, reopened, and every step verified by recomputing it;
2. a capsule exported to ``OUTDIR/chain.dlcapsule`` and inspected;
3. after a restart (new main window, same profile): the exported revision is
   still listed, the capsule opens as the workspace and every step verifies.

Writes ``OUTDIR/capsule_smoke_check.json`` and exits with 1 when a check fails.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import tempfile


def _main_window():
    # pylint: disable=import-outside-toplevel
    from datalab.gui.main import DLMainWindow

    win = DLMainWindow(console=False)
    win.historypanel.set_tracking_enabled(False)
    return win


def _verify_all(win) -> list[str]:
    return [
        win.verify_provenance_activity(activity["activity_id"])["verdict"]
        for activity in win.provenance.ledger.activities
    ]


def run(outdir: str) -> dict:
    """Run the checks and return their results."""
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import sigima.params
    import sigima.proc.signal as sips
    from datalab_capsule import exchange
    from sigima.objects import create_signal

    from datalab.config import ensure_initialized
    from datalab.env import execenv
    from datalab.objectmodel import get_uuid
    from datalab.utils.qthelpers import datalab_app_context

    execenv.unattended = True
    ensure_initialized(load_user_config=False)
    workspace = osp.join(outdir, "chain.h5")
    capsule = osp.join(outdir, "chain.dlcapsule")
    result: dict = {}
    with datalab_app_context(exec_loop=False):
        win = _main_window()
        try:
            panel = win.signalpanel
            x = np.linspace(0.0, 1.0, 100)
            panel.add_object(create_signal("S0", x, np.sin(6 * x), units=("s", "V")))
            for func, param in (
                (sips.addition_constant, sigima.params.ConstantParam.create(value=1.0)),
                (sips.normalize, sigima.params.NormalizeParam.create(method="maximum")),
            ):
                uuids = panel.objmodel.get_object_ids()
                panel.objview.select_objects([uuids[-1]])
                panel.processor.compute_1_to_1(func, param=param, edit=False)
            win.save_h5_workspace(workspace)
            win.load_h5_workspace([workspace], reset_all=True)
            result["reopened_verdicts"] = _verify_all(win)
            fingerprint = win.export_capsule(capsule, name="smoke check")
            with open(capsule, "rb") as file:
                inspection = exchange.inspect_capsule(file.read())
            result["capsule_valid"] = inspection["integrity"]["valid"]
            result["fingerprint"] = fingerprint
            result["objects"] = [get_uuid(o) for o in panel.objmodel]
        finally:
            win.set_modified(False)
            win.close()
        win = _main_window()
        try:
            revisions = [r.fingerprint for r in win.capsuleio.revisions.revisions()]
            result["revision_kept_after_restart"] = fingerprint in revisions
            win.open_capsule(capsule)
            result["capsule_verdicts"] = _verify_all(win)
        finally:
            win.set_modified(False)
            win.close()
    return result


def main() -> int:
    """Run in a temporary profile and report."""
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    outdir = osp.abspath(sys.argv[1])
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        # DataLab keeps its configuration (and the capsule revisions) there
        for name in ("USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA"):
            os.environ[name] = profile
        result = run(outdir)
    # pylint: disable=import-outside-toplevel
    import importlib.metadata as metadata

    result["versions"] = {
        name: metadata.version(name)
        for name in ("datalab-platform", "DataLab-Capsule", "Sigima", "guidata")
    }
    failures = [
        name
        for name, ok in (
            ("reopened workspace", result["reopened_verdicts"] == ["exact"] * 2),
            ("capsule integrity", result["capsule_valid"]),
            ("revision after restart", result["revision_kept_after_restart"]),
            ("capsule replay", result["capsule_verdicts"] == ["exact"] * 2),
        )
        if not ok
    ]
    result["failures"] = failures
    with open(osp.join(outdir, "capsule_smoke_check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
