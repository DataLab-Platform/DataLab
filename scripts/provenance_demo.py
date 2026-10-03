# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Run the provenance chain scenario on Desktop and write its evidence files.

Usage (from the repository root)::

    python scripts/run_with_env.py python scripts/provenance_demo.py OUTDIR [OTHER.h5]

Writes into OUTDIR:

- ``desktop_chain.h5``: workspace of the chain scenario (S0 -> S1 opaque -> S2
  -> S3, S0 -> S4), saved with its provenance ledger (kept when it exists);
- ``desktop_chain.dlcapsule``: capsule of that workspace;
- ``desktop_reports.json``: verification reports of every recorded activity,
  obtained after reopening each workspace (``desktop_chain.h5`` and, when
  given, OTHER.h5, e.g. a workspace saved by DataLab-Web) in a fresh window;
- ``desktop_versions.json``: environment of this run.

Replays really compute: the reports file records the number of computations
run by the execution service for each workspace.
"""

from __future__ import annotations

import json
import os.path as osp
import sys

from datalab_capsule.archive import create_from_hdf5

from datalab.env import execenv
from datalab.tests.features.common.provenance_unit_test import app
from datalab.tests.features.hdf5.provenance_roundtrip_unit_test import build_chain


def _reports(path: str) -> dict:
    """Reopen *path* in a fresh window and verify every recorded activity."""
    with app() as win:
        win.load_h5_workspace([path], reset_all=True)
        execution = win.signalpanel.processor.execution
        before = execution.computations
        reports = [
            {
                "activity_id": activity["activity_id"],
                "name": (activity["call"]["operation"] or {}).get("id")
                or activity["implementation"]["python_name"],
                "parameters": activity["call"]["parameters"],
                "report": win.verify_provenance_activity(activity["activity_id"]),
            }
            for activity in win.provenance.ledger.activities
        ]
        return {
            "workspace": osp.basename(path),
            "file_status": win.provenance.file_status,
            "state_status": win.provenance.state_status,
            "computations": execution.computations - before,
            "activities": reports,
        }


def main() -> None:
    """Write the evidence files."""
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    outdir = sys.argv[1]
    others = sys.argv[2:]
    execenv.unattended = True
    path = osp.join(outdir, "desktop_chain.h5")
    # An existing workspace is kept, so that other evidence can refer to it.
    if not osp.exists(path):
        with app() as win:
            build_chain(win)
            win.save_h5_workspace(path)
        with open(osp.join(outdir, "desktop_chain.dlcapsule"), "wb") as file:
            file.write(create_from_hdf5(path, name="Desktop chain scenario"))
    with app() as win:
        environment = win.provenance.environment
    reports = [_reports(workspace) for workspace in [path, *others]]
    for name, data in (
        ("desktop_reports.json", reports),
        ("desktop_versions.json", environment),
    ):
        with open(osp.join(outdir, name), "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    for item in reports:
        verdicts = [a["report"]["verdict"] for a in item["activities"]]
        print(f"{item['workspace']}: {verdicts}, {item['computations']} computations")


if __name__ == "__main__":
    main()
