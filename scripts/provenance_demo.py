# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Run the provenance chain scenario on Desktop and write its evidence files.

Usage (from the repository root)::

    python scripts/run_with_env.py python scripts/provenance_demo.py OUTDIR
        [OTHER.h5] [--reuse]

Writes into OUTDIR (created when missing):

- ``desktop_chain.h5``: workspace of the chain scenario (S0 -> S1 -> S2 -> S3,
  S0 -> S4, S1 being the addition of a constant), saved with its provenance
  ledger;
- ``desktop_chain.dlcapsule``: capsule of that workspace;
- ``desktop_reports.json``: verification reports of every recorded activity,
  obtained after reopening each workspace (``desktop_chain.h5`` and, when
  given, OTHER.h5, e.g. a workspace saved by DataLab-Web) in a fresh window;
- ``desktop_versions.json``: environment of this run.

By default, OUTDIR must not hold evidence of an earlier run: the chain is built
again, so that every file comes from this run. With ``--reuse``, an existing
``desktop_chain.h5`` (and its capsule) is kept, so that evidence produced from
it elsewhere (e.g. a child revision saved by DataLab-Web) still refers to it;
the reports and the environment are always written again. A workspace keeps
what was recorded when it was built: in workspaces built before operation
contracts were recorded, S1 is opaque and is never replayed, whereas a new run
records it with its operation and replays it.

Replays really compute: the reports file records the number of computations
run by the execution service for each workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

from datalab_capsule.archive import create_from_hdf5

from datalab.env import execenv
from datalab.tests.features.common.provenance_unit_test import app
from datalab.tests.features.hdf5.provenance_roundtrip_unit_test import build_chain

CHAIN = "desktop_chain.h5"
OUTPUTS = (
    CHAIN,
    "desktop_chain.dlcapsule",
    "desktop_reports.json",
    "desktop_versions.json",
)


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
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    parser.add_argument("outdir", metavar="OUTDIR")
    parser.add_argument("others", metavar="OTHER.h5", nargs="*")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=f"keep an existing {CHAIN} and its capsule instead of refusing them",
    )
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    existing = [name for name in OUTPUTS if osp.exists(osp.join(outdir, name))]
    if existing and not args.reuse:
        parser.error(
            f"{outdir} already holds {', '.join(existing)}: choose an empty "
            f"folder, or pass --reuse to keep {CHAIN}"
        )
    execenv.unattended = True
    path = osp.join(outdir, CHAIN)
    if not osp.exists(path):
        with app() as win:
            build_chain(win)
            win.save_h5_workspace(path)
        with open(osp.join(outdir, "desktop_chain.dlcapsule"), "wb") as file:
            file.write(create_from_hdf5(path, name="Desktop chain scenario"))
    with app() as win:
        environment = win.provenance.environment
    reports = [_reports(workspace) for workspace in [path, *args.others]]
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
