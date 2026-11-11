# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D2: reproducibility
=====================================

Provenance capsules (RO-Crate): an image analysis (denoising, edge and blob
detection) is packed into a capsule, the workspace is cleared, the capsule is
opened again (objects, parameters and history restored, every step replayed
exactly), then a parameter is changed and a child revision is exported.

Capsules are written to a temporary folder (no file dialog), removed at the end of
the scenario. The scenario records the History and the revisions of the capsules
in a temporary store only: the History record mode and the revision store of the
user are restored afterwards.

The image chain of this scenario (:func:`create_cells_image`, :func:`build_chain`)
also generates the showcase capsule of DataLab Web
(``scripts/gen_showcase_capsule.py`` of DataLab-Web): each of its operations is
replayed exactly by DataLab Web.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sigima.objects
import sigima.proc.image as sipi
from datalab_capsule import exchange
from datalab_capsule.archive import read_capsule
from datalab_capsule.manifest import inspect_manifest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.adapters_metadata import GeometryAdapter
from datalab.capsulestore import RevisionStore
from datalab.config import _
from datalab.gui.showcase.engine import ShowcaseScenario
from datalab.widgets.capsule import CapsuleDialog, short_id

if TYPE_CHECKING:
    from sigima.objects import ImageObj

    from datalab.gui.main import DLMainWindow
    from datalab.gui.showcase.engine import ShowcaseContext

#: Title of the raw image
CELLS_TITLE = "Cells"
#: Size (pixels) of the raw image
CELLS_SIZE = 256
#: Number of cells in the raw image
CELLS_COUNT = 14
#: Denoising strength (total variation weight) of the capsule
DENOISE_WEIGHT = 0.1
#: Denoising strength of the child revision
REVISED_WEIGHT = 0.25
#: Name of the capsule (generated without translation, e.g. for DataLab Web)
CAPSULE_NAME = "Cells analysis"


def create_cells_image(seed: int = 7) -> ImageObj:
    """Create a raw image of cells: bright disks on a background, with noise

    Args:
        seed: Seed of the random generator (the image is reproducible)

    Returns:
        8-bit image of :data:`CELLS_COUNT` cells
    """
    size = CELLS_SIZE
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size), dtype=float)
    data = 20.0 + 25.0 * xx / size
    cells: list[tuple[float, float, float]] = []
    while len(cells) < CELLS_COUNT:
        radius = rng.uniform(6.0, 17.0)
        x0, y0 = rng.uniform(radius + 8.0, size - radius - 8.0, 2)
        if all(np.hypot(x0 - x, y0 - y) > radius + r + 6.0 for x, y, r in cells):
            cells.append((x0, y0, radius))
    for x0, y0, radius in cells:
        distance = np.hypot(xx - x0, yy - y0)
        data += 150.0 / (1.0 + np.exp((distance - radius) / 1.5))
    data += rng.normal(0.0, 22.0, data.shape)
    data = np.clip(data, 0.0, 255.0).astype(np.uint8)
    return sigima.objects.create_image(CELLS_TITLE, data)


def denoise_param(weight: float = DENOISE_WEIGHT) -> sipi.DenoiseTVParam:
    """Return the total variation denoising parameters

    Args:
        weight: Denoising strength

    Returns:
        Parameters
    """
    return sipi.DenoiseTVParam.create(weight=weight)


def blob_param() -> sipi.BlobLOGParam:
    """Return the blob detection (Laplacian of Gaussian) parameters"""
    return sipi.BlobLOGParam.create(
        min_sigma=4.0, max_sigma=13.0, threshold_rel=0.2, overlap=0.5
    )


def count_blobs(obj: ImageObj) -> int:
    """Return the number of blobs detected on an image

    Args:
        obj: Image

    Returns:
        Number of blobs (0 if blobs were not detected)
    """
    return sum(
        len(adapter.result.coords)
        for adapter in GeometryAdapter.iterate_from_obj(obj)
        if adapter.result.func_name == "blob_log"
    )


@dataclass
class CellsChain:
    """Objects of the image chain

    Args:
        raw: Raw image
        denoised: Denoised image, with the detected blobs
        edges: Edges of the denoised image
    """

    raw: ImageObj
    denoised: ImageObj
    edges: ImageObj


def _edit(ctx: ShowcaseContext) -> bool | None:
    """Return the ``edit`` argument showing the parameter dialogs at showcase speed

    In check mode, no parameter dialog is shown (they would be rejected at once
    by the unattended mode of the test suite).
    """
    return None if ctx.check else True


def build_chain(ctx: ShowcaseContext) -> CellsChain:
    """Build the image chain: raw image, denoising, edge and blob detection

    Args:
        ctx: Showcase context (check mode for a chain built without captions)

    Returns:
        Objects of the chain
    """
    win = ctx.win
    panel = win.imagepanel
    win.set_current_panel("image")
    raw = create_cells_image()
    win.add_object(raw)
    ctx.pause(0.5)
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(_("Start from a raw image of cells: it is noisy"), hold=2.5)
    ctx.say(_("Reduce the noise first (total variation denoising)"), hold=1.5)
    denoised = ctx.run_feature("denoise_tv", denoise_param(), edit=_edit(ctx))
    ctx.checkpoint(denoised is not raw, "Denoising created no image")
    ctx.pause(1.0)
    ctx.say(_("Then detect the edges of the cells"), hold=1.5)
    edges = ctx.run_feature("sobel")
    ctx.checkpoint(edges is not denoised, "Edge detection created no image")
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(_("Edges of the cells (Sobel filter)"), hold=2.0)
    panel.objview.select_objects([denoised])
    ctx.pause(0.5)
    ctx.say(_("Finally, detect the cells as blobs on the denoised image"), hold=1.5)
    ctx.run_feature("blob_log", blob_param(), edit=_edit(ctx))
    blobs = count_blobs(denoised)
    ctx.checkpoint(
        blobs == CELLS_COUNT, f"{blobs} blobs detected, {CELLS_COUNT} expected"
    )
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(_("%d cells detected, drawn on the image") % blobs, hold=3.0)
    return CellsChain(raw, denoised, edges)


def checksum(obj: ImageObj) -> str:
    """Return the SHA-256 checksum of the data of an image (type and shape included)

    Args:
        obj: Image

    Returns:
        Hexadecimal checksum
    """
    data = np.ascontiguousarray(obj.data)
    digest = hashlib.sha256(f"{data.dtype.str}{data.shape}".encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """What a capsule restores: objects, processing parameters and history

    Args:
        objects: ``(title, data checksum)`` of the images, sorted
        activities: ``(operation, parameters)`` of the provenance activities
        history: Number of History actions
        blobs: Number of blobs detected on the images
    """

    objects: tuple[tuple[str, str], ...]
    activities: tuple[tuple[str, str], ...]
    history: int
    blobs: int

    @classmethod
    def take(cls, win: DLMainWindow) -> WorkspaceSnapshot:
        """Take a snapshot of the workspace

        Args:
            win: DataLab main window

        Returns:
            Snapshot
        """
        images = win.imagepanel.objmodel.get_all_objects()
        activities = []
        for activity in win.provenance.ledger.activities:
            call = activity["call"]
            operation = call["operation"]
            activities.append(
                (
                    operation["id"] if operation else "",
                    json.dumps(call["parameters"], sort_keys=True, default=str),
                )
            )
        return cls(
            objects=tuple(sorted((obj.title, checksum(obj)) for obj in images)),
            activities=tuple(activities),
            history=len(win.historypanel),
            blobs=sum(count_blobs(obj) for obj in images),
        )


def find_image(win: DLMainWindow, operation: str) -> ImageObj | None:
    """Return the image produced last by an operation of the provenance ledger

    Args:
        win: DataLab main window
        operation: Operation ID (e.g. ``'sigima.image.denoise_tv'``)

    Returns:
        Image, or None if the operation produced no image of the workspace
    """
    ledger = win.provenance.ledger
    for activity in reversed(ledger.activities):
        call_operation = activity["call"]["operation"]
        if not call_operation or call_operation["id"] != operation:
            continue
        for output in activity["outputs"]:
            state = ledger.states[output["state_id"]]
            obj = win.find_object_by_uuid(state["object_uuid"])
            if obj is not None:
                return obj
    return None


def find_weight(win: DLMainWindow) -> float | None:
    """Return the weight of the last denoising activity of the provenance ledger

    Args:
        win: DataLab main window

    Returns:
        Weight, or None without denoising activity
    """
    for activity in reversed(win.provenance.ledger.activities):
        operation = activity["call"]["operation"]
        if operation and operation["id"] == "sigima.image.denoise_tv":
            return activity["call"]["parameters"]["weight"]
    return None


def export_capsule(
    win: DLMainWindow,
    filename: str,
    name: str = CAPSULE_NAME,
    description: str | None = None,
) -> str:
    """Export the workspace as a capsule and check it

    Args:
        win: DataLab main window
        filename: Capsule file name
        name: Capsule name
        description: Capsule description

    Returns:
        Manifest fingerprint of the capsule

    Raises:
        AssertionError: If the capsule is not intact
    """
    fingerprint = win.export_capsule(filename, name=name, description=description)
    with open(filename, "rb") as file:
        inspection = exchange.inspect_capsule(file.read())
    if not inspection["integrity"]["valid"]:
        raise AssertionError(f"Capsule {filename} is not intact")
    return fingerprint


def format_size(filename: str) -> str:
    """Return the size of a file as a short text (e.g. ``'1.2 MB'``)"""
    return _("%.1f MB") % (osp.getsize(filename) / 1e6)


@contextmanager
def recording_history(win: DLMainWindow) -> Generator[None, None, None]:
    """Record the History within the context (record mode restored afterwards)

    Args:
        win: DataLab main window

    Yields:
        None
    """
    action = win.historypanel.ui.actions["record"]
    previous = action.isChecked()
    action.setChecked(True)
    try:
        yield
    finally:
        action.setChecked(previous)


@contextmanager
def temporary_revisions(win: DLMainWindow, folder: str) -> Generator[None, None, None]:
    """Keep the capsule revisions seen within the context in a temporary store

    Args:
        win: DataLab main window
        folder: Folder of the temporary store

    Yields:
        None
    """
    capsuleio = win.capsuleio
    previous = capsuleio.revisions
    capsuleio.revisions = RevisionStore(osp.join(folder, "revisions"))
    try:
        yield
    finally:
        capsuleio.revisions = previous


def show_capsule_dialog(ctx: ShowcaseContext) -> CapsuleDialog:
    """Show the Workspace capsule dialog, without blocking the scenario

    The dialog is placed at the top of the main window, above the captions.

    Args:
        ctx: Showcase context

    Returns:
        Dialog (to be closed by the caller)
    """
    win = ctx.win
    dialog = CapsuleDialog(win)
    width = min(dialog.width(), int(win.width() * 0.7))
    height = min(dialog.height(), int(win.height() * 0.62))
    dialog.resize(width, height)
    top_left = win.mapToGlobal(QC.QPoint(0, 0))
    dialog.move(
        top_left.x() + (win.width() - width) // 2,
        top_left.y() + win.menuBar().height() + int(win.height() * 0.05),
    )
    dialog.show()
    dialog.raise_()
    ctx.pause(1.0)
    return dialog


def replay_all(ctx: ShowcaseContext, dialog: CapsuleDialog) -> list[str]:
    """Replay each activity of the workspace in the Local replay tab

    Args:
        ctx: Showcase context
        dialog: Workspace capsule dialog

    Returns:
        Verdicts of the activities
    """
    tab = dialog.replay_tab
    dialog.tabs.setCurrentWidget(tab)
    ctx.pause(1.0)
    verdicts = []
    for row, activity_id in enumerate(tab.activity_ids()):
        cell = tab.table.cellWidget(row, 3)
        ctx.checkpoint(cell is not None, f"No replay button for {activity_id}")
        ctx.click(cell.findChildren(QW.QPushButton)[0])
        report = tab.reports.get(activity_id)
        ctx.checkpoint(report is not None, f"Activity {activity_id} not replayed")
        verdicts.append(report["verdict"])
        ctx.pause(0.6)
    return verdicts


def change_denoising(ctx: ShowcaseContext, denoised: ImageObj) -> None:
    """Change the denoising strength in the Processing tab, then apply it

    Args:
        ctx: Showcase context
        denoised: Denoised image
    """
    panel = ctx.win.imagepanel
    objprop = panel.objprop
    panel.objview.select_objects([denoised])
    ctx.pause(0.3)
    editor = objprop.processing_param_editor
    ctx.checkpoint(editor is not None, "Denoising parameters cannot be edited")
    objprop.tabwidget.setCurrentWidget(objprop.processing_scroll)
    line_edit = None
    for widget in editor.edit.widgets:
        if widget.item.item.get_name() == "weight":
            line_edit = widget.edit
    ctx.checkpoint(isinstance(line_edit, QW.QLineEdit), "No denoising weight field")
    old, new = f"{DENOISE_WEIGHT:g}", f"{REVISED_WEIGHT:g}"
    with ctx.spotlight(objprop):
        ctx.say(
            _("Change one parameter: a stronger denoising (weight %s → %s)")
            % (old, new),
            hold=1.5,
        )
        line_edit.selectAll()
        ctx.type_text(line_edit, new, cps=6.0)
        ctx.pause(0.5)
        ctx.checkpoint(editor.apply_button.isEnabled(), "Denoising weight not changed")
        ctx.click(editor.apply_button)
        ctx.pause(1.0)


def run(ctx: ShowcaseContext) -> None:
    """Play the scenario

    Args:
        ctx: Showcase context
    """
    win = ctx.win
    history_dock = win.docks[win.historypanel]
    history_shown = history_dock.toggleViewAction().isChecked()
    folder = tempfile.mkdtemp(prefix="datalab-showcase-d2-")
    try:
        with recording_history(win), temporary_revisions(win, folder):
            _play(ctx, folder)
    finally:
        # The capsules are removed: a capsule exported later is no revision of them
        win.provenance.capsule_origin = None
        shutil.rmtree(folder, ignore_errors=True)
        if not history_shown:
            history_dock.hide()


def _play(ctx: ShowcaseContext, folder: str) -> None:
    """Play the scenario, writing the capsules to a temporary folder

    Args:
        ctx: Showcase context
        folder: Temporary folder
    """
    win = ctx.win
    panel = win.imagepanel
    history_dock = win.docks[win.historypanel]

    # 1. Image chain
    build_chain(ctx)
    history_dock.show()
    history_dock.raise_()
    with ctx.spotlight(win.historypanel, panel.objview):
        ctx.say(_("Each step is recorded, with its parameters"), hold=2.5)
    win.set_current_panel("image")
    before = WorkspaceSnapshot.take(win)
    ctx.checkpoint(
        len(before.objects) == 3 and len(before.activities) == 3,
        f"Unexpected chain: {before}",
    )

    # 2. Capsule export
    parent = osp.join(folder, "cells.dlcapsule")
    name = _("Cells analysis")
    ctx.say(_("Pack data, processing and provenance into a capsule"), hold=1.5)
    ctx.point_action(win.capsule_action)
    parent_fp = export_capsule(
        win, parent, name, _("Raw cells image, denoised, edges and blobs")
    )
    ctx.say(
        _("Capsule “%(name)s” exported (%(size)s, RO-Crate)\nFingerprint: %(fp)s")
        % {"name": name, "size": format_size(parent), "fp": short_id(parent_fp)},
        hold=3.5,
    )

    # 3. Workspace reset
    ctx.reset_workspace()
    ctx.checkpoint(not win.has_objects(), "Objects left after the reset")
    ctx.checkpoint(len(win.historypanel) == 0, "History left after the reset")
    ctx.say(_("The workspace is cleared: everything is gone"), hold=2.5)

    # 4. Capsule opening, replay
    ctx.say(_("Open the capsule..."), hold=1.0)
    ctx.point_action(win.capsule_action)
    win.open_capsule(parent)
    win.set_current_panel("image")
    after = WorkspaceSnapshot.take(win)
    ctx.checkpoint(
        after.objects == before.objects, "Objects or data differ after the opening"
    )
    ctx.checkpoint(
        after.activities == before.activities,
        "Processing parameters differ after the opening",
    )
    ctx.checkpoint(after.blobs == before.blobs, "Blobs differ after the opening")
    ctx.checkpoint(
        before.history > 0 and after.history >= before.history,
        "History actions lost after the opening",
    )
    denoised = find_image(win, "sigima.image.denoise_tv")
    edges = find_image(win, "sigima.image.sobel")
    ctx.checkpoint(denoised is not None and edges is not None, "Chain not restored")
    panel.objview.select_objects([denoised])
    history_dock.raise_()
    with ctx.spotlight(panel.objview, win.historypanel):
        ctx.say(
            _(
                "Everything is back: %d images, parameters and history, "
                "with identical checksums"
            )
            % len(after.objects),
            hold=3.5,
        )
    win.set_current_panel("image")
    dialog = show_capsule_dialog(ctx)
    try:
        ctx.say(_("Replay each step and compare it with the stored result"), hold=1.5)
        verdicts = replay_all(ctx, dialog)
        ctx.checkpoint(
            verdicts == ["exact"] * len(before.activities),
            f"Replay verdicts: {verdicts}",
        )
        ctx.say(_("Exact match, bit for bit: the analysis is reproducible"), hold=3.0)
    finally:
        dialog.close()
        dialog.deleteLater()

    # 5. Child revision
    change_denoising(ctx, denoised)
    weight = find_weight(win)
    ctx.checkpoint(weight == REVISED_WEIGHT, f"Denoising weight {weight} not changed")
    report = panel.processor.recompute_processing(edges, interactive=False)
    ctx.checkpoint(report.success, f"Edges not recomputed: {report.message}")
    ctx.checkpoint(
        panel.processor.recompute_analysis(denoised), "Blobs not detected again"
    )
    panel.objview.select_objects([denoised])
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(
            _("Dependent results are updated: %d cells detected")
            % count_blobs(denoised),
            hold=2.5,
        )
    child = osp.join(folder, "cells-revision.dlcapsule")
    ctx.say(_("Export a child revision of the capsule"), hold=1.0)
    ctx.point_action(win.capsule_action)
    child_fp = win.export_capsule(child, name=name, as_revision=True)
    revision_of = inspect_manifest(read_capsule(child).manifest)["revision_of"]
    ctx.checkpoint(revision_of == parent_fp, "The child revision names no parent")
    dialog = show_capsule_dialog(ctx)
    try:
        tab = dialog.revisions_tab
        dialog.tabs.setCurrentWidget(tab)
        lineage = {rev.fingerprint: rev.revision_of for rev in tab.revisions}
        ctx.checkpoint(
            lineage.get(child_fp) == parent_fp, f"Unexpected lineage: {lineage}"
        )
        tab.select(parent_fp, child_fp)
        ctx.pause(0.5)
        ctx.click(tab.compare_button)
        ctx.checkpoint(tab.diff is not None, "Revisions not compared")
        ctx.say(
            _("Revision %(child)s ← parent %(parent)s: the lineage is kept")
            % {"child": short_id(child_fp), "parent": short_id(parent_fp)},
            hold=3.5,
        )
    finally:
        dialog.close()
        dialog.deleteLater()

    # 6. DataLab Web
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(
            _("The same capsule opens in DataLab Web, right in a browser"), hold=3.5
        )


SCENARIO = ShowcaseScenario(
    id="d2",
    title=_("Reproducibility: provenance capsules (RO-Crate)"),
    subtitle=_("Share an analysis that anyone can replay and verify"),
    run=run,
    duration_hint=95.0,
)
