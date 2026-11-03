# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Workspace capsule dialog
========================

Export, inspect, replay and compare workspace capsules, and exchange them
through an optional registry: the same dialog as DataLab-Web's
(**File > Workspace capsule...**), with the same tabs.

Three results are shown apart and never mixed:

- **Integrity** of a capsule: computed from its files (manifest, workspace,
  stored states); nothing is recomputed.
- **Supplied documents**: verification reports and reviews given with the
  capsule, checked against it; their verdicts are what their authors claim.
- **Local replay**: an activity of the open workspace recomputed by DataLab and
  compared with its stored result.

All the capsule logic lives in DataLab-Capsule
(:mod:`datalab_capsule.exchange`, :mod:`datalab_capsule.registry_client`).

.. autoclass:: CapsuleDialog
"""

from __future__ import annotations

import json
import os.path as osp
import re
from collections.abc import Callable
from html import escape
from typing import TYPE_CHECKING, Any

from datalab_capsule import exchange
from datalab_capsule.archive import MAX_ATTACHMENT_BYTES
from datalab_capsule.registry_client import RegistryClient
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getopenfilenames, getsavefilename

from datalab.capsulestore import (
    CapsuleRevision,
    get_registry_settings,
    set_registry_settings,
)
from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui.capsuleio import attachment_entries, workspace_leaf
from datalab.widgets.expandabletext import apply_palette_color, apply_subdued_color
from datalab.widgets.provenance import eligibility_text, origin_text, verdict_text

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

__all__ = ["CapsuleDialog", "report_details", "short_id"]

CAPSULE_FILTER = _("DataLab capsules") + " (*.dlcapsule);;" + _("All files") + " (*)"
JSON_FILTER = _("JSON files") + " (*.json)"
HDF5_FILTER = _("HDF5 files") + " (*.h5 *.hdf5)"


def short_id(identifier: str | None) -> str:
    """Return the first 12 characters of an activity, state or fingerprint id."""
    if not identifier:
        return ""
    raw = re.sub(r"^#(activity|state)-", "", identifier)
    return re.sub(r"^sha256:", "", raw)[:12]


def format_size(size: int) -> str:
    """Return a file size as a short text."""
    if size < 1024:
        return _("%d B") % size
    return _("%.1f KiB") % (size / 1024)


def report_details(report: dict[str, Any]) -> list[str]:
    """Return the reason and output details of a verification report."""
    details = [str(report["reason"])] if report.get("reason") else []
    for output in report.get("outputs") or []:
        if output.get("detail"):
            details.append(f"{output.get('role')}: {output['detail']}")
    return details


def _base_dir() -> str:
    """Return the folder last used to open or save files."""
    path = Conf.base_dir.get() or ""
    return path if osp.isdir(path) else osp.dirname(path)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def _note(text: str) -> QW.QLabel:
    """Return a subdued, word-wrapped label."""
    label = QW.QLabel(text)
    # Never interpreted as HTML: shown texts may come from a capsule or a registry
    label.setTextFormat(QC.Qt.PlainText)
    label.setWordWrap(True)
    apply_subdued_color(label)
    return label


def _error(text: str = "") -> QW.QLabel:
    """Return a word-wrapped label showing an error."""
    label = _text(text)
    apply_palette_color(label, QG.QColor("#d32f2f"))
    return label


def _text(text: str = "") -> QW.QLabel:
    """Return a selectable, word-wrapped label."""
    label = QW.QLabel(text)
    # Never interpreted as HTML: shown texts may come from a capsule or a registry
    label.setTextFormat(QC.Qt.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
    return label


def _table(headers: list[str]) -> QW.QTableWidget:
    table = QW.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QW.QAbstractItemView.NoSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fit_height(table: QW.QTableWidget) -> None:
    """Size a table to its rows (tables shown in a scrolled page)."""
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    height += sum(table.rowHeight(row) for row in range(table.rowCount()))
    if table.horizontalScrollBar().isVisible():
        height += table.horizontalScrollBar().height()
    table.setFixedHeight(height + 2)


def _set_row(table: QW.QTableWidget, row: int, values: list[Any]) -> None:
    """Fill a table row: strings become items, widgets are set as is."""
    for column, value in enumerate(values):
        if isinstance(value, QW.QWidget):
            table.setCellWidget(row, column, value)
        else:
            item = QW.QTableWidgetItem("" if value is None else str(value))
            table.setItem(row, column, item)


def _buttons(*buttons: QW.QPushButton) -> QW.QWidget:
    widget = QW.QWidget()
    layout = QW.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for button in buttons:
        layout.addWidget(button)
    layout.addStretch()
    return widget


def _button(text: str, slot: Callable[[], Any]) -> QW.QPushButton:
    button = QW.QPushButton(text)
    button.clicked.connect(lambda _checked=False: slot())
    return button


def _save_json(value: Any, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class _Tab(QW.QWidget):
    """Base of the dialog tabs: shared access to the dialog and main window."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__()
        self.dialog = dialog
        self.mainwindow = dialog.mainwindow

    def run(self, task: Callable[[], Any]) -> bool:
        """Run *task* (see :meth:`CapsuleDialog.run`)."""
        return self.dialog.run(task)

    def save_filename(self, title: str, name: str, filters: str) -> str | None:
        """Ask for a file name to save to (None if cancelled)."""
        filename, _filter = getsavefilename(
            self, title, osp.join(_base_dir(), name), filters
        )
        return filename or None

    def open_filename(self, title: str, filters: str) -> str | None:
        """Ask for a file to read (None if cancelled)."""
        filename, _filter = getopenfilename(self, title, _base_dir(), filters)
        return filename or None

    def refresh(self) -> None:
        """Update the tab after the workspace or the revisions changed."""


class AttachmentsSection(QW.QGroupBox):
    """Attachments of a capsule, with their checked download."""

    def __init__(
        self, attachments: list[dict[str, Any]], save: Callable[[str], Any]
    ) -> None:
        super().__init__(_("Attachments"))
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _note(
                _(
                    "Files added by the author, outside the processing. They are "
                    "checked against the manifest and saved only when you ask; they "
                    "are never opened."
                )
            )
        )
        self.table = _table(
            [_("Name"), _("Type"), _("Size"), _("SHA-256"), _("Description"), ""]
        )
        self.save_buttons: dict[str, QW.QPushButton] = {}
        for row, attachment in enumerate(attachments):
            self.table.insertRow(row)
            name = attachment["name"]
            button = _button(_("Save..."), lambda name=name: save(name))
            self.save_buttons[name] = button
            _set_row(
                self.table,
                row,
                [
                    name,
                    attachment.get("media_type"),
                    format_size(int(attachment.get("size") or 0)),
                    str(attachment.get("sha256") or "")[:12],
                    attachment.get("description") or "",
                    _buttons(button),
                ],
            )
            self.table.item(row, 3).setToolTip(str(attachment.get("sha256") or ""))
        _fit_height(self.table)
        layout.addWidget(self.table)


class ExportTab(_Tab):
    """Export the workspace as a capsule."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__(dialog)
        self.attachments: list[str] = []
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _text(
                _(
                    "A capsule holds the workspace and the provenance of its "
                    "processing, so that someone else can check and replay it."
                )
            )
        )
        form = QW.QFormLayout()
        self.name_edit = QW.QLineEdit()
        self.description_edit = QW.QLineEdit()
        form.addRow(_("Name"), self.name_edit)
        form.addRow(_("Description"), self.description_edit)
        layout.addLayout(form)
        self.revision_check = QW.QCheckBox()
        self.revision_check.setChecked(True)
        self.revision_check.toggled.connect(self.__update)
        layout.addWidget(self.revision_check)
        self.parent_note = _note("")
        layout.addWidget(self.parent_note)
        self.referenced_check = QW.QCheckBox(
            _("Reference the workspace instead of including it")
        )
        self.referenced_check.toggled.connect(self.__update)
        layout.addWidget(self.referenced_check)
        self.ref_widget = QW.QWidget()
        ref_layout = QW.QFormLayout(self.ref_widget)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_edit = QW.QLineEdit()
        self.ref_edit.setPlaceholderText("https://")
        self.ref_edit.textChanged.connect(self.__update)
        ref_layout.addRow(_("Workspace address (absolute IRI)"), self.ref_edit)
        layout.addWidget(self.ref_widget)
        group = QW.QGroupBox(_("Attachments"))
        group_layout = QW.QVBoxLayout(group)
        group_layout.addWidget(
            _note(
                _(
                    "Light files (protocol, notes, small tables) declared in the "
                    "capsule with their size and fingerprint. They are never "
                    "opened or executed."
                )
            )
        )
        self.attachment_table = _table([_("Name"), _("Size"), ""])
        self.attachment_table.setMaximumHeight(160)
        group_layout.addWidget(self.attachment_table)
        self.attachment_error = _error()
        self.attachment_error.hide()
        group_layout.addWidget(self.attachment_error)
        group_layout.addWidget(
            _buttons(_button(_("Add attachments..."), self.add_attachments))
        )
        layout.addWidget(group)
        self.export_button = _button(_("Export capsule..."), self.export)
        self.status = _text()
        layout.addWidget(_buttons(self.export_button))
        layout.addWidget(self.status)
        layout.addStretch()
        self.refresh()

    @property
    def origin(self) -> dict[str, Any] | None:
        """Capsule the workspace was opened from, if any."""
        return self.mainwindow.provenance.capsule_origin

    def refresh(self) -> None:
        """Show the parent revision of the workspace, if any."""
        origin = self.origin
        if origin is not None:
            self.revision_check.setText(
                _("Revision of %(name)s (%(fingerprint)s)")
                % {
                    "name": origin["name"],
                    "fingerprint": short_id(origin["fingerprint"]),
                }
            )
        self.__update()

    def __update(self) -> None:
        origin = self.origin
        self.revision_check.setVisible(origin is not None)
        names = {name for name, _path in attachment_entries(self.attachments)}
        parent_only = [
            name for name in (origin or {}).get("attachments", []) if name not in names
        ]
        show_note = bool(origin and self.revision_check.isChecked() and parent_only)
        self.parent_note.setVisible(show_note)
        if show_note:
            self.parent_note.setText(
                _(
                    "The parent revision has attachments that are not copied: %s. "
                    "Select them again below if they still apply."
                )
                % ", ".join(parent_only)
            )
        referenced = self.referenced_check.isChecked()
        self.ref_widget.setVisible(referenced)
        self.export_button.setEnabled(
            not referenced or bool(self.ref_edit.text().strip())
        )

    def __refresh_attachments(self) -> None:
        self.attachment_table.setRowCount(0)
        for row, (name, path) in enumerate(attachment_entries(self.attachments)):
            self.attachment_table.insertRow(row)
            remove = _button(
                _("Remove"), lambda path=path: self.remove_attachment(path)
            )
            _set_row(
                self.attachment_table,
                row,
                [name, format_size(osp.getsize(path)), _buttons(remove)],
            )
        self.attachment_table.resizeColumnsToContents()
        self.__update()

    def add_attachments(self, filenames: list[str] | None = None) -> None:
        """Add attachment files (asked for when *filenames* is None)."""
        if filenames is None:
            filenames, _filter = getopenfilenames(
                self, _("Add attachments..."), _base_dir()
            )
        too_large = []
        for filename in filenames or []:
            if osp.getsize(filename) > MAX_ATTACHMENT_BYTES:
                too_large.append(osp.basename(filename))
            elif filename not in self.attachments:
                self.attachments.append(filename)
        self.attachment_error.setVisible(bool(too_large))
        self.attachment_error.setText(
            _("Attachments are limited to 16 MiB: %s") % ", ".join(too_large)
        )
        self.__refresh_attachments()

    def remove_attachment(self, path: str) -> None:
        """Remove an attachment file."""
        self.attachments.remove(path)
        self.__refresh_attachments()

    def export(
        self, filename: str | None = None, workspace_filename: str | None = None
    ) -> str | None:
        """Export the workspace as a capsule.

        Args:
            filename: Capsule file name (asked for when None).
            workspace_filename: File name of a referenced workspace (asked for
             when None).

        Returns:
            Manifest fingerprint, or None if cancelled or refused.
        """
        name = self.name_edit.text().strip()
        referenced = self.referenced_check.isChecked()
        workspace_ref = self.ref_edit.text().strip() if referenced else None
        if filename is None:
            default = _safe_filename(name or "workspace") + ".dlcapsule"
            filename = self.save_filename(
                _("Export capsule..."), default, CAPSULE_FILTER
            )
            if filename is None:
                return None
        if referenced and workspace_filename is None:
            # The workspace is saved separately, to be published at the IRI.
            workspace_filename = self.save_filename(
                _("Save the referenced workspace"),
                workspace_leaf(workspace_ref or ""),
                HDF5_FILTER,
            )
            if workspace_filename is None:
                return None
        result: list[str] = []

        def task() -> None:
            result.append(
                self.mainwindow.export_capsule(
                    filename,
                    name=name or None,
                    description=self.description_edit.text().strip() or None,
                    as_revision=self.revision_check.isChecked(),
                    workspace_ref=workspace_ref,
                    workspace_filename=workspace_filename,
                    attachments=list(self.attachments),
                )
            )

        self.status.setText("")
        if not self.run(task):
            return None
        self.status.setText(
            _("Capsule %(fingerprint)s saved to %(filename)s")
            % {"fingerprint": short_id(result[0]), "filename": filename}
        )
        self.dialog.refresh()
        return result[0]


class ReviewWriter(QW.QGroupBox):
    """Write a review of the inspected revision (a separate JSON document)."""

    def __init__(self, tab: InspectTab, manifest: dict[str, Any]) -> None:
        super().__init__(_("Write a review"))
        self.tab = tab
        self.manifest = manifest
        self.cited: list[dict[str, Any]] = []
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _note(
                _(
                    "A review is a separate file about this revision; the capsule "
                    "is not changed."
                )
            )
        )
        form = QW.QFormLayout()
        self.reviewer_edit = QW.QLineEdit()
        self.statement_edit = QW.QPlainTextEdit()
        self.statement_edit.setMaximumHeight(80)
        self.verdict_combo = QW.QComboBox()
        for label, value in (
            (_("No verdict"), None),
            (_("endorsed"), "endorsed"),
            (_("disputed"), "disputed"),
            (_("comment"), "comment"),
        ):
            self.verdict_combo.addItem(label, value)
        form.addRow(_("Reviewer"), self.reviewer_edit)
        form.addRow(_("Statement"), self.statement_edit)
        form.addRow(_("Verdict"), self.verdict_combo)
        layout.addLayout(form)
        self.cited_label = QW.QLabel()
        layout.addWidget(
            _buttons(self.cited_label, _button(_("Cite a report..."), self.cite_report))
        )
        self.save_button = _button(_("Save the review..."), self.save)
        self.publish_button = _button(_("Publish the review"), self.publish)
        self.publish_button.setVisible(bool(get_registry_settings()[0]))
        layout.addWidget(_buttons(self.save_button, self.publish_button))
        self.published = _text()
        layout.addWidget(self.published)
        self.reviewer_edit.textChanged.connect(self.__update)
        self.statement_edit.textChanged.connect(self.__update)
        self.__update()

    def __update(self) -> None:
        self.cited_label.setText(_("Cited reports: %d") % len(self.cited))
        ready = bool(
            self.reviewer_edit.text().strip()
            and self.statement_edit.toPlainText().strip()
        )
        self.save_button.setEnabled(ready)
        self.publish_button.setEnabled(ready)

    def cite_report(self, filename: str | None = None) -> None:
        """Cite a verification report (unreadable files are not cited)."""
        if filename is None:
            filename = self.tab.open_filename(_("Cite a report..."), JSON_FILTER)
            if filename is None:
                return
        try:
            with open(filename, encoding="utf-8") as file:
                self.cited.append(json.load(file))
        except (OSError, ValueError):
            # Unreadable files are reported by the inspection, not cited here.
            return
        self.__update()

    def review(self) -> dict[str, Any]:
        """Return the review written in the form."""
        return exchange.review_of(
            self.manifest,
            reviewer=self.reviewer_edit.text().strip(),
            statement=self.statement_edit.toPlainText().strip(),
            verdict=self.verdict_combo.currentData(),
            reports=self.cited,
        )

    def save(self, filename: str | None = None) -> dict[str, Any] | None:
        """Save the review to a JSON file (asked for when *filename* is None)."""
        result: list[dict[str, Any]] = []

        def task() -> None:
            review = self.review()
            target = filename or self.tab.save_filename(
                _("Save the review..."),
                f"review-{short_id(review['review_fingerprint'])}.json",
                JSON_FILTER,
            )
            if target:
                _save_json(review, target)
                result.append(review)

        self.tab.run(task)
        return result[0] if result else None

    def publish(self) -> None:
        """Publish the review to the registry."""
        self.published.setText("")

        def task() -> None:
            address, token = get_registry_settings()
            reply = RegistryClient(address, token or None).publish_review(
                json.dumps(self.review())
            )
            self.published.setText(
                _("Review %s published") % short_id(reply["fingerprint"])
            )

        self.tab.run(task)


class InspectTab(_Tab):
    """Inspect a capsule, its integrity and supplied documents; open it."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__(dialog)
        self.capsule: str | None = None
        self.workspace: str | None = None
        self.reports: list[tuple[str, str]] = []
        self.reviews: list[tuple[str, str]] = []
        self.inspection: dict[str, Any] | None = None
        self.review_writer: ReviewWriter | None = None
        self.attachments_section: AttachmentsSection | None = None
        layout = QW.QVBoxLayout(self)
        self.choose_button = _button(_("Choose a capsule..."), self.choose)
        self.workspace_button = _button(
            _("Select the referenced workspace..."), self.choose_workspace
        )
        self.report_button = _button(
            _("Add a verification report..."), lambda: self.add_document("report")
        )
        self.review_button = _button(
            _("Add a review..."), lambda: self.add_document("review")
        )
        layout.addWidget(
            _buttons(
                self.choose_button,
                self.workspace_button,
                self.report_button,
                self.review_button,
            )
        )
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        self.content = QW.QWidget()
        self.content_layout = QW.QVBoxLayout(self.content)
        scroll.setWidget(self.content)
        layout.addWidget(scroll)
        self.publish_button = _button(_("Publish to the registry"), self.publish)
        self.open_button = _button(_("Open as workspace"), self.open)
        self.published = _text()
        layout.addWidget(self.published)
        layout.addWidget(_buttons(self.publish_button, self.open_button))
        self.__render()

    @property
    def summary(self) -> dict[str, Any] | None:
        """Summary of the inspected capsule, if readable."""
        return None if self.inspection is None else self.inspection["summary"]

    @property
    def referenced(self) -> bool:
        """True when the inspected capsule references its workspace."""
        summary = self.summary
        return bool(summary and summary["workspace"]["referenced"])

    @property
    def intact(self) -> bool:
        """True when the inspected capsule is intact."""
        return bool(self.inspection and self.inspection["integrity"]["valid"])

    def refresh(self) -> None:
        """Show or hide registry actions after the settings changed."""
        self.__update()

    def __update(self) -> None:
        has_capsule = self.capsule is not None
        self.workspace_button.setVisible(has_capsule and self.referenced)
        self.report_button.setVisible(has_capsule)
        self.review_button.setVisible(has_capsule)
        self.report_button.setEnabled(self.summary is not None)
        self.review_button.setEnabled(self.summary is not None)
        shown = self.inspection is not None
        self.publish_button.setVisible(shown and bool(get_registry_settings()[0]))
        self.publish_button.setEnabled(self.intact)
        self.open_button.setVisible(shown)
        self.open_button.setEnabled(
            self.intact and not (self.referenced and self.workspace is None)
        )
        if self.review_writer is not None:
            self.review_writer.publish_button.setVisible(
                bool(get_registry_settings()[0])
            )

    def inspect(self) -> bool:
        """Inspect the chosen capsule with the supplied workspace and documents."""
        capsule = self.capsule
        if capsule is None:
            return False

        def task() -> None:
            capsuleio = self.mainwindow.capsuleio
            self.inspection = exchange.inspect_capsule(
                capsuleio.read_capsule_file(capsule),
                workspace=None
                if self.workspace is None
                else capsuleio.read_workspace_file(self.workspace),
                reports=[text for _name, text in self.reports],
                reviews=[text for _name, text in self.reviews],
                policy=capsuleio.policy,
            )
            if self.inspection["manifest"] is not None:
                self.mainwindow.capsuleio.record_revision(
                    self.inspection["manifest"], "inspected"
                )

        done = self.run(task)
        self.__render()
        self.dialog.refresh_revisions()
        return done

    def choose(self, filename: str | None = None) -> bool:
        """Choose a capsule (asked for when *filename* is None) and inspect it."""
        if filename is None:
            filename = self.open_filename(_("Choose a capsule..."), CAPSULE_FILTER)
            if filename is None:
                return False
        self.capsule = filename
        self.workspace = None
        self.inspection = None
        self.reports, self.reviews = [], []
        self.published.setText("")
        return self.inspect()

    def choose_workspace(self, filename: str | None = None) -> bool:
        """Select the workspace of a capsule that references it."""
        if filename is None:
            filename = self.open_filename(
                _("Select the referenced workspace..."), HDF5_FILTER
            )
            if filename is None:
                return False
        self.workspace = filename
        return self.inspect()

    def add_document(self, kind: str, filename: str | None = None) -> bool:
        """Add a verification report (``"report"``) or a review (``"review"``)."""
        if filename is None:
            title = (
                _("Add a verification report...")
                if kind == "report"
                else _("Add a review...")
            )
            filename = self.open_filename(title, JSON_FILTER)
            if filename is None:
                return False
        try:
            with open(filename, encoding="utf-8") as file:
                text = file.read()
        except (OSError, UnicodeDecodeError) as exc:
            self.dialog.show_error(str(exc))
            return False
        documents = self.reports if kind == "report" else self.reviews
        documents.append((osp.basename(filename), text))
        return self.inspect()

    def save_attachment(self, name: str, filename: str | None = None) -> bool:
        """Save a checked attachment of the capsule (never opened)."""
        if self.capsule is None:
            return False
        if filename is None:
            filename = self.save_filename(_("Save..."), name, _("All files") + " (*)")
            if filename is None:
                return False
        capsule = self.capsule

        def task() -> None:
            capsuleio = self.mainwindow.capsuleio
            data = exchange.capsule_attachment(
                capsuleio.read_capsule_file(capsule), name, capsuleio.policy
            )
            with open(filename, "wb") as file:
                file.write(data)

        return self.run(task)

    def publish(self) -> bool:
        """Publish the manifest, its attachments and the valid supplied reviews."""
        if self.capsule is None or self.inspection is None or self.summary is None:
            return False
        self.published.setText("")
        capsule, inspection, summary = self.capsule, self.inspection, self.summary

        def task() -> None:
            address, token = get_registry_settings()
            client = RegistryClient(address, token or None)
            capsuleio = self.mainwindow.capsuleio
            data = capsuleio.read_capsule_file(capsule)
            fingerprint = client.publish_manifest(
                json.dumps(inspection["manifest"]).encode("utf-8")
            )["fingerprint"]
            for attachment in summary["attachments"]:
                name = attachment["name"]
                client.upload_attachment(
                    fingerprint,
                    name,
                    exchange.capsule_attachment(data, name, capsuleio.policy),
                )
            valid = (inspection.get("documents") or {}).get("reviews") or []
            count = 0
            for index, (_name, text) in enumerate(self.reviews):
                if index < len(valid) and valid[index]["status"] == "valid":
                    client.publish_review(text)
                    count += 1
            self.published.setText(
                _(
                    "Published %(fingerprint)s with %(attachments)d attachments "
                    "and %(reviews)d reviews"
                )
                % {
                    "fingerprint": short_id(fingerprint),
                    "attachments": len(summary["attachments"]),
                    "reviews": count,
                }
            )

        return self.run(task)

    def open(self, confirm: bool = True) -> bool:
        """Open the inspected capsule as the workspace (replacing it)."""
        if self.capsule is None:
            return False
        if (
            confirm
            and not execenv.unattended
            and self.mainwindow.has_objects()
            and QW.QMessageBox.question(
                self,
                _("Open as workspace"),
                _(
                    "The current workspace (signals and images) will be replaced "
                    "by the workspace of this capsule. Continue?"
                ),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
            )
            != QW.QMessageBox.Yes
        ):
            return False
        capsule, workspace = self.capsule, self.workspace
        done = self.run(lambda: self.mainwindow.open_capsule(capsule, workspace))
        self.dialog.refresh()
        return done

    def __clear(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.review_writer = None
        self.attachments_section = None

    def __render(self) -> None:
        self.__clear()
        inspection = self.inspection
        if inspection is not None:
            summary = inspection["summary"]
            if summary is not None:
                self.content_layout.addWidget(self.__summary_view(summary))
                if summary["attachments"]:
                    self.attachments_section = AttachmentsSection(
                        summary["attachments"], self.save_attachment
                    )
                    self.attachments_section.setEnabled(self.intact)
                    self.content_layout.addWidget(self.attachments_section)
            self.content_layout.addWidget(self.__integrity_view(inspection))
            documents = self.__documents_view(inspection)
            if documents is not None:
                self.content_layout.addWidget(documents)
            if summary is not None and inspection["manifest"] is not None:
                self.review_writer = ReviewWriter(self, inspection["manifest"])
                self.content_layout.addWidget(self.review_writer)
        self.content_layout.addStretch()
        self.__update()

    @staticmethod
    def __summary_view(summary: dict[str, Any]) -> QW.QGroupBox:
        group = QW.QGroupBox(summary["name"] or "")
        layout = QW.QVBoxLayout(group)
        if summary.get("description"):
            layout.addWidget(_text(summary["description"]))
        form = QW.QFormLayout()
        workspace = summary["workspace"]
        form.addRow(_("Fingerprint"), _text(summary["fingerprint"]))
        form.addRow(_("Revision of"), _text(summary["revision_of"] or _("none")))
        form.addRow(
            _("Workspace"),
            _text(
                _("referenced at %s") % workspace["id"]
                if workspace["referenced"]
                else _("included")
            ),
        )
        form.addRow(_("Licence"), _text(summary.get("license") or ""))
        layout.addLayout(form)
        table = _table([_("Activity"), _("Operation"), _("Origin"), _("Replay")])
        for row, activity in enumerate(summary["activities"]):
            table.insertRow(row)
            _set_row(
                table,
                row,
                [
                    short_id(activity["id"]),
                    activity.get("name") or "",
                    origin_text(activity.get("origin") or ""),
                    _("replayable")
                    if activity.get("replayable")
                    else _("not replayable"),
                ],
            )
        _fit_height(table)
        layout.addWidget(table)
        return group

    @staticmethod
    def __integrity_view(inspection: dict[str, Any]) -> QW.QGroupBox:
        integrity = inspection["integrity"]
        group = QW.QGroupBox(_("Integrity"))
        layout = QW.QVBoxLayout(group)
        layout.addWidget(_note(_("Computed from the files; nothing is recomputed.")))
        status = _text(_("Intact") if integrity["valid"] else _("Not intact"))
        status.setObjectName("integrity_status")
        font = status.font()
        font.setBold(True)
        status.setFont(font)
        layout.addWidget(status)
        workspace = integrity.get("workspace")
        if workspace:
            label = {
                "verified": _("checked"),
                "integrity_only": _("SHA-256 and size checked; stored states not read"),
                "not_available": _("not supplied: stored states not checked"),
            }.get(workspace, workspace)
            layout.addWidget(_text(_("Workspace: %s") % label))
        if integrity.get("states_located") is not None:
            layout.addWidget(
                _text(
                    _("Stored states checked: %(count)d, altered: %(altered)d")
                    % {
                        "count": integrity["states_located"],
                        "altered": len(integrity["states_altered"]),
                    }
                )
            )
        if integrity["problems"]:
            problems = _error("\n".join(f"• {p}" for p in integrity["problems"]))
            layout.addWidget(problems)
        return group

    def __documents_view(self, inspection: dict[str, Any]) -> QW.QGroupBox | None:
        documents = inspection.get("documents")
        if not documents or not (self.reports or self.reviews):
            return None
        group = QW.QGroupBox(_("Supplied documents"))
        layout = QW.QVBoxLayout(group)
        layout.addWidget(
            _note(
                _(
                    "Checked against this capsule. Their verdicts are what their "
                    "authors claim; they were not recomputed here."
                )
            )
        )
        lines = []

        def status(entry: dict[str, Any]) -> str:
            return _("consistent") if entry["status"] == "valid" else entry["status"]

        for index, entry in enumerate(documents["reports"]):
            name = self.reports[index][0] if index < len(self.reports) else ""
            lines.append(
                _(
                    "Report %(name)s: activity %(activity)s, claimed verdict "
                    "%(verdict)s, %(status)s"
                )
                % {
                    "name": name,
                    "activity": short_id(entry.get("activity_id")),
                    "verdict": entry.get("verdict") or "",
                    "status": status(entry),
                }
            )
        for index, entry in enumerate(documents["reviews"]):
            name = self.reviews[index][0] if index < len(self.reviews) else ""
            lines.append(
                _("Review %(name)s: %(status)s")
                % {"name": name, "status": status(entry)}
            )
        for entry in documents["unreadable"]:
            lines.append(_("Unreadable document: %s") % entry["status"])
        label = _text("\n".join(f"• {line}" for line in lines))
        label.setObjectName("documents_status")
        layout.addWidget(label)
        return group


class ReplayTab(_Tab):
    """Recompute activities of the open workspace and compare them."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__(dialog)
        self.reports: dict[str, dict[str, Any]] = {}
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _note(
                _(
                    "Recomputes an activity of the open workspace with DataLab "
                    "and compares it with its stored result. The workspace is "
                    "not changed."
                )
            )
        )
        self.empty = _text(_("The open workspace has no recorded activity."))
        layout.addWidget(self.empty)
        self.table = _table([_("Activity"), _("Operation"), _("Local result"), ""])
        layout.addWidget(self.table)
        self.refresh()

    def activity_ids(self) -> list[str]:
        """Identifiers of the recorded activities of the workspace."""
        return [a["activity_id"] for a in self.mainwindow.provenance.ledger.activities]

    def refresh(self) -> None:
        """Read the ledger again (reports of activities still present are kept)."""
        activities = self.mainwindow.provenance.ledger.activities
        known = {a["activity_id"] for a in activities}
        self.reports = {k: v for k, v in self.reports.items() if k in known}
        self.empty.setVisible(not activities)
        self.table.setVisible(bool(activities))
        self.table.setRowCount(0)
        for row, activity in enumerate(activities):
            activity_id = activity["activity_id"]
            operation = (activity.get("call") or {}).get("operation")
            report = self.reports.get(activity_id)
            result = ""
            if report is not None:
                verdict = verdict_text(report["verdict"])
                result = f"{verdict} ({eligibility_text(report['eligibility'])})"
                details = report_details(report)
                if details:
                    result += "\n" + "\n".join(f"• {d}" for d in details)
            buttons = [
                _button(_("Replay"), lambda a=activity_id: self.replay(a)),
            ]
            if report is not None:
                buttons.append(
                    _button(_("Save report..."), lambda a=activity_id: self.save(a))
                )
            self.table.insertRow(row)
            _set_row(
                self.table,
                row,
                [
                    short_id(activity_id),
                    operation["id"] if operation else _("opaque"),
                    result,
                    _buttons(*buttons),
                ],
            )
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def replay(self, activity_id: str) -> dict[str, Any] | None:
        """Replay an activity and show its verification report."""

        def task() -> None:
            self.reports[activity_id] = self.mainwindow.verify_provenance_activity(
                activity_id
            )

        self.run(task)
        self.refresh()
        return self.reports.get(activity_id)

    def save(self, activity_id: str, filename: str | None = None) -> bool:
        """Save the verification report of an activity to a JSON file."""
        report = self.reports.get(activity_id)
        if report is None:
            return False
        if filename is None:
            filename = self.save_filename(
                _("Save report..."),
                f"report-{short_id(report['activity_id'])}.json",
                JSON_FILTER,
            )
            if filename is None:
                return False
        return self.run(lambda: _save_json(report, filename))


class RevisionsTab(_Tab):
    """Revisions seen on this computer; compare any two of them."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__(dialog)
        self.revisions: list[CapsuleRevision] = []
        self.diff: dict[str, Any] | None = None
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _note(
                _(
                    "Revisions exported, inspected or retrieved on this computer. "
                    "A revision names its parent by fingerprint."
                )
            )
        )
        self.table = _table([_("Name"), _("Fingerprint"), _("Parent"), _("Source")])
        layout.addWidget(self.table)
        self.old_combo = QW.QComboBox()
        self.new_combo = QW.QComboBox()
        self.compare_button = _button(_("Compare"), self.compare)
        self.old_combo.currentIndexChanged.connect(self.__update)
        self.new_combo.currentIndexChanged.connect(self.__update)
        layout.addWidget(_buttons(self.old_combo, self.new_combo, self.compare_button))
        self.diff_view = QW.QTextBrowser()
        self.diff_view.hide()
        layout.addWidget(self.diff_view)
        self.refresh()

    def refresh(self) -> None:
        """Read the revisions again."""
        self.revisions = self.mainwindow.capsuleio.revisions.revisions()
        sources = {
            "exported": _("exported"),
            "registry": _("from the registry"),
            "inspected": _("inspected"),
        }
        self.table.setRowCount(0)
        for row, revision in enumerate(self.revisions):
            self.table.insertRow(row)
            _set_row(
                self.table,
                row,
                [
                    revision.name,
                    short_id(revision.fingerprint),
                    short_id(revision.revision_of),
                    sources.get(revision.source, revision.source),
                ],
            )
        self.table.resizeColumnsToContents()
        for combo, placeholder in (
            (self.old_combo, _("Older revision")),
            (self.new_combo, _("Newer revision")),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(placeholder, None)
            for revision in self.revisions:
                combo.addItem(
                    f"{revision.name} ({short_id(revision.fingerprint)})",
                    revision.fingerprint,
                )
            index = combo.findData(current) if current else 0
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)
        self.__update()

    def __update(self) -> None:
        self.compare_button.setEnabled(
            bool(self.old_combo.currentData() and self.new_combo.currentData())
        )

    def select(self, older: str, newer: str) -> None:
        """Select the two revisions to compare, by fingerprint."""
        self.old_combo.setCurrentIndex(self.old_combo.findData(older))
        self.new_combo.setCurrentIndex(self.new_combo.findData(newer))

    def compare(self) -> dict[str, Any] | None:
        """Compare the selected revisions."""
        by_fingerprint = {r.fingerprint: r for r in self.revisions}
        old = by_fingerprint.get(self.old_combo.currentData())
        new = by_fingerprint.get(self.new_combo.currentData())
        if old is None or new is None:
            return None

        def task() -> None:
            self.diff = exchange.diff_checked_manifests(old.manifest, new.manifest)

        if self.run(task):
            self.diff_view.setHtml(self.diff_html(self.diff))
            self.diff_view.show()
        return self.diff

    @staticmethod
    def diff_html(diff: dict[str, Any]) -> str:
        """Return the comparison of two revisions as HTML."""
        html = ""
        if diff["revision_of_old"]:
            parent = _("The newer revision names the older one as its parent.")
            html += f"<p>{escape(parent)}</p>"
        if diff["identical"]:
            identical = _("The two revisions are identical.")
            return html + f"<p>{escape(identical)}</p>"
        conclusions = {
            "differences": _("Scientific changes were found."),
            "none": _("No scientific change, the same workspace file and attachments."),
            "none_detected": _(
                "No change in the recorded activities and states. The workspace "
                "files or the attachments differ: content outside the recorded "
                "provenance is not compared, so equality is not established."
            ),
        }
        conclusion = diff["scientific_conclusion"]
        html += f"<p><b>{escape(conclusions.get(conclusion, conclusion))}</b></p>"
        for key, title in (
            ("scientific", _("Scientific changes")),
            ("presentation", _("Presentation changes")),
            ("storage", _("Storage and integrity")),
            ("environment", _("Environment changes")),
            ("attachments", _("Attachments")),
        ):
            changes = diff[key]
            html += f"<h4>{escape(title)} ({len(changes)})</h4><ul>"
            for change in changes:
                identifier = (
                    change["id"]
                    if change.get("item") == "attachment"
                    else short_id(change.get("id"))
                )
                field = f" {change['field']}" if change.get("field") else ""
                html += (
                    f"<li>{escape(str(change.get('change')))} "
                    f"{escape(str(change.get('item')))} "
                    f"<code>{escape(str(identifier))}</code>{escape(field)}</li>"
                )
            html += "</ul>"
        return html


class RegistryTab(_Tab):
    """Optional registry: settings, list and retrieval of revisions."""

    def __init__(self, dialog: CapsuleDialog) -> None:
        super().__init__(dialog)
        self.listed: list[dict[str, Any]] = []
        self.retrieved: dict[str, Any] | None = None
        address, token = get_registry_settings()
        layout = QW.QVBoxLayout(self)
        layout.addWidget(
            _note(
                _(
                    "Optional: a registry publishes manifests, their attachments "
                    "and reviews, never workspaces. Everything it returns is "
                    "checked here."
                )
            )
        )
        form = QW.QFormLayout()
        self.address_edit = QW.QLineEdit(address)
        self.address_edit.setPlaceholderText("https://")
        self.token_edit = QW.QLineEdit(token)
        self.token_edit.setEchoMode(QW.QLineEdit.Password)
        form.addRow(_("Registry address"), self.address_edit)
        form.addRow(_("Publishing token (kept for this session only)"), self.token_edit)
        layout.addLayout(form)
        self.list_button = _button(_("List revisions"), self.list_revisions)
        layout.addWidget(
            _buttons(_button(_("Use these settings"), self.save), self.list_button)
        )
        self.status = _text()
        layout.addWidget(self.status)
        self.table = _table(
            [
                _("Name"),
                _("Fingerprint"),
                _("Parent"),
                _("Attachments"),
                _("Reviews"),
                "",
            ]
        )
        self.table.hide()
        layout.addWidget(self.table)
        self.retrieved_scroll = QW.QScrollArea()
        self.retrieved_scroll.setWidgetResizable(True)
        self.retrieved_widget = QW.QWidget()
        self.retrieved_layout = QW.QVBoxLayout(self.retrieved_widget)
        self.retrieved_scroll.setWidget(self.retrieved_widget)
        self.retrieved_scroll.hide()
        layout.addWidget(self.retrieved_scroll)
        layout.addStretch()
        self.__update()

    def __update(self) -> None:
        self.list_button.setEnabled(bool(get_registry_settings()[0]))

    def client(self) -> RegistryClient:
        """Return a client of the registry set."""
        address, token = get_registry_settings()
        return RegistryClient(address, token or None)

    def save(self) -> bool:
        """Use the address and token typed."""

        def task() -> None:
            set_registry_settings(self.address_edit.text(), self.token_edit.text())
            self.address_edit.setText(get_registry_settings()[0])

        done = self.run(task)
        self.status.setText("")
        self.listed, self.retrieved = [], None
        self.table.hide()
        self.__render_retrieved()
        self.__update()
        self.dialog.refresh()
        return done

    def list_revisions(self) -> bool:
        """Connect to the registry and list its revisions."""

        def task() -> None:
            client = self.client()
            info = client.status()
            self.status.setText(
                _("Connected to %(registry)s %(version)s")
                % {"registry": info.get("registry"), "version": info.get("version")}
            )
            self.listed = client.list_revisions()

        done = self.run(task)
        self.table.setVisible(bool(done))
        self.table.setRowCount(0)
        for row, revision in enumerate(self.listed if done else []):
            self.table.insertRow(row)
            retrieve = _button(
                _("Retrieve"),
                lambda fp=revision["fingerprint"]: self.retrieve(fp),
            )
            _set_row(
                self.table,
                row,
                [
                    revision.get("name"),
                    short_id(revision["fingerprint"]),
                    short_id(revision.get("revision_of")),
                    len(revision.get("attachments") or []),
                    revision.get("reviews", 0),
                    _buttons(retrieve),
                ],
            )
        self.table.resizeColumnsToContents()
        return done

    def retrieve(self, fingerprint: str) -> bool:
        """Retrieve a revision; check its manifest and reviews."""

        def task() -> None:
            client = self.client()
            checked = exchange.check_registry_revision(
                client.get_manifest(fingerprint),
                fingerprint,
                client.list_reviews(fingerprint),
            )
            self.mainwindow.capsuleio.record_revision(checked["manifest"], "registry")
            listed = next(
                (r for r in self.listed if r["fingerprint"] == fingerprint), {}
            )
            self.retrieved = {
                "fingerprint": fingerprint,
                "summary": checked["summary"],
                "reviews": checked["reviews"],
                "stored": {
                    a["name"]
                    for a in listed.get("attachments") or []
                    if a.get("stored")
                },
            }

        done = self.run(task)
        self.__render_retrieved()
        self.dialog.refresh_revisions()
        return done

    def save_attachment(self, name: str, filename: str | None = None) -> bool:
        """Download an attachment, checked against the manifest, and save it."""
        retrieved = self.retrieved
        if retrieved is None:
            return False
        declared = next(
            (a for a in retrieved["summary"]["attachments"] if a["name"] == name), None
        )
        if declared is None:
            return False
        if filename is None:
            filename = self.save_filename(_("Save..."), name, _("All files") + " (*)")
            if filename is None:
                return False

        def task() -> None:
            data = self.client().get_checked_attachment(
                retrieved["fingerprint"], name, declared["sha256"], declared["size"]
            )
            with open(filename, "wb") as file:
                file.write(data)

        return self.run(task)

    def __render_retrieved(self) -> None:
        while self.retrieved_layout.count():
            item = self.retrieved_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        retrieved = self.retrieved
        self.retrieved_scroll.setVisible(retrieved is not None)
        if retrieved is not None:
            summary = retrieved["summary"]
            group = QW.QGroupBox(summary["name"] or "")
            group.setObjectName("retrieved")
            layout = QW.QVBoxLayout(group)
            layout.addWidget(
                _text(
                    _(
                        "Manifest checked: valid, fingerprint %s. Added to the "
                        "revisions of this computer."
                    )
                    % short_id(retrieved["fingerprint"])
                )
            )
            reviews_title = _("Reviews")
            layout.addWidget(QW.QLabel(f"<b>{escape(reviews_title)}</b>"))
            lines = []
            for entry in retrieved["reviews"]:
                review = entry["review"]
                text = (
                    f"{review['reviewer']['name']}: {review['statement']} "
                    if review
                    else ""
                )
                status = _("valid") if entry["status"] == "valid" else entry["status"]
                lines.append(f"• {text}({status})")
            reviews = _text("\n".join(lines) if lines else _("none"))
            reviews.setObjectName("retrieved_reviews")
            layout.addWidget(reviews)
            stored = [
                a for a in summary["attachments"] if a["name"] in retrieved["stored"]
            ]
            if stored:
                layout.addWidget(AttachmentsSection(stored, self.save_attachment))
            self.retrieved_layout.addWidget(group)
        self.retrieved_layout.addStretch()


class CapsuleDialog(QW.QDialog):
    """Workspace capsule dialog: export, inspect, replay, revisions, registry.

    Args:
        mainwindow: DataLab main window
        parent: Parent widget (default: the main window)
    """

    def __init__(self, mainwindow: DLMainWindow, parent: QW.QWidget | None = None):
        super().__init__(mainwindow if parent is None else parent)
        self.mainwindow = mainwindow
        self.setWindowTitle(_("Workspace capsule"))
        self.setObjectName("capsule_dialog")
        layout = QW.QVBoxLayout(self)
        self.error_label = _error()
        self.error_label.hide()
        self.tabs = QW.QTabWidget()
        self.export_tab = ExportTab(self)
        self.inspect_tab = InspectTab(self)
        self.replay_tab = ReplayTab(self)
        self.revisions_tab = RevisionsTab(self)
        self.registry_tab = RegistryTab(self)
        for tab, title in (
            (self.export_tab, _("Export")),
            (self.inspect_tab, _("Inspect")),
            (self.replay_tab, _("Local replay")),
            (self.revisions_tab, _("Revisions")),
            (self.registry_tab, _("Registry")),
        ):
            self.tabs.addTab(tab, title)
        layout.addWidget(self.tabs)
        layout.addWidget(self.error_label)
        buttons = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(820, 640)

    @property
    def error(self) -> str | None:
        """Error shown by the dialog, if any."""
        return None if self.error_label.isHidden() else self.error_label.text()

    def show_error(self, message: str | None) -> None:
        """Show an error message (None hides it)."""
        self.error_label.setText(message or "")
        self.error_label.setVisible(bool(message))

    def run(self, task: Callable[[], Any]) -> bool:
        """Run *task* with a wait cursor; show its error, if any.

        Returns:
            True if the task completed without error.
        """
        self.show_error(None)
        self.tabs.setEnabled(False)
        QW.QApplication.setOverrideCursor(QC.Qt.WaitCursor)
        try:
            task()
        except Exception as exc:  # pylint: disable=broad-except
            self.show_error(str(exc) or type(exc).__name__)
            return False
        finally:
            QW.QApplication.restoreOverrideCursor()
            self.tabs.setEnabled(True)
        return True

    def refresh_revisions(self) -> None:
        """Read the revisions seen on this computer again."""
        self.revisions_tab.refresh()

    def refresh(self) -> None:
        """Update every tab (workspace, revisions or registry settings changed)."""
        for tab in (
            self.export_tab,
            self.inspect_tab,
            self.replay_tab,
            self.revisions_tab,
            self.registry_tab,
        ):
            tab.refresh()
