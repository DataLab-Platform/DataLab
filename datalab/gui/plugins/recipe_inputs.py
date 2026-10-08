# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Presentation of recipe inputs: requirements, issues, and assignment dialog."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from typing import Union

from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.objectmodel import get_short_id
from datalab.plugins.recipe_binding import (
    RecipeInputIssue,
    RecipeInputIssueCode,
    RecipeReadiness,
    RecipeReadinessStatus,
    is_compatible,
)
from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeDiagnosticLevel,
    RecipeInputSlot,
    RecipeObjectType,
)
from datalab.widgets.expandabletext import apply_subdued_color

__all__ = [
    "RecipeInputDialog",
    "format_diagnostic",
    "format_input_issue",
    "format_inputs_html",
    "format_readiness",
    "format_slot_kind",
]

DataObject = Union[SignalObj, ImageObj]
BindingsAssessor = Callable[[Mapping[str, Sequence[DataObject]]], RecipeReadiness]


def format_slot_kind(slot: RecipeInputSlot) -> str:
    """Return the number and type of objects expected by an input slot."""
    signal = slot.object_type is RecipeObjectType.SIGNAL
    if slot.cardinality is RecipeCardinality.ONE:
        text = _("One signal") if signal else _("One image")
    elif slot.min_count > 1:
        text = (
            _("At least %d signals") if signal else _("At least %d images")
        ) % slot.min_count
    else:
        text = _("One or more signals") if signal else _("One or more images")
    if not slot.required:
        text = _("%s (optional)") % text
    return text


def format_inputs_html(descriptor: RecipeDescriptor) -> str:
    """Return the expected inputs of a recipe as rich text."""
    blocks: list[str] = []
    for slot in descriptor.inputs:
        lines = [
            f"<b>{html.escape(slot.display_title)}</b> — "
            f"{html.escape(format_slot_kind(slot))}"
        ]
        if slot.description:
            lines.append(html.escape(slot.description))
        for requirement in slot.metadata:
            label = _("required metadata") if requirement.required else _("hint")
            key = html.escape(requirement.key)
            text = f"<i>{html.escape(label)}</i>: <code>{key}</code>"
            if requirement.description:
                text += f" — {html.escape(requirement.description)}"
            lines.append(text)
        blocks.append("<br>".join(lines))
    if not blocks:
        return html.escape(_("No input data"))
    return "<br><br>".join(blocks)


def format_input_issue(issue: RecipeInputIssue, descriptor: RecipeDescriptor) -> str:
    """Return a user-facing description of a recipe input issue."""
    try:
        slot_title = descriptor.get_input(issue.slot_id).display_title
    except KeyError:
        slot_title = issue.slot_id
    details = dict(issue.details)
    values = {
        "slot": slot_title,
        "count": details.get("count", 0),
        "min_count": details.get("min_count", 1),
        "key": details.get("key", ""),
        "titles": ", ".join(str(title) for title in details.get("titles", ())),
    }
    texts = {
        RecipeInputIssueCode.MISSING: _("%(slot)s: no object assigned"),
        RecipeInputIssueCode.AMBIGUOUS: _("%(slot)s: choose the objects to use"),
        RecipeInputIssueCode.TOO_FEW: _(
            "%(slot)s: %(count)d object(s) assigned, at least %(min_count)d required"
        ),
        RecipeInputIssueCode.TOO_MANY: _(
            "%(slot)s: accepts only one object (%(count)d assigned)"
        ),
        RecipeInputIssueCode.WRONG_TYPE: _(
            "%(slot)s: %(count)d object(s) of the wrong type (%(titles)s)"
        ),
        RecipeInputIssueCode.MISSING_METADATA: _(
            "%(slot)s: metadata '%(key)s' missing on %(count)d object(s) (%(titles)s)"
        ),
        RecipeInputIssueCode.DUPLICATE: _(
            "%(slot)s: %(count)d object(s) assigned twice (%(titles)s)"
        ),
    }
    return texts[issue.code] % values


def format_diagnostic(diagnostic: RecipeDiagnostic) -> str:
    """Return a user-facing description of a recipe diagnostic."""
    labels = {
        RecipeDiagnosticLevel.ERROR: _("Error"),
        RecipeDiagnosticLevel.WARNING: _("Warning"),
        RecipeDiagnosticLevel.INFO: _("Information"),
    }
    return f"{labels[diagnostic.level]}: {diagnostic.message}"


def format_readiness(
    readiness: RecipeReadiness, descriptor: RecipeDescriptor
) -> tuple[str, list[str]]:
    """Return the summary and the reasons of a recipe readiness."""
    summaries = {
        RecipeReadinessStatus.READY: _("Ready to run on the current selection"),
        RecipeReadinessStatus.WARNINGS: _("Ready to run, with warnings"),
        RecipeReadinessStatus.NEEDS_ASSIGNMENT: _(
            "Ready to run once the inputs are assigned"
        ),
        RecipeReadinessStatus.NOT_READY: _("The current selection cannot be analyzed"),
        RecipeReadinessStatus.NO_INPUT: _("Select the input data in the workspace"),
    }
    reasons = [format_input_issue(issue, descriptor) for issue in readiness.issues]
    if any(
        issue.code is RecipeInputIssueCode.MISSING_METADATA
        for issue in readiness.issues
    ):
        reasons.append(_("To set it, use Edit > Metadata > Add metadata..."))
    if readiness.status is not RecipeReadinessStatus.NO_INPUT:
        reasons.extend(
            format_diagnostic(diagnostic) for diagnostic in readiness.diagnostics
        )
    return summaries[readiness.status], reasons


class RecipeInputDialog(QW.QDialog):
    """Assign candidate objects to the input slots of a recipe.

    Args:
        parent: parent widget
        descriptor: recipe descriptor
        candidates: objects that may be assigned
        readiness: initial assessment, providing the proposed bindings
        assess: function assessing edited bindings
    """

    def __init__(
        self,
        parent: QW.QWidget | None,
        descriptor: RecipeDescriptor,
        candidates: Sequence[DataObject],
        readiness: RecipeReadiness,
        assess: BindingsAssessor,
    ) -> None:
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.descriptor = descriptor
        self.__assess = assess
        self.__candidates = tuple(candidates)
        self.__compatible: dict[str, list[DataObject]] = {}
        self.combos: dict[str, QW.QComboBox] = {}
        self.lists: dict[str, QW.QListWidget] = {}
        self.setWindowTitle(_("Inputs of '%s'") % descriptor.title)
        self.setMinimumWidth(480)

        layout = QW.QVBoxLayout(self)
        if descriptor.description:
            description = QW.QLabel(descriptor.description)
            description.setWordWrap(True)
            apply_subdued_color(description)
            layout.addWidget(description)
        for slot in descriptor.inputs:
            layout.addWidget(
                self._create_slot_group(slot, readiness.bindings.get(slot.id, ()))
            )
        self.issues_label = QW.QLabel()
        self.issues_label.setWordWrap(True)
        self.issues_label.setTextFormat(QC.Qt.RichText)
        layout.addWidget(self.issues_label)
        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QW.QDialogButtonBox.Ok)
        layout.addWidget(button_box)
        self.update_assessment()

    def _candidate_label(self, slot: RecipeInputSlot, obj: DataObject) -> str:
        """Return the label of a candidate, naming its missing metadata."""
        label = f"{get_short_id(obj)}: {obj.title}"
        missing = [
            requirement.key
            for requirement in slot.metadata
            if requirement.required and obj.metadata.get(requirement.key) is None
        ]
        if missing:
            label += "  " + _("(missing: %s)") % ", ".join(missing)
        return label

    def _create_slot_group(
        self, slot: RecipeInputSlot, bound: Sequence[DataObject]
    ) -> QW.QGroupBox:
        """Create the assignment widgets of one input slot."""
        group = QW.QGroupBox(f"{slot.display_title} — {format_slot_kind(slot)}")
        layout = QW.QVBoxLayout(group)
        if slot.description:
            description = QW.QLabel(slot.description)
            description.setWordWrap(True)
            apply_subdued_color(description)
            layout.addWidget(description)
        compatible = [obj for obj in self.__candidates if is_compatible(slot, obj)]
        self.__compatible[slot.id] = compatible
        bound_ids = {id(obj) for obj in bound}
        if slot.cardinality is RecipeCardinality.ONE:
            combo = QW.QComboBox()
            combo.addItem(_("Not assigned"), None)
            for index, obj in enumerate(compatible):
                combo.addItem(self._candidate_label(slot, obj), index)
                if id(obj) in bound_ids:
                    combo.setCurrentIndex(combo.count() - 1)
            combo.currentIndexChanged.connect(self.update_assessment)
            self.combos[slot.id] = combo
            layout.addWidget(combo)
        else:
            widget = QW.QListWidget()
            for obj in compatible:
                item = QW.QListWidgetItem(self._candidate_label(slot, obj))
                item.setFlags(item.flags() | QC.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QC.Qt.Checked if id(obj) in bound_ids else QC.Qt.Unchecked
                )
                widget.addItem(item)
            widget.setMaximumHeight(160)
            widget.itemChanged.connect(self.update_assessment)
            self.lists[slot.id] = widget
            layout.addWidget(widget)
        if not compatible:
            label = QW.QLabel(_("No compatible object"))
            apply_subdued_color(label)
            layout.addWidget(label)
        return group

    def get_bindings(self) -> dict[str, tuple[DataObject, ...]]:
        """Return the objects assigned to each input slot."""
        bindings: dict[str, tuple[DataObject, ...]] = {}
        for slot_id, combo in self.combos.items():
            index = combo.currentData()
            bindings[slot_id] = (
                () if index is None else (self.__compatible[slot_id][index],)
            )
        for slot_id, widget in self.lists.items():
            candidates = self.__compatible[slot_id]
            bindings[slot_id] = tuple(
                candidates[row]
                for row in range(widget.count())
                if widget.item(row).checkState() == QC.Qt.Checked
            )
        return bindings

    def update_assessment(self, *_args: object) -> None:
        """Assess the current assignment and show its problems."""
        readiness = self.__assess(self.get_bindings())
        _summary, reasons = format_readiness(readiness, self.descriptor)
        self.issues_label.setText(
            "<br>".join(f"• {html.escape(reason)}" for reason in reasons)
        )
        self.issues_label.setVisible(bool(reasons))
        self.ok_button.setEnabled(readiness.can_run)
