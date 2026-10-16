# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Provenance view
===============

Read-only view of the provenance of a signal or image, shown in the *Provenance*
tab of the object properties and from the History panel, as cards (one per
recorded step) with links next to what they act on.

The view reads the workspace provenance ledger (see
:meth:`datalab_capsule.recorder.Recorder.describe_object`): the activity that
produced the latest recorded state of the object, its parameters, its ordered
inputs and their ancestors (expanded on demand), what is still available, and
the analyses recorded on the object. Three operations are kept apart:

- consulting reads the ledger only (no fingerprint, no computation);
- *Check data* compares the object, the direct inputs and outputs of the
  activity and the earlier steps shown with their recorded states, without
  computing;
- *Verify computation* recomputes the recorded activity as a separate candidate
  and compares it with the recorded result (see
  :meth:`datalab.gui.main.DLMainWindow.verify_provenance_activity`).

None of them changes the workspace, the ledger or the History. Results of a
check or a verification are dropped as soon as the workspace or the selection
changes.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.objectmodel import get_short_id, get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

__all__ = [
    "EDITION_NAMES",
    "ExecutionProvenanceDialog",
    "ProvenanceWidget",
    "edition_name",
    "format_parameter_value",
]

#: Longest text shown for one value (details are bounded).
MAX_TEXT = 120
#: Items of a list shown before ellipsis.
MAX_ITEMS = 8


def format_parameter_value(value: Any) -> str:  # pylint: disable=R0911
    """Return a bounded text for a parameter value encoded by the codec.

    Arrays are shown by type and shape, never by content; long lists and
    strings are shortened.
    """
    if isinstance(value, dict):
        if "$ndarray" in value:
            array = value["$ndarray"]
            shape = "×".join(str(n) for n in array.get("shape", []))
            return _("array %s (%s)") % (shape or "0", array.get("dtype", "?"))
        if "$float" in value:
            return str(value["$float"])
        if "$complex" in value:
            real, imag = (format_parameter_value(v) for v in value["$complex"])
            return f"({real}{'' if imag.startswith('-') else '+'}{imag}j)"
        if "$tuple" in value:
            items = [format_parameter_value(v) for v in value["$tuple"]]
            return "(" + ", ".join(items) + ")"
        for key in ("$datetime", "$date"):
            if key in value:
                return str(value[key])
        inner = value.get("$dict", value)
        items = [f"{k}={format_parameter_value(v)}" for k, v in inner.items()]
        text = "{" + ", ".join(items[:MAX_ITEMS])
        return text + (", …}" if len(items) > MAX_ITEMS else "}")
    if isinstance(value, list):
        items = [format_parameter_value(v) for v in value[:MAX_ITEMS]]
        suffix = ", …" if len(value) > MAX_ITEMS else ""
        return "[" + ", ".join(items) + suffix + f"] ({len(value)})"
    text = "—" if value is None else str(value)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _content_message(content: str) -> str:
    return {
        "unchecked": _("Data to be checked"),
        "matches": _("The object holds the recorded state"),
        "differs": _(
            "The current content differs from the last recorded state: the "
            "recorded production is not attributed to it"
        ),
        "absent": _("The object no longer exists"),
        "unavailable": _("The data of the object are not available"),
        "unknown": _("The state cannot be checked (no fingerprint)"),
        "unrecorded": _("No state of this object is recorded"),
    }.get(content, content)


def _input_message(status: str) -> str:
    return {
        "unchecked": _("Data to be checked"),
        "available": _("Recorded state available"),
        "changed": _("This input has changed since the computation"),
        "altered": _("This input has changed since the computation"),
        "missing": _("Original input missing"),
        "unavailable": _("The data of this input are not available"),
        "unsupported": _("This input cannot be checked"),
        "unknown": _("State at execution time unknown"),
    }.get(status, status)


def _output_message(status: str) -> str:
    return {
        "unchecked": _("Data to be checked"),
        "available": _("Recorded result available"),
        "altered": _("The result was modified since the computation"),
        "missing": _(
            "Recorded result absent: recomputation possible, comparison with "
            "the original result unavailable"
        ),
        "unavailable": _("The data of the result are not available"),
        "replaced": _("Result replaced by another analysis"),
    }.get(status, status)


def _origin_text(origin: str) -> str:
    return {
        "ordinary": _("Processing"),
        "recompute_in_place": _("Applied again in place"),
        "history_replay": _("History replay"),
        "legacy_metadata": _("Record of an old workspace"),
    }.get(origin, origin)


def _verdict_text(verdict: str) -> str:
    return {
        "exact": _("Exact"),
        "within_tolerance": _("Within tolerance"),
        "different": _("Different"),
        "not_verified": _("Not verified"),
    }.get(verdict, verdict)


#: Product names of the editions recorded in the ledger (``environment.edition``)
EDITION_NAMES = {"desktop": "DataLab Desktop", "web": "DataLab Web"}


def edition_name(edition: str | None) -> str:
    """Return the product name of a recorded edition (e.g. ``DataLab Desktop``)."""
    if not edition:
        return ""
    return EDITION_NAMES.get(edition, edition)


def _short_date(timestamp: str | None) -> str:
    """Return a recorded UTC timestamp as a short local date and time."""
    if not timestamp:
        return ""
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return str(timestamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def _esc(text: Any) -> str:
    """Return *text* as HTML text (titles and values are never markup)."""
    return html.escape(str(text), quote=True)


class ProvenanceWidget(QW.QWidget):
    """Read-only provenance view of a signal or image, presented as cards.

    Each recorded step is a card: the operation and when it was recorded, its
    parameters, its inputs and results with their status, and links acting on
    what they are next to (show the recorded step of an input, go to an
    object, verify the step, select its History action, show its details).
    Text wraps: nothing is truncated in a narrow panel.

    Args:
        mainwindow: DataLab main window.
        parent: Parent widget.
    """

    def __init__(
        self, mainwindow: DLMainWindow, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.mainwindow = mainwindow
        self._object_uuid: str | None = None
        self._activity_id: str | None = None
        self._fallback: str = ""
        self._description: dict[str, Any] | None = None
        self._generation = 0
        # True once the data were checked: steps expanded afterwards are
        # checked too (reset when another object or activity is shown).
        self._checked = False
        # Earlier steps expanded by the user, and their descriptions (by path)
        self._expanded: set[str] = set()
        self._steps: dict[str, dict[str, Any] | None] = {}
        # Steps whose details are shown
        self._details: set[str] = set()
        # Links of the current rendering: href -> (action, payload, text)
        self._links: dict[str, tuple[str, Any, str]] = {}
        self._html: list[str] = []
        self._muted = self._warning = ""
        self._border = self._card = ""

        self.summary = QW.QLabel()
        self.summary.setObjectName("provenance_summary")
        self.summary.setTextFormat(QC.Qt.PlainText)
        self.summary.setWordWrap(True)
        self.browser = QW.QTextBrowser()
        self.browser.setObjectName("provenance_browser")
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(
            lambda url: self.activate_link(url.toString())
        )
        self.report = QW.QLabel()
        self.report.setObjectName("provenance_report")
        self.report.setTextFormat(QC.Qt.PlainText)
        self.report.setWordWrap(True)
        self.report.setVisible(False)

        self.check_button = QW.QPushButton(
            get_icon("show_results.svg"), _("Check data")
        )
        self.check_button.setObjectName("provenance_check")
        self.check_button.setToolTip(
            _(
                "Compare the object, the direct inputs of its production and the "
                "earlier steps shown with their recorded states, without computing"
            )
        )
        self.check_button.clicked.connect(self.check_data)
        self.verify_button = QW.QPushButton(
            get_icon("replay.svg"), _("Verify computation")
        )
        self.verify_button.setObjectName("provenance_verify")
        self.verify_button.setToolTip(
            _(
                "Recompute the recorded production separately and compare it with "
                "the recorded result; the workspace is not changed"
            )
        )
        self.verify_button.clicked.connect(lambda _checked=False: self.verify())

        buttons = QW.QHBoxLayout()
        buttons.addWidget(self.check_button)
        buttons.addWidget(self.verify_button)
        buttons.addStretch()
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.summary)
        layout.addWidget(self.browser)
        layout.addLayout(buttons)
        layout.addWidget(self.report)
        self.__update_buttons()

    # -- Public API -----------------------------------------------------------

    @property
    def description(self) -> dict[str, Any] | None:
        """Description currently shown (see ``Recorder.describe_object``)."""
        return self._description

    @property
    def links(self) -> list[dict[str, Any]]:
        """Links of the current rendering (``href``, ``action``, ``text``...)."""
        return [
            {"href": href, "action": action, "payload": payload, "text": text}
            for href, (action, payload, text) in self._links.items()
        ]

    def text(self) -> str:
        """Return the text of the view (as read by a screen reader)."""
        return self.browser.toPlainText()

    def has_content(self) -> bool:
        """Return True if the ledger records something about the shown object."""
        desc = self._description
        if desc is None:
            return False
        if self._activity_id is not None:
            return True
        return bool(
            desc.get("production") is not None
            or desc.get("legacy")
            or desc.get("analyses")
        )

    def set_object(self, obj: SignalObj | ImageObj | None, fallback: str = "") -> bool:
        """Show the provenance of *obj* (consultation only).

        Showing the same object again (e.g. after it was modified) keeps the
        steps expanded and the details shown.

        Args:
            obj: Signal or image, or None to clear the view.
            fallback: Text shown when the ledger records no production (e.g.
             how the object was created).

        Returns:
            True if the ledger records a production, an old record or an
            analysis of the object.
        """
        self._generation += 1
        uuid = get_uuid(obj) if obj is not None else None
        same = uuid is not None and uuid == self._object_uuid
        self._activity_id = None
        self._fallback = fallback
        self._object_uuid = uuid
        self._checked = False
        self.__refresh(check_data=False, keep_layout=same)
        return self.has_content()

    def set_activity(self, activity_id: str) -> None:
        """Show one recorded activity (e.g. an execution of a History action)."""
        self._generation += 1
        same = activity_id == self._activity_id
        self._object_uuid = None
        self._activity_id = activity_id
        self._fallback = ""
        self._checked = False
        self.__refresh(check_data=False, keep_layout=same)

    def invalidate(self) -> None:
        """Drop check and verification results; read the ledger again."""
        if self._object_uuid is None and self._activity_id is None:
            return
        self._generation += 1
        self._checked = False
        self.__refresh(check_data=False, keep_layout=True)

    def check_data(self) -> None:
        """Compare the data with their recorded states (no computation).

        The steps shown (or expanded afterwards) are checked too; the steps
        expanded and the details shown stay so.
        """
        if self._object_uuid is None and self._activity_id is None:
            return
        generation = self._generation
        QW.QApplication.setOverrideCursor(QC.Qt.WaitCursor)
        try:
            description = self.__describe(check_data=True)
            if generation == self._generation and description is not None:
                self._checked = True
                self._steps.clear()
                self._description = description
                self.__render(keep_scroll=True)
        finally:
            QW.QApplication.restoreOverrideCursor()

    def verify(self, activity_id: str | None = None) -> dict[str, Any] | None:
        """Verify a recorded activity (by default, the production shown).

        Args:
            activity_id: Activity to verify (a step of the view), or None for
             the production of the object (or the activity shown).

        Returns:
            The verification report, or None if nothing was verified (or the
            view changed meanwhile).
        """
        activity_id = activity_id or self.__main_activity_id()
        if activity_id is None:
            return None
        generation = self._generation
        report = self.mainwindow.verify_provenance_activity(activity_id)
        if generation != self._generation:
            return None
        try:
            described = self.mainwindow.provenance.describe_activity(activity_id)
            title = self.__operation_title(described)
        except KeyError:
            title = activity_id
        self.__show_report(report, title)
        return report

    def go_to_object(self, uuid: str) -> None:
        """Select, in the signal or image panel, the object holding *uuid*."""
        obj = self.mainwindow.find_object_by_uuid(uuid)
        if not isinstance(obj, (SignalObj, ImageObj)):
            return
        panel = (
            self.mainwindow.signalpanel
            if isinstance(obj, SignalObj)
            else self.mainwindow.imagepanel
        )
        self.mainwindow.set_current_panel(panel)
        panel.objview.set_current_item_id(uuid)

    def show_history_action(self, activity_id: str) -> None:
        """Select the History action whose last execution recorded *activity_id*."""
        action = self.__history_action(activity_id)
        if action is None:
            return
        historypanel = self.mainwindow.historypanel
        dock = self.mainwindow.docks.get(historypanel)
        if dock is not None:
            dock.show()
            dock.raise_()
        historypanel.navigation.select_action_in_tree(action)

    def activate_link(self, href: str) -> None:
        """Run the action of a link of the view (same path as a click)."""
        target = self._links.get(href)
        if target is None:
            return
        action, payload, _text = target
        if action == "expand":
            self._expanded.add(payload)
        elif action == "collapse":
            self._expanded.discard(payload)
            self._steps.pop(payload, None)
        elif action == "details":
            self._details.symmetric_difference_update({payload})
        elif action == "verify":
            self.verify(payload)
            return
        elif action == "goto":
            self.go_to_object(payload)
            return
        elif action == "action":
            self.show_history_action(payload)
            return
        self.__render(keep_scroll=True)

    # -- Ledger access --------------------------------------------------------

    def __describe(self, check_data: bool) -> dict[str, Any] | None:
        provenance = self.mainwindow.provenance
        try:
            if self._activity_id is not None:
                activity = provenance.describe_activity(self._activity_id, check_data)
                return {"activity": activity}
            if self._object_uuid is not None:
                return provenance.describe_object(self._object_uuid, check_data)
        except KeyError:
            return None
        return None

    def __step(self, key: str, state_id: str, trail: list[str]) -> dict | None:
        """Return the description of an expanded earlier step (cached)."""
        if key not in self._steps:
            try:
                self._steps[key] = self.mainwindow.provenance.describe_state(
                    state_id, check_data=self._checked, trail=trail
                )
            except KeyError:
                self._steps[key] = None
        return self._steps[key]

    def __history_action(self, activity_id: str | None) -> Any:
        historypanel = self.mainwindow.historypanel
        if activity_id is None or historypanel is None:
            return None
        return historypanel.find_action_for_activity(activity_id)

    def __main_activity_id(self) -> str | None:
        if self._activity_id is not None:
            return self._activity_id
        production = (self._description or {}).get("production")
        return None if production is None else production["activity_id"]

    def __refresh(self, check_data: bool, keep_layout: bool = False) -> None:
        self.report.clear()
        self.report.setVisible(False)
        if not keep_layout:
            self._expanded.clear()
            self._details.clear()
        self._steps.clear()
        self._description = self.__describe(check_data)
        self.__render(keep_scroll=keep_layout)

    # -- Display --------------------------------------------------------------

    def __object_name(self, uuid: str | None) -> str | None:
        obj = None if uuid is None else self.mainwindow.find_object_by_uuid(uuid)
        if not isinstance(obj, (SignalObj, ImageObj)):
            return None
        try:
            return f"{get_short_id(obj)}: {obj.title}"
        except ValueError:
            # Object numbers are briefly absent while metadata are deleted
            # (see BaseDataPanel.delete_metadata); the view is refreshed after.
            return obj.title

    def __operation_title(self, activity: dict[str, Any]) -> str:
        operation = activity.get("operation")
        if operation is None:
            implementation = activity.get("implementation") or {}
            name = implementation.get("python_name")
            return name or _("Unknown operation")
        title = operation["id"]
        try:
            contract = self.mainwindow.provenance.resolve_function(
                operation["id"], operation["contract_version"]
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return title
        for panel in (self.mainwindow.signalpanel, self.mainwindow.imagepanel):
            # pylint: disable=import-outside-toplevel
            from datalab.processingcall import feature_for_function

            feature = feature_for_function(panel.processor, contract.function)
            if feature is not None and feature.title:
                return f"{feature.title} ({title})"
        return title

    def __colors(self) -> tuple[str, str]:
        """Return the muted and warning colors for the current palette.

        Also sets the border and background of the step cards.
        """
        palette = self.browser.palette()
        text = palette.color(QG.QPalette.Text)
        base = palette.color(QG.QPalette.Base)

        def blend(ratio: float) -> str:
            return QG.QColor(
                round(base.red() + (text.red() - base.red()) * ratio),
                round(base.green() + (text.green() - base.green()) * ratio),
                round(base.blue() + (text.blue() - base.blue()) * ratio),
            ).name()

        self._border = blend(0.3)
        self._card = blend(0.05)
        warning = "#f0a040" if base.lightness() < 128 else "#b05000"
        return blend(0.5), warning

    def __link(self, action: str, payload: Any, text: str) -> str:
        href = f"provenance:{len(self._links)}"
        self._links[href] = (action, payload, text)
        return f'<a href="{href}">{_esc(text)}</a>'

    def __render(self, keep_scroll: bool = False) -> None:
        """Render the description shown as HTML cards."""
        scroll = self.browser.verticalScrollBar().value()
        self._links = {}
        self._muted, self._warning = self.__colors()
        self._html = []
        self.__build(self._description)
        self.browser.setHtml("".join(self._html))
        if keep_scroll:
            self.browser.verticalScrollBar().setValue(scroll)
        self.__update_buttons()

    def __line(self, content: str, level: int = 0, top: int = 0) -> None:
        """Add one paragraph of HTML *content*, indented by *level*."""
        self._html.append(
            f'<p style="margin-left:{level * 14}px; margin-top:{top}px; '
            f'margin-bottom:0px">{content}</p>'
        )

    def __muted(self, text: str) -> str:
        return f'<span style="color:{self._muted}">{_esc(text)}</span>'

    def __status(self, status: str, message: str) -> str:
        """Return a status message, highlighted when attention is needed."""
        if status in ("changed", "altered", "missing", "differs", "absent"):
            return f'<span style="color:{self._warning}">{_esc(message)}</span>'
        return self.__muted(message)

    def __heading(self, text: str) -> None:
        self.__line(f"<b>{_esc(text)}</b>", 0, top=8)

    def __build(self, description: dict[str, Any] | None) -> None:
        if description is None:
            self.summary.setText(_("No provenance is recorded for this object."))
            if self._fallback:
                self.__fallback()
            return
        if "activity" in description:
            activity = description["activity"]
            self.summary.setText(
                _("Recorded execution: %s") % self.__operation_title(activity)
            )
            self.__add_activity(activity, [], 0, "a")
            return
        lines = [_content_message(description["anchor"]["content"])]
        if description["anchor"].get("file_status") == "altered":
            lines.append(_("The saved data of this object were altered in the file."))
        if description["production"] is None and description["state"] is not None:
            lines.append(_("This state was observed, without recorded production."))
        self.summary.setText("\n".join(lines))
        trail = (
            [] if description["state"] is None else [description["state"]["state_id"]]
        )
        if description["production"] is not None:
            self.__heading(_("Production"))
            self.__add_activity(description["production"], trail, 0, "p")
        elif self._fallback:
            self.__fallback()
        if description.get("legacy"):
            self.__heading(_("Records of an old workspace"))
            for index, activity in enumerate(description["legacy"]):
                self.__add_activity(activity, trail, 0, f"l{index}")
        if description.get("analyses"):
            self.__heading(_("Recorded analyses"))
            for index, activity in enumerate(description["analyses"]):
                self.__add_activity(activity, trail, 0, f"r{index}")
        earlier = description.get("object_states", [])[1:]
        if earlier:
            self.__heading(_("Earlier recorded states"))
            for state in earlier:
                key = "e/" + state["state_id"]
                content = (
                    _esc(_("State %s") % state["sequence"])
                    + " — "
                    + self.__muted(
                        _("Produced by a recorded activity")
                        if state["produced_by"]
                        else _("Observed state")
                    )
                )
                if state["produced_by"]:
                    content += " " + self.__step_link(key, state["state_id"])
                self.__line(content, 1)
                if key in self._expanded:
                    self.__add_step(key, state["state_id"], trail, 2)

    def __fallback(self) -> None:
        self.__heading(_("Metadata history (not verified)"))
        for line in self._fallback.splitlines():
            if line.strip():
                self.__line(_esc(line.strip()), 1)

    def __step_link(self, key: str, state_id: str) -> str:
        del state_id  # the key identifies the step
        # The triangle says that the link shows or hides a part of the view
        if key in self._expanded:
            return self.__link("collapse", key, "▾ " + _("Hide recorded step"))
        return self.__link("expand", key, "▸ " + _("Show recorded step"))

    def __add_step(self, key: str, state_id: str, trail: list[str], level: int) -> None:
        """Add an earlier step (the production of a recorded state)."""
        description = self.__step(key, state_id, trail)
        if description is None:
            return
        production = description["production"]
        if production is None:
            self.__line(
                self.__muted(_("Observed state, without recorded production")), level
            )
        elif "depth_limit" in description["limits"]:
            self.__line(self.__muted(_("Too many levels: not expanded")), level)
        else:
            self.__add_activity(production, [*trail, state_id], level, key)

    def __add_activity(
        self, activity: dict[str, Any], trail: list[str], level: int, key: str
    ) -> None:
        """Add the card of one recorded activity (framed: an earlier step shown
        inside another card is visibly a part that can be hidden again)."""
        details = " — ".join(
            part
            for part in (
                _origin_text(activity["origin"]),
                edition_name(activity.get("edition")),
                _short_date(activity.get("finished_at")),
            )
            if part
        )
        self._html.append(
            f'<table width="100%" border="1" cellspacing="0" cellpadding="4" '
            f'style="margin-left:{level * 14}px; margin-top:4px; '
            f"border-style:solid; border-color:{self._border}; "
            f'background-color:{self._card}"><tr><td>'
        )
        self.__line(
            f"<b>{_esc(self.__operation_title(activity))}</b><br>"
            + self.__muted(details),
            0,
        )
        inner = 1
        support = activity["support"]
        if support.get("reason"):
            self.__line(self.__muted(support["reason"]), inner)
        parameters = activity.get("parameters")
        if parameters:
            for name, value in parameters.items():
                self.__line(
                    self.__muted(f"{name} = ") + _esc(format_parameter_value(value)),
                    inner,
                )
        elif parameters is None and activity["operation"] is not None:
            self.__line(self.__muted(_("Parameters: not recorded")), inner)
        if activity.get("no_result"):
            self.__line(_esc(_("Nothing was found (not an error)")), inner)
        self.__add_inputs(activity, trail, inner, key)
        self.__add_outputs(activity, inner)
        if activity.get("limits"):
            self.__line(
                self.__muted(_("Limits: %s") % ", ".join(activity["limits"])), inner
            )
        legacy = activity.get("legacy")
        if legacy:
            issues = ", ".join(legacy.get("issues") or []) or _("No issue")
            self.__line(self.__muted(_("Old record: %s") % issues), inner)
        links = [self.__link("verify", activity["activity_id"], _("Verify this step"))]
        if self.__history_action(activity["activity_id"]) is not None:
            links.append(
                self.__link("action", activity["activity_id"], _("History action"))
            )
        details_key = "d/" + key
        shown = details_key in self._details
        links.append(
            self.__link(
                "details",
                details_key,
                "▾ " + _("Hide details") if shown else "▸ " + _("Show details"),
            )
        )
        self.__line(" · ".join(links), inner)
        if shown:
            self.__add_details(activity, inner + 1)
        self._html.append("</td></tr></table>")

    def __add_inputs(
        self, activity: dict[str, Any], trail: list[str], level: int, key: str
    ) -> None:
        inputs = activity["inputs"]
        count = activity.get("input_count", len(inputs))
        if not inputs:
            return
        self.__line(f"<b>{_esc(_('Inputs'))}</b> " + self.__muted(f"({count})"), level)
        for position, entry in enumerate(inputs):
            # An absent input is not the object now holding its identifier
            # (identity break after appending a workspace).
            uuid = entry["object_uuid"] if entry["object_present"] else None
            name = (uuid and self.__object_name(uuid)) or _("Original input missing")
            content = _esc(f"{position + 1}. {entry['role']}: {name}")
            status = _input_message(entry["data"])
            if entry.get("cycle"):
                status += " — " + _("already shown above (cycle)")
            content += " — " + self.__status(entry["data"], status)
            step_key = f"{key}/{position}"
            links = []
            if entry.get("has_ancestors") and entry["state_id"] is not None:
                links.append(self.__step_link(step_key, entry["state_id"]))
            if uuid is not None and uuid != self._object_uuid and name:
                links.append(self.__go_to(uuid))
            if links:
                content += "<br>" + " · ".join(links)
            self.__line(content, level + 1)
            if step_key in self._expanded and entry["state_id"] is not None:
                self.__add_step(step_key, entry["state_id"], trail, level + 1)
        if count > len(inputs):
            self.__line(self.__muted(_("%d inputs in total") % count), level + 1)

    def __add_outputs(self, activity: dict[str, Any], level: int) -> None:
        outputs = activity.get("outputs") or []
        if not outputs:
            return
        self.__line(f"<b>{_esc(_('Results'))}</b>", level)
        for output in outputs:
            uuid = output.get("object_uuid")
            if output["recorded"] == "artifact":
                text = f"{output['role']}: {output['key']}"
            else:
                uuid = uuid if output["object_present"] else None
                text = f"{output['role']}: " + (
                    (uuid and self.__object_name(uuid)) or _("Object no longer exists")
                )
            content = (
                _esc(text)
                + " — "
                + self.__status(output["status"], _output_message(output["status"]))
            )
            if uuid is not None and uuid != self._object_uuid:
                if self.mainwindow.find_object_by_uuid(uuid) is not None:
                    content += "<br>" + self.__go_to(uuid)
            self.__line(content, level + 1)

    def __go_to(self, uuid: str) -> str:
        return self.__link("goto", uuid, _("Go to object"))

    def __add_details(self, activity: dict[str, Any], level: int) -> None:
        context = activity.get("context") or {}
        environment = activity.get("environment") or {}
        for name in ("options", "x_alignment", "roi", "mask"):
            value = context.get(name)
            if value:
                self.__line(
                    self.__muted(f"{name} = ") + _esc(format_parameter_value(value)),
                    level,
                )
        for read in context.get("metadata") or []:
            status = read.get("status")
            self.__line(
                _esc(_("Metadata read: %s") % read.get("key"))
                + " — "
                + self.__muted(status or _("Data to be checked")),
                level,
            )
        if environment:
            version = environment.get("edition_version") or "?"
            name = edition_name(environment.get("edition")) or "?"
            self.__line(
                self.__muted(_("Software") + ": ") + _esc(f"{name} {version}"), level
            )
            for package, package_version in (environment.get("packages") or {}).items():
                self.__line(self.__muted(f"{package}: ") + _esc(package_version), level)
        when = activity.get("finished_at")
        if when:
            self.__line(self.__muted(_("Recorded at: ")) + _esc(when), level)

    def __show_report(self, report: dict[str, Any], title: str) -> None:
        lines = [
            _("Verification of %s: %s")
            % (title, _verdict_text(report.get("verdict", ""))),
        ]
        if report.get("reason"):
            lines.append(str(report["reason"]))
        for entry in report.get("inputs") or []:
            lines.append(
                f"  {entry.get('role')}: {_input_message(entry.get('status', ''))}"
            )
        for output in report.get("outputs") or []:
            line = f"  {output.get('role')}: {_verdict_text(output.get('verdict', ''))}"
            if output.get("detail"):
                line += f" ({output['detail']})"
            lines.append(line)
        environment = report.get("environment") or {}
        if environment.get("match") == "different":
            lines.append(_("The software environment differs from the recorded one."))
        self.report.setText("\n".join(lines))
        self.report.setVisible(True)

    def __update_buttons(self) -> None:
        self.check_button.setEnabled(self._description is not None)
        self.verify_button.setEnabled(self.__main_activity_id() is not None)


class ExecutionProvenanceDialog(QW.QDialog):
    """Provenance of the last successful execution of a History action.

    One tab per recorded activity of the execution (e.g. one per result of a
    command applied to several objects). The activities are those linked to
    the action (:attr:`~datalab.history.action.HistoryAction.provenance_activity_ids`);
    when the action was edited without being executed again, the dialog says
    that the provenance describes the last successful execution.

    Args:
        mainwindow: DataLab main window.
        action: History action.
        parent: Parent widget.
    """

    def __init__(
        self, mainwindow: DLMainWindow, action: Any, parent: QW.QWidget | None = None
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from datalab.processingcall import recipe_differs_from_execution

        super().__init__(parent)
        self.setObjectName("execution_provenance_dialog")
        name = action.title or action.func_name or ""
        self.setWindowTitle(_("Provenance of the execution: %s") % name)
        ledger = mainwindow.provenance.ledger
        ids = list(action.provenance_activity_ids)
        known = []
        for activity_id in ids:
            try:
                known.append(ledger.activity(activity_id))
            except KeyError:
                continue
        notes = []
        if not ids:
            notes.append(
                _(
                    "No recorded execution is linked to this action (action "
                    "recorded before this link existed, copied or imported, or "
                    "not executed in this workspace)."
                )
            )
        elif not known:
            notes.append(
                _(
                    "The provenance of this execution is not available in this "
                    "workspace (e.g. History imported without its provenance)."
                )
            )
        elif recipe_differs_from_execution(mainwindow, action):
            notes.append(
                _(
                    "The parameters of this action were edited since its last "
                    "successful execution: the provenance below describes that "
                    "execution."
                )
            )
        self.note = QW.QLabel("\n".join(notes))
        self.note.setObjectName("execution_provenance_note")
        self.note.setTextFormat(QC.Qt.PlainText)
        self.note.setWordWrap(True)
        self.note.setVisible(bool(notes))
        self.tabs = QW.QTabWidget()
        self.views: list[ProvenanceWidget] = []
        for number, activity in enumerate(known, start=1):
            view = ProvenanceWidget(mainwindow, self)
            view.set_activity(activity["activity_id"])
            self.tabs.addTab(view, self.__label(mainwindow, activity, number))
            self.views.append(view)
        self.tabs.setVisible(bool(known))
        buttons = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QW.QVBoxLayout(self)
        layout.addWidget(self.note)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)
        self.resize(760, 560)

    @staticmethod
    def __label(mainwindow: DLMainWindow, activity: dict, number: int) -> str:
        """Return the tab label of an activity: its result, or its number."""
        described = mainwindow.provenance.describe_activity(activity["activity_id"])
        for output in described["outputs"]:
            if output["recorded"] == "state":
                obj = (
                    mainwindow.find_object_by_uuid(output["object_uuid"])
                    if output["object_present"]
                    else None
                )
                if isinstance(obj, (SignalObj, ImageObj)):
                    return f"{get_short_id(obj)}: {obj.title}"
            else:
                return f"#{number}: {output['key']}"
        return f"#{number}"
