# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Provenance view
===============

Read-only view of the provenance of a signal or image, shown in the *History*
tab of the object properties and from the History panel.

The view reads the workspace provenance ledger (see
:meth:`datalab_capsule.recorder.Recorder.describe_object`): the activity that
produced the latest recorded state of the object, its parameters, its ordered
inputs and their ancestors (expanded on demand), what is still available, and
the analyses recorded on the object. Three operations are kept apart:

- consulting reads the ledger only (no fingerprint, no computation);
- *Check data* compares the object and the direct inputs and outputs of the
  activity with their recorded states, without computing;
- *Verify computation* recomputes the recorded activity as a separate candidate
  and compares it with the recorded result (see
  :meth:`datalab.gui.main.DLMainWindow.verify_provenance_activity`).

None of them changes the workspace, the ledger or the History. Results of a
check or a verification are dropped as soon as the workspace or the selection
changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.objectmodel import get_short_id, get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

__all__ = ["ExecutionProvenanceDialog", "ProvenanceWidget", "format_parameter_value"]

#: Longest text shown for one value (details are bounded).
MAX_TEXT = 120
#: Items of a list shown before ellipsis.
MAX_ITEMS = 8

_ROLE = QC.Qt.UserRole
_PLACEHOLDER = {"placeholder": True}


def format_parameter_value(value: Any) -> str:
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


class ProvenanceWidget(QW.QWidget):
    """Read-only provenance view of a signal or image.

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

        self.summary = QW.QLabel()
        self.summary.setObjectName("provenance_summary")
        self.summary.setTextFormat(QC.Qt.PlainText)
        self.summary.setWordWrap(True)
        self.tree = QW.QTreeWidget()
        self.tree.setObjectName("provenance_tree")
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([_("Item"), _("Details")])
        self.tree.setUniformRowHeights(False)
        self.tree.itemExpanded.connect(self.__item_expanded)
        self.tree.currentItemChanged.connect(self.__update_buttons)
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
                "Compare the object and the direct inputs of its production with "
                "their recorded states, without computing"
            )
        )
        self.check_button.clicked.connect(self.check_data)
        self.verify_button = QW.QPushButton(
            get_icon("replay.svg"), _("Verify computation")
        )
        self.verify_button.setObjectName("provenance_verify")
        self.verify_button.setToolTip(
            _(
                "Recompute the recorded activity separately and compare it with "
                "the recorded result; the workspace is not changed"
            )
        )
        self.verify_button.clicked.connect(self.verify)
        self.select_button = QW.QPushButton(_("Select current object"))
        self.select_button.setObjectName("provenance_select")
        self.select_button.setToolTip(
            _("Select the object currently holding this identifier")
        )
        self.select_button.clicked.connect(self.select_current_object)
        self.action_button = QW.QPushButton(
            get_icon("history.svg"), _("History action")
        )
        self.action_button.setObjectName("provenance_history_action")
        self.action_button.setToolTip(
            _("Select the History action whose last execution recorded this step")
        )
        self.action_button.clicked.connect(self.show_history_action)

        buttons = QW.QHBoxLayout()
        for button in (
            self.check_button,
            self.verify_button,
            self.select_button,
            self.action_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.summary)
        layout.addWidget(self.tree)
        layout.addLayout(buttons)
        layout.addWidget(self.report)
        self.__update_buttons()

    # -- Public API -----------------------------------------------------------

    @property
    def description(self) -> dict[str, Any] | None:
        """Description currently shown (see ``Recorder.describe_object``)."""
        return self._description

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

        Args:
            obj: Signal or image, or None to clear the view.
            fallback: Text shown when the ledger records no production (e.g.
             how the object was created).

        Returns:
            True if the ledger records a production, an old record or an
            analysis of the object.
        """
        self._generation += 1
        self._activity_id = None
        self._fallback = fallback
        self._object_uuid = get_uuid(obj) if obj is not None else None
        self.__refresh(check_data=False)
        return self.has_content()

    def set_activity(self, activity_id: str) -> None:
        """Show one recorded activity (e.g. an execution of a History action)."""
        self._generation += 1
        self._object_uuid = None
        self._activity_id = activity_id
        self._fallback = ""
        self.__refresh(check_data=False)

    def invalidate(self) -> None:
        """Drop check and verification results; read the ledger again."""
        if self._object_uuid is None and self._activity_id is None:
            return
        self._generation += 1
        self.__refresh(check_data=False)

    def check_data(self) -> None:
        """Compare the data with their recorded states (no computation)."""
        if self._object_uuid is None and self._activity_id is None:
            return
        generation = self._generation
        QW.QApplication.setOverrideCursor(QC.Qt.WaitCursor)
        try:
            description = self.__describe(check_data=True)
        finally:
            QW.QApplication.restoreOverrideCursor()
        if generation == self._generation and description is not None:
            self.__show(description)

    def verify(self) -> dict[str, Any] | None:
        """Verify the recorded activity of the selected item (or the production).

        Returns:
            The verification report, or None if nothing was verified (or the
            view changed meanwhile).
        """
        activity_id = self.__selected_activity_id()
        if activity_id is None:
            return None
        generation = self._generation
        report = self.mainwindow.verify_provenance_activity(activity_id)
        if generation != self._generation:
            return None
        self.__show_report(report)
        return report

    def select_current_object(self) -> None:
        """Select the current object holding the identifier of the selected item."""
        uuid = self.__selected_value("object_uuid")
        obj = None if uuid is None else self.mainwindow.find_object_by_uuid(uuid)
        if not isinstance(obj, (SignalObj, ImageObj)):
            return
        panel = (
            self.mainwindow.signalpanel
            if isinstance(obj, SignalObj)
            else self.mainwindow.imagepanel
        )
        self.mainwindow.set_current_panel(panel)
        panel.objview.set_current_item_id(uuid)

    def __history_action(self) -> Any:
        """Return the History action linked to the selected activity, if any."""
        activity_id = self.__selected_activity_id()
        historypanel = self.mainwindow.historypanel
        if activity_id is None or historypanel is None:
            return None
        return historypanel.find_action_for_activity(activity_id)

    def show_history_action(self) -> None:
        """Select the History action whose last execution recorded the activity."""
        action = self.__history_action()
        if action is None:
            return
        historypanel = self.mainwindow.historypanel
        dock = self.mainwindow.docks.get(historypanel)
        if dock is not None:
            dock.show()
            dock.raise_()
        historypanel.navigation.select_action_in_tree(action)

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

    def __refresh(self, check_data: bool) -> None:
        self.report.clear()
        self.report.setVisible(False)
        description = self.__describe(check_data)
        self._description = description
        self.__show(description)

    # -- Display --------------------------------------------------------------

    def __object_name(self, uuid: str | None) -> str | None:
        obj = None if uuid is None else self.mainwindow.find_object_by_uuid(uuid)
        if isinstance(obj, (SignalObj, ImageObj)):
            return f"{get_short_id(obj)}: {obj.title}"
        return None

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

    def __show(self, description: dict[str, Any] | None) -> None:
        self._description = description
        self.tree.clear()
        if description is None:
            self.summary.setText(_("No provenance is recorded for this object."))
            self.__update_buttons()
            return
        if "activity" in description:
            activity = description["activity"]
            self.summary.setText(
                _("Recorded execution: %s") % self.__operation_title(activity)
            )
            self.__add_activity(self.tree.invisibleRootItem(), activity, [])
            self.tree.expandToDepth(0)
            self.__update_buttons()
            return
        lines = [_content_message(description["anchor"]["content"])]
        if description["anchor"].get("file_status") == "altered":
            lines.append(_("The saved data of this object were altered in the file."))
        if description["production"] is None and description["state"] is not None:
            lines.append(_("This state was observed, without recorded production."))
        self.summary.setText("\n".join(lines))
        root = self.tree.invisibleRootItem()
        trail = (
            [] if description["state"] is None else [description["state"]["state_id"]]
        )
        if description["production"] is not None:
            item = self.__item(root, _("Production"), "")
            self.__add_activity(item, description["production"], trail)
            item.setExpanded(True)
        elif self._fallback:
            item = self.__item(root, _("Metadata history (not verified)"), "")
            for line in self._fallback.splitlines():
                self.__item(item, line.strip(), "")
        if description.get("legacy"):
            item = self.__item(root, _("Records of an old workspace"), "")
            for activity in description["legacy"]:
                self.__add_activity(item, activity, trail)
        if description.get("analyses"):
            item = self.__item(root, _("Recorded analyses"), "")
            for activity in description["analyses"]:
                self.__add_activity(item, activity, trail, analysis=True)
        earlier = description.get("object_states", [])[1:]
        if earlier:
            item = self.__item(root, _("Earlier recorded states"), "")
            for state in earlier:
                child = self.__item(
                    item,
                    _("State %s") % state["sequence"],
                    _("Produced by a recorded activity")
                    if state["produced_by"]
                    else _("Observed state"),
                    {"state_id": state["state_id"], "trail": trail},
                )
                if state["produced_by"]:
                    self.__item(child, "…", "", _PLACEHOLDER)
        self.__update_buttons()

    def __item(
        self,
        parent: QW.QTreeWidgetItem,
        text: str,
        details: str,
        data: dict[str, Any] | None = None,
    ) -> QW.QTreeWidgetItem:
        item = QW.QTreeWidgetItem(parent, [text, details])
        item.setToolTip(1, details)
        if data is not None:
            item.setData(0, _ROLE, data)
        return item

    def __add_activity(
        self,
        parent: QW.QTreeWidgetItem,
        activity: dict[str, Any],
        trail: list[str],
        analysis: bool = False,
    ) -> None:
        when = activity.get("finished_at") or ""
        details = " — ".join(
            part
            for part in (
                _origin_text(activity["origin"]),
                activity.get("edition"),
                when,
            )
            if part
        )
        item = self.__item(
            parent,
            self.__operation_title(activity),
            details,
            {"activity_id": activity["activity_id"]},
        )
        support = activity["support"]
        if support.get("reason"):
            self.__item(item, _("Verification"), support["reason"])
        parameters = activity.get("parameters")
        if parameters:
            params_item = self.__item(item, _("Parameters"), "")
            for name, value in parameters.items():
                self.__item(params_item, str(name), format_parameter_value(value))
            params_item.setExpanded(True)
        elif parameters is None and activity["operation"] is not None:
            self.__item(item, _("Parameters"), _("Not recorded"))
        inputs_item = self.__item(
            item, _("Inputs"), str(activity.get("input_count", len(activity["inputs"])))
        )
        for position, entry in enumerate(activity["inputs"]):
            # An absent input is not the object now holding its identifier
            # (identity break after appending a workspace).
            uuid = entry["object_uuid"] if entry["object_present"] else None
            name = (uuid and self.__object_name(uuid)) or _("Original input missing")
            status = _input_message(entry["data"])
            if entry.get("cycle"):
                status += " — " + _("already shown above (cycle)")
            child = self.__item(
                inputs_item,
                f"{position + 1}. {entry['role']}: {name}",
                status,
                {
                    "object_uuid": uuid,
                    "state_id": entry["state_id"],
                    "trail": trail,
                },
            )
            if entry.get("has_ancestors"):
                self.__item(child, "…", "", _PLACEHOLDER)
        inputs_item.setExpanded(True)
        outputs = activity.get("outputs") or []
        if outputs or activity.get("no_result"):
            outputs_item = self.__item(item, _("Results"), "")
            if activity.get("no_result"):
                self.__item(
                    outputs_item, _("No result"), _("Nothing was found (not an error)")
                )
            for output in outputs:
                uuid = output.get("object_uuid")
                if output["recorded"] == "artifact":
                    text = f"{output['role']}: {output['key']}"
                else:
                    uuid = uuid if output["object_present"] else None
                    text = f"{output['role']}: " + (
                        (uuid and self.__object_name(uuid))
                        or _("Object no longer exists")
                    )
                self.__item(
                    outputs_item,
                    text,
                    _output_message(output["status"]),
                    {"object_uuid": uuid},
                )
            if analysis:
                outputs_item.setExpanded(True)
        self.__add_context(item, activity)
        if activity.get("limits"):
            self.__item(item, _("Limits"), ", ".join(activity["limits"]))
        legacy = activity.get("legacy")
        if legacy:
            self.__item(
                item,
                _("Old record"),
                ", ".join(legacy.get("issues") or []) or _("No issue"),
            )

    def __add_context(self, parent: QW.QTreeWidgetItem, activity: dict) -> None:
        context = activity.get("context") or {}
        environment = activity.get("environment") or {}
        if not context and not environment:
            return
        item = self.__item(parent, _("Details"), "")
        for key in ("options", "x_alignment", "roi", "mask"):
            value = context.get(key)
            if value:
                self.__item(item, key, format_parameter_value(value))
        for read in context.get("metadata") or []:
            status = read.get("status")
            text = _("Metadata read: %s") % read.get("key")
            self.__item(item, text, status or _("Data to be checked"))
        if environment:
            version = environment.get("edition_version") or "?"
            self.__item(
                item,
                _("Software"),
                f"{environment.get('edition', '?')} {version}",
            )
            for name, version in (environment.get("packages") or {}).items():
                self.__item(item, name, str(version))

    def __item_expanded(self, item: QW.QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).data(0, _ROLE) != _PLACEHOLDER:
            return
        item.takeChild(0)
        data = item.data(0, _ROLE) or {}
        state_id = data.get("state_id")
        if state_id is None:
            return
        try:
            description = self.mainwindow.provenance.describe_state(
                state_id, trail=data.get("trail") or []
            )
        except KeyError:
            return
        production = description["production"]
        if production is None:
            self.__item(item, _("Observed state, without recorded production"), "")
            return
        if "depth_limit" in description["limits"]:
            self.__item(item, _("Too many levels: not expanded"), "")
            return
        trail = [*(data.get("trail") or []), state_id]
        self.__add_activity(item, production, trail)

    def __show_report(self, report: dict[str, Any]) -> None:
        lines = [
            _("Verification: %s") % _verdict_text(report.get("verdict", "")),
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

    # -- Selection ------------------------------------------------------------

    def __selected_value(self, key: str) -> Any:
        item = self.tree.currentItem()
        while item is not None:
            data = item.data(0, _ROLE) or {}
            if data.get(key) is not None:
                return data[key]
            item = item.parent()
        return None

    def __selected_activity_id(self) -> str | None:
        activity_id = self.__selected_value("activity_id")
        if activity_id is not None:
            return activity_id
        if self._activity_id is not None:
            return self._activity_id
        description = self._description or {}
        production = description.get("production")
        return None if production is None else production["activity_id"]

    def __update_buttons(self, *_args) -> None:
        has = self._description is not None
        self.check_button.setEnabled(has)
        self.verify_button.setEnabled(self.__selected_activity_id() is not None)
        uuid = self.__selected_value("object_uuid")
        self.select_button.setEnabled(
            uuid is not None
            and uuid != self._object_uuid
            and self.mainwindow.find_object_by_uuid(uuid) is not None
        )
        self.action_button.setEnabled(self.__history_action() is not None)


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
