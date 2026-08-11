# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Transactional Desktop runner for plugin-provided scientific recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import guidata.dataset as gds
import sigima
from sigima.objects import GeometryResult, ImageObj, SignalObj, TableResult

from datalab import __version__ as datalab_version
from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.objectmodel import ObjectGroup, get_uuid
from datalab.recipes import (
    RECIPE_RUN_RECORD_OPTION,
    RecipeCardinality,
    RecipeDescriptor,
    RecipeExecutionContext,
    RecipeInputs,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
    RecipeRunRecord,
    RecipeRunStatus,
    RecipeValidationError,
)

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel

__all__ = ["RecipeCommitError", "RecipeRunner"]


class RecipeCommitError(RuntimeError):
    """Raised after a failed recipe workspace commit has been rolled back.

    Attributes:
        rollback_error: Exception raised by the rollback itself, or None when
         the workspace was successfully restored.
    """

    def __init__(self, message: str, rollback_error: Exception | None = None) -> None:
        super().__init__(message)
        self.rollback_error = rollback_error


class RecipeRunner:
    """Validate, execute, and transactionally commit a recipe on Desktop."""

    def __init__(self, mainwindow: DLMainWindow) -> None:
        self.mainwindow = mainwindow

    @staticmethod
    def _validate_parameters(
        descriptor: RecipeDescriptor, parameters: gds.DataSet | None
    ) -> None:
        """Validate parameters against the class declared by the recipe."""
        if descriptor.parameter_class is None:
            if parameters is not None:
                raise RecipeValidationError(
                    f"Recipe {descriptor.recipe_id!r} does not accept parameters"
                )
        elif not isinstance(parameters, descriptor.parameter_class):
            raise RecipeValidationError(
                f"Recipe {descriptor.recipe_id!r} requires parameters of type "
                f"{descriptor.parameter_class.__name__}"
            )

    @staticmethod
    def _normalize_inputs(
        descriptor: RecipeDescriptor,
        inputs: Mapping[str, Sequence[SignalObj | ImageObj]],
    ) -> RecipeInputs:
        """Validate and normalize recipe slot assignments to object tuples."""
        if not isinstance(inputs, Mapping):
            raise RecipeValidationError("Recipe inputs must be a mapping")
        declared_ids = {slot.id for slot in descriptor.inputs}
        unknown_ids = set(inputs) - declared_ids
        if unknown_ids:
            unknown = ", ".join(sorted(repr(slot_id) for slot_id in unknown_ids))
            raise RecipeValidationError(f"Unknown recipe input slot(s): {unknown}")

        normalized: dict[str, tuple[SignalObj | ImageObj, ...]] = {}
        for slot in descriptor.inputs:
            values = inputs.get(slot.id, ())
            if not isinstance(values, Sequence):
                raise RecipeValidationError(
                    f"Recipe input slot {slot.id!r} must contain a sequence"
                )
            objects = tuple(values)
            if slot.cardinality is RecipeCardinality.ONE and len(objects) > 1:
                raise RecipeValidationError(
                    f"Recipe input slot {slot.id!r} accepts exactly one object"
                )
            if slot.required and not objects:
                raise RecipeValidationError(
                    f"Required recipe input slot {slot.id!r} is empty"
                )
            expected_type = (
                SignalObj if slot.object_type is RecipeObjectType.SIGNAL else ImageObj
            )
            if any(not isinstance(obj, expected_type) for obj in objects):
                raise RecipeValidationError(
                    f"Recipe input slot {slot.id!r} accepts only "
                    f"{slot.object_type.value} objects"
                )
            normalized[slot.id] = objects
        return normalized

    def _panel_for_object(self, obj: SignalObj | ImageObj) -> BaseDataPanel:
        """Return the Desktop panel owning an output object's type."""
        return (
            self.mainwindow.signalpanel
            if isinstance(obj, SignalObj)
            else self.mainwindow.imagepanel
        )

    def _prepare_outcome(
        self,
        descriptor: RecipeDescriptor,
        outcome: RecipeOutcome,
        parameters: gds.DataSet | None,
        inputs: RecipeInputs,
        started_at: str,
    ) -> RecipeOutcome:
        """Validate outputs and attach scalar results before workspace mutation."""
        anchors = {output.id: output.value for output in outcome.objects}
        seen_uuids: set[str] = set()
        for output in outcome.objects:
            obj = output.value
            obj_uuid = get_uuid(obj)
            if obj_uuid in seen_uuids:
                raise RecipeValidationError(
                    f"Recipe output objects share UUID {obj_uuid!r}"
                )
            seen_uuids.add(obj_uuid)
            if any(
                panel.objmodel.has_uuid(obj_uuid)
                for panel in (
                    self.mainwindow.signalpanel,
                    self.mainwindow.imagepanel,
                )
            ):
                raise RecipeValidationError(
                    f"Recipe output {output.id!r} already belongs to the workspace"
                )

        prepared_results: list[RecipeResultOutput] = []
        for output in outcome.results:
            func_name = f"{descriptor.recipe_id}:{output.id}"
            result = dataclasses.replace(
                output.value,
                func_name=func_name,
                attrs=dict(output.value.attrs),
            )
            anchor = anchors[output.anchor_id]
            if isinstance(result, TableResult):
                TableAdapter(result).add_to(anchor, parameters)
            elif isinstance(result, GeometryResult):
                GeometryAdapter(result).add_to(anchor, parameters)
            prepared_results.append(dataclasses.replace(output, value=result))

        record = RecipeRunRecord(
            run_id=str(uuid4()),
            plugin_id=descriptor.plugin_id,
            plugin_version=descriptor.plugin_version,
            recipe_id=descriptor.recipe_id,
            recipe_version=descriptor.version,
            parameters_json=(
                gds.dataset_to_json(parameters) if parameters is not None else "null"
            ),
            input_uuids={
                slot_id: tuple(get_uuid(obj) for obj in objects)
                for slot_id, objects in inputs.items()
            },
            output_uuids={
                output.id: get_uuid(output.value) for output in outcome.objects
            },
            datalab_version=datalab_version,
            sigima_version=sigima.__version__,
            status=RecipeRunStatus.COMPLETED,
            started_at=started_at,
            finished_at=self._utc_now(),
        )
        for output in outcome.objects:
            output.value.set_metadata_option(
                RECIPE_RUN_RECORD_OPTION,
                record.to_dict(),
            )
        return dataclasses.replace(outcome, results=tuple(prepared_results))

    @staticmethod
    def _utc_now() -> str:
        """Return the current UTC time in an ISO 8601 JSON-friendly form."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _rollback(
        self,
        added_objects: list[tuple[BaseDataPanel, SignalObj | ImageObj]],
        added_groups: list[tuple[BaseDataPanel, ObjectGroup]],
    ) -> None:
        """Remove every model, tree, and plot artifact created by a commit."""
        touched_panels = list(
            dict.fromkeys(panel for panel, _item in (*added_groups, *added_objects))
        )
        with ExitStack() as cleanup:
            for panel in touched_panels:
                cleanup.callback(panel.SIG_OBJECT_REMOVED.emit)
                cleanup.callback(panel.selection_changed, update_items=True)
                cleanup.callback(panel.objview.update_tree)
            for panel, group in added_groups:
                cleanup.callback(self._remove_added_group, panel, group)
            for panel, obj in added_objects:
                cleanup.callback(
                    panel._remove_added_object,  # pylint: disable=protected-access
                    obj,
                )

    @staticmethod
    def _remove_added_group(panel: BaseDataPanel, group: ObjectGroup) -> None:
        """Remove a group created by the current commit."""
        panel.objview.remove_item(get_uuid(group), refresh=False)
        if group in panel.objmodel.get_groups():
            panel.objmodel.remove_group(group)

    def _commit(self, outcome: RecipeOutcome, group_title: str) -> None:
        """Commit recipe outputs to both panels, compensating any failure."""
        if not self.mainwindow.confirm_memory_state():
            raise RecipeCommitError("Recipe commit cancelled due to low memory")

        modified_before = self.mainwindow.is_modified()
        current_panel_before = self.mainwindow.get_current_panel()
        added_groups: list[tuple[BaseDataPanel, ObjectGroup]] = []
        added_objects: list[tuple[BaseDataPanel, SignalObj | ImageObj]] = []
        groups_by_panel: dict[BaseDataPanel, ObjectGroup] = {}
        objects_by_panel: dict[BaseDataPanel, list[SignalObj | ImageObj]] = {}
        try:
            for output in outcome.objects:
                panel = self._panel_for_object(output.value)
                objects_by_panel.setdefault(panel, []).append(output.value)
            for panel in objects_by_panel:
                group = panel.add_group(group_title)
                groups_by_panel[panel] = group
                added_groups.append((panel, group))
            for panel, objects in objects_by_panel.items():
                group = groups_by_panel[panel]
                panel._add_objects(  # pylint: disable=protected-access
                    objects,
                    group_id=get_uuid(group),
                    set_current=False,
                )
                added_objects.extend((panel, obj) for obj in objects)
        except Exception as exc:
            rollback_error: Exception | None = None
            try:
                self._rollback(added_objects, added_groups)
            except Exception as error:  # pylint: disable=broad-except
                rollback_error = error
            finally:
                try:
                    self.mainwindow.set_modified(modified_before)
                finally:
                    self.mainwindow.set_current_panel(current_panel_before)
            if rollback_error is not None:
                raise RecipeCommitError(
                    f"{exc} (rollback also failed: {rollback_error})",
                    rollback_error=rollback_error,
                ) from exc
            raise RecipeCommitError(str(exc)) from exc

    def run(
        self,
        descriptor: RecipeDescriptor,
        inputs: Mapping[str, Sequence[SignalObj | ImageObj]],
        parameters: gds.DataSet | None = None,
        context: RecipeExecutionContext | None = None,
        group_title: str | None = None,
    ) -> RecipeOutcome:
        """Validate, execute, and atomically commit one recipe invocation."""
        if not isinstance(descriptor, RecipeDescriptor):
            raise TypeError("Recipe runner requires a RecipeDescriptor")
        self._validate_parameters(descriptor, parameters)
        normalized_inputs = self._normalize_inputs(descriptor, inputs)
        title = descriptor.title if group_title is None else group_title
        if not isinstance(title, str) or not title.strip():
            raise RecipeValidationError("Recipe output group title must be non-empty")
        if context is None:
            context = RecipeExecutionContext()
        elif not isinstance(context, RecipeExecutionContext):
            raise TypeError("Recipe context must be a RecipeExecutionContext")
        started_at = self._utc_now()
        context.raise_if_cancelled()
        outcome = descriptor.run(normalized_inputs, parameters, context)
        if not isinstance(outcome, RecipeOutcome):
            raise TypeError("Recipe callable must return a RecipeOutcome")
        context.raise_if_cancelled()
        prepared_outcome = self._prepare_outcome(
            descriptor,
            outcome,
            parameters,
            normalized_inputs,
            started_at,
        )
        context.raise_if_cancelled()
        self._commit(prepared_outcome, title)
        return prepared_outcome
