# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Host-independent assignment of objects to recipe inputs, and readiness.

DataLab Desktop and DataLab-Web share this module: both propose bindings from
the selected objects, assess whether a recipe can run on them, and phrase the
returned issues in their own user interface.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Union

import guidata.dataset as gds
from sigima.objects import ImageObj, SignalObj

from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeDiagnosticLevel,
    RecipeInputs,
    RecipeInputSlot,
    RecipeObjectType,
    RecipeValidationError,
)

__all__ = [
    "INPUT_CHECK_FAILED_CODE",
    "RecipeBindingProposal",
    "RecipeInputIssue",
    "RecipeInputIssueCode",
    "RecipeReadiness",
    "RecipeReadinessStatus",
    "assess_recipe_inputs",
    "check_recipe_inputs",
    "create_recipe_parameters",
    "find_input_issues",
    "is_compatible",
    "propose_bindings",
]

DataObject = Union[SignalObj, ImageObj]

#: Diagnostic code of an exception raised by a recipe input check
INPUT_CHECK_FAILED_CODE = "input-check-failed"

_MAX_TITLES = 3


class RecipeInputIssueCode(str, enum.Enum):
    """Generic reason why objects cannot be bound to a recipe input slot."""

    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    TOO_FEW = "too_few"
    TOO_MANY = "too_many"
    WRONG_TYPE = "wrong_type"
    MISSING_METADATA = "missing_metadata"
    DUPLICATE = "duplicate"


@dataclasses.dataclass(frozen=True)
class RecipeInputIssue:
    """Generic problem found on one recipe input slot.

    ``details`` holds what hosts need to phrase the problem: ``count`` and
    ``min_count`` (``TOO_FEW``), ``count`` (``TOO_MANY``), ``key``, ``count``
    and ``titles`` of the objects lacking it (``MISSING_METADATA``), ``count``
    and ``titles`` (``WRONG_TYPE``, ``DUPLICATE``).
    """

    code: RecipeInputIssueCode
    slot_id: str
    details: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the code and freeze details."""
        object.__setattr__(self, "code", RecipeInputIssueCode(self.code))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


class RecipeReadinessStatus(str, enum.Enum):
    """Whether a recipe can run on a set of candidate objects."""

    #: No candidate object has a type accepted by the recipe
    NO_INPUT = "no_input"
    #: Inputs are available but must be assigned to slots by the user
    NEEDS_ASSIGNMENT = "needs_assignment"
    #: Inputs violate the slot declarations or the recipe input checks
    NOT_READY = "not_ready"
    #: The recipe can run, but its input checks returned warnings
    WARNINGS = "warnings"
    #: The recipe can run
    READY = "ready"


@dataclasses.dataclass(frozen=True)
class RecipeBindingProposal:
    """Objects proposed for each recipe input slot."""

    bindings: Mapping[str, tuple[DataObject, ...]]
    ambiguous_slots: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RecipeReadiness:
    """Result of the assessment of recipe inputs."""

    status: RecipeReadinessStatus
    bindings: Mapping[str, tuple[DataObject, ...]]
    issues: tuple[RecipeInputIssue, ...] = ()
    diagnostics: tuple[RecipeDiagnostic, ...] = ()

    @property
    def can_run(self) -> bool:
        """Return True if the recipe can run with the assessed bindings."""
        return self.status in (
            RecipeReadinessStatus.READY,
            RecipeReadinessStatus.WARNINGS,
        )


def is_compatible(slot: RecipeInputSlot, obj: object) -> bool:
    """Return True if an object has the type accepted by an input slot."""
    expected = SignalObj if slot.object_type is RecipeObjectType.SIGNAL else ImageObj
    return isinstance(obj, expected)


def _unique(objects: Sequence[DataObject]) -> tuple[DataObject, ...]:
    """Return objects without repetition, in their original order."""
    seen: set[int] = set()
    unique: list[DataObject] = []
    for obj in objects:
        if id(obj) not in seen:
            seen.add(id(obj))
            unique.append(obj)
    return tuple(unique)


def _titles(objects: Sequence[DataObject]) -> tuple[str, ...]:
    """Return the titles of the first objects, to name them in messages."""
    return tuple(str(getattr(obj, "title", "") or "") for obj in objects[:_MAX_TITLES])


def _suggested_bindings(
    descriptor: RecipeDescriptor,
    candidates: tuple[DataObject, ...],
) -> dict[str, tuple[DataObject, ...]]:
    """Return the validated binding suggestions of a recipe."""
    if descriptor.suggest_bindings is None or not candidates:
        return {}
    suggestions = descriptor.suggest_bindings(candidates)
    if not isinstance(suggestions, Mapping):
        raise RecipeValidationError("Recipe binding suggestions must be a mapping")
    slot_ids = {slot.id for slot in descriptor.inputs}
    candidate_ids = {id(obj) for obj in candidates}
    validated: dict[str, tuple[DataObject, ...]] = {}
    for slot_id, objects in suggestions.items():
        if slot_id not in slot_ids:
            raise RecipeValidationError(
                f"Binding suggestions reference unknown slot {slot_id!r}"
            )
        if isinstance(objects, (str, bytes)) or not isinstance(objects, Sequence):
            raise RecipeValidationError(
                f"Binding suggestion {slot_id!r} must be a sequence"
            )
        values = tuple(objects)
        if any(id(obj) not in candidate_ids for obj in values):
            raise RecipeValidationError(
                f"Binding suggestion {slot_id!r} references an object outside "
                "the candidates"
            )
        validated[slot_id] = values
    return validated


def propose_bindings(
    descriptor: RecipeDescriptor,
    candidates: Sequence[DataObject],
) -> RecipeBindingProposal:
    """Propose objects for each input slot of a recipe.

    The recipe's binding suggestions come first. Otherwise, all compatible
    candidates go to the only slot accepting their type; slots sharing a type
    with others are left to the user (ambiguous slots).

    Args:
        descriptor: recipe descriptor
        candidates: candidate objects, e.g. the current selection

    Returns:
        Proposed bindings and the slots requiring a user decision
    """
    unique = tuple(
        obj
        for obj in _unique(candidates)
        if any(is_compatible(slot, obj) for slot in descriptor.inputs)
    )
    suggested = _suggested_bindings(descriptor, unique)
    bindings: dict[str, tuple[DataObject, ...]] = {}
    ambiguous: list[str] = []
    for slot in descriptor.inputs:
        compatible = tuple(obj for obj in unique if is_compatible(slot, obj))
        same_type_slots = sum(
            other.object_type is slot.object_type for other in descriptor.inputs
        )
        if slot.id in suggested:
            bindings[slot.id] = suggested[slot.id]
        elif same_type_slots == 1 and (
            slot.cardinality is RecipeCardinality.MANY or len(compatible) == 1
        ):
            bindings[slot.id] = compatible
        else:
            bindings[slot.id] = ()
            if compatible:
                ambiguous.append(slot.id)
    return RecipeBindingProposal(MappingProxyType(bindings), tuple(ambiguous))


def find_input_issues(
    descriptor: RecipeDescriptor,
    bindings: Mapping[str, Sequence[DataObject]],
    ambiguous_slots: Sequence[str] = (),
) -> tuple[RecipeInputIssue, ...]:
    """Check bindings against the declaration of the recipe input slots.

    Args:
        descriptor: recipe descriptor
        bindings: objects bound to each slot
        ambiguous_slots: unbound slots awaiting a user assignment

    Returns:
        Problems found, in slot declaration order
    """
    issues: list[RecipeInputIssue] = []
    bound_ids: set[int] = set()
    for slot in descriptor.inputs:
        objects = tuple(bindings.get(slot.id, ()))
        if not objects:
            if slot.id in ambiguous_slots:
                issues.append(RecipeInputIssue(RecipeInputIssueCode.AMBIGUOUS, slot.id))
            elif slot.required:
                issues.append(RecipeInputIssue(RecipeInputIssueCode.MISSING, slot.id))
            continue
        wrong = [obj for obj in objects if not is_compatible(slot, obj)]
        if wrong:
            issues.append(
                RecipeInputIssue(
                    RecipeInputIssueCode.WRONG_TYPE,
                    slot.id,
                    {"count": len(wrong), "titles": _titles(wrong)},
                )
            )
        duplicates: list[DataObject] = []
        slot_ids: set[int] = set()
        for obj in objects:
            if id(obj) in bound_ids or id(obj) in slot_ids:
                duplicates.append(obj)
            slot_ids.add(id(obj))
        bound_ids.update(slot_ids)
        if duplicates:
            issues.append(
                RecipeInputIssue(
                    RecipeInputIssueCode.DUPLICATE,
                    slot.id,
                    {"count": len(duplicates), "titles": _titles(duplicates)},
                )
            )
        if slot.cardinality is RecipeCardinality.ONE and len(objects) > 1:
            issues.append(
                RecipeInputIssue(
                    RecipeInputIssueCode.TOO_MANY, slot.id, {"count": len(objects)}
                )
            )
        elif len(objects) < slot.min_count:
            issues.append(
                RecipeInputIssue(
                    RecipeInputIssueCode.TOO_FEW,
                    slot.id,
                    {"count": len(objects), "min_count": slot.min_count},
                )
            )
        for requirement in slot.metadata:
            if not requirement.required:
                continue
            lacking = [
                obj
                for obj in objects
                if getattr(obj, "metadata", {}).get(requirement.key) is None
            ]
            if lacking:
                issues.append(
                    RecipeInputIssue(
                        RecipeInputIssueCode.MISSING_METADATA,
                        slot.id,
                        {
                            "key": requirement.key,
                            "count": len(lacking),
                            "titles": _titles(lacking),
                        },
                    )
                )
    return tuple(issues)


def check_recipe_inputs(
    descriptor: RecipeDescriptor,
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
) -> tuple[RecipeDiagnostic, ...]:
    """Run the input checks of a recipe on complete inputs.

    An exception raised by the checks (e.g. by a validation function reused
    from the recipe) is reported as an error diagnostic.

    Args:
        descriptor: recipe descriptor
        inputs: objects bound to each slot
        parameters: recipe parameters, or None for a parameterless recipe

    Returns:
        Diagnostics returned by the recipe
    """
    if descriptor.check_inputs is None:
        return ()
    try:
        diagnostics = tuple(descriptor.check_inputs(inputs, parameters))
    except Exception as exc:  # pylint: disable=broad-except
        message = str(exc).strip() or type(exc).__name__
        return (
            RecipeDiagnostic(
                RecipeDiagnosticLevel.ERROR, INPUT_CHECK_FAILED_CODE, message
            ),
        )
    if not all(isinstance(item, RecipeDiagnostic) for item in diagnostics):
        raise TypeError("Recipe input checks must return RecipeDiagnostic values")
    return diagnostics


def assess_recipe_inputs(
    descriptor: RecipeDescriptor,
    candidates: Sequence[DataObject] = (),
    parameters: gds.DataSet | None = None,
    bindings: Mapping[str, Sequence[DataObject]] | None = None,
) -> RecipeReadiness:
    """Assess whether a recipe can run on candidate objects or bindings.

    Args:
        descriptor: recipe descriptor
        candidates: candidate objects, used when ``bindings`` is None
        parameters: parameters passed to the recipe input checks
        bindings: explicit bindings (e.g. edited by the user)

    Returns:
        Readiness status, bindings, generic issues, and recipe diagnostics
    """
    if bindings is None:
        proposal = propose_bindings(descriptor, candidates)
        bound = dict(proposal.bindings)
        ambiguous = proposal.ambiguous_slots
        if (
            descriptor.inputs
            and not ambiguous
            and not any(bound.values())
            and not any(
                is_compatible(slot, obj)
                for slot in descriptor.inputs
                for obj in candidates
            )
        ):
            return RecipeReadiness(
                RecipeReadinessStatus.NO_INPUT,
                MappingProxyType(bound),
                find_input_issues(descriptor, bound),
            )
    else:
        bound = {slot_id: tuple(objects) for slot_id, objects in bindings.items()}
        ambiguous = ()
    frozen = MappingProxyType(bound)
    issues = find_input_issues(descriptor, bound, ambiguous)
    if issues:
        status = (
            RecipeReadinessStatus.NEEDS_ASSIGNMENT
            if all(issue.code is RecipeInputIssueCode.AMBIGUOUS for issue in issues)
            else RecipeReadinessStatus.NOT_READY
        )
        return RecipeReadiness(status, frozen, issues)
    inputs = {slot.id: tuple(bound.get(slot.id, ())) for slot in descriptor.inputs}
    diagnostics = check_recipe_inputs(descriptor, inputs, parameters)
    levels = {diagnostic.level for diagnostic in diagnostics}
    if RecipeDiagnosticLevel.ERROR in levels:
        status = RecipeReadinessStatus.NOT_READY
    elif RecipeDiagnosticLevel.WARNING in levels:
        status = RecipeReadinessStatus.WARNINGS
    else:
        status = RecipeReadinessStatus.READY
    return RecipeReadiness(status, frozen, (), diagnostics)


def create_recipe_parameters(
    descriptor: RecipeDescriptor,
    values: Mapping[str, object] | None = None,
) -> gds.DataSet | None:
    """Create default recipe parameters, updated with known values.

    Args:
        descriptor: recipe descriptor
        values: parameter values, e.g. suited to an example

    Returns:
        Parameters, or None for a parameterless recipe

    Raises:
        RecipeValidationError: a value does not match a recipe parameter
    """
    values = {} if values is None else dict(values)
    if descriptor.parameter_class is None:
        if values:
            raise RecipeValidationError(
                f"Recipe {descriptor.recipe_id!r} does not accept parameters"
            )
        return None
    parameters = descriptor.parameter_class()
    names = {item.get_name() for item in parameters.get_items()}
    for name, value in values.items():
        if name not in names:
            raise RecipeValidationError(
                f"Unknown parameter {name!r} for recipe {descriptor.recipe_id!r}"
            )
        setattr(parameters, name, value)
    return parameters
