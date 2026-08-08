# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Headless contracts for plugin-provided scientific recipes."""

from __future__ import annotations

import dataclasses
import enum
import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Union

import guidata.dataset as gds
from packaging.version import InvalidVersion, Version
from sigima.objects import GeometryResult, ImageObj, SignalObj, TableResult

__all__ = [
    "RecipeCancellationCallback",
    "RecipeCancellationError",
    "RecipeCardinality",
    "RecipeDescriptor",
    "RecipeDiagnostic",
    "RecipeDiagnosticLevel",
    "RecipeExecutionContext",
    "RecipeInputSlot",
    "RecipeInputs",
    "RecipeObjectOutput",
    "RecipeObjectType",
    "RecipeOutcome",
    "RecipeProgressCallback",
    "RecipeResultOutput",
    "RecipeRun",
]


_LOCAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


def _validate_local_id(value: str, field_name: str) -> None:
    """Validate a stable identifier local to a recipe."""
    if not isinstance(value, str) or not _LOCAL_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain lowercase letters, digits, '.', '_' or '-'"
        )


def _find_duplicate(values: Sequence[str]) -> str | None:
    """Return the first duplicate string in a sequence, if any."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


class RecipeObjectType(str, enum.Enum):
    """Scientific object type accepted by a recipe input slot."""

    SIGNAL = "signal"
    IMAGE = "image"


class RecipeCardinality(str, enum.Enum):
    """Number of objects accepted by a recipe input slot."""

    ONE = "one"
    MANY = "many"


class RecipeDiagnosticLevel(str, enum.Enum):
    """Severity of a structured recipe diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RecipeCancellationError(RuntimeError):
    """Raised when recipe execution observes a cancellation request."""


RecipeProgressCallback = Callable[[float, Optional[str]], None]
RecipeCancellationCallback = Callable[[], bool]


@dataclasses.dataclass(frozen=True)
class RecipeExecutionContext:
    """Technology-neutral progress and cancellation callbacks for a recipe."""

    progress_callback: RecipeProgressCallback | None = None
    cancellation_callback: RecipeCancellationCallback | None = None

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return bool(
            self.cancellation_callback is not None and self.cancellation_callback()
        )

    def raise_if_cancelled(self) -> None:
        """Raise :class:`RecipeCancellationError` when cancellation is requested."""
        if self.is_cancelled:
            raise RecipeCancellationError("Recipe execution was cancelled")

    def report_progress(self, progress: float, message: str | None = None) -> None:
        """Report normalized progress between zero and one."""
        if isinstance(progress, bool):
            raise ValueError("Recipe progress must be between 0.0 and 1.0")
        try:
            normalized_progress = float(progress)
        except (TypeError, ValueError) as exc:
            raise ValueError("Recipe progress must be between 0.0 and 1.0") from exc
        if (
            not math.isfinite(normalized_progress)
            or not 0.0 <= normalized_progress <= 1.0
        ):
            raise ValueError("Recipe progress must be between 0.0 and 1.0")
        if message is not None and not isinstance(message, str):
            raise TypeError("Recipe progress message must be a string or None")
        if self.progress_callback is not None:
            self.progress_callback(normalized_progress, message)


@dataclasses.dataclass(frozen=True)
class RecipeInputSlot:
    """Typed input slot declared by a recipe."""

    id: str
    object_type: RecipeObjectType
    cardinality: RecipeCardinality
    required: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize the slot declaration."""
        _validate_local_id(self.id, "Recipe input slot ID")
        try:
            object.__setattr__(self, "object_type", RecipeObjectType(self.object_type))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported recipe object type: {self.object_type!r}"
            ) from exc
        try:
            object.__setattr__(self, "cardinality", RecipeCardinality(self.cardinality))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported recipe input cardinality: {self.cardinality!r}"
            ) from exc
        if not isinstance(self.required, bool):
            raise TypeError("Recipe input slot required flag must be a bool")


@dataclasses.dataclass(frozen=True)
class RecipeDiagnostic:
    """Structured, machine-identifiable diagnostic emitted by a recipe."""

    level: RecipeDiagnosticLevel
    code: str
    message: str
    details: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze diagnostic data."""
        try:
            object.__setattr__(self, "level", RecipeDiagnosticLevel(self.level))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported recipe diagnostic level: {self.level!r}"
            ) from exc
        _validate_local_id(self.code, "Recipe diagnostic code")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Recipe diagnostic message must be a non-empty string")
        if not isinstance(self.details, Mapping) or not all(
            isinstance(key, str) for key in self.details
        ):
            raise TypeError("Recipe diagnostic details must map string keys to values")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclasses.dataclass(frozen=True)
class RecipeObjectOutput:
    """Named signal or image produced by a recipe."""

    id: str
    value: SignalObj | ImageObj

    def __post_init__(self) -> None:
        """Validate the output identifier and scientific object type."""
        _validate_local_id(self.id, "Recipe object output ID")
        if not isinstance(self.value, (SignalObj, ImageObj)):
            raise TypeError("Recipe object output must contain a SignalObj or ImageObj")


@dataclasses.dataclass(frozen=True)
class RecipeResultOutput:
    """Named scalar result attached to a named object output."""

    id: str
    value: TableResult | GeometryResult
    anchor_id: str

    def __post_init__(self) -> None:
        """Validate the result and its local anchor identifier."""
        _validate_local_id(self.id, "Recipe result output ID")
        _validate_local_id(self.anchor_id, "Recipe result anchor ID")
        if not isinstance(self.value, (TableResult, GeometryResult)):
            raise TypeError(
                "Recipe result output must contain a TableResult or GeometryResult"
            )


@dataclasses.dataclass(frozen=True)
class RecipeOutcome:
    """Objects, anchored scalar results, and diagnostics produced by a recipe."""

    objects: Sequence[RecipeObjectOutput] = ()
    results: Sequence[RecipeResultOutput] = ()
    diagnostics: Sequence[RecipeDiagnostic] = ()

    def __post_init__(self) -> None:
        """Freeze output collections and validate IDs and anchor references."""
        objects = tuple(self.objects)
        results = tuple(self.results)
        diagnostics = tuple(self.diagnostics)
        if not all(isinstance(output, RecipeObjectOutput) for output in objects):
            raise TypeError("Recipe outcome objects must be RecipeObjectOutput values")
        if not all(isinstance(output, RecipeResultOutput) for output in results):
            raise TypeError("Recipe outcome results must be RecipeResultOutput values")
        if not all(
            isinstance(diagnostic, RecipeDiagnostic) for diagnostic in diagnostics
        ):
            raise TypeError(
                "Recipe outcome diagnostics must be RecipeDiagnostic values"
            )

        duplicate_object_id = _find_duplicate([output.id for output in objects])
        if duplicate_object_id is not None:
            raise ValueError(
                f"Duplicate recipe object output ID: {duplicate_object_id!r}"
            )
        duplicate_result_id = _find_duplicate([output.id for output in results])
        if duplicate_result_id is not None:
            raise ValueError(
                f"Duplicate recipe result output ID: {duplicate_result_id!r}"
            )
        object_ids = {output.id for output in objects}
        for result in results:
            if result.anchor_id not in object_ids:
                raise ValueError(
                    f"Recipe result {result.id!r} references unknown object output "
                    f"{result.anchor_id!r}"
                )

        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "diagnostics", diagnostics)


RecipeInputs = Mapping[str, tuple[Union[SignalObj, ImageObj], ...]]
RecipeRun = Callable[
    [RecipeInputs, Optional[gds.DataSet], RecipeExecutionContext], RecipeOutcome
]


@dataclasses.dataclass(frozen=True)
class RecipeDescriptor:
    """Versioned, typed declaration of a plugin-provided headless recipe.

    A missing ``parameter_class`` declares a parameterless recipe whose run
    callable receives ``None`` as its second argument. Otherwise, that argument
    must be an instance of the declared ``DataSet`` subclass.
    """

    recipe_id: str
    title: str
    version: str
    run: RecipeRun
    description: str = ""
    inputs: Sequence[RecipeInputSlot] = ()
    parameter_class: type[gds.DataSet] | None = None

    def __post_init__(self) -> None:
        """Validate identity, metadata, input slots, parameters, and callable."""
        if not isinstance(self.recipe_id, str) or self.recipe_id.count(":") != 1:
            raise ValueError("Recipe ID must be namespaced as '<plugin-id>:<local-id>'")
        plugin_id, local_id = self.recipe_id.split(":", maxsplit=1)
        if (
            not plugin_id
            or plugin_id.strip() != plugin_id
            or any(char.isspace() for char in plugin_id)
        ):
            raise ValueError("Recipe ID plugin namespace must be non-empty")
        _validate_local_id(local_id, "Recipe local ID")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Recipe title must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("Recipe description must be a string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Recipe version must be a non-empty string")
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid recipe version: {self.version!r}") from exc
        if not callable(self.run):
            raise TypeError("Recipe run must be callable")

        inputs = tuple(self.inputs)
        if not all(isinstance(slot, RecipeInputSlot) for slot in inputs):
            raise TypeError("Recipe inputs must be RecipeInputSlot values")
        duplicate_slot_id = _find_duplicate([slot.id for slot in inputs])
        if duplicate_slot_id is not None:
            raise ValueError(f"Duplicate recipe input slot ID: {duplicate_slot_id!r}")
        if self.parameter_class is not None and (
            not isinstance(self.parameter_class, type)
            or not issubclass(self.parameter_class, gds.DataSet)
        ):
            raise TypeError("Recipe parameter class must be a DataSet subclass")
        object.__setattr__(self, "inputs", inputs)

    @property
    def plugin_id(self) -> str:
        """Return the owning plugin ID encoded in the recipe ID."""
        return self.recipe_id.split(":", maxsplit=1)[0]

    @property
    def local_id(self) -> str:
        """Return the recipe-local ID encoded in the recipe ID."""
        return self.recipe_id.split(":", maxsplit=1)[1]
