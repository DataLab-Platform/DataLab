# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Processing call adapter
=======================

A :class:`ProcessingCall` is the call that produced a processed signal or image,
restored so that it can be shown, edited and applied again: the computing
feature and the processor that runs it, the parameters, and every input in
order (roles and repetitions kept). It is shared by the Processing tab and the
History panel; it is not a new contract model: the identity, the roles and the
parameter codec of an operation come from its Sigima contract.

The workspace provenance ledger is read first: the call that produced the latest
recorded state of the object is restored with the parameters it actually used,
decoded by the local contract (whether or not its verification is qualified).
The processing metadata stored in the object are only used when the ledger does
not record such a call (workspaces saved without provenance, functions without
contract such as most plugin functions): they keep the ordinary re-application
of old results, without any historical claim.

Re-applying a call is an ordinary processing with the current inputs and options
(see :meth:`datalab.gui.processor.base.BaseProcessor.recompute_processing`);
verifying a recorded activity is a separate operation.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from guidata.dataset import update_dataset
from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.gui.processor.base import (
    FeatureNotFoundError,
    ProcessingParameters,
    extract_processing_parameters,
)
from datalab.gui.processor.geometry_postprocess import unwrap_computation_function
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.processor.base import BaseProcessor, ComputingFeature

__all__ = [
    "CallInput",
    "ProcessingCall",
    "feature_for_function",
    "resolve_processing_call",
]

#: Pattern of the processing metadata of an output, by feature pattern (an output
#: of a 1-to-n command is produced by one concrete 1-to-1 call).
_PATTERNS = {
    "1_to_1": "1-to-1",
    "1_to_n": "1-to-1",
    "2_to_1": "2-to-1",
    "n_to_1": "n-to-1",
}
#: Input roles of the processing metadata, by pattern.
_METADATA_ROLES = {"1-to-1": ("source",), "2-to-1": ("source", "operand")}


@dataclasses.dataclass
class CallInput:
    """One input of a processing call.

    Attributes:
        role: Input role (``source``, ``operand``, ``sources``...).
        object_uuid: UUID of the input object at execution time.
        obj: Current object holding this UUID, or None if it was deleted.
        state_id: Recorded input state (ledger calls only).
    """

    role: str
    object_uuid: str
    obj: SignalObj | ImageObj | None
    state_id: str | None = None


@dataclasses.dataclass
class ProcessingCall:
    """Restored processing call of an output object.

    Attributes:
        pattern: ``"1-to-1"``, ``"2-to-1"`` or ``"n-to-1"``.
        func_name: Feature identifier, or the recorded operation identifier when
         the feature is not available.
        feature: Computing feature, or None if not available here.
        processor: Processor running the feature (the one of its inputs).
        param: Parameter argument (a copy), or None.
        inputs: Every input, in call order.
        plugin_origin: Plugin origin of the feature, if any.
        activity_id: Recorded activity (ledger calls only).
        origin: ``"ledger"`` or ``"metadata"``.
        unavailable_reason: Why the call cannot be applied again here, if so.
    """

    pattern: str
    func_name: str
    feature: ComputingFeature | None
    processor: BaseProcessor | None
    param: Any
    inputs: list[CallInput]
    plugin_origin: dict[str, Any] | None = None
    activity_id: str | None = None
    origin: str = "metadata"
    unavailable_reason: str | None = None

    @property
    def missing_inputs(self) -> list[CallInput]:
        """Inputs whose object was deleted (never replaced by another object)."""
        return [item for item in self.inputs if item.obj is None]

    @property
    def objects(self) -> list[SignalObj | ImageObj | None]:
        """Current input objects, in call order."""
        return [item.obj for item in self.inputs]

    @property
    def roles(self) -> list[str]:
        """Input roles, in call order."""
        return [item.role for item in self.inputs]

    @property
    def title(self) -> str:
        """Title of the processing (feature title or recorded name)."""
        if self.feature is not None and self.feature.title:
            return self.feature.title
        return self.func_name.rsplit(":", 1)[-1]

    @property
    def param_editable(self) -> bool:
        """True if the parameters can be edited as one parameter set."""
        return isinstance(self.param, gds.DataSet)

    @property
    def reason(self) -> str | None:
        """Why the call cannot be applied again, or None."""
        if self.unavailable_reason is not None:
            return self.unavailable_reason
        if self.missing_inputs:
            if len(self.inputs) == 1:
                return _("Source object no longer exists.")
            return _("An input object no longer exists.")
        if self.feature is None or self.processor is None:
            return _("This processing is not available in this version of DataLab.")
        return None

    @property
    def can_reapply(self) -> bool:
        """True if the call can be applied again to the current inputs."""
        return self.reason is None

    def processing_parameters(self, param: Any) -> ProcessingParameters:
        """Return the processing metadata of an output of this call.

        Args:
            param: Parameter argument actually used.
        """
        uuids = [item.object_uuid for item in self.inputs]
        single = self.pattern == "1-to-1"
        return ProcessingParameters(
            func_name=self.feature.feature_id if self.feature else self.func_name,
            pattern=self.pattern,
            param=param,
            source_uuid=uuids[0] if single else None,
            source_uuids=None if single else uuids,
            plugin_origin=self.plugin_origin,
        )


def _processor_for(mainwindow: DLMainWindow, kind: str | None) -> BaseProcessor | None:
    if kind == "signal":
        return mainwindow.signalpanel.processor
    if kind == "image":
        return mainwindow.imagepanel.processor
    return None


def _kind_of(obj: Any) -> str | None:
    if isinstance(obj, SignalObj):
        return "signal"
    if isinstance(obj, ImageObj):
        return "image"
    return None


def feature_for_function(
    processor: BaseProcessor, function: Any
) -> ComputingFeature | None:
    """Return the feature of *processor* running the Sigima *function*.

    Wrapped feature functions (see
    :func:`~datalab.gui.processor.geometry_postprocess.unwrap_computation_function`)
    run the function they wrap. When several features run it, built-in
    features are preferred.
    """
    matches = [
        feature
        for feature in processor.computing_registry.values()
        if feature.function is not None
        and unwrap_computation_function(feature.function) is function
    ]
    built_ins = [feature for feature in matches if feature.plugin_origin is None]
    return (built_ins or matches or [None])[0]


def _pattern_of_roles(roles: list[str]) -> str:
    if roles == ["source"]:
        return "1-to-1"
    if roles == ["source", "operand"]:
        return "2-to-1"
    return "n-to-1"


def _typed_param(feature: ComputingFeature, param: Any) -> Any:
    """Return *param* as an instance of the parameter class of *feature*."""
    paramclass = feature.paramclass
    if (
        paramclass is None
        or not isinstance(param, gds.DataSet)
        or isinstance(param, paramclass)
    ):
        return param
    typed = paramclass()
    update_dataset(typed, param)
    return typed


def _from_ledger(
    mainwindow: DLMainWindow, obj: SignalObj | ImageObj, produced: dict[str, Any]
) -> ProcessingCall:
    """Return the call recorded in the ledger (see ``Recorder.producing_call``)."""
    states = mainwindow.provenance.ledger.states
    inputs = [
        CallInput(
            item["role"],
            item["object_uuid"],
            mainwindow.find_object_by_uuid(item["object_uuid"])
            if item["present"]
            else None,
            item["state_id"],
        )
        for item in produced["inputs"]
    ]
    kind = states[inputs[0].state_id]["kind"] if inputs else None
    processor = _processor_for(mainwindow, kind)
    operation = produced["operation"]
    call = ProcessingCall(
        pattern=_pattern_of_roles([item.role for item in inputs]),
        func_name=operation["id"],
        feature=None,
        processor=processor,
        param=None,
        inputs=inputs,
        activity_id=produced["activity_id"],
        origin="ledger",
    )
    contract = produced["contract"]
    if contract is None:
        call.unavailable_reason = _(
            "Operation %s is not available in this version: %s"
        ) % (operation["id"], produced["reason"])
        return call
    stored = extract_processing_parameters(obj)
    feature = None
    if processor is not None:
        if stored is not None and stored.func_name:
            # The feature named with the result is used when it runs this
            # operation; when it is no longer available (e.g. a removed
            # plugin), no other feature is substituted.
            try:
                named = processor.get_feature(
                    stored.func_name, plugin_origin=stored.plugin_origin
                )
            except FeatureNotFoundError as exc:
                call.unavailable_reason = str(exc)
                return call
            except ValueError:
                named = None
            if (
                named is not None
                and named.function is not None
                and unwrap_computation_function(named.function) is contract.function
            ):
                feature = named
        if feature is None:
            feature = feature_for_function(processor, contract.function)
    if feature is None:
        call.unavailable_reason = (
            _("Operation %s has no processing feature in this version of DataLab.")
            % operation["id"]
        )
        return call
    call.feature = feature
    call.func_name = feature.feature_id
    # The plugin origin stored with the result is kept when it describes the
    # same feature (as the processing metadata always did).
    same = stored is not None and stored.func_name == feature.feature_id
    call.plugin_origin = (
        stored.plugin_origin
        if same and stored.plugin_origin is not None
        else feature.plugin_origin
    )
    call.pattern = _PATTERNS.get(feature.pattern, call.pattern)
    expected = {"1-to-1": 1, "2-to-1": 2}.get(call.pattern)
    if (expected is not None and len(inputs) != expected) or not inputs:
        call.unavailable_reason = _(
            "The recorded inputs do not match this processing feature."
        )
        return call
    values = produced["parameters"]
    if values is None:
        if contract.parameters:
            call.unavailable_reason = _(
                "The parameters of this processing were not recorded."
            )
        return call
    try:
        arguments = contract.decode_parameters(values)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        call.unavailable_reason = _("The recorded parameters cannot be used: %s") % exc
        return call
    if len(arguments) > 1:
        call.unavailable_reason = _(
            "This processing takes several parameter sets: it cannot be applied "
            "again from here."
        )
        return call
    call.param = _typed_param(feature, arguments[0]) if arguments else None
    return call


def _from_metadata(
    mainwindow: DLMainWindow, obj: SignalObj | ImageObj
) -> ProcessingCall | None:
    """Return the call stored in the processing metadata of *obj*, if any."""
    stored = extract_processing_parameters(obj)
    if stored is None or stored.pattern not in ("1-to-1", "2-to-1", "n-to-1"):
        return None
    if stored.pattern == "1-to-1":
        uuids = [stored.source_uuid] if stored.source_uuid else []
    else:
        uuids = [uid for uid in stored.source_uuids or [] if uid]
    roles = list(_METADATA_ROLES.get(stored.pattern, ()))
    if len(roles) != len(uuids):
        roles = ["sources"] * len(uuids)
    inputs = [
        CallInput(role, uid, mainwindow.find_object_by_uuid(uid))
        for role, uid in zip(roles, uuids)
    ]
    present = [item.obj for item in inputs if item.obj is not None]
    processor = _processor_for(mainwindow, _kind_of(present[0])) if present else None
    call = ProcessingCall(
        pattern=stored.pattern,
        func_name=stored.func_name,
        feature=None,
        processor=processor,
        param=copy.deepcopy(stored.param),
        inputs=inputs,
        plugin_origin=stored.plugin_origin,
    )
    if not uuids:
        call.unavailable_reason = _(
            "Processing metadata is incomplete (missing source UUID)."
        )
        return call
    if processor is not None:
        param = call.param
        try:
            call.feature = processor.get_feature(
                stored.func_name,
                plugin_origin=stored.plugin_origin,
                paramclass_name=type(param).__name__
                if isinstance(param, gds.DataSet)
                else None,
            )
        except ValueError as exc:
            call.unavailable_reason = str(exc)
    return call


def resolve_processing_call(
    mainwindow: DLMainWindow, obj: SignalObj | ImageObj
) -> ProcessingCall | None:
    """Return the processing call that produced *obj*, or None.

    The ledger is read first (see the module documentation): the call that
    produced the latest recorded state of *obj*, unless the file said that the
    saved content does not match it; otherwise the processing metadata of *obj*.

    Args:
        mainwindow: DataLab main window.
        obj: Processed signal or image.

    Returns:
        The restored call, or None when *obj* records no processing.
    """
    produced = mainwindow.provenance.producing_call(get_uuid(obj))
    if produced is not None and produced["output_status"] is None:
        return _from_ledger(mainwindow, obj, produced)
    return _from_metadata(mainwindow, obj)
