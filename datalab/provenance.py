# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Workspace provenance
====================

The :class:`ProvenanceService` records, in a workspace-level ledger, every signal
1-to-1 processing executed by DataLab, whatever the History "Record" setting. The
ledger model, fingerprints, replay preparation and reports come from
DataLab-Capsule; operation contracts come from Sigima.

Capture never breaks processing: if recording fails, the result is kept, no
activity is recorded and the failure is logged and counted.

.. autoclass:: ProvenanceService
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import uuid
from collections.abc import Callable, Generator
from importlib import metadata
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from datalab_capsule.calls import make_call
from datalab_capsule.compare import build_report, compare_environments, compare_exact
from datalab_capsule.environment import collect_environment, environment_id
from datalab_capsule.integrity import signal_state_facts
from datalab_capsule.ledger import Ledger, utc_timestamp
from datalab_capsule.replay import IneligibleError, Plan, prepare_activity
from sigima.objects import SignalObj
from sigima.proc.contracts import (
    IncompatibleContractError,
    InvalidParametersError,
    OperationContract,
    ParameterEncodingError,
    UnknownOperationError,
    contract_for_function,
    get_operation_contract,
    parameters_from_values,
    parameters_to_values,
)

import datalab
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab_capsule.replay import Refusal

__all__ = ["PendingActivity", "ProvenanceService"]

_logger = logging.getLogger(__name__)

EDITION = "desktop"


@dataclasses.dataclass
class PendingActivity:
    """An execution whose input state was recorded, awaiting its output.

    Attributes:
        call: Operation call bound to the input state.
        implementation: Informative implementation descriptor.
        limits: Reasons why the activity is not replayable.
        command_id: Identifier shared by the executions of one user command.
        origin: Activity origin.
        started_at: Start time.
    """

    call: dict[str, Any]
    implementation: dict[str, Any]
    limits: list[str]
    command_id: str | None
    origin: str
    started_at: str


def implementation_of(func: Callable) -> dict[str, Any]:
    """Return the informative implementation descriptor of *func*.

    It is never resolved into a contract nor imported back.
    """
    module = getattr(func, "__module__", None) or ""
    package = module.split(".", 1)[0] or None
    try:
        version = metadata.version(package) if package else None
    except metadata.PackageNotFoundError:
        version = None
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return {
        "package": package,
        "version": version,
        "python_name": f"{module}.{qualname}" if module else qualname,
    }


def signal_rows(obj: SignalObj) -> dict[str, Any]:
    """Return the rows compared by the ``exact`` rule."""
    rows = {"x": obj.x, "y": obj.y}
    if obj.dx is not None:
        rows["dx"] = obj.dx
    if obj.dy is not None:
        rows["dy"] = obj.dy
    return rows


class _Deferred:
    """Pending activities held until temporary outputs get their final UUIDs."""

    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.items: list[tuple[PendingActivity, str]] = []


class ProvenanceService:
    """Workspace provenance ledger of DataLab Desktop.

    Args:
        find_object: Object lookup by UUID across panels (injected, so that this
         module never depends on panels or History).
    """

    def __init__(self, find_object: Callable[[str], Any]) -> None:
        self._find_object = find_object
        self._environment: dict[str, Any] | None = None
        self._deferred: _Deferred | None = None
        self.ledger = Ledger()
        self.state_status: dict[str, str] = {}
        self.notices: list[str] = []
        self.capture_failures = 0

    # -- Workspace lifecycle ------------------------------------------------

    def reset(self) -> None:
        """Start an empty ledger for a new workspace."""
        self.ledger = Ledger()
        self.state_status = {}
        self.notices = []
        self._deferred = None

    @property
    def environment(self) -> dict[str, Any]:
        """Environment record of this DataLab session."""
        if self._environment is None:
            self._environment = collect_environment(EDITION, datalab.__version__)
        return self._environment

    # -- Capture ------------------------------------------------------------

    def _capture_failed(self, exc: Exception) -> None:
        self.capture_failures += 1
        _logger.warning("Provenance capture failed: %s", exc, exc_info=True)

    def observe(self, obj: SignalObj) -> str:
        """Return the state of *obj*, reusing its latest state if unchanged."""
        return self.ledger.observe(get_uuid(obj), signal_state_facts(obj))

    def _build_call(
        self, func: Callable, param: gds.DataSet | None, state_id: str, obj: SignalObj
    ) -> tuple[dict[str, Any], list[str]]:
        """Return the operation call and its limits for one execution."""
        limits: list[str] = []
        contract = contract_for_function(func)
        if (
            contract is not None
            and contract.qualified
            and contract.check_preconditions([obj]) is None
        ):
            values = parameters_to_values(param)
            return (
                make_call(
                    contract.operation_id,
                    contract.contract_version,
                    values,
                    [("source", state_id)],
                ),
                limits,
            )
        try:
            values = parameters_to_values(param) if param is not None else {}
        except ParameterEncodingError:
            values = None
            limits.append("parameters_not_encoded")
        return make_call(None, None, values, [("source", state_id)]), limits

    def begin(
        self,
        func: Callable,
        param: gds.DataSet | None,
        source: Any,
        command_id: str | None = None,
        origin: str = "ordinary",
    ) -> PendingActivity | None:
        """Record the input state of a signal 1-to-1 execution, before it runs.

        Returns:
            A pending activity, or None when the execution is not captured.
        """
        if not isinstance(source, SignalObj):
            return None
        try:
            state_id = self.observe(source)
            call, limits = self._build_call(func, param, state_id, source)
            return PendingActivity(
                call=call,
                implementation=implementation_of(func),
                limits=limits,
                command_id=command_id,
                origin=origin,
                started_at=utc_timestamp(),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._capture_failed(exc)
            return None

    def _record(self, pending: PendingActivity, output: SignalObj, origin: str) -> Any:
        return self.ledger.record_activity(
            call=pending.call,
            outputs=[("result", get_uuid(output), signal_state_facts(output))],
            environment=self.environment,
            edition=EDITION,
            origin=origin,
            implementation=pending.implementation,
            command_id=pending.command_id,
            limits=pending.limits,
            started_at=pending.started_at,
        )

    def complete(
        self, pending: PendingActivity | None, output: Any
    ) -> dict[str, Any] | None:
        """Record a completed execution once its output was inserted.

        Outputs that are not signals, or were not inserted in the workspace (e.g.
        History output suppression), are not recorded.

        Returns:
            The recorded activity, or None.
        """
        if pending is None or not isinstance(output, SignalObj):
            return None
        try:
            output_uuid = get_uuid(output)
            if self._find_object(output_uuid) is not output:
                return None
            if self._deferred is not None:
                self._deferred.items.append((pending, output_uuid))
                return None
            return self._record(pending, output, pending.origin)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._capture_failed(exc)
            return None

    @contextlib.contextmanager
    def deferred(self, origin: str) -> Generator[_Deferred, None, None]:
        """Hold completed activities until :meth:`finalize_deferred` remaps them.

        Used by History replay, whose temporary outputs are committed onto the
        recorded objects after the computation. Pending activities left when the
        context exits are dropped.
        """
        previous = self._deferred
        self._deferred = _Deferred(origin)
        try:
            yield self._deferred
        finally:
            self._deferred = previous

    def finalize_deferred(self, deferred: _Deferred, remap: dict[str, str]) -> int:
        """Record held activities with their final output UUIDs.

        Args:
            deferred: Context returned by :meth:`deferred`.
            remap: Temporary output UUID -> final object UUID. Activities whose
             temporary output is not remapped are dropped.

        Returns:
            Number of recorded activities.
        """
        recorded = 0
        items, deferred.items = deferred.items, []
        for pending, temp_uuid in items:
            final_uuid = remap.get(temp_uuid)
            output = None if final_uuid is None else self._find_object(final_uuid)
            if not isinstance(output, SignalObj):
                continue
            try:
                self._record(pending, output, deferred.origin)
                recorded += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._capture_failed(exc)
        return recorded

    # -- Verification -------------------------------------------------------

    @staticmethod
    def _resolve_contract(operation_id: str, version: int) -> OperationContract:
        try:
            contract = get_operation_contract(operation_id, version)
        except UnknownOperationError as exc:
            raise IneligibleError("unsupported_operation", str(exc)) from exc
        except IncompatibleContractError as exc:
            raise IneligibleError("unsupported_contract", str(exc)) from exc
        if not contract.qualified:
            raise IneligibleError("unsupported_operation", "Contract not qualified")
        return contract

    @staticmethod
    def _decode(contract: OperationContract, values: Any) -> gds.DataSet | None:
        try:
            return parameters_from_values(contract, values)
        except InvalidParametersError as exc:
            raise IneligibleError("invalid_parameters", str(exc)) from exc

    def prepare(self, activity_id: str) -> Plan | Refusal:
        """Prepare a recorded activity for replay (no GUI selection needed)."""
        return prepare_activity(
            self.ledger,
            activity_id,
            resolve_contract=self._resolve_contract,
            find_object=self._find_object,
            observe_object=signal_state_facts,
            check_preconditions=lambda contract, objs: contract.check_preconditions(
                objs
            ),
            decode_parameters=self._decode,
            state_status=self.state_status,
        )

    def _reference(self, activity: dict[str, Any]) -> tuple[dict | None, Any]:
        """Return the reference status and object of an activity's result."""
        output = next(
            (
                o
                for o in activity["outputs"]
                if "state_id" in o and o["role"] == "result"
            ),
            None,
        )
        if output is None:
            return None, None
        state = self.ledger.states[output["state_id"]]
        obj = self._find_object(state["object_uuid"])
        status = "available"
        if obj is None or self.state_status.get(state["state_id"]) == "unavailable":
            status, obj = "missing", None
        elif self.state_status.get(state["state_id"]) == "altered" or (
            signal_state_facts(obj)["fingerprint"] != state["fingerprint"]
        ):
            status = "altered"
        return {"state_id": state["state_id"], "status": status}, obj

    def verify(
        self,
        activity_id: str,
        execute_candidate: Callable[[Callable, Any, Any], SignalObj | None],
    ) -> dict[str, Any]:
        """Recompute a recorded activity as a separate candidate and compare it.

        The reference data and the original activity never change; the
        verification run only appears in the returned report.

        Args:
            activity_id: Activity to verify.
            execute_candidate: ``(function, source, parameters) -> result``; runs
             the computation without inserting the result.

        Returns:
            Verification report.
        """
        activity = self.ledger.activity(activity_id)
        env_id = activity["environment_id"]
        environment = compare_environments(
            env_id,
            self.ledger.environments.get(env_id),
            environment_id(self.environment),
            self.environment,
        )
        reference, ref_obj = self._reference(activity)
        prepared = self.prepare(activity_id)
        if not isinstance(prepared, Plan):
            return build_report(
                activity_id=activity_id,
                restoration=prepared.restoration,
                eligibility=prepared.eligibility,
                inputs=list(prepared.input_statuses),
                reference=reference,
                environment=environment,
                reason=prepared.reason,
            )
        inputs = [
            {"role": role, "state_id": state_id, "status": "available"}
            for role, state_id, _obj in prepared.inputs
        ]
        source = prepared.inputs[0][2]
        candidate = execute_candidate(
            prepared.contract.function, source, prepared.parameters
        )
        if candidate is None:
            return build_report(
                activity_id=activity_id,
                restoration="replayable",
                eligibility="ready",
                inputs=inputs,
                reference=reference,
                environment=environment,
                reason="The candidate computation failed or was cancelled",
            )
        comparison = None
        if ref_obj is not None:
            comparison = compare_exact(signal_rows(candidate), signal_rows(ref_obj))
            if (candidate.xunit, candidate.yunit) != (ref_obj.xunit, ref_obj.yunit):
                comparison["verdict"] = "different"
        return build_report(
            activity_id=activity_id,
            restoration="replayable",
            eligibility="ready",
            inputs=inputs,
            reference=reference,
            environment=environment,
            comparison=comparison,
            candidate_state_ids=[("result", str(uuid.uuid4()))],
        )
