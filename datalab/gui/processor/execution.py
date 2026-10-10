# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Execution service of 1-to-1 processing (see parent package
   :mod:`datalab.gui.processor`)

One service per processor runs every 1-to-1 execution started from a live
selection (commit mode) and every verification candidate (candidate mode). It
wraps the existing infrastructure: process isolation, preview reuse, output
handling, processing metadata and panel insertion. Signal 1-to-1 executions are
captured in the workspace provenance ledger. Selection, groups, dialogs and
progress stay in the processor. The service schedules nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import guidata.dataset as gds
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.objectmodel import get_short_id, get_uuid, patch_title_with_ids
from datalab.utils.qthelpers import create_progress_bar

if TYPE_CHECKING:
    from datalab.gui.processor.base import BaseProcessor
    from datalab.gui.processor.catcher import CompOut

__all__ = ["CANCELLED", "SKIPPED", "ExecutionService"]

#: The execution was cancelled: the caller stops the batch.
CANCELLED = "cancelled"
#: The execution produced no object (error or warning): the caller continues.
SKIPPED = "skipped"


class ExecutionService:
    """1-to-1 execution service owned by a processor.

    Args:
        processor: Owning processor.
        exec_func: The processor's private executor (process isolation aware).
    """

    def __init__(
        self,
        processor: BaseProcessor,
        exec_func: Callable[[Callable, tuple, QW.QProgressDialog], CompOut | None],
    ) -> None:
        self.processor = processor
        self._exec_func = exec_func
        #: Number of computations actually run (preview reuse excluded).
        self.computations = 0

    @property
    def provenance(self):
        """Workspace provenance service."""
        return self.processor.mainwindow.provenance

    def run_1_to_1(
        self,
        obj: SignalObj | ImageObj,
        func: Callable,
        param: gds.DataSet | None,
        progress: QW.QProgressDialog,
        label: str,
        command_id: str | None,
        place: Callable[[SignalObj | ImageObj], str | None],
        preview: CompOut | None = None,
        feature_id: str | None = None,
    ) -> SignalObj | ImageObj | str:
        """Run one 1-to-1 execution in commit mode.

        Args:
            obj: Source object.
            func: Computation function.
            param: Effective parameters, or None.
            progress: Progress dialog.
            label: Progress label of this execution.
            command_id: Identifier shared by the executions of one command.
            place: Returns the group of the result (called before insertion).
            preview: Accepted preview result to reuse instead of computing.
            feature_id: Stable feature identifier stored in processing metadata.

        Returns:
            The inserted result, :data:`CANCELLED` or :data:`SKIPPED`.
        """
        # pylint: disable=import-outside-toplevel
        from datalab.gui.processor.base import (
            ProcessingParameters,
            insert_processing_parameters,
        )

        processor = self.processor
        pending = self.provenance.begin(func, param, obj, command_id)
        if preview is not None:
            result = preview
        else:
            args = (obj,) if param is None else (obj, param)
            self.computations += 1
            result = self._exec_func(func, args, progress)
        if result is None:
            return CANCELLED
        new_obj = processor.handle_output(result, _("Computing: %s") % label, progress)
        if new_obj is None:
            return SKIPPED
        assert isinstance(new_obj, (SignalObj, ImageObj))
        patch_title_with_ids(new_obj, [obj], get_short_id)
        processor._handle_keep_results(new_obj)  # pylint: disable=protected-access
        pp = ProcessingParameters(
            func_name=processor.get_feature_id(func, feature_id),
            pattern="1-to-1",
            param=param,
            source_uuid=get_uuid(obj),
            plugin_origin=processor._get_plugin_origin_for(func, feature_id),  # pylint: disable=protected-access
        )
        insert_processing_parameters(new_obj, pp)
        processor.panel.objprop.mark_as_freshly_processed(new_obj)
        group_id = place(new_obj)
        processor._add_object_to_appropriate_panel(  # pylint: disable=protected-access
            new_obj, group_id=group_id, use_group_for_non_native=True
        )
        self.provenance.complete(pending, new_obj)
        return new_obj

    def execute_candidate(
        self, func: Callable, inputs: list[SignalObj | ImageObj], param: Any
    ) -> SignalObj | ImageObj | None:
        """Compute a result in candidate mode: nothing is inserted or recorded.

        Args:
            func: Computation function.
            inputs: Function inputs, in role order.
            param: Parameters, or None.

        Returns:
            The candidate, or None on error or cancellation.
        """
        with create_progress_bar(
            self.processor.panel, _("Recomputing..."), max_=1
        ) as progress:
            args = tuple(inputs) if param is None else (*inputs, param)
            self.computations += 1
            compout = self._exec_func(func, args, progress)
            if compout is None:
                return None
            result = self.processor.handle_output(compout, _("Recomputing"), progress)
        return result if isinstance(result, (SignalObj, ImageObj)) else None
