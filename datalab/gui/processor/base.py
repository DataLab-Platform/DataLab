# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Base processor object (see parent package :mod:`datalab.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import multiprocessing
import time
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Optional

import guidata.dataset as gds
import numpy as np
from guidata.dataset import update_dataset
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.config import options as sigima_options
from sigima.enums import Interpolation1DMethod
from sigima.objects import (
    GeometryResult,
    ImageObj,
    SignalObj,
    TableResult,
    TypeROI,
    TypeROIParam,
    concat_geometries,
)
from sigima.proc.decorator import is_computation_function
from sigima.tools.signal.interpolation import interpolate

from datalab import env
from datalab.adapters_metadata import (
    GeometryAdapter,
    ResultData,
    TableAdapter,
    show_resultdata,
)
from datalab.config import Conf, _
from datalab.gui.processor.catcher import CompOut, wng_err_func
from datalab.objectmodel import get_short_id, get_uuid, patch_title_with_ids
from datalab.utils.qthelpers import create_progress_bar, qt_try_except
from datalab.widgets.warningerror import show_warning_error

if TYPE_CHECKING:
    from multiprocessing.pool import AsyncResult

    from plotpy.plot import PlotWidget

    from datalab.gui.panel.image import ImagePanel
    from datalab.gui.panel.signal import SignalPanel


# Enable multiprocessing support for Windows, with frozen executable (e.g. PyInstaller)
multiprocessing.freeze_support()

# Set start method to 'spawn' for Linux (default is 'fork' which is not safe here
# because of the use of Qt and multithreading) - for other OS, the default is
# 'spawn' anyway
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # This exception is raised if the method is already set (this may happen because
    # this module is imported more than once, e.g. when running tests)
    pass


COMPUTATION_TIP = _(
    "DataLab relies on various libraries to perform the computation. During the "
    "computation, errors may occur because of the data (e.g. division by zero, "
    "unexpected data type, etc.) or because of the libraries (e.g. memory error, "
    "etc.). If you encounter an error, before reporting it, please ensure that "
    "the computation is correct, by checking the data and the parameters."
)


POOL: Pool | None = None


def run_with_env(func: Callable, args: tuple, env_json: str) -> CompOut:
    """Wrapper to apply environment config before calling func

    Args:
        func: function to call
        args: function arguments

    Returns:
        Computation output object containing the result, error message,
         or warning message.
    """
    sigima_options.set_env(env_json)
    sigima_options.ensure_loaded_from_env()  # recharge depuis l'env
    return wng_err_func(func, args)


class WorkerState(Enum):
    """Worker states for computation lifecycle."""

    IDLE = auto()  # Ready to start new computation
    STARTING = auto()  # Computation starting (prevents race conditions)
    RUNNING = auto()  # Computation in progress
    FINISHED = auto()  # Computation completed, result available


class WorkerStateMachine:
    """State machine for managing worker computation lifecycle.

    This class handles state transitions for worker computations,
    ensuring valid state flow and preventing invalid operations.
    """

    def __init__(self) -> None:
        """Initialize the state machine in IDLE state."""
        self._current_state = WorkerState.IDLE

    @property
    def current_state(self) -> WorkerState:
        """Get the current state.

        Returns:
            Current WorkerState.
        """
        return self._current_state

    def transition_to(self, target_state: WorkerState) -> None:
        """Transition to the specified target state.

        Args:
            target_state: The state to transition to.

        Raises:
            ValueError: If the transition is not valid from the current state.
        """
        # Define valid state transitions
        valid_transitions = {
            WorkerState.IDLE: {WorkerState.STARTING},
            WorkerState.STARTING: {WorkerState.RUNNING},
            WorkerState.RUNNING: {WorkerState.FINISHED},
            WorkerState.FINISHED: {WorkerState.IDLE},
        }

        # Allow transitions to the same state (no-op)
        if target_state == self._current_state:
            return

        # Check if transition is valid
        allowed_targets = valid_transitions.get(self._current_state, set())
        if target_state not in allowed_targets:
            raise ValueError(
                f"Invalid transition from {self._current_state} to {target_state}. "
                f"Valid transitions: {allowed_targets}"
            )

        self._current_state = target_state

    def reset_to_idle(self) -> None:
        """Reset state to IDLE unconditionally.

        This is used for restart/cancel operations where we need
        to force the state back to IDLE regardless of current state.
        """
        self._current_state = WorkerState.IDLE


class Worker:
    """Multiprocessing worker, to run long-running tasks in a separate process"""

    def __init__(self) -> None:
        self.asyncresult: AsyncResult = None
        self.state_machine = WorkerStateMachine()

    @staticmethod
    def create_pool() -> None:
        """Create multiprocessing pool"""
        global POOL  # pylint: disable=global-statement
        # Create a pool with one process
        POOL = Pool(processes=1)  # pylint: disable=not-callable,consider-using-with

    @staticmethod
    def terminate_pool(wait: bool = False) -> None:
        """Terminate multiprocessing pool.

        Args:
            wait: wait for all tasks to finish. Defaults to False.
        """
        global POOL  # pylint: disable=global-statement
        if POOL is not None:
            if wait:
                # Close the pool properly (wait for all tasks to finish)
                POOL.close()
            else:
                # Terminate the pool and stop the timer
                POOL.terminate()
            POOL.join()
            POOL = None

    def restart_pool(self) -> None:
        """Terminate and recreate the pool"""
        # Terminate the process and stop the timer
        Worker.terminate_pool(wait=False)
        # Recreate the pool for the next computation
        Worker.create_pool()
        # Reset worker state after pool restart
        self.asyncresult = None
        self.state_machine.reset_to_idle()

    def run(self, func: Callable, args: tuple[Any]) -> None:
        """Run computation.

        Args:
            func: function to run
            args: arguments

        Raises:
            ValueError: If not in IDLE state or pool is not available.
        """
        # Check if we can start computation
        if self.state_machine.current_state != WorkerState.IDLE:
            current_state = self.state_machine.current_state
            raise ValueError(f"Cannot start computation from {current_state} state")

        # Transition to starting state
        self.state_machine.transition_to(WorkerState.STARTING)

        global POOL  # pylint: disable=global-statement,global-variable-not-assigned
        if POOL is None:
            raise ValueError("Multiprocessing pool is not available")

        # Start the computation
        env_json = sigima_options.get_env()
        self.asyncresult = POOL.apply_async(run_with_env, (func, args, env_json))

        # Transition to running state
        self.state_machine.transition_to(WorkerState.RUNNING)

    def restart(self) -> None:
        """Restart/cancel current computation"""
        current_state = self.state_machine.current_state

        if current_state == WorkerState.IDLE:
            return  # Already idle, nothing to restart
        if current_state == WorkerState.STARTING:
            # If we're still starting, just go back to idle
            self.asyncresult = None
        elif current_state == WorkerState.RUNNING:
            # Cancel the running computation - use restart_pool for consistency
            self.restart_pool()
            return  # restart_pool already handles state reset
        if current_state == WorkerState.FINISHED:
            # Clean up and go to idle
            self.asyncresult = None

        # Let state machine handle the transition to idle
        self.state_machine.reset_to_idle()

    def close(self) -> None:
        """Close worker: close pool properly and wait for all tasks to finish"""
        # Close multiprocessing Pool properly, but only if no computation is running,
        # to avoid blocking the GUI at exit (so, when wait=True, we wait for the
        # task to finish before closing the pool but there is actually no task running,
        # so the pool is closed immediately but *properly*)
        Worker.terminate_pool(wait=self.asyncresult is None)

    def is_computation_finished(self) -> bool:
        """Return True if computation is finished.

        Returns:
            bool: True if computation is finished
        """
        current_state = self.state_machine.current_state

        if current_state == WorkerState.IDLE:
            return True  # No computation has been started
        if current_state == WorkerState.STARTING:
            return False  # Computation is starting, not finished yet
        if current_state == WorkerState.FINISHED:
            return True  # Already finished
        if current_state == WorkerState.RUNNING:
            if self.asyncresult is None:
                return False  # Should not happen, but defensive
            finished = self.asyncresult.ready()
            if finished:
                # Transition to finished state
                self.state_machine.transition_to(WorkerState.FINISHED)
            return finished
        raise ValueError(f"Invalid worker state: {current_state}")

    def get_result(self) -> CompOut:
        """Return computation result.

        Returns:
            CompOut: computation result

        Raises:
            ValueError: If not in FINISHED state or no result available.
        """
        # Check if we can get result
        if self.state_machine.current_state != WorkerState.FINISHED:
            current_state = self.state_machine.current_state
            raise ValueError(f"Cannot get result from {current_state} state")

        if self.asyncresult is None:
            raise ValueError("No result available")

        # Get result and clean up (ensure cleanup happens even if exception occurs)
        try:
            result = self.asyncresult.get()
            return result
        finally:
            # Always clean up, even if get() raises an exception
            self.asyncresult = None
            self.state_machine.transition_to(WorkerState.IDLE)

    def has_result_available(self) -> bool:
        """Check if computation finished successfully and result is available.

        Returns:
            True if computation completed successfully and result can be retrieved.
        """
        return self.state_machine.current_state == WorkerState.FINISHED


def is_pairwise_mode() -> bool:
    """Return True if operation mode is pairwise.

    Returns:
        bool: True if operation mode is pairwise
    """
    state = Conf.proc.operation_mode.get() == "pairwise"
    return state


@dataclass
class ComputingFeature:
    """Computing feature dataclass.

    Args:
        pattern: pattern
        function: function
        paramclass: parameter class
        title: title
        icon_name: icon name
        comment: comment
        edit: whether to edit the parameters
        obj2_name: name of the second object
        skip_xarray_compat: whether to skip X-array compatibility check for this feature
    """

    pattern: Literal["1_to_1", "1_to_0", "1_to_n", "n_to_1", "2_to_1"]
    function: Optional[Callable] = None
    paramclass: Optional[type[gds.DataSet]] = None
    title: Optional[str] = None
    icon_name: Optional[str] = None
    comment: Optional[str] = None
    edit: Optional[bool] = None
    obj2_name: Optional[str] = None
    skip_xarray_compat: Optional[bool] = None

    def __post_init__(self):
        """Validate the function after initialization."""
        if self.function is not None and not is_computation_function(self.function):
            raise ValueError(
                f"'{self.function.__name__}' is not a valid computation function."
            )

    @property
    def name(self) -> str:
        """Return the name of the computing feature."""
        if self.function is None:
            raise ValueError(
                "ComputingFeature must have a 'function' to derive its name."
            )
        return self.function.__name__

    @property
    def action_title(self) -> str:
        """Return the action title of the computing feature."""
        title = self.title
        if (
            self.paramclass is not None and (self.edit is None or self.edit)
        ) or self.pattern == "1_to_0":
            title += "..."
        return title


class BaseProcessor(QC.QObject, Generic[TypeROI, TypeROIParam]):
    """Object handling data processing: operations, processing, analysis.

    Args:
        panel: panel
        plotwidget: plot widget
    """

    SIG_ADD_SHAPE = QC.Signal(str)
    PARAM_DEFAULTS: dict[str, gds.DataSet] = {}

    def __init__(self, panel: SignalPanel | ImagePanel, plotwidget: PlotWidget):
        super().__init__()
        self.panel = panel
        self.mainwindow = panel.mainwindow
        self.plotwidget = plotwidget
        self.worker: Worker | None = None
        self.set_process_isolation_enabled(Conf.main.process_isolation_enabled.get())
        self.computing_registry: dict[str, ComputingFeature] = {}
        self.register_computations()

    def close(self):
        """Close processor properly"""
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def set_process_isolation_enabled(self, enabled: bool) -> None:
        """Set process isolation enabled.

        Args:
            enabled: enabled
        """
        if enabled:
            if self.worker is None:
                self.worker = Worker()
                Worker.create_pool()
        else:
            if self.worker is not None:
                self.worker.terminate_pool()
                self.worker = None

    def _is_signal_panel(self) -> bool:
        """Check if the current panel is a signal panel.

        Returns:
            True if processing signals, False if processing images
        """
        return self.panel.PARAMCLASS == SignalObj

    def _check_signal_xarray_compatibility(
        self, signals: list[SignalObj], progress: QW.QProgressDialog | None = None
    ) -> tuple[list[SignalObj], bool] | None:
        """Check X-array compatibility for multiple signals and handle conflicts.

        Args:
            signals: List of signal objects to check
            progress: Progress dialog (if method is called from a long-running task,
             we need to handle the progress dialog: the dialog will show up after a
             short delay on top of the message box if we don't handle it here)

        Returns:
            Tuple of (signals, yes_to_all_selected) where signals is the list of
            signals (potentially with interpolated signals) and yes_to_all_selected
            is True if user chose "Yes to All". Returns None if user canceled.
        """
        if not self._is_signal_panel() or len(signals) <= 1:
            return signals, False

        initial_duration = 0
        if progress is not None:
            initial_duration = progress.minimumDuration()
            # Set progress dialog minimum duration to a very high value to effectively
            # hide it if it shows up (we handle the dialog manually here)
            progress.setMinimumDuration(2000000)
            QW.QApplication.processEvents()

        # Get X arrays for comparison
        x_arrays = [sig.x for sig in signals]

        # Check if all X arrays are identical
        x_arrays_identical = True
        if len(x_arrays) > 1:
            # Compare sizes first
            sizes = [len(x) for x in x_arrays]
            if len(set(sizes)) > 1:
                x_arrays_identical = False
            else:
                # Same sizes - check if xmin and xmax are also the same
                xmins = [x.min() for x in x_arrays]
                xmaxs = [x.max() for x in x_arrays]
                # Use relative tolerance for floating point comparison
                if not (
                    np.allclose(xmins, xmins[0], rtol=1e-12)
                    and np.allclose(xmaxs, xmaxs[0], rtol=1e-12)
                ):
                    x_arrays_identical = False

        # If X arrays are identical, proceed normally
        if x_arrays_identical:
            if initial_duration > 0:
                # Restore initial progress dialog duration
                progress.setMinimumDuration(initial_duration)
            return signals, False

        # X arrays differ - handle based on configuration
        behavior = Conf.proc.xarray_compat_behavior.get("ask")
        yes_to_all_selected = False

        if behavior == "ask" and not env.execenv.unattended:
            # Create custom message box with "Yes to All" option
            msg_box = QW.QMessageBox(self.mainwindow)
            msg_box.setWindowTitle(_("X-array incompatibility"))
            msg_box.setText(
                _(
                    "The selected signals have different X arrays.\n\n"
                    "To perform the computation, signals need to be interpolated "
                    "to match a common X array.\n\n"
                    "Do you want to continue with automatic interpolation?"
                )
            )
            msg_box.setIcon(QW.QMessageBox.Icon.Question)

            # Add custom buttons
            msg_box.addButton(_("Yes"), QW.QMessageBox.ButtonRole.YesRole)
            yes_all_button = msg_box.addButton(
                _("Yes to All"), QW.QMessageBox.ButtonRole.YesRole
            )
            no_button = msg_box.addButton(_("No"), QW.QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(no_button)

            # Execute dialog and get user choice
            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == no_button:
                return None
            elif clicked_button == yes_all_button:
                yes_to_all_selected = True

        # Perform interpolation to the smallest X array
        sizes = [len(x) for x in x_arrays]
        min_size_idx = np.argmin(sizes)
        target_x = x_arrays[min_size_idx]

        interpolated_signals = []
        for i, sig in enumerate(signals):
            if i == min_size_idx:
                # Keep the target signal as-is
                interpolated_signals.append(sig)
            else:
                # Create interpolated copy
                interpolated_sig = sig.copy(
                    title=f"{sig.title} (interpolated)", all_metadata=True
                )
                x_orig, y_orig = sig.x, sig.y

                # Interpolate using linear method (safe default)
                y_new = interpolate(
                    x_orig,
                    y_orig,
                    target_x,
                    Interpolation1DMethod.LINEAR,
                    fill_value=None,
                )

                interpolated_sig.set_xydata(target_x, y_new)
                interpolated_signals.append(interpolated_sig)

        signals = interpolated_signals

        if initial_duration > 0:
            # Restore initial progress dialog duration
            progress.setMinimumDuration(initial_duration)

        return signals, yes_to_all_selected

    @abc.abstractmethod
    def register_operations(self) -> None:
        """Register operations."""

    @abc.abstractmethod
    def register_processing(self) -> None:
        """Register processing functions."""

    @abc.abstractmethod
    def register_analysis(self) -> None:
        """Register analysis functions."""

    def register_computations(self) -> None:
        """Register computations."""
        self.register_operations()
        self.register_processing()
        self.register_analysis()

    def has_param_defaults(self, paramclass: type[gds.DataSet]) -> bool:
        """Return True if parameter defaults are available.

        Args:
            paramclass: parameter class

        Returns:
            bool: True if parameter defaults are available
        """
        return paramclass.__name__ in self.PARAM_DEFAULTS

    def update_param_defaults(self, param: gds.DataSet) -> None:
        """Update parameter defaults.

        Args:
            param: parameters
        """
        key = param.__class__.__name__
        pdefaults = self.PARAM_DEFAULTS.get(key)
        if pdefaults is not None:
            update_dataset(param, pdefaults)
        self.PARAM_DEFAULTS[key] = param

    def init_param(
        self,
        param: gds.DataSet,
        paramclass: type[gds.DataSet],
        title: str,
        comment: str | None = None,
    ) -> tuple[bool, gds.DataSet]:
        """Initialize processing parameters.

        Args:
            param: parameter
            paramclass: parameter class
            title: title
            comment: comment

        Returns:
            Tuple (edit, param) where edit is True if parameters have been edited,
            False otherwise.
        """
        edit = param is None
        if edit:
            param = paramclass(title, comment)
            self.update_param_defaults(param)
            if hasattr(param, "update_from_obj"):
                obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
                param.update_from_obj(obj)
        return edit, param

    def handle_output(
        self, compout: CompOut, context: str, progress: QW.QProgressDialog
    ) -> SignalObj | ImageObj | GeometryResult | TableResult | None:
        """Handle computation output: if error, display error message,
        if warning, display warning message.

        Args:
            compout: computation output
            context: context (e.g. "Computing: Gaussian filter")
            progress: progress dialog

        Returns:
            Output object: a signal or image object, or a geometry/table result object,
             or None if error
        """
        if compout.error_msg or compout.warning_msg:
            mindur = progress.minimumDuration()
            progress.setMinimumDuration(1000000)
            if compout.error_msg:
                show_warning_error(
                    self.panel, "error", context, compout.error_msg, COMPUTATION_TIP
                )
            if compout.warning_msg:
                show_warning_error(self.panel, "warning", context, compout.warning_msg)
            progress.setMinimumDuration(mindur)
            if compout.error_msg:
                return None
        result = compout.result
        return result

    def _merge_geometry_results_for_n_to_1(
        self, result_obj: SignalObj | ImageObj, src_obj_list: list[SignalObj | ImageObj]
    ) -> None:
        """Merge geometry results from source objects into the result object.

        This method handles geometry result merging for n_to_1 operations when
        keep_results is enabled, providing a clean alternative to monkey patching.

        Args:
            result_obj: The result object from the computation
            src_obj_list: The list of source objects used in the computation
        """
        # Only merge if keep_results is enabled and we have multiple source objects
        if not Conf.proc.keep_results.get() or len(src_obj_list) <= 1:
            return

        # Group geometry results by title for merging
        geometry_by_title = {}

        # Collect all geometry results from all source objects
        for src_obj in src_obj_list:
            for geom_adapter in GeometryAdapter.iterate_from_obj(src_obj):
                title = geom_adapter.title
                if title not in geometry_by_title:
                    geometry_by_title[title] = []
                geometry_by_title[title].append(geom_adapter.result)

        # Only proceed if we have geometry results to merge
        if not geometry_by_title:
            return

        # Remove any existing geometry results from the result object
        result_keys_to_remove = []
        for key in result_obj.metadata.keys():
            if GeometryAdapter.match(key, result_obj.metadata[key]):
                result_keys_to_remove.append(key)

        for key in result_keys_to_remove:
            result_obj.metadata.pop(key, None)

        # Merge and add back concatenated geometry results
        for title, geometries in geometry_by_title.items():
            if len(geometries) > 1:
                # Concatenate multiple geometry results
                merged_geometry = concat_geometries(title, geometries)
                adapter = GeometryAdapter(merged_geometry)
                adapter.add_to(result_obj)
            elif len(geometries) == 1:
                # Just one geometry result, add it
                adapter = GeometryAdapter(geometries[0])
                adapter.add_to(result_obj)

    def _handle_keep_results(self, result_obj: SignalObj | ImageObj) -> None:
        """Handle keep_results logic by removing all results if keep_results is False.

        This method implements the logic that was previously in Sigima's dst_1_to_1,
        dst_n_to_1, and dst_2_to_1 functions, where results were deleted from the
        destination object when keep_results was False.

        Args:
            result_obj: The result object from the computation
        """
        if not Conf.proc.keep_results.get():
            # Remove all table and geometry results when keep_results is disabled
            TableAdapter.remove_all_from(result_obj)
            GeometryAdapter.remove_all_from(result_obj)

    def __exec_func(
        self,
        func: Callable,
        args: tuple,
        progress: QW.QProgressDialog,
    ) -> CompOut | None:
        """Execute function, eventually in a separate process.

        Args:
            func: function to execute
            args: function arguments
            progress: progress dialog

        Returns:
            Computation output object or None if canceled
        """
        QW.QApplication.processEvents()
        if not progress.wasCanceled():
            if self.worker is None:
                # No process isolation: run function directly
                return wng_err_func(func, args)
            # Process isolation: run function in a separate process
            self.worker.run(func, args)
            while not self.worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0)  # Just yields to other threads - no forced delay
                if progress.wasCanceled():  # User canceled the operation
                    self.worker.restart()  # Cancel computation and reset to idle
                    break
            # Only get result if computation actually finished (not canceled)
            if self.worker.has_result_available():
                return self.worker.get_result()
        return None

    def _compute_1_to_1_subroutine(
        self, funcs: list[Callable], params: list, title: str
    ) -> None:
        """Generic subroutine for 1-to-1 processing.

        Args:
            funcs: list of functions to execute
            params: list of parameters
            title: title of progress bar
        """
        assert len(funcs) == len(params)
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        grps = self.panel.objview.get_sel_groups()
        new_gids = {}
        with create_progress_bar(
            self.panel, title, max_=len(objs) * len(params)
        ) as progress:
            for i_row, obj in enumerate(objs):
                for i_param, (param, func) in enumerate(zip(params, funcs)):
                    name = func.__name__
                    i_title = f"{title} ({i_row + 1}/{len(objs)})"
                    progress.setLabelText(i_title)
                    pvalue = (i_row + 1) * (i_param + 1)
                    pvalue = 0 if pvalue == 1 else pvalue
                    progress.setValue(pvalue)
                    args = (obj,) if param is None else (obj, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(
                        result, _("Computing: %s") % i_title, progress
                    )
                    if new_obj is None:
                        continue

                    # Handle keep_results logic for 1_to_1 operations
                    if isinstance(new_obj, (SignalObj, ImageObj)):
                        self._handle_keep_results(new_obj)

                    patch_title_with_ids(new_obj, [obj], get_short_id)

                    # Is new object a native object (i.e. a Signal object for a Signal
                    # Panel, or an Image object for an Image Panel) ?
                    # (example of non-native object use case: image profile extraction)
                    is_new_obj_native = isinstance(new_obj, self.panel.PARAMCLASS)

                    new_gid = None
                    if grps and is_new_obj_native:
                        # If groups are selected, then it means that there is no
                        # individual object selected: we work on groups only
                        old_gid = self.panel.objmodel.get_object_group_id(obj)
                        new_gid = new_gids.get(old_gid)
                        if new_gid is None:
                            # Create a new group for each selected group
                            old_g = self.panel.objmodel.get_group(old_gid)
                            new_g = self.panel.add_group(
                                f"{name}({get_short_id(old_g)})"
                            )
                            new_gids[old_gid] = new_gid = get_uuid(new_g)
                    if is_new_obj_native:
                        self.panel.add_object(new_obj, group_id=new_gid)
                    else:
                        self.panel.mainwindow.add_object(new_obj)
        # Select newly created groups, if any
        for group_id in new_gids.values():
            self.panel.objview.set_current_item_id(group_id, extend=True)

    def __get_src_grps_gids_objs_nbobj_valid(
        self, min_group_nb: int
    ) -> tuple[list, list, dict, int]:
        """In pairwise mode only: get source groups, group ids, objects,
        and number of objects. Check if the number of objects is valid.

        Args:
            min_group_nb: minimum number of groups (typically, 2 for `n1` functions
            and 1 for `n1n` functions)

        Returns:
            Tuple (source groups, group ids, objects, number of objects, valid)
        """
        # In pairwise mode, we need to create a new object for each pair of objects
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        objmodel = self.panel.objmodel
        src_grps = sorted(
            {objmodel.get_group_from_object(obj) for obj in objs},
            key=objmodel.get_number,
        )
        src_gids = [get_uuid(grp) for grp in src_grps]

        # [src_objs dictionary] keys: old group id, values: list of old objects
        src_objs: dict[str, list[SignalObj | ImageObj]] = {}
        for src_gid in src_gids:
            src_objs[src_gid] = [
                obj for obj in objs if objmodel.get_object_group_id(obj) == src_gid
            ]

        nbobj = len(src_objs[src_gids[0]])

        valid = len(src_grps) >= min_group_nb
        if not valid:
            # In pairwise mode, we need selected objects in at least two groups.
            if env.execenv.unattended:
                raise ValueError(
                    "Pairwise mode: objects must be selected in at least two groups"
                )
            QW.QMessageBox.warning(
                self.mainwindow,
                _("Warning"),
                _(
                    "In pairwise mode, you need to select objects "
                    "in at least two groups."
                ),
            )
        if valid:
            valid = all(len(src_objs[src_gid]) == nbobj for src_gid in src_gids)
            if not valid:
                if env.execenv.unattended:
                    raise ValueError(
                        "Pairwise mode: invalid number of objects in each group"
                    )
                QW.QMessageBox.warning(
                    self.mainwindow,
                    _("Warning"),
                    _(
                        "In pairwise mode, you need to select "
                        "the same number of objects in each group."
                    ),
                )
        return src_grps, src_gids, src_objs, nbobj, valid

    def compute_1_to_1(
        self,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: type[gds.DataSet] | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> None:
        """Generic processing method: 1 object in → 1 object out.

        Applies a function independently to each selected object in the active panel.
        The result of each computation is a new object appended to the same panel.

        Args:
            func: Function to execute, that takes either `(dst_obj, src_obj)` or
             `(dst_obj, src_obj, param)` as arguments, where `dst_obj` is the output
             object, `src_obj` is the input object, and `param` is an optional
             parameter set.
            param: Optional parameter instance.
            paramclass: Optional parameter class for editing.
            title: Optional progress bar title.
            comment: Optional comment for parameter dialog.
            edit: Whether to open the parameter editor before execution.

        .. note::
            With k selected objects, the method produces k outputs (one per input).

        .. note::
            This method does not support pairwise mode.
        """
        if (edit is None or param is None) and paramclass is not None:
            old_edit = edit
            edit, param = self.init_param(param, paramclass, title, comment)
            if old_edit is not None:
                edit = old_edit
        if param is not None:
            if edit and not param.edit(parent=self.mainwindow):
                return
        self._compute_1_to_1_subroutine([func], [param], title)

    def compute_multiple_1_to_1(
        self,
        funcs: list[Callable],
        params: list[gds.DataSet] | None = None,
        title: str | None = None,
        edit: bool | None = None,
    ) -> None:
        """Generic processing method: 1 object in → n objects out.

        Applies multiple functions to each selected object, generating multiple
        outputs per object. The resulting objects are appended to the active panel.

        Args:
            funcs: List of functions to apply. Each function takes either
             `(dst_obj, src_obj)` or `(dst_obj, src_obj, param)` as arguments,
             where `dst_obj` is the output object, `src_obj` is the input object,
             and `param` is an optional parameter set.
            params: List of parameter instances corresponding to each function.
            title: Optional progress bar title.
            edit: Whether to open the parameter editor before execution.

        .. note::
            With k selected objects and n outputs per function,
            the method produces k × n outputs.

        .. note::
            This method does not support pairwise mode.
        """
        if params is None:
            params = [None] * len(funcs)
        else:
            group = gds.DataSetGroup(params, title=_("Parameters"))
            if edit and not group.edit(parent=self.mainwindow):
                return
            if len(funcs) != len(params):
                raise ValueError("Number of functions must match number of parameters")
        self._compute_1_to_1_subroutine(funcs, params, title)

    def compute_1_to_n(
        self,
        func: Callable,
        params: list[gds.DataSet],
        title: str | None = None,
        edit: bool | None = None,
    ) -> None:
        """Generic processing method: 1 object in → n objects out.

        Applies a single function to each selected object, with n different parameters
        set, thus generating n outputs per object. The resulting objects are appended to
        the active panel.

        Args:
            func: Single function to apply, that takes either `(dst_obj, src_obj)`
             or `(dst_obj, src_obj, param)` as arguments,
             where `dst_obj` is the output object, `src_obj` is the input object,
             and `param` is an optional parameter set.
            params: List of parameter instances.
            title: Optional progress bar title.
            edit: Whether to open the parameter editor before execution.

        .. note::
            With k selected objects and n parameter sets,
            the method produces k × n outputs.

        .. note::
            This method does not support pairwise mode.
        """
        assert params is not None
        if edit:
            group = gds.DataSetGroup(params, title=_("Parameters"))
            if not group.edit(parent=self.mainwindow):
                return
        self._compute_1_to_1_subroutine([func] * len(params), params, title)

    def compute_1_to_0(
        self,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: type[gds.DataSet] | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> dict[str, GeometryResult | TableResult]:
        """Generic processing method: 1 object in → no object out.

        Applies a function to each selected object, returning metadata or measurement
        results (e.g. peak coordinates, statistical properties) without generating
        new objects. Results are stored in the object's metadata and returned as a
        dictionary.

        Args:
            func: Function to execute, that takes either `(obj)` or `(obj, param)` as
             arguments, where `obj` is the input object and `param` is an optional
             parameter set.
            param: Optional parameter instance.
            paramclass: Optional parameter class for editing.
            title: Optional progress bar title.
            comment: Optional comment for parameter dialog.
            edit: Whether to open the parameter editor before execution.

        Returns:
            Dictionary mapping each object UUID to a GeometryResult or TableResult
            instance.

        .. note::
            With k selected objects, the method performs k analyses and produces
            no output objects.

        .. note::
            This method does not support pairwise mode.
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.mainwindow):
                return None
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        current_obj = self.panel.objview.get_current_object()
        title = func.__name__ if title is None else title
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            rdata = ResultData()
            for idx, obj in enumerate(objs):
                pvalue = idx + 1
                pvalue = 0 if pvalue == 1 else pvalue
                progress.setValue(pvalue)
                args = (obj,) if param is None else (obj, param)

                # Execute function
                compout = self.__exec_func(func, args, progress)
                if compout is None:
                    break
                result = self.handle_output(
                    compout, _("Computing: %s") % title, progress
                )
                if result is None:
                    continue

                # Using the adapters:
                if isinstance(result, GeometryResult):
                    adapter = GeometryAdapter(result)
                elif isinstance(result, TableResult):
                    adapter = TableAdapter(result)
                else:
                    # For "compute 1 to 0" functions, the result is either a
                    # GeometryResult or TableResult:
                    raise TypeError("Unsupported result type")

                # Add result shape to object's metadata
                adapter.add_to(obj)
                if param is not None:
                    obj.metadata[f"{adapter.title}Param"] = str(param)

                # Append result to result data for later display
                rdata.append(adapter, obj)

                if obj is current_obj:
                    self.panel.selection_changed(update_items=True)
                else:
                    self.panel.refresh_plot(get_uuid(obj), True, False)

        if rdata:
            show_resultdata(self.mainwindow, rdata, f"{objs[0].PREFIX}_results")
        return rdata

    def compute_n_to_1(
        self,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: type[gds.DataSet] | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> None:
        """Generic processing method: n objects in → 1 object out.

        Aggregates multiple selected objects into a single result using the provided
        function. In pairwise mode, applies the function to object pairs (grouped by
        index) and generates one output per pair.

        Args:
            func: Function to apply, that takes either `(dst_obj, src_obj_list)` or
             `(dst_obj, src_obj_list, param)` as arguments, where `dst_obj` is the
             output object, `src_obj_list` is the input object list,
             and `param` is an optional parameter set.
            param: Optional parameter instance.
            paramclass: Optional parameter class for editing.
            title: Optional progress bar title.
            comment: Optional comment for parameter dialog.
            edit: Whether to open the parameter editor before execution.

        .. note::
            With n selected objects:

            - in default mode, produces 1 output.
            - in pairwise mode, produces n outputs (one per pair).
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.mainwindow):
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)
        objmodel = self.panel.objmodel
        pairwise = is_pairwise_mode()
        name = func.__name__

        if pairwise:
            src_grps, src_gids, src_objs, _nbobj, valid = (
                self.__get_src_grps_gids_objs_nbobj_valid(min_group_nb=2)
            )
            if not valid:
                return
            dst_gname = (
                f"{name}({','.join([get_short_id(grp) for grp in src_grps])})|pairwise"
            )
            group_exclusive = len(self.panel.objview.get_sel_groups()) != 0
            if not group_exclusive:
                # This is not a group exclusive selection
                dst_gname += "[...]"
            dst_gid = get_uuid(self.panel.add_group(dst_gname))
            n_pairs = len(src_objs[src_gids[0]])
            max_i_pair = min(
                n_pairs, max(len(src_objs[get_uuid(grp)]) for grp in src_grps)
            )
            # Track "Yes to All" choice for this compute operation
            auto_interpolate_for_operation = False

            with create_progress_bar(self.panel, title, max_=n_pairs) as progress:
                for i_pair, src_obj1 in enumerate(src_objs[src_gids[0]][:max_i_pair]):
                    progress.setValue(i_pair + 1)
                    progress.setLabelText(title)
                    src_objs_pair = [src_obj1]
                    for src_gid in src_gids[1:]:
                        src_obj = src_objs[src_gid][i_pair]
                        src_objs_pair.append(src_obj)

                    # Check signal x-array compatibility for n-to-1 operations
                    if auto_interpolate_for_operation:
                        # "Yes to All" selected, automatically interpolate
                        # by temporarily changing the configuration
                        with Conf.proc.xarray_compat_behavior.temp("interpolate"):
                            result = self._check_signal_xarray_compatibility(
                                src_objs_pair, progress=progress
                            )
                    else:
                        # Normal compatibility check with dialog
                        result = self._check_signal_xarray_compatibility(
                            src_objs_pair, progress=progress
                        )

                    if result is None:
                        # User canceled or compatibility check failed
                        return

                    checked_objs, yes_to_all_selected = result
                    if yes_to_all_selected:
                        auto_interpolate_for_operation = True

                    src_objs_pair = checked_objs
                    if param is None:
                        args = (src_objs_pair,)
                    else:
                        args = (src_objs_pair, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(
                        result, _("Calculating: %s") % title, progress
                    )
                    if new_obj is None:
                        break
                    # Handle geometry result merging for n_to_1 operations (pairwise)
                    if isinstance(new_obj, (SignalObj, ImageObj)):
                        self._handle_keep_results(new_obj)
                        self._merge_geometry_results_for_n_to_1(new_obj, src_objs_pair)
                    patch_title_with_ids(new_obj, src_objs_pair, get_short_id)
                    self.panel.add_object(new_obj, group_id=dst_gid)

        else:
            # In single operand mode, we create a single object for all selected objects

            # [src_objs dictionary] keys: old group id, values: list of old objects
            src_objs: dict[str, list[SignalObj | ImageObj]] = {}

            grps = self.panel.objview.get_sel_groups()
            if grps:
                # (Group exclusive selection)
                # At least one group is selected: create a new group
                dst_gname = f"{name}({','.join([get_uuid(grp) for grp in grps])})"
                dst_gid = get_uuid(self.panel.add_group(dst_gname))
            else:
                # (Object exclusive selection)
                # No group is selected: use each object's group
                dst_gid = None

            for src_obj in objs:
                src_gid = objmodel.get_object_group_id(src_obj)
                src_objs.setdefault(src_gid, []).append(src_obj)

            # Track "Yes to All" choice for this compute operation
            auto_interpolate_for_operation = False

            with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
                progress.setValue(0)
                progress.setLabelText(title)
                for src_gid, src_obj_list in src_objs.items():
                    # Check signal x-array compatibility for n-to-1 operations
                    if auto_interpolate_for_operation:
                        # "Yes to All" selected, automatically interpolate
                        with Conf.proc.xarray_compat_behavior.temp("interpolate"):
                            result = self._check_signal_xarray_compatibility(
                                src_obj_list, progress=progress
                            )
                    else:
                        # Normal compatibility check with dialog
                        result = self._check_signal_xarray_compatibility(
                            src_obj_list, progress=progress
                        )

                    if result is None:
                        # User canceled or compatibility check failed
                        return

                    checked_objs, yes_to_all_selected = result
                    if yes_to_all_selected:
                        auto_interpolate_for_operation = True

                    src_obj_list = checked_objs

                    if param is None:
                        args = (src_obj_list,)
                    else:
                        args = (src_obj_list, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(
                        result, _("Calculating: %s") % title, progress
                    )
                    if new_obj is None:
                        break
                    # Handle geometry result merging for n_to_1 operations
                    if isinstance(new_obj, (SignalObj, ImageObj)):
                        self._handle_keep_results(new_obj)
                        self._merge_geometry_results_for_n_to_1(new_obj, src_obj_list)
                    group_id = dst_gid if dst_gid is not None else src_gid
                    patch_title_with_ids(new_obj, src_obj_list, get_short_id)
                    self.panel.add_object(new_obj, group_id=group_id)

        # Select newly created group, if any
        if dst_gid is not None:
            self.panel.objview.set_current_item_id(dst_gid)

    def compute_2_to_1(
        self,
        obj2: SignalObj | ImageObj | list[SignalObj | ImageObj] | None,
        obj2_name: str,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: type[gds.DataSet] | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
        skip_xarray_compat: bool | None = None,
    ) -> None:
        """Generic processing method: binary operation 1+1 → 1.

        Applies a binary function between each selected object and a second operand.
        Supports both single operand mode (same operand for all objects)
        and pairwise mode (one-to-one matching between two object lists).

        Args:
            obj2: Second operand (single object or list for pairwise mode).
            obj2_name: Display name for the second operand (used in selection dialog).
            func: Function to apply, that takes either `(dst_obj, src_obj1, src_obj2)`
             or `(dst_obj, src_obj1, src_obj2, param)` as arguments, where
             `dst_obj` is the output object, `src_obj1` is the first input object,
             `src_obj2` is the second input object (operand), and `param` is an
             optional parameter set.
            param: Optional parameter instance.
            paramclass: Optional parameter class for editing.
            title: Optional progress bar title.
            comment: Optional comment for parameter dialog.
            edit: Whether to open the parameter editor before execution.
            skip_xarray_compat: If True, skip x-array compatibility checks
             (only for signal panels).

        .. note::
            With k selected objects:

            - in single operand mode and 1 secondary object: produces k outputs.
            - in pairwise mode with k secondary objects: produces k outputs
              (one per pair).
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.mainwindow):
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)
        objmodel = self.panel.objmodel
        pairwise = is_pairwise_mode()

        if obj2 is None:
            objs2 = []
        elif isinstance(obj2, list):
            objs2 = obj2
            assert pairwise
        else:
            objs2 = [obj2]

        dlg_title = _("Select %s") % obj2_name

        if pairwise:
            group_exclusive = len(self.panel.objview.get_sel_groups()) != 0

            src_grps, src_gids, src_objs, nbobj, valid = (
                self.__get_src_grps_gids_objs_nbobj_valid(min_group_nb=1)
            )
            if not valid:
                return
            if not objs2:
                objs2 = self.panel.get_objects_with_dialog(
                    dlg_title,
                    _(
                        "<u>Note:</u> operation mode is <i>pairwise</i>: "
                        "%s object(s) expected (i.e. as many as in the first group)"
                    )
                    % nbobj,
                    nbobj,
                )
                if objs2 is None:
                    return

            name = func.__name__
            n_pairs = len(src_objs[src_gids[0]])
            max_i_pair = min(
                n_pairs, max(len(src_objs[get_uuid(grp)]) for grp in src_grps)
            )
            grp2_id = objmodel.get_object_group_id(objs2[0])
            grp2 = objmodel.get_group(grp2_id)

            # Initialize pair mapping for potential interpolations
            pair_maps = {}

            # Check x-array compatibility for signal processing (pairwise mode)
            if self._is_signal_panel() and not skip_xarray_compat:
                # Check compatibility between objects from both groups
                all_pairs = []
                for src_gid in src_gids:
                    for i_pair in range(max_i_pair):
                        src_obj1 = src_objs[src_gid][i_pair]
                        src_obj2 = objs2[i_pair]
                        if isinstance(src_obj1, SignalObj) and isinstance(
                            src_obj2, SignalObj
                        ):
                            all_pairs.append((src_obj1, src_obj2))

                # Track "Yes to All" choice for this compute operation
                auto_interpolate_for_operation = False

                # Check all pairs for compatibility and create interpolation maps
                for src_obj1, src_obj2 in all_pairs:
                    if auto_interpolate_for_operation:
                        # "Yes to All" selected, automatically interpolate
                        with Conf.proc.xarray_compat_behavior.temp("interpolate"):
                            result = self._check_signal_xarray_compatibility(
                                [src_obj1, src_obj2]
                            )
                    else:
                        # Normal compatibility check with dialog
                        result = self._check_signal_xarray_compatibility(
                            [src_obj1, src_obj2]
                        )

                    if result is None:
                        return  # User cancelled or error occurred

                    checked_pair, yes_to_all_selected = result
                    if yes_to_all_selected:
                        auto_interpolate_for_operation = True

                    # Store mapping for this specific pair
                    pair_maps[(src_obj1, src_obj2)] = checked_pair

            with create_progress_bar(self.panel, title, max_=len(src_gids)) as progress:
                for i_group, src_gid in enumerate(src_gids):
                    progress.setValue(i_group + 1)
                    progress.setLabelText(title)
                    if group_exclusive:
                        # This is a group exclusive selection
                        src_grp = objmodel.get_group(src_gid)
                        grp_short_ids = [get_uuid(grp) for grp in (src_grp, grp2)]
                        dst_gname = f"{name}({','.join(grp_short_ids)})|pairwise"
                    else:
                        dst_gname = f"{name}[...]"
                    dst_gid = get_uuid(self.panel.add_group(dst_gname))
                    for i_pair in range(max_i_pair):
                        orig_obj1, orig_obj2 = src_objs[src_gid][i_pair], objs2[i_pair]

                        # Use interpolated signals if available, keep original refs
                        actual_obj1, actual_obj2 = orig_obj1, orig_obj2
                        if (orig_obj1, orig_obj2) in pair_maps:
                            interpolated_pair = pair_maps[(orig_obj1, orig_obj2)]
                            actual_obj1 = interpolated_pair[0]
                            actual_obj2 = interpolated_pair[1]

                        args = [actual_obj1, actual_obj2]
                        if param is not None:
                            args.append(param)
                        result = self.__exec_func(func, tuple(args), progress)
                        if result is None:
                            break
                        new_obj = self.handle_output(
                            result, _("Calculating: %s") % title, progress
                        )
                        if new_obj is None:
                            continue

                        # Handle keep_results logic for 2_to_1 operations (pairwise)
                        if isinstance(new_obj, (SignalObj, ImageObj)):
                            self._handle_keep_results(new_obj)

                        # Use original objects for title generation
                        patch_title_with_ids(
                            new_obj, [orig_obj1, orig_obj2], get_short_id
                        )
                        self.panel.add_object(new_obj, group_id=dst_gid)

        else:
            if not objs2:
                objs2 = self.panel.get_objects_with_dialog(
                    dlg_title,
                    _(
                        "<u>Note:</u> operation mode is <i>single operand</i>: "
                        "1 object expected"
                    ),
                )
                if objs2 is None:
                    return
            obj2 = objs2[0]

            # Initialize signal mapping for potential interpolations
            signal_map = {}

            # Check x-array compatibility for signal processing (single operand mode)
            orig_obj2 = obj2  # Keep reference to original obj2 for title generation
            if (
                self._is_signal_panel()
                and isinstance(obj2, SignalObj)
                and not skip_xarray_compat
            ):
                signal_objs = [obj for obj in objs if isinstance(obj, SignalObj)]
                if signal_objs:
                    # Check compatibility and get potentially interpolated signals
                    result = self._check_signal_xarray_compatibility(
                        signal_objs + [obj2]
                    )
                    if result is None:
                        return  # User cancelled or error occurred

                    checked_objs, _yes_to_all_selected = result
                    # Note: In single operand mode, "Yes to All" doesn't apply
                    # since there's only one compatibility check

                    # Replace obj2 with the potentially interpolated version
                    obj2 = checked_objs[-1]  # obj2 was added last

                    # Create a mapping of original to interpolated signals
                    for orig_obj, checked_obj in zip(signal_objs, checked_objs[:-1]):
                        signal_map[orig_obj] = checked_obj

            with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
                for index, obj in enumerate(objs):
                    progress.setValue(index + 1)
                    progress.setLabelText(title)

                    # Use interpolated signal if available
                    actual_obj = obj
                    if (
                        self._is_signal_panel()
                        and isinstance(obj, SignalObj)
                        and obj in signal_map
                    ):
                        actual_obj = signal_map[obj]

                    args = (
                        (actual_obj, obj2)
                        if param is None
                        else (actual_obj, obj2, param)
                    )
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(
                        result, _("Calculating: %s") % title, progress
                    )
                    if new_obj is None:
                        continue

                    # Handle keep_results logic for 2_to_1 operations (single operand)
                    if isinstance(new_obj, (SignalObj, ImageObj)):
                        self._handle_keep_results(new_obj)

                    group_id = objmodel.get_object_group_id(obj)
                    # Use original objects for title generation
                    patch_title_with_ids(new_obj, [obj, orig_obj2], get_short_id)
                    self.panel.add_object(new_obj, group_id=group_id)

    def register_1_to_1(
        self,
        function: Callable,
        title: str,
        paramclass: type[gds.DataSet] | None = None,
        icon_name: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> ComputingFeature:
        """Register a 1-to-1 processing function.

        The `register_1_to_1` method is used to register a function that takes one
        object as input and produces one object as output. The function is called
        with the input object and an optional parameter set. The result of the
        function is returned.

        Args:
            function: function to register
            title: title of the function
            paramclass: parameter class. Defaults to None.
            icon_name: icon name. Defaults to None.
            comment: comment. Defaults to None.
            edit: whether to open the parameter editor before execution.

        Returns:
            Registered feature.
        """
        feature = ComputingFeature(
            pattern="1_to_1",
            function=function,
            title=title,
            paramclass=paramclass,
            icon_name=icon_name,
            comment=comment,
            edit=edit,
        )
        self.add_feature(feature)
        return feature

    def register_1_to_0(
        self,
        function: Callable,
        title: str,
        paramclass: type[gds.DataSet] | None = None,
        icon_name: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> ComputingFeature:
        """Register a 1-to-0 processing function.

        The function takes one object as input and produces no output.
        The function is called with the input object and an optional parameter set.
        The result of the function is returned.

        Args:
            function: function to register
            title: title of the function
            paramclass: parameter class. Defaults to None.
            icon_name: icon name. Defaults to None.
            comment: comment. Defaults to None.
            edit: whether to open the parameter editor before execution.

        Returns:
            Registered feature.
        """
        feature = ComputingFeature(
            pattern="1_to_0",
            function=function,
            title=title,
            paramclass=paramclass,
            icon_name=icon_name,
            comment=comment,
            edit=edit,
        )
        self.add_feature(feature)
        return feature

    def register_1_to_n(
        self, function: Callable, title: str, icon_name: str | None = None
    ) -> ComputingFeature:
        """Register a 1-to-n processing function.

        The function takes one object as input and produces multiple objects as output.
        The function is called with the input object and an optional parameter set.
        The result of the function is returned.

        Args:
            function: function to register
            title: title of the function
            icon_name: icon name. Defaults to None.

        Returns:
            Registered feature.
        """
        feature = ComputingFeature(
            pattern="1_to_n",
            function=function,
            title=title,
            icon_name=icon_name,
        )
        self.add_feature(feature)
        return feature

    def register_n_to_1(
        self,
        function: Callable,
        title: str,
        paramclass: type[gds.DataSet] | None = None,
        icon_name: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> ComputingFeature:
        """Register a n-to-1 processing function.

        The function takes multiple objects as input and produces one object as output.
        The function is called with the input objects and an optional parameter set.
        The result of the function is returned.

        Args:
            function: function to register
            title: title of the function
            paramclass: parameter class. Defaults to None.
            icon_name: icon name. Defaults to None.
            comment: comment. Defaults to None.
            edit: whether to open the parameter editor before execution.

        Returns:
            Registered feature.
        """
        feature = ComputingFeature(
            pattern="n_to_1",
            function=function,
            title=title,
            paramclass=paramclass,
            icon_name=icon_name,
            comment=comment,
            edit=edit,
        )
        self.add_feature(feature)
        return feature

    def register_2_to_1(
        self,
        function: Callable,
        title: str,
        paramclass: type[gds.DataSet] | None = None,
        icon_name: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
        obj2_name: str | None = None,
        skip_xarray_compat: bool | None = None,
    ) -> ComputingFeature:
        """Register a 2-to-1 processing function.

        The function takes two objects as input and produces one object as output.
        The function is called with the input objects and an optional parameter set.
        The result of the function is returned.

        Args:
            function: function to register
            title: title of the function
            paramclass: parameter class. Defaults to None.
            icon_name: icon name. Defaults to None.
            comment: comment. Defaults to None.
            edit: whether to open the parameter editor before execution.
            obj2_name: name of the second object. Defaults to None.
            skip_xarray_compat: whether to skip X-array compatibility check.
             Defaults to None. Set to True for operations like interpolation where
             different X-arrays are expected and desired.

        Returns:
            Registered feature.
        """
        feature = ComputingFeature(
            pattern="2_to_1",
            function=function,
            title=title,
            paramclass=paramclass,
            icon_name=icon_name,
            comment=comment,
            edit=edit,
            obj2_name=obj2_name,
            skip_xarray_compat=skip_xarray_compat,
        )
        self.add_feature(feature)
        return feature

    def add_feature(self, feature: ComputingFeature) -> None:
        """Add a computing feature to the registry.

        Args:
            feature: ComputingFeature instance to add.
        """
        self.computing_registry[feature.function] = feature

    def get_feature(self, function_or_name: Callable | str) -> ComputingFeature:
        """Get a computing feature by name or function.

        Args:
            function_or_name: Name of the feature or the function itself.

        Returns:
            Computing feature instance.
        """
        try:
            return self.computing_registry[function_or_name]
        except KeyError as exc:
            for _func, feature in self.computing_registry.items():
                if feature.name == function_or_name:
                    return feature
            raise ValueError(f"Unknown computing feature: {function_or_name}") from exc

    @qt_try_except()
    def run_feature(
        self, key: str | Callable | ComputingFeature, *args, **kwargs
    ) -> dict[str, GeometryResult | TableResult] | None:
        """Run a computing feature that has been previously registered.

        This method is a generic dispatcher for all compute methods.
        It uses the central registry to find the appropriate compute method
        based on the pattern (`1_to_1`, `1_to_0`, `n_to_1`, `2_to_1`, `1_to_n`).
        It then calls the appropriate compute method with the provided arguments.

        Depending on the pattern, this method can take different arguments:

        .. code-block:: python

            import sigima.proc.signal as sigima_signal
            import sigima.params

            proc = win.signalpanel.processor  # where `win` is DataLab's main window

            # For patterns `1_to_1`, `1_to_0`, `n_to_1`:
            proc.run_feature(sigima_signal.normalize)
            param = sigima.params.MovingAverageParam(n=3)
            proc.run_feature(sigima_signal.moving_average, param)
            proc.run_feature(computation_function, param, edit=False)

            # For pattern `2_to_1`:
            proc.run_feature(sigima_signal.difference, obj2)
            param = sigima.params.InterpolationParam(method="cubic")
            proc.run_feature(sigima_signal.interpolation, obj2, param)

            # For pattern `1_to_n`:
            params = roi.to_params(obj)
            proc.run_feature(sigima_signal.extract_roi, params=params)

        Args:
            key: The key to look up in the registry. It can be a string, a callable,
             or a ComputingFeature instance.
            *args: Positional arguments to pass to the compute method.
            **kwargs: Keyword arguments to pass to the compute method.

        Returns:
            The result of the computation or None.
        """
        if not isinstance(key, ComputingFeature):
            feature = self.get_feature(key)
        else:
            feature = key

        # Some keyword parameters may be overridden
        edit = kwargs.pop("edit", feature.edit)
        title = kwargs.pop("title", feature.title)
        comment = kwargs.pop("comment", feature.comment)

        pattern = feature.pattern

        if pattern in {"1_to_1", "1_to_0", "n_to_1"}:
            compute_method = getattr(self, f"compute_{pattern}")
            param = kwargs.pop("param", args[0] if args else None)
            assert isinstance(param, (gds.DataSet, type(None))), (
                f"For pattern '{pattern}', 'param' must be a DataSet or None"
            )
            return compute_method(
                feature.function,
                param=param,
                paramclass=feature.paramclass,
                title=title,
                comment=comment,
                edit=edit,
            )
        if pattern == "2_to_1":
            obj2 = kwargs.pop("obj2", args[0] if args else None)
            assert isinstance(obj2, (SignalObj, ImageObj, list, type(None))), (
                "For pattern '2_to_1', 'obj2' must be a SignalObj, ImageObj, "
                "list of SignalObj/ImageObj, or None"
            )
            param = kwargs.pop("param", args[1] if args and len(args) > 1 else None)
            assert isinstance(param, (gds.DataSet, type(None))), (
                "For pattern '2_to_1', 'param' must be a DataSet or None"
            )
            return self.compute_2_to_1(
                obj2,
                feature.obj2_name or _("Second operand"),
                feature.function,
                param=param,
                paramclass=feature.paramclass,
                title=title,
                comment=comment,
                edit=edit,
                skip_xarray_compat=feature.skip_xarray_compat,
            )
        if pattern == "1_to_n":
            params = kwargs.get("params", args[0] if args else [])
            if not isinstance(params, list) or any(
                not isinstance(param, gds.DataSet) for param in params
            ):
                raise ValueError(
                    "For pattern '1_to_n', 'params' must be "
                    "a list of DataSet or a DataSetGroup"
                )
            return self.compute_1_to_n(
                feature.function,
                params=params,
                title=title,
                edit=edit,
            )
        raise ValueError(f"Unsupported compute pattern: {pattern}")

    # ------Data Processing-------------------------------------------------------------

    @qt_try_except()
    def compute_roi_extraction(self, roi: TypeROI | None = None) -> None:
        """Extract Region Of Interest (ROI) from data with:

        - :py:func:`sigima.proc.image.compute_extract_roi` for single ROI
        - :py:func:`sigima.proc.image.compute_extract_rois` for multiple ROIs"""
        # Expected behavior:
        # -----------------
        # * If `roi` is not None or not empty, skip the ROI dialog
        # * If first selected obj has a ROI, use this ROI as default but open
        #   ROI Editor dialog anyway
        # * If multiple objs are selected, then apply the first obj ROI to all
        if roi is None or roi.is_empty():
            roi = self.edit_roi_graphically(mode="extract")
        if roi is None or roi.is_empty():
            return
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        params = roi.to_params(obj)
        if Conf.proc.extract_roi_singleobj.get() and len(params) > 1:
            # Extract multiple ROIs into a single object (remove all the ROIs),
            # if the "Extract all ROIs into a single image object"
            # option is checked and if there are more than one ROI
            self._extract_multiple_roi_in_single_object(params)
        else:
            # Extract each ROI into a separate object (keep the ROI in the case of
            # a circular ROI), if the "Extract all ROIs into a single image object"
            # option is not checked or if there is only one ROI (See Issue #31)
            self.run_feature("extract_roi", params=params, edit=False)

    @abc.abstractmethod
    @qt_try_except()
    def _extract_multiple_roi_in_single_object(
        self, params: list[TypeROIParam]
    ) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""

    # ------Analysis-------------------------------------------------------------------

    def edit_roi_graphically(
        self, mode: Literal["apply", "extract", "define"] = "apply"
    ) -> TypeROI | None:
        """Define Region Of Interest (ROI).

        Args:
            mode: Mode of operation, either "apply" (define ROI, then apply it to
             selected objects), "extract" (define ROI, then extract data from it),
             or "define" (define ROI without applying or extracting).

        Returns:
            ROI object or None if ROI dialog has been canceled.
        """
        assert mode in ("apply", "extract", "define"), (
            f"Invalid mode: {mode}. Must be either 'apply', 'extract' or 'define'."
        )
        # Expected behavior:
        # -----------------
        # * If first selected obj has a ROI, use this ROI as default but open
        #   ROI Editor dialog anyway
        # * If multiple objs are selected, then apply the first obj ROI to all
        results = self.panel.get_roi_editor_output(mode=mode)
        if results is None:
            return None
        edited_roi, modified = results
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        obj = objs[-1]
        params = edited_roi.to_params(obj)
        group = gds.DataSetGroup(params, title=_("Regions of Interest"))
        if (
            env.execenv.unattended  # Unattended mode (automated unit tests)
            or edited_roi.is_empty()  # No ROI has been defined
            or group.edit(parent=self.mainwindow)  # ROI dialog has been accepted
        ):
            if modified:
                # If ROI has been modified, save ROI (not in "extract" mode)
                if edited_roi.is_empty() and mode != "define":
                    # If ROI is empty, remove it from all selected objects
                    # (not in "define" mode because the ROI is just defined and used
                    # in a processing function for example: it's not bound to any
                    # object yet)
                    for obj_i in objs:
                        obj_i.roi = None
                else:
                    edited_roi = edited_roi.__class__.from_params(obj, params)
                    if mode == "apply":
                        # Apply ROI to all selected objects
                        for obj_i in objs:
                            obj_i.roi = edited_roi
                self.SIG_ADD_SHAPE.emit(get_uuid(obj))
                # self.panel.selection_changed(update_items=True)
                self.panel.refresh_plot(
                    "selected",
                    update_items=True,
                    only_visible=False,
                    only_existing=True,
                )
        return edited_roi

    def edit_roi_numerically(self) -> TypeROI:
        """Edit Regions Of Interest (ROIs) numerically.

        Opens a dialog to edit the parameters of the selected ROIs.
        If no ROIs are selected, it will prompt the user to select ROIs.

        Returns:
            The edited ROI object if the dialog is accepted, otherwise the original ROI.
        """
        obj = self.panel.objview.get_sel_objects()[0]
        assert obj.roi is not None, _("No ROI selected for editing.")
        params = obj.roi.to_params(obj)
        group = gds.DataSetGroup(params, title=_("Regions of Interest"))
        if group.edit(parent=self.mainwindow):
            edited_roi = obj.roi.__class__.from_params(obj, params)
            obj.roi = edited_roi
            self.SIG_ADD_SHAPE.emit(get_uuid(obj))
            self.panel.refresh_plot(
                "selected",
                update_items=True,
                only_visible=False,
                only_existing=True,
            )
            return edited_roi
        return obj.roi

    def delete_regions_of_interest(self) -> None:
        """Delete Regions Of Interest"""
        if (
            env.execenv.unattended
            or QW.QMessageBox.question(
                self.mainwindow,
                _("Remove all ROIs"),
                _("Are you sure you want to remove all ROIs?"),
            )
            == QW.QMessageBox.Yes
        ):
            for obj in self.panel.objview.get_sel_objects():
                if obj.roi is not None:
                    obj.roi = None
                    self.panel.selection_changed(update_items=True)
