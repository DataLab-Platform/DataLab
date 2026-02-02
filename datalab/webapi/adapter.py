# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Adapter
===============

Thread-safe adapter for accessing the DataLab workspace from Web API handlers.

The Web API server runs in a separate thread from the Qt GUI. This adapter
provides safe access to workspace operations by using Qt's signal/slot mechanism
to marshal calls to the main thread when necessary.

Design
------

The adapter provides the same interface as the workspace but ensures:

1. Read operations are safe (DataLab's data model is mostly immutable)
2. Write operations are marshaled to the Qt main thread
3. Errors are properly propagated back to the calling thread

Usage
-----

The adapter is instantiated once when the Web API starts and passed to route
handlers via FastAPI dependency injection.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Union

from qtpy.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    DataObject = Union[SignalObj, ImageObj]


class MainThreadExecutor(QObject):
    """Helper to execute functions on the main thread using Qt signals.

    This class uses Qt's signal/slot mechanism to safely marshal function calls
    from worker threads (like the Uvicorn server thread) to the Qt main thread.
    This is essential because Qt GUI operations must be performed on the main
    thread.

    The implementation uses a signal connected with Qt.QueuedConnection to post
    work to the main thread's event loop, and a threading.Event to synchronize
    the calling thread with the result.

    Important: This class must be instantiated on the main thread!
    """

    # Signal to request execution on main thread
    _execute_signal = Signal(object, object)  # (func, result_holder)

    def __init__(self) -> None:
        super().__init__()
        # Connect signal to slot - this connection will queue calls to main thread
        self._execute_signal.connect(self._execute_on_main_thread)
        # Store the main thread for comparison
        self._main_thread = QThread.currentThread()

    @Slot(object, object)
    def _execute_on_main_thread(self, func, result_holder: dict) -> None:
        """Slot that executes the function on the main thread.

        Args:
            func: Zero-argument callable to execute.
            result_holder: Dict to store result/exception and signal completion.
        """
        try:
            result_holder["result"] = func()
            result_holder["exception"] = None
        except Exception as e:  # pylint: disable=broad-exception-caught
            result_holder["result"] = None
            result_holder["exception"] = e
        finally:
            # Signal that execution is complete
            result_holder["event"].set()

    def run_on_main_thread(self, func) -> Any:
        """Run a function on the Qt main thread and wait for the result.

        If already on the main thread, executes directly. Otherwise, uses
        Qt's signal/slot mechanism to marshal the call to the main thread.

        Args:
            func: Zero-argument callable to execute.

        Returns:
            The result of the function.

        Raises:
            Any exception raised by the function.
        """
        # Check if we're already on the main thread
        if QThread.currentThread() == self._main_thread:
            return func()

        # Create result holder with synchronization event
        result_holder = {
            "result": None,
            "exception": None,
            "event": threading.Event(),
        }

        # Emit signal to queue execution on main thread
        self._execute_signal.emit(func, result_holder)

        # Wait for execution to complete
        # Use a timeout to avoid hanging forever if something goes wrong
        if not result_holder["event"].wait(timeout=30.0):
            raise TimeoutError("Main thread execution timed out after 30 seconds")

        # Process pending events to ensure UI updates are applied
        QCoreApplication.processEvents()

        # Re-raise any exception from the main thread
        if result_holder["exception"] is not None:
            raise result_holder["exception"]  # pylint: disable=raising-bad-type

        return result_holder["result"]


class WorkspaceAdapter(QObject):
    """Thread-safe adapter for workspace access.

    This class wraps access to DataLab's workspace, ensuring that all
    modifying operations are executed on the Qt main thread.

    Attributes:
        main_window: Reference to the DataLab main window.
    """

    def __init__(self, main_window=None) -> None:
        """Initialize the adapter.

        This should be called from the main thread to ensure the executor
        is properly initialized with working Qt signals.

        Args:
            main_window: The DataLab main window. If None, operations will fail.
        """
        super().__init__()
        self._main_window = main_window
        # Create executor on main thread to ensure proper Qt signal connection
        self._executor = MainThreadExecutor()

    def set_main_window(self, main_window) -> None:
        """Set the main window reference.

        Args:
            main_window: The DataLab main window.
        """
        self._main_window = main_window

    def _ensure_main_window(self) -> None:
        """Ensure main window is available."""
        if self._main_window is None:
            raise RuntimeError("DataLab main window not available")

    # =========================================================================
    # Read operations (marshaled to Qt main thread for thread safety)
    # =========================================================================

    def list_objects(self) -> list[tuple[str, str]]:
        """List all objects in the workspace.

        This operation is marshaled to the Qt main thread for thread safety.

        Returns:
            List of (name, panel) tuples for all objects.
        """
        self._ensure_main_window()

        def do_list():
            result = []
            # Access signal panel
            sig_panel = self._main_window.signalpanel
            if sig_panel is not None:
                for obj in sig_panel.objmodel:
                    result.append((obj.title, "signal"))

            # Access image panel
            img_panel = self._main_window.imagepanel
            if img_panel is not None:
                for obj in img_panel.objmodel:
                    result.append((obj.title, "image"))
            return result

        return self._executor.run_on_main_thread(do_list)

    def get_object(self, name: str) -> DataObject:
        """Get an object by name.

        This operation is marshaled to the Qt main thread for thread safety.

        Args:
            name: Object name/title.

        Returns:
            The requested object.

        Raises:
            KeyError: If object not found.
        """
        self._ensure_main_window()

        def do_get():
            # Search in signal panel
            sig_panel = self._main_window.signalpanel
            if sig_panel is not None:
                for obj in sig_panel.objmodel:
                    if obj.title == name:
                        return obj.copy()

            # Search in image panel
            img_panel = self._main_window.imagepanel
            if img_panel is not None:
                for obj in img_panel.objmodel:
                    if obj.title == name:
                        return obj.copy()

            raise KeyError(f"Object '{name}' not found")

        return self._executor.run_on_main_thread(do_get)

    def object_exists(self, name: str) -> bool:
        """Check if an object exists.

        Args:
            name: Object name/title.

        Returns:
            True if object exists.
        """
        try:
            self.get_object(name)
            return True
        except KeyError:
            return False

    def get_object_panel(self, name: str) -> str | None:
        """Get the panel containing an object.

        This operation is marshaled to the Qt main thread for thread safety.

        Args:
            name: Object name/title.

        Returns:
            "signal" or "image", or None if not found.
        """
        self._ensure_main_window()

        def do_lookup():
            sig_panel = self._main_window.signalpanel
            if sig_panel is not None:
                for obj in sig_panel.objmodel:
                    if obj.title == name:
                        return "signal"

            img_panel = self._main_window.imagepanel
            if img_panel is not None:
                for obj in img_panel.objmodel:
                    if obj.title == name:
                        return "image"

            return None

        return self._executor.run_on_main_thread(do_lookup)

    # =========================================================================
    # Write operations (must be marshaled to Qt main thread)
    # =========================================================================

    def add_object(self, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace.

        This operation is marshaled to the Qt main thread as a single atomic
        operation to ensure thread safety.

        Args:
            obj: Object to add.
            overwrite: If True, replace existing object with same name.

        Raises:
            ValueError: If object exists and overwrite is False.
        """
        self._ensure_main_window()
        name = obj.title
        obj_type = type(obj).__name__

        if obj_type not in ("SignalObj", "ImageObj"):
            raise TypeError(f"Unsupported object type: {obj_type}")

        # Check if object exists
        if self.object_exists(name):
            if not overwrite:
                raise ValueError(f"Object '{name}' already exists")
            # Remove existing object first using the working remove method
            self._remove_object_sync(name)

        # Add the new object
        self._add_object_sync(obj)

    def _add_object_sync(self, obj: DataObject) -> None:
        """Add object (called from main thread or marshaled via executor)."""
        obj_type = type(obj).__name__

        if obj_type == "SignalObj":
            panel = self._main_window.signalpanel
        elif obj_type == "ImageObj":
            panel = self._main_window.imagepanel
        else:
            raise TypeError(f"Unsupported object type: {obj_type}")

        # Use executor to run on main thread if necessary
        self._executor.run_on_main_thread(lambda: panel.add_object(obj))

    def remove_object(self, name: str) -> None:
        """Remove an object from the workspace.

        This operation is marshaled to the Qt main thread.

        Args:
            name: Object name/title.

        Raises:
            KeyError: If object not found.
        """
        self._ensure_main_window()

        if not self.object_exists(name):
            raise KeyError(f"Object '{name}' not found")

        self._remove_object_sync(name)

    def _remove_object_sync(self, name: str) -> None:
        """Remove object (called from main thread or marshaled via executor)."""
        panel_name = self.get_object_panel(name)
        if panel_name is None:
            return

        main_window = self._main_window

        # Use executor to run on main thread, including all panel access
        def do_remove():
            # All panel access happens inside the executor
            if panel_name == "signal":
                panel = main_window.signalpanel
            else:
                panel = main_window.imagepanel

            # Find the object
            target_obj = None
            for obj in panel.objmodel:
                if obj.title == name:
                    target_obj = obj
                    break

            if target_obj is None:
                return

            obj_uuid = get_uuid(target_obj)

            # Remove using the same approach as remove_all_objects but for single object
            # Remove from plot handler
            panel.plothandler.remove_item(obj_uuid)
            # Remove from tree view
            panel.objview.remove_item(obj_uuid, refresh=False)
            # Remove from object model
            panel.objmodel.remove_object(target_obj)
            # Update tree
            panel.objview.update_tree()
            # Emit signal
            panel.SIG_OBJECT_REMOVED.emit()

        self._executor.run_on_main_thread(do_remove)

    def update_metadata(self, name: str, metadata: dict) -> None:
        """Update object metadata.

        This operation modifies Qt objects and should be marshaled to the
        Qt main thread for thread safety.

        Args:
            name: Object name/title.
            metadata: Dictionary of metadata fields to update.

        Raises:
            KeyError: If object not found.
        """
        self._ensure_main_window()

        panel_name = self.get_object_panel(name)
        if panel_name is None:
            raise KeyError(f"Object '{name}' not found")

        if panel_name == "signal":
            panel = self._main_window.signalpanel
        else:
            panel = self._main_window.imagepanel

        def do_update():
            # Find and update object
            for obj in panel.objmodel:
                if obj.title == name:
                    for key, value in metadata.items():
                        if value is not None and hasattr(obj, key):
                            setattr(obj, key, value)
                    # Refresh display
                    panel.SIG_REFRESH_PLOT.emit("selected", True)
                    break

        self._executor.run_on_main_thread(do_update)

    def clear(self) -> None:
        """Clear all objects from the workspace.

        This operation is marshaled to the Qt main thread.
        """
        self._ensure_main_window()

        def do_clear():
            # Clear both panels using remove_all_objects (no confirmation dialog)
            for panel in [self._main_window.signalpanel, self._main_window.imagepanel]:
                if panel is not None:
                    panel.remove_all_objects()

        self._executor.run_on_main_thread(do_clear)

    # =========================================================================
    # Computation operations (for calc API)
    # =========================================================================

    def select_objects(
        self, names: list[str], panel: str | None = None
    ) -> tuple[list[str], str]:
        """Select objects by name in a panel.

        This operation is marshaled to the Qt main thread for thread safety.

        Args:
            names: List of object names/titles to select.
            panel: Panel name ("signal" or "image"). None = auto-detect or current.

        Returns:
            Tuple of (list of selected names, panel name).

        Raises:
            KeyError: If any object not found.
            ValueError: If objects span multiple panels.
        """
        self._ensure_main_window()

        def do_select():
            # Determine panel for each object
            panels_found = set()
            obj_indices = []

            for name in names:
                obj_panel = self.get_object_panel(name)
                if obj_panel is None:
                    raise KeyError(f"Object '{name}' not found")
                panels_found.add(obj_panel)

            if len(panels_found) > 1:
                raise ValueError(
                    "Cannot select objects from multiple panels. "
                    f"Found objects in: {panels_found}"
                )

            if panel is not None:
                target_panel = panel
            elif panels_found:
                target_panel = panels_found.pop()
            else:
                target_panel = "signal"

            # Get the panel widget
            if target_panel == "signal":
                panel_widget = self._main_window.signalpanel
            else:
                panel_widget = self._main_window.imagepanel

            # Find object indices (1-based) by name
            for name in names:
                for idx, obj in enumerate(panel_widget.objmodel):
                    if obj.title == name:
                        obj_indices.append(idx + 1)  # 1-based indexing
                        break

            # Select the objects using the panel's method
            if obj_indices:
                panel_widget.objview.select_objects(obj_indices)

            return names, target_panel

        return self._executor.run_on_main_thread(do_select)

    def get_selected_objects(self, panel: str | None = None) -> list[str]:
        """Get names of currently selected objects.

        Args:
            panel: Panel name. None = current panel.

        Returns:
            List of selected object names.
        """
        self._ensure_main_window()

        def do_get_selected():
            if panel == "signal":
                panel_widget = self._main_window.signalpanel
            elif panel == "image":
                panel_widget = self._main_window.imagepanel
            else:
                # Use current panel
                panel_widget = self._main_window.tabwidget.currentWidget()
                if not hasattr(panel_widget, "objmodel"):
                    return []

            return [obj.title for obj in panel_widget.objview.get_sel_objects()]

        return self._executor.run_on_main_thread(do_get_selected)

    def calc(self, name: str, param: dict | None = None) -> tuple[bool, list[str]]:
        """Call a computation function on currently selected objects.

        This operation is marshaled to the Qt main thread for thread safety.

        Args:
            name: Computation function name (e.g., "normalize", "fft").
            param: Optional parameters as a dictionary.

        Returns:
            Tuple of (success, list of new object names created).

        Raises:
            ValueError: If computation function not found.
        """
        self._ensure_main_window()

        def do_calc():
            # Get objects before calc to track new ones
            before_names = set(self._get_all_object_names())

            # Convert param dict to DataSet if provided
            param_dataset = None
            if param is not None:
                param_dataset = self._dict_to_dataset(name, param)

            # Call the main window's calc method with edit=False to prevent
            # blocking modal dialogs when called from the API
            try:
                self._main_window.calc(name, param_dataset, edit=False)
                success = True
            except ValueError:
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                raise RuntimeError(f"Computation '{name}' failed: {e}") from e

            # Get objects after calc to find new ones
            after_names = set(self._get_all_object_names())
            new_names = list(after_names - before_names)

            return success, new_names

        return self._executor.run_on_main_thread(do_calc)

    def _get_all_object_names(self) -> list[str]:
        """Get all object names from all panels."""
        names = []
        for panel in [self._main_window.signalpanel, self._main_window.imagepanel]:
            if panel is not None:
                for obj in panel.objmodel:
                    names.append(obj.title)
        return names

    def _dict_to_dataset(self, func_name: str, param_dict: dict):
        """Convert a parameter dictionary to a DataSet object.

        This looks up the parameter class for the given function and
        creates an instance with the provided values.

        Args:
            func_name: Computation function name.
            param_dict: Dictionary of parameter values.

        Returns:
            DataSet instance, or None if no parameters needed.
        """
        import guidata.dataset as gds  # pylint: disable=import-outside-toplevel

        # Try to find the parameter class from the processor
        # First, look in the current panel's processor
        panel = self._main_window.tabwidget.currentWidget()
        if hasattr(panel, "processor"):
            try:
                feature = panel.processor.get_feature(func_name)
                if feature.paramclass is not None:
                    # Create instance and set values
                    param_obj = feature.paramclass()
                    for key, value in param_dict.items():
                        if hasattr(param_obj, key):
                            setattr(param_obj, key, value)
                    return param_obj
            except ValueError:
                pass

        # Fallback: try to import common parameter classes from sigima
        try:
            import sigima.params  # pylint: disable=import-outside-toplevel

            # Try to find matching param class (e.g., "normalize" -> NormalizeParam)
            param_class_name = func_name.title().replace("_", "") + "Param"
            if hasattr(sigima.params, param_class_name):
                param_class = getattr(sigima.params, param_class_name)
                return param_class.create(**param_dict)
        except ImportError:
            pass

        # If we can't find a param class, create a simple DataSet
        if param_dict:

            class DynamicParam(gds.DataSet):
                """Dynamic parameter class created at runtime."""

            param_obj = DynamicParam()
            for key, value in param_dict.items():
                setattr(param_obj, key, value)
            return param_obj

        return None
