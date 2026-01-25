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
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    DataObject = SignalObj | ImageObj


class MainThreadExecutor(QObject):
    """Helper to execute functions on the main thread.

    This class uses Qt signals to marshal function calls to the main thread.
    The signal carries a callable and result container, allowing the caller
    to wait for completion and retrieve results.
    """

    # Signal that carries the work to be done
    execute_requested = Signal(object, object)  # (func, result_container)

    def __init__(self) -> None:
        super().__init__()
        self.execute_requested.connect(self._execute_on_main_thread)

    @Slot(object, object)
    def _execute_on_main_thread(self, func, result_container: dict) -> None:
        """Execute function and store result."""
        try:
            result_container["result"] = func()
            result_container["error"] = None
        except Exception as e:
            result_container["result"] = None
            result_container["error"] = e
        finally:
            result_container["done"].set()

    def run_on_main_thread(self, func) -> Any:
        """Run a function on the main thread and wait for result.

        Args:
            func: Zero-argument callable to execute.

        Returns:
            The result of the function.

        Raises:
            Any exception raised by the function.
        """
        if threading.current_thread() is threading.main_thread():
            # Already on main thread, just call it
            return func()

        # Create result container with threading event
        result_container = {"done": threading.Event(), "result": None, "error": None}

        # Emit signal (will be queued to main thread)
        self.execute_requested.emit(func, result_container)

        # Wait for completion
        result_container["done"].wait()

        # Re-raise any exception
        if result_container["error"] is not None:
            raise result_container["error"]

        return result_container["result"]


# Global executor instance (created lazily)
_executor: MainThreadExecutor | None = None


def get_executor() -> MainThreadExecutor:
    """Get or create the main thread executor."""
    global _executor
    if _executor is None:
        _executor = MainThreadExecutor()
    return _executor


class WorkspaceAdapter(QObject):
    """Thread-safe adapter for workspace access.

    This class wraps access to DataLab's workspace, ensuring that all
    modifying operations are executed on the Qt main thread.

    Attributes:
        main_window: Reference to the DataLab main window.
    """

    def __init__(self, main_window=None) -> None:
        """Initialize the adapter.

        Args:
            main_window: The DataLab main window. If None, operations will fail.
        """
        super().__init__()
        self._main_window = main_window

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
    # Read operations (thread-safe, no Qt marshaling needed)
    # =========================================================================

    def list_objects(self) -> list[tuple[str, str]]:
        """List all objects in the workspace.

        Returns:
            List of (name, panel) tuples for all objects.
        """
        self._ensure_main_window()
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

    def get_object(self, name: str) -> DataObject:
        """Get an object by name.

        Args:
            name: Object name/title.

        Returns:
            The requested object.

        Raises:
            KeyError: If object not found.
        """
        self._ensure_main_window()

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

        Args:
            name: Object name/title.

        Returns:
            "signal" or "image", or None if not found.
        """
        self._ensure_main_window()

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

    # =========================================================================
    # Write operations (must be marshaled to Qt main thread)
    # =========================================================================

    def add_object(self, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace.

        This operation is marshaled to the Qt main thread.

        Args:
            obj: Object to add.
            overwrite: If True, replace existing object with same name.

        Raises:
            ValueError: If object exists and overwrite is False.
        """
        self._ensure_main_window()
        name = obj.title

        if not overwrite and self.object_exists(name):
            raise ValueError(f"Object '{name}' already exists")

        if overwrite and self.object_exists(name):
            self._remove_object_sync(name)

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
        executor = get_executor()
        executor.run_on_main_thread(lambda: panel.add_object(obj))

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

        if panel_name == "signal":
            panel = self._main_window.signalpanel
        else:
            panel = self._main_window.imagepanel

        # Use executor to run on main thread, including index lookup
        def do_remove():
            # Find object index - must be done on main thread
            obj_idx = None
            for idx, obj in enumerate(panel.objmodel):
                if obj.title == name:
                    obj_idx = idx
                    break

            if obj_idx is None:
                return

            # Select via objview (1-based index for selection)
            panel.objview.select_objects([obj_idx + 1])
            panel.remove_object(force=True)

        executor = get_executor()
        executor.run_on_main_thread(do_remove)

    def update_metadata(self, name: str, metadata: dict) -> None:
        """Update object metadata.

        This operation is marshaled to the Qt main thread.

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

        # Find and update object
        for obj in panel.objmodel:
            if obj.title == name:
                for key, value in metadata.items():
                    if value is not None and hasattr(obj, key):
                        setattr(obj, key, value)
                # Refresh display
                panel.SIG_REFRESH_PLOT.emit("selected", True)
                break

    def clear(self) -> None:
        """Clear all objects from the workspace.

        This operation is marshaled to the Qt main thread.
        """
        self._ensure_main_window()

        executor = get_executor()

        # Clear both panels
        for panel in [self._main_window.signalpanel, self._main_window.imagepanel]:
            if panel is not None:
                executor.run_on_main_thread(lambda p=panel: p.delete_all_objects())
