# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab base proxy module
-------------------------
"""

# How to add a new method to the proxy:
# -------------------------------------
#
# 1.  Add the method to the AbstractDLControl class, as an abstract method
#
# 2a. If the method requires any data conversion to get through the XML-RPC layer,
#     implement the method in both LocalProxy and RemoteClient classes
#
# 2b. If the method does not require any data conversion, implement the method
#     directly in the BaseProxy class, so that it is available to both LocalProxy
#     and RemoteClient classes without any code duplication
#
# 3.  Implement the method in the DLMainWindow class
#
# 4.  Implement the method in the RemoteServer class (it will be automatically
#     registered as an XML-RPC method, like all methods of AbstractDLControl)

from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import guidata.dataset as gds
import numpy as np
from sigima import ImageObj, SignalObj

if TYPE_CHECKING:
    from collections.abc import Iterator

    from datalab.control.remote import ServerProxy
    from datalab.gui.main import DLMainWindow


class AbstractDLControl(abc.ABC):
    """Abstract base class for controlling DataLab (main window or remote server)"""

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self.get_object_uuids())

    def __getitem__(self, nb_id_title: int | str | None = None) -> SignalObj | ImageObj:
        """Return object"""
        return self.get_object(nb_id_title)

    def __iter__(self) -> Iterator[SignalObj | ImageObj]:
        """Iterate over objects"""
        uuids = self.get_object_uuids()
        for uuid in uuids:
            yield self.get_object(uuid)

    def __str__(self) -> str:
        """Return object string representation"""
        return super().__repr__()

    def __repr__(self) -> str:
        """Return object representation"""
        titles = self.get_object_titles()
        uuids = self.get_object_uuids()
        text = f"{str(self)} (DataLab, {len(titles)} items):\n"
        for uuid, title in zip(uuids, titles):
            text += f"  {uuid}: {title}\n"
        return text

    def __bool__(self) -> bool:
        """Return True if model is not empty"""
        return bool(self.get_object_uuids())

    def __contains__(self, id_title: str) -> bool:
        """Return True if object (UUID or title) is in model"""
        return id_title in (self.get_object_titles() + self.get_object_uuids())

    @classmethod
    def get_public_methods(cls) -> list[str]:
        """Return all public methods of the class, except itself.

        Returns:
            List of public methods
        """
        return [
            method
            for method in dir(cls)
            if not method.startswith(("_", "context_"))
            and method != "get_public_methods"
        ]

    @abc.abstractmethod
    def get_version(self) -> str:
        """Return DataLab public version.

        Returns:
            DataLab version
        """

    @abc.abstractmethod
    def close_application(self) -> None:
        """Close DataLab application"""

    @abc.abstractmethod
    def raise_window(self) -> None:
        """Raise DataLab window"""

    @abc.abstractmethod
    def get_current_panel(self) -> str:
        """Return current panel name.

        Returns:
            Panel name (valid values: "signal", "image", "macro"))
        """

    @abc.abstractmethod
    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel: Panel name (valid values: "signal", "image", "macro"))
        """

    @abc.abstractmethod
    def reset_all(self) -> None:
        """Reset all application data"""

    @abc.abstractmethod
    def remove_object(self, force: bool = False) -> None:
        """Remove current object from current panel.

        Args:
            force: if True, remove object without confirmation. Defaults to False.
        """

    @abc.abstractmethod
    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh state.

        Args:
            state: Auto refresh state
        """

    # Returns a context manager to temporarily disable autorefresh
    @contextmanager
    def context_no_refresh(self) -> Generator[None, None, None]:
        """Return a context manager to temporarily disable auto refresh.

        Returns:
            Context manager

        Example:

            >>> with proxy.context_no_refresh():
            ...     proxy.add_image("image1", data1)
            ...     proxy.calc("fft")
            ...     proxy.calc("wiener")
            ...     proxy.calc("ifft")
            ...     # Auto refresh is disabled during the above operations
        """
        self.toggle_auto_refresh(False)
        try:
            yield
        finally:
            self.toggle_auto_refresh(True)

    @abc.abstractmethod
    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles state.

        Args:
            state: Show titles state
        """

    @abc.abstractmethod
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename: HDF5 file name
        """

    @abc.abstractmethod
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files: List of HDF5 files to open. Defaults to None.
            import_all: Import all objects from HDF5 files. Defaults to None.
            reset_all: Reset all application data. Defaults to None.
        """

    @abc.abstractmethod
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename: HDF5 file name
            reset_all: Reset all application data. Defaults to None.
        """

    @abc.abstractmethod
    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images).

        Args:
            filenames: list of file names
        """

    @abc.abstractmethod
    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """

    @abc.abstractmethod
    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title: Signal title
            xdata: X data
            ydata: Y data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            group_id: group id in which to add the signal. Defaults to ""
            set_current: if True, set the added signal as current

        Returns:
            True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """

    @abc.abstractmethod
    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        zunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title: Image title
            data: Image data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            zunit: Z unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            zlabel: Z label. Defaults to ""
            group_id: group id in which to add the image. Defaults to ""
            set_current: if True, set the added image as current

        Returns:
            True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """

    @abc.abstractmethod
    def add_object(
        self, obj: SignalObj | ImageObj, group_id: str = "", set_current: bool = True
    ) -> None:
        """Add object to DataLab.

        Args:
            obj: Signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current

        Returns:
            True if object was added successfully, False otherwise
        """

    @abc.abstractmethod
    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """

    @abc.abstractmethod
    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """

    @abc.abstractmethod
    def select_objects(
        self,
        selection: list[int | str],
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """

    @abc.abstractmethod
    def select_groups(
        self, selection: list[int | str] | None = None, panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """

    @abc.abstractmethod
    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """

    @abc.abstractmethod
    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Groups titles, lists of inner objects uuids and titles
        """

    @abc.abstractmethod
    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

        Returns:
            List of object titles

        Raises:
            ValueError: if panel not found
        """

    @abc.abstractmethod
    def get_object(
        self, nb_id_title: int | str | None = None, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """

    @abc.abstractmethod
    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            List of object uuids

        Raises:
            ValueError: if panel not found
        """

    @abc.abstractmethod
    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """

    @abc.abstractmethod
    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
            refresh_plot: refresh plot. Defaults to True.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
        """

    @abc.abstractmethod
    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title: Label title. Defaults to None.
             If None, the title is the object title.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
        """

    @abc.abstractmethod
    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (run
             current macro, or does nothing if there is no macro).
        """

    @abc.abstractmethod
    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (stop
             current macro, or does nothing if there is no macro).
        """

    @abc.abstractmethod
    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """

    @abc.abstractmethod
    def calc(self, name: str, param: gds.DataSet | None = None) -> gds.DataSet:
        """Call computation feature ``name``

        .. note::

            This calls either the processor's ``compute_<name>`` method (if it exists),
            or the processor's ``<name>`` computation feature (if it is registered,
            using the ``run_feature`` method).
            It looks for the function in all panels, starting with the current one.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """


class BaseProxy(AbstractDLControl, metaclass=abc.ABCMeta):
    """Common base class for DataLab proxies

    Args:
        datalab: DLMainWindow instance or ServerProxy instance. If None, then the proxy
         implementation will have to set it later (e.g. see RemoteClient).
    """

    def __init__(self, datalab: DLMainWindow | ServerProxy | None = None) -> None:
        self._datalab = datalab

    def get_version(self) -> str:
        """Return DataLab public version.

        Returns:
            DataLab version
        """
        return self._datalab.get_version()

    def close_application(self) -> None:
        """Close DataLab application"""
        self._datalab.close_application()

    def raise_window(self) -> None:
        """Raise DataLab window"""
        self._datalab.raise_window()

    def get_current_panel(self) -> str:
        """Return current panel name.

        Returns:
            Panel name (valid values: "signal", "image", "macro"))
        """
        return self._datalab.get_current_panel()

    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel: Panel name (valid values: "signal", "image", "macro"))
        """
        self._datalab.set_current_panel(panel)

    def reset_all(self) -> None:
        """Reset all application data"""
        self._datalab.reset_all()

    def remove_object(self, force: bool = False) -> None:
        """Remove current object from current panel.

        Args:
            force: if True, remove object without confirmation. Defaults to False.
        """
        self._datalab.remove_object(force)

    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh state.

        Args:
            state: Auto refresh state
        """
        self._datalab.toggle_auto_refresh(state)

    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles state.

        Args:
            state: Show titles state
        """
        self._datalab.toggle_show_titles(state)

    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename: HDF5 file name
        """
        self._datalab.save_to_h5_file(filename)

    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files: List of HDF5 files to open. Defaults to None.
            import_all: Import all objects from HDF5 files. Defaults to None.
            reset_all: Reset all application data. Defaults to None.
        """
        self._datalab.open_h5_files(h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename: HDF5 file name
            reset_all: Reset all application data. Defaults to None.
        """
        self._datalab.import_h5_file(filename, reset_all)

    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images).

        Args:
            filenames: list of file names
        """
        self._datalab.load_from_files(filenames)

    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """
        self._datalab.load_from_directory(path)

    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """
        return self._datalab.get_sel_object_uuids(include_groups)

    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """
        self._datalab.add_group(title, panel, select)

    def select_objects(
        self,
        selection: list[int | str],
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """
        self._datalab.select_objects(selection, panel)

    def select_groups(
        self, selection: list[int | str] | None = None, panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """
        self._datalab.select_groups(selection, panel)

    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """
        self._datalab.delete_metadata(refresh_plot, keep_roi)

    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Tuple: groups titles, lists of inner objects uuids and titles
        """
        return self._datalab.get_group_titles_with_object_info()

    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

        Returns:
            List of object titles

        Raises:
            ValueError: if panel not found
        """
        return self._datalab.get_object_titles(panel)

    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            List of object uuids

        Raises:
            ValueError: if panel not found
        """
        return self._datalab.get_object_uuids(panel, group)

    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title: Label title. Defaults to None.
             If None, the title is the object title.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
        """
        self._datalab.add_label_with_title(title, panel)

    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
            number_or_title: Macro number, or macro title.
             Defaults to None (current macro).

        Raises:
            ValueError: if macro not found
        """
        self._datalab.run_macro(number_or_title)

    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number_or_title: Macro number, or macro title.
             Defaults to None (current macro).

        Raises:
            ValueError: if macro not found
        """
        self._datalab.stop_macro(number_or_title)

    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """
        return self._datalab.import_macro_from_file(filename)
