# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 I/O
========

The :mod:`datalab.gui.h5io` module provides the HDF5 file open/save into/from
DataLab data model/main window.

.. autoclass:: H5InputOutput
"""

from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING, Any

import h5py
from datalab_capsule.hdf5 import ProvenanceFormatError, has_block
from datalab_capsule.recorder import Recorder
from guidata.qthelpers import exec_dialog
from qtpy import QtWidgets as QW
from sigima.objects import SignalObj
from sigimax.widgets.h5browser import H5BrowserDialog

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.base import H5ImportBatch
from datalab.h5 import H5Importer
from datalab.h5.native import NativeH5Reader, atomic_native_writer
from datalab.utils.qthelpers import create_progress_bar, qt_try_loadsave_file

if TYPE_CHECKING:
    from datalab_capsule.ledger import Ledger
    from sigima.objects import ImageObj
    from sigimax.h5.common import BaseNode

    from datalab.gui.main import DLMainWindow


def workspace_has_provenance(filename: str) -> bool:
    """Return True if a HDF5 file holds a provenance block (nothing is loaded).

    Args:
        filename: HDF5 filename (unreadable files give False).
    """
    try:
        with h5py.File(filename, "r") as h5file:
            return has_block(h5file)
    except (OSError, ValueError):
        return False


def append_provenance_notice() -> str:
    """Return the notice shown before appending a workspace holding provenance."""
    return _(
        "The data will be added, but their earlier provenance will not be "
        "imported. Open the file replacing the current workspace to keep it."
    )


class H5InputOutput:
    """Object handling HDF5 file open/save into/from DataLab data model/main window

    Args:
        mainwindow: Main window
    """

    def __init__(self, mainwindow: DLMainWindow) -> None:
        self.mainwindow = mainwindow
        #: True when the user was already told, for the current opening, that
        #: the provenance of appended workspaces is not imported (no repeat).
        self.append_informed = False
        #: Files whose appending the user cancelled (see :meth:`__confirm_append`)
        self.cancelled_files: list[str] = []

    @staticmethod
    def __progbartitle(fname: str) -> str:
        """Return progress bar title"""
        return _("Loading data from %s...") % osp.basename(fname)

    def save_file(self, filename: str) -> None:
        """Save all signals and images from DataLab model into a HDF5 file

        The file is replaced only once completely written: if saving fails, the
        previous file (if any) is left untouched.
        """
        with atomic_native_writer(filename) as writer:
            for panel in self.mainwindow.panels:
                panel.serialize_to_hdf5(writer)
            self.mainwindow.provenance.save(writer.h5)

    @staticmethod
    def __read_provenance(
        reader: NativeH5Reader,
    ) -> tuple[Ledger, dict[str, str], dict[str, Any] | None] | None:
        """Read and validate the provenance block before anything is replaced.

        Returns:
            ``(ledger, state_status, capsule_origin)``, or None without block.

        Raises:
            ProvenanceFormatError: If the block is invalid (never a KeyError, which
             would turn the file into a generic HDF5 import).
        """
        try:
            return Recorder.read(reader.h5)
        except ProvenanceFormatError as exc:
            reader.close()
            raise ProvenanceFormatError(
                _("The provenance data of this workspace are invalid: %s") % exc
            ) from exc

    def __data_objects(self) -> list[SignalObj | ImageObj]:
        """Return the objects of the signal and image panels."""
        return [
            obj
            for panel in (self.mainwindow.signalpanel, self.mainwindow.imagepanel)
            for obj in panel.objmodel.get_all_objects()
        ]

    def __confirm_append(self) -> bool:
        """Warn, before anything changes, that appended provenance is ignored.

        Shown once per opening, unless the question about clearing the workspace
        already said so. Returns False if the user cancels.
        """
        if self.append_informed or execenv.unattended:
            return True
        box = QW.QMessageBox(
            QW.QMessageBox.Warning,
            _("Warning"),
            append_provenance_notice(),
            QW.QMessageBox.NoButton,
            self.mainwindow,
        )
        add = box.addButton(_("Add without provenance"), QW.QMessageBox.AcceptRole)
        box.addButton(QW.QMessageBox.Cancel)
        box.setDefaultButton(add)
        exec_dialog(box)
        if box.clickedButton() is not add:
            return False
        self.append_informed = True
        return True

    def open_file_headless(self, filename: str, reset_all: bool) -> bool:
        """Open native DataLab HDF5 file without any GUI elements.

        This method can be safely called from any thread (e.g., the console thread)
        as it does not create any Qt widgets or dialogs. When a file holding
        provenance is appended (*reset_all* False), its provenance is not
        imported and the notice ``datalab_capsule.recorder.APPEND_NOTICE`` is
        added to ``mainwindow.provenance.notices`` (and logged).

        Args:
            filename: HDF5 filename
            reset_all: Reset all application data before importing

        Returns:
            True if file was successfully opened as a native DataLab file,
            False if the file format is not compatible (KeyError was raised)

        Raises:
            ProvenanceFormatError: If the provenance block is invalid (raised
             before the workspace is replaced).
        """
        reader = None
        try:
            reader = NativeH5Reader(filename)
            block = self.__read_provenance(reader)
            saved_version = reader.saved_version
            if reset_all:
                self.mainwindow.reset_all()
            before = {id(obj) for obj in self.__data_objects()}
            batches: list[H5ImportBatch] = []
            for panel in self.mainwindow.panels:
                batch = panel.deserialize_from_hdf5(reader, reset_all)
                if isinstance(batch, H5ImportBatch):
                    batches.append(batch)
            H5ImportBatch.finalize_imports(batches, refresh_panels=True)
            appended = [o for o in self.__data_objects() if id(o) not in before]
            self.mainwindow.provenance.load(
                block, reset_all, saved_version, appended=appended
            )
            return True
        except KeyError:
            return False
        finally:
            if reader is not None:
                reader.close()

    def open_file(self, filename: str, import_all: bool, reset_all: bool) -> None:
        """Open HDF5 file

        When the file holds provenance and is appended to the current workspace,
        the user is warned before anything changes (see :meth:`__confirm_append`)
        and may cancel. A file that is not a DataLab workspace is imported as a
        generic HDF5 file.
        """
        if not self.__open_native(filename, reset_all):
            self.import_files([filename], import_all, reset_all)

    def open_workspace_file(self, filename: str) -> None:
        """Open a DataLab workspace file, replacing the workspace.

        Unlike :meth:`open_file`, a file that is not a DataLab workspace is never
        imported as a generic HDF5 file.

        Raises:
            ValueError: If the file is not a DataLab workspace.
            ProvenanceFormatError: If the provenance block is invalid.
        """
        if not self.__open_native(filename, True):
            raise ValueError(_("This file is not a DataLab workspace"))

    def __open_native(self, filename: str, reset_all: bool) -> bool:
        """Open a native DataLab HDF5 file (see :meth:`open_file`).

        Returns:
            False if the file is not a DataLab workspace (KeyError), else True
            (also when the user cancels appending it).
        """
        progress = None
        reader = None
        try:
            reader = NativeH5Reader(filename)
            block = self.__read_provenance(reader)
            saved_version = reader.saved_version
            if block is not None and not reset_all and not self.__confirm_append():
                self.cancelled_files.append(filename)
                return True
            if reset_all:
                self.mainwindow.reset_all()
            before = {id(obj) for obj in self.__data_objects()}
            batches: list[H5ImportBatch] = []
            with create_progress_bar(
                self.mainwindow, self.__progbartitle(filename), 2
            ) as progress:
                for idx, panel in enumerate(self.mainwindow.panels):
                    progress.setValue(idx + 1)
                    QW.QApplication.processEvents()
                    batch = panel.deserialize_from_hdf5(reader, reset_all)
                    if isinstance(batch, H5ImportBatch):
                        batches.append(batch)
                    if progress.wasCanceled():
                        break
            H5ImportBatch.finalize_imports(batches, refresh_panels=True)
            appended = [o for o in self.__data_objects() if id(o) not in before]
            self.mainwindow.provenance.load(
                block, reset_all, saved_version, appended=appended
            )
            return True
        except KeyError:
            if progress is not None:
                # KeyError was encoutered when deserializing datasets (DataLab data
                # model is not compatible with this version)
                progress.close()
            return False
        finally:
            if reader is not None:
                reader.close()

    def __add_object_from_node(self, node: BaseNode) -> None:
        """Add DataLab object from h5 node"""
        obj = node.get_native_object()
        if obj is None:
            return
        if isinstance(obj, SignalObj):
            self.mainwindow.signalpanel.add_object(obj)
        else:
            self.mainwindow.imagepanel.add_object(obj)

    def import_files(
        self, filenames: list[str], import_all: bool, reset_all: bool
    ) -> None:
        """Import HDF5 files"""
        h5browser = H5BrowserDialog(self.mainwindow)
        try:
            self.__import_files_from_browser(
                h5browser, filenames, import_all, reset_all
            )
        finally:
            # Close the HDF5 files, even if an import error is raised
            h5browser.cleanup()

    def __import_files_from_browser(
        self,
        h5browser: H5BrowserDialog,
        filenames: list[str],
        import_all: bool,
        reset_all: bool,
    ) -> None:
        """Open HDF5 files in browser, then import the selected datasets"""
        for filename in filenames:
            with qt_try_loadsave_file(self.mainwindow, filename, "load"):
                h5browser.open_file(filename)
        if h5browser.is_empty():
            if not execenv.unattended:
                QW.QMessageBox.warning(
                    self.mainwindow,
                    _("Warning"),
                    _("No supported data available in HDF5 file(s)."),
                )
            return
        if execenv.unattended:
            # Unattended mode: import all datasets (for testing)
            import_all = True
        if import_all or exec_dialog(h5browser):
            if import_all:
                nodes = h5browser.get_all_nodes()
            else:
                nodes = h5browser.get_nodes()
            if nodes is not None:
                if reset_all:
                    self.mainwindow.reset_all()
                errors: list[str] = []
                with qt_try_loadsave_file(self.mainwindow, "*.h5", "load"):
                    with create_progress_bar(self.mainwindow, "", len(nodes)) as prog:
                        for idx, node in enumerate(nodes):
                            prog.setLabelText(self.__progbartitle(node.h5file.filename))
                            prog.setValue(idx + 1)
                            QW.QApplication.processEvents()
                            if prog.wasCanceled():
                                break
                            try:
                                self.__add_object_from_node(node)
                            except Exception as exc:  # pylint: disable=broad-except
                                # Data of third-party files: import the other
                                # datasets, then report all errors at once
                                errors.append(f"{node.id}: {exc}")
                    self.__report_import_errors(errors)

    def __report_import_errors(self, errors: list[str]) -> None:
        """Report the datasets that could not be imported

        Args:
            errors: description of each error ("dataset: reason")

        Raises:
            ValueError: in unattended mode, if any dataset could not be imported
        """
        if not errors:
            return
        if execenv.unattended:
            raise ValueError("\n".join(errors))
        shown = errors[:10]
        if len(errors) > len(shown):
            shown.append(_("... and %d more") % (len(errors) - len(shown)))
        QW.QMessageBox.warning(
            self.mainwindow,
            _("Warning"),
            _("Some datasets could not be imported:") + "\n\n" + "\n".join(shown),
        )

    def import_dataset_from_file(self, filename: str, dsetname: str) -> None:
        """Import dataset from HDF5 file"""
        h5importer = H5Importer(filename)
        try:
            node = h5importer.get(dsetname)
            self.__add_object_from_node(node)
        except KeyError as exc:
            raise KeyError(f"Dataset not found: {dsetname}") from exc
        finally:
            h5importer.close()
