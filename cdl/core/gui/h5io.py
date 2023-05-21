# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab HDF5 open/save module
"""

from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.core.io.h5 import H5Importer
from cdl.core.io.native import NativeH5Reader, NativeH5Writer
from cdl.core.model.signal import SignalParam
from cdl.env import execenv
from cdl.utils.qthelpers import create_progress_bar, exec_dialog, qt_try_loadsave_file
from cdl.widgets.h5browser import H5BrowserDialog

if TYPE_CHECKING:
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.io.h5.common import BaseNode


class H5InputOutput:
    """Object handling HDF5 file open/save
    into/from DataLab data model/main window"""

    def __init__(self, mainwindow: CDLMainWindow) -> None:
        self.mainwindow = mainwindow
        self.h5browser: H5BrowserDialog = None
        self.uint32_wng: bool = None

    @staticmethod
    def __progbartitle(fname: str) -> str:
        """Return progress bar title"""
        return _("Loading data from %s...") % osp.basename(fname)

    def save_file(self, filename: str) -> None:
        """Save all signals and images from DataLab model into a HDF5 file"""
        writer = NativeH5Writer(filename)
        for panel in self.mainwindow.panels:
            panel.serialize_to_hdf5(writer)
        writer.close()

    def open_file(self, filename: str, import_all: bool, reset_all: bool) -> None:
        """Open HDF5 file"""
        progress = None
        try:
            reader = NativeH5Reader(filename)
            if reset_all:
                self.mainwindow.reset_all()
            with create_progress_bar(
                self.mainwindow, self.__progbartitle(filename), 2
            ) as progress:
                for idx, panel in enumerate(self.mainwindow.panels):
                    progress.setValue(idx + 1)
                    QW.QApplication.processEvents()
                    panel.deserialize_from_hdf5(reader)
                    if progress.wasCanceled():
                        break
            reader.close()
        except KeyError:
            if progress is not None:
                # KeyError was encoutered when deserializing datasets (DataLab data
                # model is not compatible with this version)
                progress.close()
            self.import_file(filename, import_all, reset_all)

    def __add_object_from_node(self, node: BaseNode) -> None:
        """Add DataLab object from h5 node"""
        obj = node.get_native_object()
        self.uint32_wng = self.uint32_wng or node.uint32_wng
        if isinstance(obj, SignalParam):
            self.mainwindow.signalpanel.add_object(obj)
        else:
            self.mainwindow.imagepanel.add_object(obj)

    def __eventually_show_warnings(self) -> None:
        """Eventually show warnings after everything is imported"""
        if self.uint32_wng:
            QW.QMessageBox.warning(
                self.mainwindow, _("Warning"), _("Clipping uint32 data to int32.")
            )

    def import_file(self, filename: str, import_all: bool, reset_all: bool) -> None:
        """Import HDF5 file"""
        if self.h5browser is None:
            self.h5browser = H5BrowserDialog(self.mainwindow)

        with qt_try_loadsave_file(self.mainwindow, filename, "load"):
            self.h5browser.setup(filename)
            if execenv.unattended:
                # Unattended mode: import all datasets (for testing)
                import_all = True
            if not import_all and not exec_dialog(self.h5browser):
                self.h5browser.cleanup()
                return
            if import_all:
                nodes = self.h5browser.get_all_nodes()
            else:
                nodes = self.h5browser.get_nodes()
            if nodes is None:
                self.h5browser.cleanup()
                return
            if reset_all:
                self.mainwindow.reset_all()
            with create_progress_bar(
                self.mainwindow, self.__progbartitle(filename), len(nodes)
            ) as progress:
                self.uint32_wng = False
                for idx, node in enumerate(nodes):
                    progress.setValue(idx + 1)
                    QW.QApplication.processEvents()
                    if progress.wasCanceled():
                        break
                    self.__add_object_from_node(node)
            self.h5browser.cleanup()
            self.__eventually_show_warnings()

    def import_dataset_from_file(self, filename: str, dsetname: str) -> None:
        """Import dataset from HDF5 file"""
        h5importer = H5Importer(filename)
        try:
            node = h5importer.get(dsetname)
            self.uint32_wng = False
            self.__add_object_from_node(node)
            self.__eventually_show_warnings()
        except KeyError as exc:
            raise KeyError(f"Dataset not found: {dsetname}") from exc
        h5importer.close()
