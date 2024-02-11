# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Text Import Wizard
--------------------------

The DataLab Import Wizard is a tool for importing data from a text source:

- Clipboard
- File (csv, txt, etc.)

It is based on the DataLab Wizard widget, which is a widget that guides the user
through a series of steps to complete a task. It is implemented as a series of
pages, each of which is a separate widget.
"""

from __future__ import annotations

import io
import os.path as osp
from typing import TYPE_CHECKING

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
import numpy as np
import qtpy.QtCore as QC
from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename

from cdl.config import Conf, _
from cdl.core.model.signal import CurveStyles
from cdl.obj import ImageObj, SignalObj, create_image, create_signal
from cdl.utils.qthelpers import create_progress_bar, save_restore_stds
from cdl.widgets.wizard import Wizard, WizardPage

if TYPE_CHECKING:
    from plotpy.items import CurveItem, MaskedImageItem
    from plotpy.plot import BasePlot
    from qtpy.QtWidgets import QWidget


class SourceWidget(QW.QGroupBox):
    """Widget to select text source: clipboard or file"""

    SIG_VALIDITY_CHANGED = QC.Signal(bool)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setTitle(_("Source"))
        layout = QW.QHBoxLayout()
        self.setLayout(layout)
        clipboard_button = QW.QRadioButton(_("Clipboard"))
        file_button = QW.QRadioButton(_("File:"))
        layout.addWidget(clipboard_button)
        layout.addWidget(file_button)
        file_button.setChecked(True)
        self.file_edit = QW.QLineEdit()
        self.file_edit.textChanged.connect(self.__text_changed)
        layout.addWidget(self.file_edit)
        self.browse_button = QW.QPushButton(_("Browse"))
        layout.addWidget(self.browse_button)
        file_button.toggled.connect(self.__update_path)
        self.browse_button.clicked.connect(self.__browse)

    def __text_changed(self, text: str) -> None:
        """Text changed"""
        self.SIG_VALIDITY_CHANGED.emit(osp.isfile(self.file_edit.text()))

    def __update_path(self, checked: bool) -> None:
        """Update the path"""
        self.file_edit.setEnabled(checked)
        self.browse_button.setEnabled(checked)
        if checked:
            self.file_edit.setFocus()
            is_valid = osp.isfile(self.file_edit.text())
        else:
            is_valid = bool(QW.QApplication.clipboard().text())
        self.SIG_VALIDITY_CHANGED.emit(is_valid)

    def __browse(self) -> None:
        """Browse for a file"""
        basedir = Conf.main.base_dir.get()
        filters = _("Text files (*.txt *.csv *.dat *.asc);;All files (*.*)")
        with save_restore_stds():
            path, _filter = getopenfilename(self, _("Select a file"), basedir, filters)
        if path:
            self.file_edit.setText(path)

    def get_source_path(self) -> str | None:
        """Return the selected source path, or None if clipboard is selected"""
        if self.file_edit.isEnabled():
            return self.file_edit.text()
        return None


class SourcePage(WizardPage):
    """Source page"""

    def __init__(self) -> None:
        super().__init__()
        self.__text = ""
        self.set_title(_("Source"))
        self.set_subtitle(_("Select the source of the data:"))
        self.source_widget = SourceWidget(self)
        self.source_widget.SIG_VALIDITY_CHANGED.connect(self.set_valid)
        self.add_to_layout(self.source_widget)
        self.add_stretch()
        self.set_valid(False)

    def get_source_text(self) -> str:
        """Return the source text"""
        return self.__text

    def validate_page(self) -> bool:
        """Validate the page"""
        self.__text = ""
        if self.source_widget.file_edit.isEnabled():
            path = self.source_widget.get_source_path()
            if path is not None and osp.isfile(path):
                try:
                    with open(path, "r") as file:
                        self.__text = file.read()
                except Exception:
                    return False
            else:
                return False
        else:
            self.__text = QW.QApplication.clipboard().text()
        return bool(self.__text)


class BaseImportParam(gds.DataSet):
    """Import parameters dataset"""

    delimiter_choice = gds.ChoiceItem(
        _("Delimiter"),
        [
            (",", ","),
            (";", ";"),
            ("\t", _("Tab")),
            (" ", _("Space")),
        ],
        help=_("Column delimiter character"),
    )
    comment_char = gds.StringItem(
        _("Comments"),
        default="#",
        help=_("Character that indicates a comment line"),
    ).set_pos(col=1)

    skip_rows = gds.IntItem(
        _("Rows to Skip"),
        default=0,
        help=_(
            "Number of rows to skip at the beginning of the file (including comments)"
        ),
    )
    max_rows = gds.IntItem(
        _("Maximum Number of Rows"),
        default=None,
        min=1,
        check=False,
        help=_("Maximum number of rows to import"),
    ).set_pos(col=1)
    transpose = gds.BoolItem(
        _("Transpose"),
        default=False,
        help=_("Transpose the data (swap rows and columns)"),
    )


class SignalImportParam(BaseImportParam):
    """Signal import parameters dataset"""

    VALID_DTYPES_STRLIST = SignalObj.get_valid_dtypenames()

    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(VALID_DTYPES_STRLIST, VALID_DTYPES_STRLIST)),
        help=_("Output signal data type."),
    ).set_pos(col=1)

    first_col_is_x = gds.BoolItem(
        _("First Column is X"),
        default=True,
        help=_(
            _(
                "First column contains the X values\n"
                "(ignored if there is only one column)"
            )
        ),
    )


class ImageImportParam(BaseImportParam):
    """Image import parameters dataset"""

    VALID_DTYPES_STRLIST = ImageObj.get_valid_dtypenames()

    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(VALID_DTYPES_STRLIST, VALID_DTYPES_STRLIST)),
        help=_("Output image data type."),
    ).set_pos(col=1)


class PreviewWidget(QW.QWidget):
    """Widget showing the raw data, the prefiltered data and a preview of the data"""

    def __init__(self, parent: QWidget, destination: str) -> None:
        super().__init__(parent)
        self.destination = destination
        main_layout = QW.QVBoxLayout()
        self.setLayout(main_layout)

        self.tabwidget = tw = QW.QTabWidget()
        self._raw_text_edit = self.create_editor()
        tw.addTab(
            self._raw_text_edit, get_icon("libre-gui-questions.svg"), _("Raw Data")
        )
        self._pre_text_edit = self.create_editor()
        tw.addTab(
            self._pre_text_edit,
            get_icon("libre-gui-file-document.svg"),
            _("Prefiltered Data"),
        )
        self._preview_table = QW.QTableWidget()
        self._preview_table.setFont(self._raw_text_edit.font())
        self._preview_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        tw.addTab(self._preview_table, get_icon("table.svg"), _("Preview"))
        main_layout.addWidget(tw)

    def create_editor(self) -> CodeEditor:
        """Create a code editor"""
        editor = CodeEditor()
        font = editor.font()
        font.setPointSize(font.pointSize() - 2)
        editor.setFont(font)
        editor.setReadOnly(True)
        editor.setWordWrapMode(QG.QTextOption.NoWrap)
        return editor

    def set_raw_data(self, data: str) -> None:
        """Set the raw data"""
        self._raw_text_edit.setPlainText(data)

    def set_prefiltered_data(self, data: str) -> None:
        """Set the prefiltered data"""
        self._pre_text_edit.setPlainText(data)

    def __clear_preview_table(self, enable: bool) -> None:
        """Clear the preview table"""
        self._preview_table.clear()
        self._preview_table.setEnabled(enable)
        idx = self.tabwidget.indexOf(self._preview_table)
        icon_name = "table.svg" if enable else "table_unavailable.svg"
        self.tabwidget.setTabIcon(idx, get_icon(icon_name))
        if enable:
            self.tabwidget.setCurrentIndex(idx)

    def set_preview_data(
        self, data: np.ndarray | None, first_col_is_x: bool | None
    ) -> None:
        """Set the preview data"""
        if data is None or len(data.shape) not in (1, 2) or data.size == 0:
            self.__clear_preview_table(False)
        else:
            self.__clear_preview_table(True)
            self._preview_table.setRowCount(len(data))
            self._preview_table.setColumnCount(len(data[0]))
            with create_progress_bar(
                self, _("Adding data to the preview table"), max_=len(data) - 1
            ) as progress:
                for i, row in enumerate(data):
                    progress.setValue(i)
                    if progress.wasCanceled():
                        break
                    for j, cell in enumerate(row):
                        self._preview_table.setItem(
                            i, j, QW.QTableWidgetItem(str(cell))
                        )
                    QW.QApplication.processEvents()
            # Add column headers, only for signals:
            if self.destination == "signal":
                assert first_col_is_x is not None
                if len(data.shape) == 1:
                    headers = ["Y"]
                elif first_col_is_x:
                    if len(data.shape) == 2:
                        headers = ["X", "Y"]
                    else:
                        headers = ["X"] + [f"Y{i+1}" for i in range(len(data[0]) - 1)]
                else:
                    headers = [f"Y{i+1}" for i in range(len(data[0]))]
                self._preview_table.setHorizontalHeaderLabels(headers)


def prefilter_data(raw_data: str, param: SignalImportParam | ImageImportParam) -> str:
    """Prefilter the data"""
    lines = raw_data.splitlines()
    # Remove the first `skip_rows` lines
    if param.skip_rows:
        lines = lines[param.skip_rows :]
    # Remove all lines starting with the comment character
    lines = [line for line in lines if not line.startswith(param.comment_char)]
    # Keep only the first `max_rows` lines
    if param.max_rows:
        lines = lines[: param.max_rows]
    return "\n".join(lines)


def str_to_array(
    raw_data: str, param: SignalImportParam | ImageImportParam
) -> np.ndarray | None:
    """Convert raw data to array"""
    if not raw_data:
        return None
    delimiter = param.delimiter_choice or param.delimiter_custom
    file_obj = io.StringIO(raw_data)
    try:
        data = np.loadtxt(
            file_obj,
            delimiter=delimiter,
            unpack=param.transpose,
        )
    except Exception:
        return None
    return data.astype(param.dtype_str)


class DataPreviewPage(WizardPage):
    """Data preview page

    Args:
        destination: Destination type ('signal' or 'image')
    """

    def __init__(self, source_page: SourcePage, destination: str) -> None:
        super().__init__()
        self.__quick_update = False
        self.source_page = source_page
        self.destination = destination
        self.__data: np.ndarray | None = None
        self.set_title(_("Data Preview"))
        self.set_subtitle(_("Preview and modify the import settings:"))

        self.param_widget = gdq.DataSetEditGroupBox(
            _("Import Parameters"),
            SignalImportParam if destination == "signal" else ImageImportParam,
        )
        self.param_widget.SIG_APPLY_BUTTON_CLICKED.connect(self.update_preview)
        self.param = self.param_widget.dataset
        self.add_to_layout(self.param_widget)

        self.preview_widget = PreviewWidget(self, destination)
        self.preview_widget.setSizePolicy(
            QW.QSizePolicy(QW.QSizePolicy.Expanding, QW.QSizePolicy.Expanding)
        )
        self.add_to_layout(self.preview_widget)

    def get_data(self) -> np.ndarray | None:
        """Return the data"""
        return self.__data

    def update_preview(self) -> None:
        """Update the preview"""
        # Raw data
        raw_data = self.source_page.get_source_text()
        self.preview_widget.set_raw_data(raw_data)
        # Prefiltered data
        pre_data = prefilter_data(raw_data, self.param)
        self.preview_widget.set_prefiltered_data(pre_data)
        # Preview
        data = str_to_array(pre_data, self.param)
        first_col_is_x = None
        if isinstance(self.param, SignalImportParam):
            first_col_is_x = self.param.first_col_is_x
        if not self.__quick_update:
            self.preview_widget.set_preview_data(data, first_col_is_x=first_col_is_x)
        self.__data = data
        self.set_valid(data is not None)

    def initialize_page(self) -> None:
        """Initialize the page"""
        super().initialize_page()
        self.param_widget.get()
        self.update_preview()

    def validate_page(self) -> bool:
        """Validate the page"""
        self.__quick_update = True
        self.param_widget.set()
        self.__quick_update = False
        if self.destination == "signal":
            nb_sig = len(self.__data.T)
            if self.param.first_col_is_x:
                nb_sig -= 1
            if (
                nb_sig > 20
                and QW.QMessageBox.warning(
                    self,
                    _("Warning"),
                    _(
                        "The number of signals to import is very high (%d).\n"
                        "This may be an error.\n\n"
                        "Are you sure you want to continue?"
                    )
                    % nb_sig,
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                )
                == QW.QMessageBox.No
            ):
                return False
        return super().validate_page()


class GraphicalRepresentationPage(WizardPage):
    """Graphical representation page"""

    def __init__(self, data_page: DataPreviewPage, destination: str) -> None:
        super().__init__()
        self.__objitmlist: list[
            tuple[SignalObj | ImageObj, CurveItem | MaskedImageItem]
        ] = []
        self.set_title(_("Graphical Representation"))
        self.set_subtitle(_("This is the final page."))
        self.data_page = data_page
        self.destination = destination
        layout = QW.QVBoxLayout()
        label = QW.QLabel(_("Graphical representation of the imported data:"))
        layout.addWidget(label)
        plot_type = "curve" if destination == "signal" else "image"
        self.plot_widget = PlotWidget(
            self,
            toolbar=True,
            options=PlotOptions(
                type=plot_type,
                show_itemlist=True,
                show_contrast=True,
            ),
        )
        plot = self.plot_widget.get_plot()
        plot.SIG_ITEMS_CHANGED.connect(self.items_changed)
        layout.addWidget(self.plot_widget)
        instruction = QW.QLabel(
            _("Unselect the %s that you do not want to import.")
            % (_("signals") if destination == "signal" else _("images"))
        )
        layout.addWidget(instruction)
        self.add_to_layout(layout)

    def items_changed(self, _plot: BasePlot) -> None:
        """Item selection changed"""
        objs = self.get_objs()
        self.set_valid(len(objs) > 0)

    def get_objs(self) -> list[SignalObj | ImageObj]:
        """Return the objects"""
        return [obj for obj, item in self.__objitmlist if item.isVisible()]

    def initialize_page(self) -> None:
        """Initialize the page"""
        data = self.data_page.get_data()
        param = self.data_page.param
        assert data is not None
        plot = self.plot_widget.get_plot()
        plot.del_all_items()
        if self.destination == "signal":
            CurveStyles.reset_styles()
            xydata = data.T
            x = np.arange(len(xydata[0]))
            if len(xydata) == 1:
                obj = create_signal("", x=x, y=xydata[0])
                item = obj.make_item()
                plot.add_item(item)
                self.__objitmlist = [(obj, item)]
            else:
                if param.first_col_is_x:
                    x = xydata[0]
                self.__objitmlist = []
                with create_progress_bar(
                    self, _("Adding data to the plot"), max_=len(xydata) - 1
                ) as progress:
                    for ycol in range(1 if param.first_col_is_x else 0, len(xydata)):
                        progress.setValue(ycol - 1)
                        if progress.wasCanceled():
                            break
                        yidx = ycol if param.first_col_is_x else ycol - 1
                        obj = create_signal("", x=x, y=xydata[yidx])
                        item = obj.make_item()
                        plot.add_item(item)
                        self.__objitmlist.append((obj, item))
                        QW.QApplication.processEvents()
        else:
            obj = create_image("", data)
            item = obj.make_item()
            plot.add_item(item)
            self.__objitmlist = [(obj, item)]
            plot.set_active_item(item)
            item.unselect()
        plot.do_autoscale()
        self.items_changed(plot)
        return super().initialize_page()

    def validate_page(self) -> bool:
        """Validate the page"""
        objs = self.get_objs()
        if len(objs) == 0:
            return False
        return super().validate_page()


class TextImportWizard(Wizard):
    """Text data import wizard

    Args:
        destination: Destination type ('signal' or 'image')
    """

    def __init__(self, destination: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        assert destination in ("signal", "image")
        self.setWindowTitle(_("DataLab Import Wizard"))
        self.source_page = SourcePage()
        self.add_page(self.source_page)
        self.data_page = DataPreviewPage(self.source_page, destination)
        self.add_page(self.data_page)
        self.plot_page = GraphicalRepresentationPage(self.data_page, destination)
        self.add_page(self.plot_page, last_page=True)

    def get_objs(self) -> list[SignalObj | ImageObj]:
        """Return the objects"""
        return self.plot_page.get_objs()
