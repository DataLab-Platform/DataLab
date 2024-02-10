# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Import Wizard
---------------------

The DataLab Import Wizard is a tool for importing data from a text source:

- Clipboard
- File (csv, txt, etc.)

The wizard is designed to be used as a widget in DataLab application. It is
implemented as a QWizard with a series of pages that guide the user through
the import process.

The first page of the wizard is a welcome page that provides a brief
description of the wizard and the import process. The second page is a
source selection page that allows the user to select the source of the data
to be imported (a radio button group with options for clipboard or file, and
a file selection button that opens a file dialog, with a text box for the
selected file). The third page is a data preview page that displays a preview
of the data to be imported and allows the user to modify the import settings
(column delimiter, row delimiter, number of rows to skip, comment character,
transpose the data, and global NumPy data type). The fourth page is a column
selection page that allows the user to select the columns to be imported, and
if the data has to be imported as signals or as an image. The fifth page is a
a graphical representation of the data to be imported (it thus requires the
signal or image objects to be created, for using the `make_item` method to
create the associated plot items): this is the final page.
"""

from __future__ import annotations

import io
import os.path as osp

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
import numpy as np
from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from plotpy.plot import PlotOptions, PlotWidget
from PyQt5.QtWidgets import QWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.obj import ImageObj, SignalObj, create_image, create_signal


class WizardPage(QW.QWidget):
    """Wizard page base class

    We create our own wizard page class instead of using QWizardPage because
    the latter does not support complete styling with `QPalette` and `QStyle`
    (e.g. `guidata`'s dark mode is not supported on Windows).

    This class reimplements the `QWizardPage` features.

    """

    SIG_INITIALIZE_PAGE = QC.Signal()

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.wizard: Wizard | None = None
        self._main_layout = QW.QVBoxLayout()
        self._user_layout = QW.QVBoxLayout()
        self._title_label = QW.QLabel("")
        font = self._title_label.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        self._title_label.setFont(font)
        self._title_label.setStyleSheet("color: #1E90FF")
        horiz_line = QW.QFrame()
        horiz_line.setFrameShape(QW.QFrame.HLine)
        horiz_line.setFrameShadow(QW.QFrame.Sunken)
        self._subtitle_label = QW.QLabel("")
        self._subtitle_label.setWordWrap(True)
        self._main_layout.addWidget(self._title_label)
        self._main_layout.addWidget(self._subtitle_label)
        self._main_layout.addWidget(horiz_line)
        self._main_layout.addLayout(self._user_layout)
        self._main_layout.addStretch()
        self.setLayout(self._main_layout)

    def set_wizard(self, wizard: Wizard) -> None:
        """Set the wizard"""
        self.wizard = wizard

    def get_wizard(self) -> Wizard:
        """Return the wizard"""
        return self.wizard

    def set_title(self, title: str) -> None:
        """Set the title of the page"""
        self._title_label.setText(title)

    def set_subtitle(self, subtitle: str) -> None:
        """Set the subtitle of the page"""
        self._subtitle_label.setText(subtitle)

    def add_to_layout(self, layout: QW.QLayout | QW.QWidget) -> None:
        """Add a layout to the user layout"""
        if isinstance(layout, QW.QWidget):
            self._user_layout.addWidget(layout)
        else:
            self._user_layout.addLayout(layout)

    def initialize_page(self) -> None:
        """Initialize the page"""
        self.SIG_INITIALIZE_PAGE.emit()

    def validate_page(self) -> bool:
        """Validate the page"""
        return True


class Wizard(QW.QDialog):
    """Wizard base class"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowIcon(get_icon("DataLab.svg"))

        _main_layout = QW.QVBoxLayout()
        self.setLayout(_main_layout)

        self._pages_widget = QW.QStackedWidget()
        self._pages_widget.currentChanged.connect(self.current_page_changed)
        _main_layout.addWidget(self._pages_widget)

        btn_layout = QW.QHBoxLayout()
        self._back_btn = QW.QPushButton(_("Back"))
        self._back_btn.clicked.connect(self.go_to_previous_page)
        self._next_btn = QW.QPushButton(_("Next"))
        self._next_btn.clicked.connect(self.go_to_next_page)
        self._finish_btn = QW.QPushButton(_("Finish"))
        self._finish_btn.clicked.connect(self.accept)
        self._cancel_btn = QW.QPushButton(_("Cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._back_btn)
        btn_layout.addWidget(self._next_btn)
        btn_layout.addWidget(self._finish_btn)
        btn_layout.addWidget(self._cancel_btn)
        _main_layout.addLayout(btn_layout)

    def add_page(self, page: WizardPage, last_page: bool = False) -> None:
        """Add a page to the wizard"""
        page.set_wizard(self)
        page.SIG_INITIALIZE_PAGE.connect(self.__update_button_states)
        self._pages_widget.addWidget(page)
        if last_page:
            self._pages_widget.widget(0).initialize_page()

    def current_page_changed(self, index: int) -> None:
        """Current page changed"""
        self.initialize_page()

    def __update_button_states(self, index: int | None = None) -> None:
        """Update button states"""
        if index is None:
            index = self._pages_widget.currentIndex()
        self._back_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < self._pages_widget.count() - 1)
        self._finish_btn.setEnabled(index == self._pages_widget.count() - 1)

    def go_to_previous_page(self) -> None:
        """Go to the previous page"""
        self._pages_widget.setCurrentIndex(self._pages_widget.currentIndex() - 1)

    def go_to_next_page(self) -> None:
        """Go to the next page"""
        if self.validate_page():
            self._pages_widget.setCurrentIndex(self._pages_widget.currentIndex() + 1)

    def initialize_page(self) -> None:
        """Initialize the page"""
        self._pages_widget.currentWidget().initialize_page()

    def validate_page(self) -> bool:
        """Validate the page"""
        return self._pages_widget.currentWidget().validate_page()

    def accept(self) -> None:
        """Accept the wizard"""
        if self.validate_page():
            super().accept()


class BaseImportParam(gds.DataSet):
    """Import parameters dataset"""

    _prop = gds.GetAttrProp("source")
    source = gds.ChoiceItem(
        _("Source"),
        [("clipboard", _("Clipboard")), ("file", _("File"))],
        help=_("Source of the data to be imported."),
    ).set_prop("display", store=_prop)
    file_path = (
        gds.FileOpenItem(
            _("Path"),
            ("txt", "csv"),
            check=False,
            help=_("Path to the file to be imported."),
        )
        .set_prop("display", active=gds.FuncProp(_prop, lambda x: x == "file"))
        .set_pos(col=1)
    )

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


class ImageImportParam(BaseImportParam):
    """Image import parameters dataset"""

    VALID_DTYPES_STRLIST = ImageObj.get_valid_dtypenames()

    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(VALID_DTYPES_STRLIST, VALID_DTYPES_STRLIST)),
        help=_("Output image data type."),
    ).set_pos(col=1)


class PreviewWidget(QW.QWidget):
    """Widget showing the raw data in a first tab, and the data preview in a second tab"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        main_layout = QW.QVBoxLayout()
        self.setLayout(main_layout)

        tab_widget = QW.QTabWidget()
        self._raw_text_edit = self.create_editor()
        tab_widget.addTab(self._raw_text_edit, _("Raw Data"))
        self._pre_text_edit = self.create_editor()
        tab_widget.addTab(self._pre_text_edit, _("Prefiltered Data"))
        self._preview_table = QW.QTableWidget()
        self._preview_table.setFont(self._raw_text_edit.font())
        self._preview_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        tab_widget.addTab(self._preview_table, _("Preview"))
        main_layout.addWidget(tab_widget)

    def create_editor(self) -> CodeEditor:
        """Create a code editor"""
        editor = CodeEditor()
        font = editor.font()
        font.setPointSize(font.pointSize() - 2)
        editor.setFont(font)
        editor.setReadOnly(True)
        return editor

    def set_raw_data(self, data: str) -> None:
        """Set the raw data"""
        self._raw_text_edit.setPlainText(data)
        self._preview_table.clear()

    def set_prefiltered_data(self, data: str) -> None:
        """Set the prefiltered data"""
        self._pre_text_edit.setPlainText(data)

    def set_preview_data(self, data: np.ndarray | None) -> None:
        """Set the preview data"""
        self._preview_table.clear()
        if data is None or len(data.shape) not in (1, 2) or data.size == 0:
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            self._preview_table.setEnabled(False)
        else:
            self._preview_table.setRowCount(len(data))
            self._preview_table.setColumnCount(len(data[0]))
            for i, row in enumerate(data):
                for j, cell in enumerate(row):
                    self._preview_table.setItem(i, j, QW.QTableWidgetItem(str(cell)))
            self._preview_table.setEnabled(True)


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

    def __init__(self, destination: str) -> None:
        super().__init__()
        self.__data: np.ndarray | None = None
        self.set_title(_("Data Preview"))
        self.set_subtitle(_("Preview and modify the import settings:"))
        self.preview_widget = PreviewWidget()
        self.add_to_layout(self.preview_widget)
        self.param_widget = gdq.DataSetEditGroupBox(
            _("Import Parameters"),
            SignalImportParam if destination == "signal" else ImageImportParam,
        )
        self.param_widget.SIG_APPLY_BUTTON_CLICKED.connect(self.update_preview)
        self.param = self.param_widget.dataset
        self.add_to_layout(self.param_widget)

    def get_data(self) -> np.ndarray | None:
        """Return the data"""
        return self.__data

    def update_preview(self) -> None:
        """Update the preview"""
        # Raw data
        raw_data = ""
        if self.param.source == "file":
            if osp.isfile(self.param.file_path):
                try:
                    with open(self.param.file_path, "r") as file:
                        raw_data = file.read()
                except Exception:
                    pass
        elif self.param.source == "clipboard":
            raw_data = QW.QApplication.clipboard().text()
        self.preview_widget.set_raw_data(raw_data)
        # Prefiltered data
        pre_data = prefilter_data(raw_data, self.param)
        self.preview_widget.set_prefiltered_data(pre_data)
        data = str_to_array(pre_data, self.param)
        self.preview_widget.set_preview_data(data)
        self.__data = data
        self.wizard._next_btn.setEnabled(data is not None)

    def initialize_page(self) -> None:
        """Initialize the page"""
        super().initialize_page()
        self.param_widget.get()
        self.update_preview()

    def validate_page(self) -> bool:
        """Validate the page"""
        self.param_widget.set()
        return super().validate_page()


class GraphicalRepresentationPage(WizardPage):
    """Graphical representation page"""

    def __init__(self, data_page: DataPreviewPage, destination: str) -> None:
        super().__init__()
        self.__objs: list[SignalObj | ImageObj] = []
        self.set_title(_("Graphical Representation"))
        self.set_subtitle(_("This is the final page."))
        self.data_page = data_page
        self.destination = destination
        layout = QW.QVBoxLayout()
        label = QW.QLabel(_("Graphical representation of the imported data:"))
        layout.addWidget(label)
        plot_type = "curve" if destination == "signal" else "image"
        self.plot_widget = PlotWidget(
            self, toolbar=True, options=PlotOptions(type=plot_type)
        )
        layout.addWidget(self.plot_widget)
        self.add_to_layout(layout)

    def get_objs(self) -> list[SignalObj | ImageObj]:
        """Return the objects"""
        return self.__objs

    def initialize_page(self) -> None:
        """Initialize the page"""
        data = self.data_page.get_data()
        assert data is not None
        plot = self.plot_widget.get_plot()
        plot.del_all_items()
        if self.destination == "signal":
            xydata = data.T
            if len(xydata) == 1:
                obj = create_signal("", x=np.range(len(xydata[0])), y=xydata[0])
                self.__objs = [obj]
            else:
                self.__objs = []
                for ycol in range(1, len(xydata)):
                    obj = create_signal("", x=xydata[0], y=xydata[ycol])
                    self.__objs.append(obj)
        else:
            obj = create_image("", data)
            self.__objs = [obj]
        item = obj.make_item()
        plot.add_item(item)
        plot.do_autoscale()
        return super().initialize_page()


class ImportWizard(Wizard):
    """Import wizard widget

    Args:
        destination: Destination type ('signal' or 'image')
    """

    def __init__(self, destination: str) -> None:
        super().__init__()
        assert destination in ("signal", "image")
        self.setWindowTitle(_("DataLab Import Wizard"))
        self.data_page = DataPreviewPage(destination)
        self.add_page(self.data_page)
        self.plot_page = GraphicalRepresentationPage(self.data_page, destination)
        self.add_page(self.plot_page, last_page=True)

    def get_objs(self) -> list[SignalObj | ImageObj]:
        """Return the objects"""
        return self.plot_page.get_objs()
