# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Wizard Widget
---------------------

The DataLab Wizard is a widget that guides the user through a series of steps
to complete a task. It is implemented as a series of pages, each of which is
a separate widget.

The `Wizard` class is the main widget that contains the pages. The `WizardPage`
class is the base class for the pages.

This module is strongly inspired from Qt's `QWizard` and `QWizardPage` classes.

.. note::

    The only motivation for reimplementing the wizard widget is to
    support complete styling with `QPalette` and `QStyle` (e.g. `guidata`'s
    dark mode is not supported on Windows).
"""

from __future__ import annotations

from guidata.configtools import get_icon
from PyQt5.QtWidgets import QWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _


class WizardPage(QW.QWidget):
    """Wizard page base class

    We create our own wizard page class instead of using QWizardPage because
    the latter does not support complete styling with `QPalette` and `QStyle`
    (e.g. `guidata`'s dark mode is not supported on Windows).

    This class reimplements the `QWizardPage` features.

    """

    SIG_INITIALIZE_PAGE = QC.Signal()
    SIG_VALID_STATE_CHANGED = QC.Signal()

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.__is_valid: bool = True
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
        # self._main_layout.addStretch()
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

    def set_valid(self, is_valid: bool) -> None:
        """Set the page as valid"""
        self.__is_valid = is_valid
        self.SIG_VALID_STATE_CHANGED.emit()

    def is_valid(self) -> bool:
        """Return whether the page is valid"""
        return self.__is_valid

    def add_to_layout(self, layout: QW.QLayout | QW.QWidget) -> None:
        """Add a layout to the user layout"""
        if isinstance(layout, QW.QWidget):
            self._user_layout.addWidget(layout)
        else:
            self._user_layout.addLayout(layout)

    def add_stretch(self) -> None:
        """Add a stretch to the user layout"""
        self._user_layout.addStretch()

    def initialize_page(self) -> None:
        """Initialize the page"""
        self.SIG_INITIALIZE_PAGE.emit()

    def validate_page(self) -> bool:
        """Validate the page"""
        return self.is_valid()


class Wizard(QW.QDialog):
    """Wizard base class"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowIcon(get_icon("DataLab.svg"))

        _main_layout = QW.QVBoxLayout()
        self.setLayout(_main_layout)

        self._pages_widget = QW.QStackedWidget()
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

        self.setSizePolicy(
            QW.QSizePolicy(QW.QSizePolicy.Minimum, QW.QSizePolicy.Minimum)
        )

    def add_page(self, page: WizardPage, last_page: bool = False) -> None:
        """Add a page to the wizard"""
        page.set_wizard(self)
        page.SIG_INITIALIZE_PAGE.connect(self.__update_button_states)
        page.SIG_VALID_STATE_CHANGED.connect(self.__update_button_states)
        self._pages_widget.addWidget(page)
        if last_page:
            self._pages_widget.widget(0).initialize_page()

    def __update_button_states(self, index: int | None = None) -> None:
        """Update button states"""
        if index is None:
            index = self._pages_widget.currentIndex()
        self._back_btn.setEnabled(index > 0)
        not_last_page = index < self._pages_widget.count() - 1
        page_valid = self._pages_widget.currentWidget().is_valid()
        self._next_btn.setEnabled(not_last_page and page_valid)
        is_last_page = index == self._pages_widget.count() - 1
        self._finish_btn.setEnabled(is_last_page and page_valid)

    def go_to_previous_page(self) -> None:
        """Go to the previous page"""
        self._pages_widget.setCurrentIndex(self._pages_widget.currentIndex() - 1)
        self.__update_button_states()

    def go_to_next_page(self) -> None:
        """Go to the next page"""
        if self.validate_page():
            self._pages_widget.setCurrentIndex(self._pages_widget.currentIndex() + 1)
            self.initialize_page()

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


class ExamplePage1(WizardPage):
    """Example wizard page 1"""

    def __init__(self) -> None:
        super().__init__()
        self.set_title(_("Welcome to the Example Wizard"))
        self.set_subtitle(
            _("This wizard will guide you through the process of importing data.")
        )

    def initialize_page(self) -> None:
        """Initialize the page"""
        print("ExamplePage1 initialized")
        super().initialize_page()

    def validate_page(self) -> bool:
        """Validate the page"""
        print("ExamplePage1 validated")
        return super().validate_page()


class ExamplePage2(WizardPage):
    """Example wizard page 2"""

    def __init__(self) -> None:
        super().__init__()
        self.set_title(_("Select the Source of the Data"))
        self.set_subtitle(
            _("Select the source of the data to be imported (clipboard or file).")
        )
        self._clipboard_rb = QW.QRadioButton(_("Clipboard"))
        self._file_rb = QW.QRadioButton(_("File"))
        self._file_rb.toggled.connect(self.file_rb_toggled)
        self._file_le = QW.QLineEdit()
        self._file_btn = QW.QPushButton(_("Browse..."))
        self._file_btn.clicked.connect(self.browse_file)
        self.add_to_layout(self._clipboard_rb)
        self.add_to_layout(self._file_rb)
        self.add_to_layout(self._file_le)
        self.add_to_layout(self._file_btn)

    def initialize_page(self) -> None:
        """Initialize the page"""
        print("ExamplePage2 initialized")
        super().initialize_page()

    def file_rb_toggled(self, checked: bool) -> None:
        """File radio button toggled"""
        self._file_le.setEnabled(checked)
        self._file_btn.setEnabled(checked)

    def browse_file(self) -> None:
        """Browse file"""
        file_name, _ = QW.QFileDialog.getOpenFileName(
            self,
            _("Select the File to Import"),
            "",
            _("CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"),
        )
        if file_name:
            self._file_le.setText(file_name)

    def validate_page(self) -> bool:
        """Validate the page"""
        if self._file_rb.isChecked() and not self._file_le.text():
            QW.QMessageBox.critical(
                self,
                _("Error"),
                _("Please select the file to import."),
                QW.QMessageBox.Ok,
            )
            return False
        return True


class ExampleWizard(Wizard):
    """Example wizard widget"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_("Example Wizard"))
        self.add_page(ExamplePage1())
        self.add_page(ExamplePage2(), last_page=True)


def test_example_wizard():
    """Test the import wizard"""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        wizard = ExampleWizard()
        wizard.exec()


if __name__ == "__main__":
    test_example_wizard()
