# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing CodraFT Macro editor widget
"""

import os.path as osp

from guidata.configtools import get_icon
from guidata.qthelpers import create_toolbutton
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from codraft.config import _

UNTITLED_NB = 0


class Macro(QC.QObject):
    """Object representing a macro: editor, path, open/save actions, etc."""

    MACRO_SAMPLE_PATH = osp.join(osp.dirname(__file__), "macrosample.py")

    def __init__(self, tabwidget: QW.QTabWidget, name: str = None):
        super().__init__()
        self.tabwidget = tabwidget
        self.setObjectName(self.get_untitled_title() if name is None else name)
        self.editor = CodeEditor(language="python")
        self.editor.set_text_from_file(self.MACRO_SAMPLE_PATH)
        self.objectNameChanged.connect(self.name_changed)
        self.tabwidget.addTab(self.editor, self.objectName())

    def name_changed(self, name):
        """Macro name has been changed"""
        index = self.tabwidget.indexOf(self)
        self.tabwidget.setTabText(index, name)

    def get_untitled_title(self):
        """Increment untitled number and return untitled macro title"""
        global UNTITLED_NB
        UNTITLED_NB += 1
        return f"untitled{UNTITLED_NB:2d}"


class MacroEditorWidget(QW.QTabWidget):
    """Macro editor widget"""

    def __init__(self, parent: QW.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle(_("Macro editor"))
        self.setWindowIcon(get_icon("libre-gui-cogs.svg"))
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self.remove_macro)
        # TODO: Add a menu instead of the "add_button" (icon: libre-gui-menu.svg)
        # TODO: Add action "Run" in menu
        # TODO: Add action "Import from file..."
        # TODO: Add action "Export to file..."
        # TODO: Add action "Remove"
        add_button = create_toolbutton(
            self,
            icon=get_icon("libre-gui-add.svg"),
            triggered=self.add_macro,
        )
        self.setCornerWidget(add_button)
        self.add_macro()

    def add_macro(self, name: str = None):
        """Add macro, optionally with name"""
        macro = Macro(self, name)

    def remove_macro(self, index: int):
        """Remove macro"""
        txt = "<br>".join(
            [
                _(
                    "When closed, the macro is <u>permanently destroyed</u>, "
                    "unless it has been exported first."
                ),
                "",
                _("Do you want to continue?"),
            ]
        )
        btns = QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No
        choice = QW.QMessageBox.warning(self, self.windowTitle(), txt, btns)
        if choice == QW.QMessageBox.StandardButton.Yes:
            self.removeTab(index)
            if self.count() == 0:
                self.add_macro()
