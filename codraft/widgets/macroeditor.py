# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing CodraFT Macro editor widget
"""

import os.path as osp

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action, create_toolbutton
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

    def name_changed(self, name):
        """Macro name has been changed"""
        index = self.tabwidget.indexOf(self.editor)
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
        self.tabBarDoubleClicked.connect(self.rename_macro)
        self.tabCloseRequested.connect(self.remove_macro)

        self.__macros = {}

        # TODO: Add action "Import from file..."
        # TODO: Add action "Export to file..."

        self.setup_actions()

    def setup_actions(self):
        """Setup macro menu actions"""
        run_act = create_action(
            self,
            _("Run macro"),
            icon=get_icon("libre-camera-flash-on.svg"),
            triggered=self.run_macro,
        )
        stp_act = create_action(
            self,
            _("Stop macro"),
            icon=get_icon("libre-camera-flash-off.svg"),
            triggered=self.stop_macro,
        )
        stp_act.setDisabled(True)
        add_act = create_action(
            self,
            _("New macro"),
            icon=get_icon("libre-gui-add.svg"),
            triggered=self.add_macro,
        )
        ren_act = create_action(
            self,
            _("Rename macro"),
            icon=get_icon("libre-gui-pencil.svg"),
            triggered=self.rename_macro,
        )
        exp_act = create_action(
            self,
            _("Export macro to file"),
            icon=get_icon("libre-gui-export.svg"),
            triggered=self.export_macro_to_file,
        )
        imp_act = create_action(
            self,
            _("Import macro from file"),
            icon=get_icon("libre-gui-import.svg"),
            triggered=self.import_macro_from_file,
        )
        rem_act = create_action(
            self,
            _("Remove macro"),
            icon=get_icon("libre-gui-action-delete.svg"),
            triggered=self.remove_macro,
        )
        actions = (
            run_act,
            stp_act,
            None,
            add_act,
            ren_act,
            exp_act,
            imp_act,
            None,
            rem_act,
        )
        menu_button = QW.QPushButton(get_icon("libre-gui-menu.svg"), "", self)
        menu_button.setFlat(True)
        menu = QW.QMenu()
        menu_button.setMenu(menu)
        add_actions(menu, actions)
        self.setCornerWidget(menu_button)
        self.add_macro(rename=False)

    def run_macro(self):
        """Run current macro"""
        # XXX: Macros should be executed in a separate process: access to CodraFT
        # is provided by the 'remote_controlling' feature (see corresponding branch).
        # So, macros should be Python scripts similar to "remoteclient_test.py".
        # Connection to the XML-RPC server should be simplified (to a single line).

    def stop_macro(self):
        """Stop current macro"""

    def add_macro(self, name: str = None, rename: bool = True):
        """Add macro, optionally with name"""
        macro = Macro(self, name)
        index = self.addTab(macro.editor, macro.objectName())
        self.__macros[id(macro)] = macro
        if rename:
            self.rename_macro(index)

    def get_macro(self, index: int):
        """Return macro at index"""
        for macro in self.__macros.values():
            if self.widget(index) is macro.editor:
                return macro

    def rename_macro(self, index: int = None):
        """Rename macro"""
        if index is None:
            index = self.currentIndex()
        macro = self.get_macro(index)
        name, valid = QW.QInputDialog.getText(
            self,
            _("Rename"),
            _("New title:"),
            QW.QLineEdit.Normal,
            macro.objectName(),
        )
        if valid:
            macro.setObjectName(name)
            self.setCurrentIndex(index)

    def export_macro_to_file(self):
        """Export macro to file"""

    def import_macro_from_file(self):
        """Import macro from file"""

    def remove_macro(self, index: int = None):
        """Remove macro"""
        if index is None:
            index = self.currentIndex()
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
