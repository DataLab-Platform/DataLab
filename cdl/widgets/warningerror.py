# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing a warning/error message box
"""

import traceback

from guidata.config import CONF
from guidata.configtools import get_font
from guidata.qthelpers import exec_dialog, get_std_icon
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import Conf, _
from cdl.utils.misc import go_to_error


def insert_spaces(text: str, nbchars: int) -> str:
    """
    Inserts spaces regularly in a string, every nbchars characters, after certain
    characters (",", ";", "-", "+", "*", ")"), and keeps searching until detecting
    one of those characters.

    Args:
        text (str): The input string.
        nbchars (int): The number of characters after which a space should be inserted.

    Returns:
        str: The modified string with spaces inserted.
    """
    special_chars = (",", ";", "-", "+", "*", ")", "_")
    new_text = ""
    index = 0
    while index < len(text):
        if (
            index + nbchars < len(text)
            and text[index + nbchars] not in special_chars
            and not any(c in special_chars for c in text[index : index + nbchars + 1])
        ):
            new_text += text[index : index + nbchars]  # Append characters
            index += nbchars
        else:
            new_text += text[index : index + nbchars] + " "  # Insert space
            index += nbchars
    return new_text


class WarningErrorMessageBox(QW.QDialog):
    """Warning/Error message box

    Args:
        parent (QW.QWidget): parent widget
        category (str): message category ("error" or "warning")
        context (str | None): context. Defaults to None.
        message (str | None): message. Defaults to None.
        tip (str | None): tip. Defaults to None.
    """

    def __init__(
        self,
        parent: QW.QWidget,
        category: str,
        context: str = None,
        message: str = None,
        tip: str = None,
    ) -> None:
        super().__init__(parent)
        assert category in ("error", "warning")
        self.setWindowTitle(parent.window().objectName())

        self.shell = PythonShellWidget(self, read_only=True)
        self.shell.go_to_error.connect(go_to_error)
        font = get_font(CONF, "console")
        font.setPointSize(9)
        self.shell.set_font(font)
        message = traceback.format_exc() if message is None else message
        self.shell.insert_text(message, at_end=True, error=True)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok)
        bbox.accepted.connect(self.accept)
        if category == "warning":
            bbox.addButton(QW.QDialogButtonBox.Ignore).clicked.connect(self.ignore)

        layout = QW.QVBoxLayout()

        if category == "error":
            width, height = 725, 400
            icon = "MessageBoxCritical"
            tb_title = _("Error message")
            tb_text = _("The following traceback may help to understand the problem:")
        else:
            width, height = 725, 200
            icon = "MessageBoxWarning"
            tb_title = _("Warning message")
            tb_text = _("Please take into account the following warning message:")

        if context is not None:
            context = insert_spaces(context, 80)
            msgprefix = _("An error has occured during the following context:")
            text = "<br>".join([msgprefix, f"<b>{context}</b>"])
            ct_groupbox = QW.QGroupBox(_("Context"), self)
            ct_layout = QW.QHBoxLayout()
            ct_image_layout = QW.QVBoxLayout()
            ct_image = QW.QLabel()
            ct_image.setPixmap(get_std_icon(icon).pixmap(24, 24))
            ct_image.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
            ct_image_layout.addWidget(ct_image)
            ct_image_layout.addStretch()
            ct_layout.addLayout(ct_image_layout)
            ct_label = QW.QLabel(text)
            ct_label.setWordWrap(True)
            ct_label.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignTop)
            ct_layout.addWidget(ct_label)
            ct_groupbox.setLayout(ct_layout)
            ct_groupbox.setSizePolicy(
                QW.QSizePolicy.MinimumExpanding, QW.QSizePolicy.Fixed
            )
            layout.addWidget(ct_groupbox)

        tb_groupbox = QW.QGroupBox(tb_title, self)
        tb_layout = QW.QVBoxLayout()
        tb_layout.addWidget(QW.QLabel(tb_text))
        tb_layout.addWidget(self.shell)
        tb_groupbox.setLayout(tb_layout)
        layout.addWidget(tb_groupbox)

        if tip is not None:
            tip_groupbox = QW.QGroupBox(_("Tip"), self)
            tip_layout = QW.QHBoxLayout()
            tip_image_layout = QW.QVBoxLayout()
            tip_image = QW.QLabel()
            tip_image.setPixmap(get_std_icon("MessageBoxInformation").pixmap(24, 24))
            tip_image.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
            tip_image_layout.addWidget(tip_image)
            tip_image_layout.addStretch()
            tip_layout.addLayout(tip_image_layout)
            tip_label = QW.QLabel(tip)
            tip_label.setWordWrap(True)
            tip_label.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignTop)
            tip_layout.addWidget(tip_label)
            tip_groupbox.setLayout(tip_layout)
            tip_groupbox.setSizePolicy(
                QW.QSizePolicy.MinimumExpanding, QW.QSizePolicy.Fixed
            )
            layout.addWidget(tip_groupbox)

        layout.addSpacing(10)
        if category == "warning":
            layout.addWidget(
                QW.QLabel(
                    _(
                        "Please click on the 'Ignore' button to "
                        "ignore this warning next time."
                    )
                )
            )
            layout.addSpacing(10)

        layout.addWidget(bbox)

        self.setLayout(layout)
        self.resize(width, height)

        bbox.button(QW.QDialogButtonBox.Ok).setFocus()

    def ignore(self):
        """Ignore warning next time"""
        Conf.proc.ignore_warnings.set(True)
        self.accept()


def show_warning_error(
    parent: QW.QWidget,
    category: str,
    context: str = None,
    message: str = None,
    tip: str = None,
) -> None:
    """Show error message

    Args:
        parent (QW.QWidget): parent widget
        category (str): message category ("error" or "warning")
        context (str | None): context. Defaults to None.
        message (str | None): message. Defaults to None.
        tip (str | None): tip. Defaults to None.
    """
    if category == "warning" and Conf.proc.ignore_warnings.get():
        return
    dlg = WarningErrorMessageBox(parent, category, context, message, tip)
    exec_dialog(dlg)
