# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing an error message box
"""

import traceback

from guidata.qthelpers import get_std_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _


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


class ErrorMessageBox(QW.QDialog):
    """Error message box"""

    def __init__(
        self, parent: QW.QWidget, context: str = None, tip: str = None
    ) -> None:
        super().__init__(parent)
        title = parent.window().objectName()
        self.setWindowTitle(title)
        self.editor = CodeEditor(self)
        font = self.editor.font()
        font.setPixelSize(12)
        self.editor.setFont(font)
        self.editor.setReadOnly(True)
        self.editor.setPlainText(traceback.format_exc() * 10)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok)
        bbox.accepted.connect(self.accept)

        layout = QW.QVBoxLayout()

        if context is not None:
            context = insert_spaces(context, 80)
            msgprefix = _("An error has occured during the following context:")
            text = "<br>".join([msgprefix, f"<b>{context}</b>"])
            ct_groupbox = QW.QGroupBox(_("Context"), self)
            ct_layout = QW.QHBoxLayout()
            ct_image_layout = QW.QVBoxLayout()
            ct_image = QW.QLabel()
            ct_image.setPixmap(get_std_icon("MessageBoxWarning").pixmap(24, 24))
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

        tb_groupbox = QW.QGroupBox(_("Traceback"), self)
        tb_layout = QW.QVBoxLayout()
        tb_text = _("The following traceback may help to understand the problem:")
        tb_layout.addWidget(QW.QLabel(tb_text))
        tb_layout.addWidget(self.editor)
        tb_groupbox.setLayout(tb_layout)
        tb_groupbox.setSizePolicy(
            QW.QSizePolicy.MinimumExpanding, QW.QSizePolicy.MinimumExpanding
        )
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
        layout.addWidget(bbox)

        self.setLayout(layout)
        self.resize(800, 500)
