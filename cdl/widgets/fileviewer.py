# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing a file viewer widget
"""

from __future__ import annotations

from pathlib import Path

from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

from cdl.config import _


def read_text_file(path: str) -> str:
    """Read text file using multiple encodings

    Args:
        path (str): path to file

    Raises:
        UnicodeDecodeError: if unable to read file using any of the encodings

    Returns:
        str: file contents
    """
    encodings = ["utf-8", "latin1", "cp1252", "utf-16", "utf-32", "ascii"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as fdesc:
                return fdesc.read()
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError(
        f"Unable to read file using the following encodings: {encodings}"
    )


def get_title_contents(path: str) -> tuple[str, str]:
    """Get title and contents for log filename

    Args:
        path (str): path to file

    Returns:
        tuple[str, str]: title and contents
    """
    contents = read_text_file(path)
    pathobj = Path(path)
    uri_path = pathobj.absolute().as_uri()
    prefix = _("Contents of file")
    text = f'{prefix} <a href="{uri_path}">{path}</a>:'
    return text, contents


class FileViewerWidget(QW.QWidget):
    """File viewer widget

    Args:
        parent (QW.QWidget | None): parent widget. Defaults to None.
    """

    def __init__(self, language: str | None = None, parent: QW.QWidget = None) -> None:
        super().__init__(parent)
        self.editor = CodeEditor(language=language)
        self.editor.setReadOnly(True)
        layout = QW.QVBoxLayout()
        self.label = QW.QLabel("")
        layout.addWidget(self.label)
        layout.addWidget(self.editor)
        self.setLayout(layout)

    def set_data(self, text: str, contents: str) -> None:
        """Set log data

        Args:
            text (str): text to display
            contents (str): contents to display
        """
        self.label.setText(text)
        self.label.setOpenExternalLinks(True)
        self.editor.setPlainText(contents)
