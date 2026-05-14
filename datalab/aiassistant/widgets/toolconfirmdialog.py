# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Confirmation dialog displayed before executing an LLM tool call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW

from datalab.config import _

if TYPE_CHECKING:
    from datalab.aiassistant.tools.registry import Tool


class ToolConfirmDialog(QW.QDialog):
    """Dialog asking the user to confirm execution of a tool call.

    Args:
        tool: Tool to be executed.
        arguments: Arguments proposed by the LLM.
        parent: Parent widget.
    """

    def __init__(
        self,
        tool: Tool,
        arguments: dict,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("AI Assistant — Confirm tool call"))
        self.resize(640, 480)

        layout = QW.QVBoxLayout(self)

        header = QW.QLabel(_("The AI assistant requests to run the following action:"))
        header.setWordWrap(True)
        layout.addWidget(header)

        title = QW.QLabel(f"<b>{tool.name}</b>")
        layout.addWidget(title)

        description = QW.QLabel(tool.description)
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addWidget(QW.QLabel(_("Parameters:")))
        params_view = QW.QPlainTextEdit()
        params_view.setReadOnly(True)
        params_view.setPlainText(json.dumps(arguments, indent=2, default=str))
        layout.addWidget(params_view, 1)

        # Special case: macro code preview is more readable as plain Python
        if tool.name == "create_and_run_macro" and "code" in arguments:
            layout.addWidget(QW.QLabel(_("Macro code:")))
            code_view = QW.QPlainTextEdit()
            code_view.setReadOnly(True)
            code_view.setPlainText(str(arguments.get("code", "")))
            layout.addWidget(code_view, 2)

        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        button_box.button(QW.QDialogButtonBox.Ok).setText(_("Run"))
        button_box.button(QW.QDialogButtonBox.Cancel).setText(_("Cancel"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @classmethod
    def confirm(
        cls,
        tool: Tool,
        arguments: dict,
        parent: QW.QWidget | None = None,
    ) -> bool:
        """Show the dialog and return True if the user clicked Run."""
        dialog = cls(tool, arguments, parent)
        return dialog.exec() == QW.QDialog.Accepted
