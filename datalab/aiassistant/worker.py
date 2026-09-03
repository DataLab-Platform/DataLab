# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Qt worker running the AI conversation off the GUI thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore as QC

if TYPE_CHECKING:
    from datalab.aiassistant.controller import AIController, TurnResult


class AIWorker(QC.QThread):
    """Run a single :meth:`AIController.send` call in a background thread.

    Signals:
        finished_turn: emitted with the :class:`TurnResult` on success.
        failed: emitted with the error message on failure.
    """

    finished_turn = QC.Signal(object)  # TurnResult
    failed = QC.Signal(str)

    def __init__(
        self,
        controller: AIController,
        user_message: str,
        parent: QC.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._user_message = user_message

    def run(self) -> None:  # pragma: no cover - thread entry point
        """Execute the controller turn and emit the appropriate signal."""
        try:
            result: TurnResult = self._controller.send(self._user_message)
        except Exception as exc:  # pylint: disable=broad-except
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_turn.emit(result)
