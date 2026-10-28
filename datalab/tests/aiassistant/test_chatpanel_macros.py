# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""AI Assistant chat panel: macros proposed by the assistant.

Covers the "Save to Macros" links of the chat panel (after a macro tool call,
in a conversation reloaded from the history and under the Python code blocks
of a reply), the "Show in Macro panel" link and the macro tool run with
``autorun=False``.
"""

from __future__ import annotations

import re

import pytest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.aiassistant.controller import TurnResult
from datalab.aiassistant.conversation import derive_title
from datalab.aiassistant.providers.base import ChatMessage, ToolCall
from datalab.aiassistant.tools.builtin import (
    _tool_create_and_run_macro,
    build_default_registry,
)
from datalab.aiassistant.tools.registry import ToolResult
from datalab.aiassistant.widgets.toolconfirmdialog import ToolConfirmDialog
from datalab.config import Conf, _
from datalab.tests import datalab_test_app_context

MACRO_TITLE = "Generate super-Gaussian"
MACRO_CODE = "import numpy as np\nprint(np.pi)"
REQUEST = "Write a macro that generates a super-gaussian curve"


@pytest.fixture(autouse=True)
def _isolated_macro_caches(tmp_path, monkeypatch):
    """Keep the user's recent-macro and recovery caches untouched."""

    def fake_path(filename: str) -> str:
        return str(tmp_path / filename)

    for module in ("recentmacros", "macrorecovery"):
        monkeypatch.setattr(
            f"datalab.utils.{module}.Configuration.get_path",
            staticmethod(fake_path),
        )


def _save_ids(panel) -> list[str]:
    """Ids of the "Save to Macros" links shown in the chat view."""
    return re.findall(r"datalab:save-macro/(\w+)", panel.history_view.toHtml())


def _click(panel, url: str) -> None:
    panel._on_anchor_clicked(QC.QUrl(url))  # pylint: disable=protected-access
    QW.QApplication.processEvents()


def _is_in_front(dock: QW.QDockWidget) -> bool:
    """Return True if the dock is the selected tab of its group"""
    return not dock.visibleRegion().isEmpty()


def test_macro_tool_without_autorun_does_not_run() -> None:
    """``autorun=False`` validates the macro and offers it without running it."""
    with datalab_test_app_context(console=False) as win:
        console = win.macropanel.console
        before = console.toPlainText()
        titles = win.macropanel.get_macro_titles()
        result = _tool_create_and_run_macro(
            None, win, title=MACRO_TITLE, code="print('ran')", autorun=False
        )
        QW.QApplication.processEvents()
        assert result["autorun"] is False
        assert result["_macro"] == {"title": MACRO_TITLE, "code": "print('ran')"}
        assert "exit_code" not in result
        assert "note" in result
        assert console.toPlainText() == before
        assert win.macropanel.get_macro_titles() == titles


def test_save_links_of_a_reloaded_conversation() -> None:
    """A reloaded conversation offers "Save to Macros" again, shows the real
    status of each tool call and lets the user show the saved macro."""
    messages = [
        ChatMessage(role="user", content=REQUEST),
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall("c1", "create_and_run_macro", {"code": MACRO_CODE}),
                ToolCall("c2", "apply_operation", {"name": "fft"}),
            ],
        ),
        ChatMessage(
            role="tool",
            content=ToolResult(
                ok=True,
                data={
                    "exit_code": 0,
                    "_macro": {"title": MACRO_TITLE, "code": MACRO_CODE},
                },
            ).to_message_content(),
            tool_call_id="c1",
            name="create_and_run_macro",
        ),
        ChatMessage(
            role="tool",
            content=ToolResult(ok=False, error="No selection").to_message_content(),
            tool_call_id="c2",
            name="apply_operation",
        ),
        ChatMessage(
            role="assistant",
            content="Or paste it:\n\n```python\nprint(2)\n```\n\n```bash\nls\n```",
        ),
    ]
    with datalab_test_app_context(console=False) as win:
        panel = win.aiassistantpanel
        panel._render_messages(messages)  # pylint: disable=protected-access
        text = panel.history_view.toPlainText()
        assert "✓ create_and_run_macro" in text
        assert "✗ apply_operation: No selection" in text

        # One link for the macro tool result, one for the Python block only
        ids = _save_ids(panel)
        assert len(ids) == 2

        win.docks[win.aiassistantpanel].raise_()
        QW.QApplication.processEvents()
        assert not _is_in_front(win.docks[win.macropanel])

        _click(panel, f"datalab:save-macro/{ids[0]}")
        _click(panel, f"datalab:save-macro/{ids[1]}")
        macro = win.macropanel.get_macro(MACRO_TITLE)
        assert macro is not None
        assert macro.get_code().splitlines() == MACRO_CODE.splitlines()
        block_macro = win.macropanel.get_macro(derive_title(REQUEST))
        assert block_macro is not None
        assert block_macro.get_code().splitlines() == ["print(2)"]
        # Saving does not hide the conversation...
        assert _is_in_front(win.docks[win.aiassistantpanel])
        # ...but offers to show the Macro panel
        assert "datalab:show-macro-panel" in panel.history_view.toHtml()
        _click(panel, "datalab:show-macro-panel")
        assert _is_in_front(win.docks[win.macropanel])


def test_save_links_of_a_live_turn() -> None:
    """A live turn offers "Save to Macros" for the macro tool result and for
    the Python code blocks of the reply, named after the user request."""
    with datalab_test_app_context(console=False) as win:
        panel = win.aiassistantpanel
        panel._last_user_text = REQUEST  # pylint: disable=protected-access
        result = TurnResult(
            assistant_message="Here it is:\n\n```\nimport numpy as np\n```",
            tool_executions=[
                (
                    "create_and_run_macro",
                    {},
                    ToolResult(
                        ok=True,
                        data={
                            "autorun": False,
                            "_macro": {"title": MACRO_TITLE, "code": MACRO_CODE},
                        },
                    ),
                )
            ],
        )
        panel._on_turn_finished(result)  # pylint: disable=protected-access
        ids = _save_ids(panel)
        assert len(ids) == 2
        for macro_id in ids:
            _click(panel, f"datalab:save-macro/{macro_id}")
        titles = win.macropanel.get_macro_titles()
        assert MACRO_TITLE in titles
        assert derive_title(REQUEST) in titles


def test_no_code_block_links_when_macro_tool_disabled() -> None:
    """Users who disabled AI macros are not offered to save code blocks."""
    with datalab_test_app_context(console=False) as win:
        Conf.ai_expose_macro_tool.set(False)
        panel = win.aiassistantpanel
        panel._append_assistant("```python\nprint(3)\n```")  # pylint: disable=W0212
        assert _save_ids(panel) == []


@pytest.mark.parametrize("autorun", [False, True, None])
def test_confirm_dialog_tells_when_the_macro_will_not_run(autorun) -> None:
    """The confirmation dialog states that a macro proposed with
    ``autorun=False`` will not be executed."""
    note = _(
        "This macro will not be executed: approving only offers it "
        "for saving to the Macro panel."
    )
    with datalab_test_app_context(console=False) as win:
        tool = build_default_registry().get("create_and_run_macro")
        arguments = {"title": MACRO_TITLE, "code": MACRO_CODE}
        if autorun is not None:
            arguments["autorun"] = autorun
        dialog = ToolConfirmDialog(tool, arguments, win)
        labels = [label.text() for label in dialog.findChildren(QW.QLabel)]
        ok_button = dialog.findChild(QW.QDialogButtonBox).button(QW.QDialogButtonBox.Ok)
        if autorun is False:
            assert any(note in text for text in labels)
            assert ok_button.text() == _("Approve")
        else:
            assert not any(note in text for text in labels)
            assert ok_button.text() == _("Run")
        dialog.deleteLater()
