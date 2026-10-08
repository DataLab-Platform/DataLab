# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
AI Assistant settings dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds

from datalab.aiassistant.providers import PROVIDERS
from datalab.config import _

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW


class AISettings(gds.DataSet):
    """AI Assistant configuration."""

    enabled = gds.BoolItem(_("Enable AI Assistant"), default=True)
    provider = gds.ChoiceItem(
        _("Provider"),
        [(name, name) for name in PROVIDERS],
        default="openai",
        help=_(
            "Use 'mock' to test the AI assistant pipeline offline without "
            "any API key (scripted replies based on simple keywords)."
        ),
    )
    model = gds.StringItem(_("Model"), default="gpt-4o-mini")
    api_key = gds.StringItem(
        _("API key"),
        default="",
        notempty=False,
        help=_(
            "Your provider API key. Stored in the DataLab INI file in plain "
            "text — keep this file private.\n\n"
            "Recommended: leave this field empty and set the provider's "
            "standard environment variable instead (e.g. OPENAI_API_KEY for "
            "OpenAI). This avoids writing the secret to disk and lets the "
            "same credential be shared with other tools."
        ),
    )
    base_url = gds.StringItem(
        _("Base URL (optional)"),
        default="",
        notempty=False,
        help=_("Override for OpenAI-compatible endpoints. Leave empty for default."),
    )
    temperature = gds.FloatItem(
        _("Temperature"), default=0.2, min=0.0, max=2.0, step=0.1
    )
    timeout = gds.FloatItem(_("HTTP timeout (s)"), default=60.0, min=5.0, max=600.0)
    max_iterations = gds.IntItem(
        _("Max tool-call iterations"), default=8, min=1, max=64
    )
    auto_approve_readonly = gds.BoolItem(
        _("Auto-approve read-only inspection tools"), default=True
    )
    expose_macro_tool = gds.BoolItem(
        _("Allow AI to create and run macros (Python code)"),
        default=True,
        help=_(
            "When enabled, the AI assistant may create and execute Python "
            "macros with full access to DataLab through the RemoteProxy API. "
            "Each macro execution still requires explicit user confirmation.\n\n"
            "Disable this option if you do not want the AI to be able to "
            "propose arbitrary code execution at all (the macro tool is then "
            "hidden from the model entirely)."
        ),
    )


# Names of all options exposed by :class:`AISettings`. Used to detect whether
# the AI assistant configuration was modified through the global Settings
# dialog (so callers can rebuild the controller).
AI_OPTION_NAMES: frozenset[str] = frozenset(
    item.get_name() for item in AISettings().get_items()
)


class AISettingsDialog:
    """Compatibility shim opening the global Settings dialog.

    AI assistant settings now live as a tab in DataLab's main Settings
    dialog (see :func:`datalab.gui.settings.edit_settings`). This helper is
    kept so the chat panel's "Settings" button keeps a single entry point
    and can detect whether any AI option was actually changed.
    """

    @staticmethod
    def edit(parent: QW.QWidget | None = None) -> bool:
        """Open the main Settings dialog.

        Returns:
            ``True`` if at least one AI assistant option was modified, so the
            caller can rebuild the controller. ``False`` otherwise (dialog
            cancelled or no AI option changed).
        """
        # Local import to avoid a hard dependency from the AI assistant
        # package on the GUI settings module at import time.
        # pylint: disable-next=import-outside-toplevel
        from datalab.gui.settings import edit_settings  # noqa: WPS433

        changed = edit_settings(parent)
        return any(option in AI_OPTION_NAMES for option in changed)
