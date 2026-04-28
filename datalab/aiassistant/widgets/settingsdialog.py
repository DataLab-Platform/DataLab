# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
AI Assistant settings dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds

from datalab.aiassistant.providers import PROVIDERS
from datalab.config import Conf, _

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


def _load_from_conf() -> AISettings:
    """Load settings from :class:`Conf.ai`."""
    settings = AISettings()
    settings.enabled = bool(Conf.ai.enabled.get(True))
    provider = str(Conf.ai.provider.get("openai"))
    if provider not in PROVIDERS:
        # Tolerate stale config values (e.g. integer index from older versions)
        provider = "openai"
    settings.provider = provider
    settings.model = str(Conf.ai.model.get("gpt-4o-mini"))
    settings.api_key = str(Conf.ai.api_key.get(""))
    settings.base_url = str(Conf.ai.base_url.get(""))
    settings.temperature = float(Conf.ai.temperature.get(0.2))
    settings.timeout = float(Conf.ai.timeout.get(60.0))
    settings.max_iterations = int(Conf.ai.max_iterations.get(8))
    settings.auto_approve_readonly = bool(Conf.ai.auto_approve_readonly.get(True))
    return settings


def _save_to_conf(settings: AISettings) -> None:
    Conf.ai.enabled.set(bool(settings.enabled))
    Conf.ai.provider.set(str(settings.provider))
    Conf.ai.model.set(str(settings.model))
    Conf.ai.api_key.set(str(settings.api_key))
    Conf.ai.base_url.set(str(settings.base_url))
    Conf.ai.temperature.set(float(settings.temperature))
    Conf.ai.timeout.set(float(settings.timeout))
    Conf.ai.max_iterations.set(int(settings.max_iterations))
    Conf.ai.auto_approve_readonly.set(bool(settings.auto_approve_readonly))


class AISettingsDialog:
    """Helper exposing :meth:`edit` returning True on accept."""

    @staticmethod
    def edit(parent: QW.QWidget | None = None) -> bool:
        """Open the settings dialog. Returns True if the user accepted."""
        settings = _load_from_conf()
        if settings.edit(parent):
            _save_to_conf(settings)
            return True
        return False
