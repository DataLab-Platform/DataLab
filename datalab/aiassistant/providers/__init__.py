# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
LLM provider implementations for the DataLab AI assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    ToolCall,
)
from datalab.aiassistant.providers.mock import MockProvider
from datalab.aiassistant.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    pass


PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def get_provider(name: str) -> type[LLMProvider]:
    """Return the LLM provider class registered under ``name``.

    Args:
        name: Provider name (e.g. ``"openai"``).

    Returns:
        Provider class.

    Raises:
        KeyError: if no provider is registered under ``name``.
    """
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise KeyError(
            f"Unknown LLM provider {name!r}. Available providers: {available}."
        ) from exc


__all__ = [
    "PROVIDERS",
    "AssistantMessage",
    "ChatMessage",
    "LLMProvider",
    "LLMProviderError",
    "MockProvider",
    "OpenAIProvider",
    "ToolCall",
    "get_provider",
]
