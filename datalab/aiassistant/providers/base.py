# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
LLM provider base class and message dataclasses.

This module is intentionally GUI-independent and only depends on the standard
library, so that it can be reused / tested in headless environments.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool call requested by the LLM.

    Args:
        id: Provider-specific identifier of the tool call (used to correlate
         results in the conversation history).
        name: Tool name.
        arguments: Tool arguments as a JSON-serializable dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    """A message in the conversation history.

    Args:
        role: Sender role (``"system"``, ``"user"``, ``"assistant"`` or
         ``"tool"``).
        content: Plain text content. May be empty when the assistant only
         emits tool calls.
        tool_calls: Tool calls emitted by the assistant (only for the
         ``"assistant"`` role).
        tool_call_id: For ``"tool"`` messages, the ID of the corresponding
         tool call.
        name: For ``"tool"`` messages, the tool name.
    """

    role: Role
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class AssistantMessage:
    """The LLM's response.

    Args:
        content: Text content (may be empty).
        tool_calls: Tool calls requested by the assistant.
        finish_reason: Provider-specific finish reason.
        raw: Raw provider response for debugging.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call fails."""


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    Args:
        api_key: Provider API key.
        model: Model name (e.g. ``"gpt-4o-mini"``).
        base_url: Optional base URL override (for OpenAI-compatible endpoints).
        temperature: Sampling temperature.
        timeout: HTTP timeout in seconds.
    """

    name: str = ""

    #: Name of the environment variable from which the API key may be read
    #: when none is configured explicitly. ``None`` disables the env-var
    #: fallback (e.g. for the mock provider). Subclasses should set this to
    #: the conventional variable name expected by the upstream SDK
    #: (e.g. ``"OPENAI_API_KEY"`` for OpenAI).
    api_key_env_var: str | None = None

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def resolve_api_key(cls, configured_key: str | None) -> str:
        """Return the API key to use, applying the env-var fallback.

        Resolution order (first non-empty value wins):

        1. ``configured_key`` — value stored in the DataLab configuration
           (typically set via the AI Assistant settings dialog).
        2. The environment variable named by :attr:`api_key_env_var`, if any.

        Storing the API key in an environment variable is the recommended
        practice: it avoids writing the secret in clear text in the DataLab
        INI file, and it lets the same machine-wide credential be shared
        with other tools (the OpenAI SDK, ``curl``, etc.).

        Args:
            configured_key: API key read from the DataLab configuration.
             May be ``None`` or empty.

        Returns:
            The resolved API key, or an empty string if none is available.
        """
        if configured_key:
            return configured_key
        if cls.api_key_env_var:
            return os.environ.get(cls.api_key_env_var, "") or ""
        return ""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantMessage:
        """Send a chat completion request and return the assistant's reply.

        Args:
            messages: Conversation history.
            tools: Tool definitions in OpenAI function-calling JSON schema
             format (see :mod:`datalab.aiassistant.tools.schema`).

        Returns:
            Assistant reply (text and/or tool calls).

        Raises:
            LLMProviderError: when the provider call fails.
        """
