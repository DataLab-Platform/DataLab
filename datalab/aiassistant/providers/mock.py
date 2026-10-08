# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Mock LLM provider for offline testing and demos.

This provider does **not** call any external service. It produces canned
replies based on simple keyword matching, exercising the full conversation
loop (text replies, tool calls, multi-step plans) without requiring a real
LLM. Use it to validate the AI assistant pipeline (UI, tool registry,
confirmation dialog, macro execution) without an API key.

Triggers (case-insensitive substring matching on the latest user message):

- ``"hello"``, ``"bonjour"`` → simple greeting (no tool call)
- ``"list"``, ``"liste"``, ``"objects"`` → ``list_objects`` tool call
- ``"signal"`` and (``"sin"`` or ``"sinus"``) → ``create_synthetic_signal``
- ``"image"`` and (``"gauss"`` or ``"2d"``) → ``create_synthetic_image``
- ``"fft"`` → 2-step plan (create signal then ``apply_operation('fft')``)
- ``"macro"`` → ``create_and_run_macro`` with a small demo script
- otherwise → echo reply describing the available triggers
"""

from __future__ import annotations

import itertools

from datalab.aiassistant.providers.base import (
    AssistantMessage,
    ChatMessage,
    LLMProvider,
    ToolCall,
)


class MockProvider(LLMProvider):
    """LLM-free provider returning scripted replies (offline demo)."""

    name = "mock"

    _id_counter = itertools.count(1)

    def __init__(
        self,
        api_key: str = "",
        model: str = "mock",
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(api_key, model, base_url, temperature, timeout)
        # Per-conversation queue of replies (used to chain a multi-step plan
        # across successive `chat()` calls within the same controller turn).
        self._pending: list[AssistantMessage] = []

    def _next_id(self) -> str:
        return f"mock_{next(self._id_counter)}"

    # pylint: disable-next=too-many-return-statements
    def chat(self, messages, tools=None):  # noqa: ARG002
        if self._pending:
            return self._pending.pop(0)

        last_user = ""
        for msg in reversed(messages):
            if isinstance(msg, ChatMessage) and msg.role == "user":
                last_user = msg.content.lower()
                break

        if any(token in last_user for token in ("hello", "bonjour", "salut")):
            return AssistantMessage(
                content="Hello! I'm running in mock mode (no external LLM)."
            )

        if any(token in last_user for token in ("list", "liste", "objects", "objets")):
            return AssistantMessage(
                content="Listing workspace objects.",
                tool_calls=[
                    ToolCall(id=self._next_id(), name="list_objects", arguments={})
                ],
            )

        if "fft" in last_user:
            # Two-step demo plan: create a signal, then apply FFT.
            self._pending.append(
                AssistantMessage(
                    content="Now applying FFT to the signal.",
                    tool_calls=[
                        ToolCall(
                            id=self._next_id(),
                            name="apply_operation",
                            arguments={
                                "name": "compute_fft",
                                "panel": "signal",
                            },
                        )
                    ],
                )
            )
            self._pending.append(
                AssistantMessage(content="Done. The FFT is now in the signal panel.")
            )
            return AssistantMessage(
                content="Creating a sine signal first.",
                tool_calls=[
                    ToolCall(
                        id=self._next_id(),
                        name="create_synthetic_signal",
                        arguments={
                            "title": "Mock sine 50 Hz",
                            "kind": "sin",
                            "npoints": 1024,
                            "xmin": 0.0,
                            "xmax": 1.0,
                            "frequency": 50.0,
                            "amplitude": 1.0,
                        },
                    )
                ],
            )

        if "signal" in last_user and ("sin" in last_user or "sinus" in last_user):
            return AssistantMessage(
                content="Creating a synthetic sine signal.",
                tool_calls=[
                    ToolCall(
                        id=self._next_id(),
                        name="create_synthetic_signal",
                        arguments={
                            "title": "Mock sine",
                            "kind": "sin",
                            "frequency": 10.0,
                        },
                    )
                ],
            )

        if "image" in last_user and ("gauss" in last_user or "2d" in last_user):
            return AssistantMessage(
                content="Creating a synthetic 2D Gaussian image.",
                tool_calls=[
                    ToolCall(
                        id=self._next_id(),
                        name="create_synthetic_image",
                        arguments={
                            "title": "Mock Gauss2D",
                            "kind": "gauss2d",
                            "width": 256,
                            "height": 256,
                        },
                    )
                ],
            )

        if "macro" in last_user:
            code = (
                "import numpy as np\n"
                "from datalab.control.proxy import RemoteProxy\n"
                "\n"
                "proxy = RemoteProxy()\n"
                "x = np.linspace(0, 1, 1024)\n"
                "y = np.sin(2 * np.pi * 5 * x)\n"
                'proxy.add_signal("Macro signal", x, y)\n'
            )
            return AssistantMessage(
                content="Creating a small demo macro.",
                tool_calls=[
                    ToolCall(
                        id=self._next_id(),
                        name="create_and_run_macro",
                        arguments={
                            "title": "Mock demo macro",
                            "code": code,
                            "autorun": True,
                        },
                    )
                ],
            )

        return AssistantMessage(
            content=(
                "Mock provider: I can react to keywords like "
                "'hello', 'list objects', 'create a sine signal', "
                "'gauss2d image', 'fft' or 'macro'. "
                "Configure a real provider in Settings to use a true LLM."
            )
        )
