# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AI assistant Markdown export module."""

from __future__ import annotations

from datalab.aiassistant.conversation import Conversation
from datalab.aiassistant.markdown_export import (
    conversation_to_markdown,
    sanitize_filename,
)
from datalab.aiassistant.providers.base import ChatMessage, TokenUsage, ToolCall


def test_conversation_to_markdown_basic_structure():
    """Renders title, meta line and per-message sections in a stable order."""
    conv = Conversation.new()
    conv.title = "Demo conversation"
    conv.usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    conv.messages = [
        ChatMessage(role="system", content="ignored"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
    ]
    md = conversation_to_markdown(conv)
    assert md.startswith("# Demo conversation\n")
    assert "Tokens: 15" in md
    assert "## You\n\nhello" in md
    assert "## Assistant\n\nhi there" in md
    # System messages must NOT leak into the export.
    assert "ignored" not in md


def test_conversation_to_markdown_renders_tool_calls_and_results():
    """Tool calls and tool results are emitted as fenced JSON blocks."""
    conv = Conversation.new()
    conv.title = "tools"
    conv.messages = [
        ChatMessage(
            role="assistant",
            content="running",
            tool_calls=[ToolCall(id="c1", name="ping", arguments={"x": 1})],
        ),
        ChatMessage(
            role="tool",
            content='{"ok": true}',
            tool_call_id="c1",
            name="ping",
        ),
    ]
    md = conversation_to_markdown(conv)
    assert "**Tool calls:**" in md
    assert "```json" in md
    assert '"name": "ping"' in md
    assert "## Tool result (ping)" in md
    # Tool result JSON must be pretty-printed.
    assert '"ok": true' in md


def test_conversation_to_markdown_handles_untitled_and_no_usage():
    """Empty title and missing usage produce a sensible header."""
    conv = Conversation.new()
    md = conversation_to_markdown(conv)
    assert md.startswith("# (untitled conversation)\n")
    assert "Tokens" not in md


def test_sanitize_filename_strips_forbidden_chars():
    """Path-unsafe characters are removed and whitespace collapsed."""
    assert sanitize_filename('hello/world:test*?"<>|') == "hello world test"
    assert sanitize_filename("   ") == "conversation"
    assert sanitize_filename("") == "conversation"


def test_sanitize_filename_clamps_length():
    """Very long filenames are clamped to ``max_len`` characters."""
    assert len(sanitize_filename("x" * 500, max_len=80)) == 80
