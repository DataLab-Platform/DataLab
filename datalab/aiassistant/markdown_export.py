# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Markdown export for AI Assistant conversations.

Renders a :class:`Conversation` to a self-contained Markdown document
suitable for sharing or pasting into a notebook. Uses fenced JSON blocks
for tool calls / results so the round-trip stays unambiguous.

Mirrors :mod:`DataLab-Web/src/aiassistant/conversationExport.ts`.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datalab.aiassistant.conversation import Conversation
    from datalab.aiassistant.providers.base import ChatMessage, ToolCall


_FORBIDDEN_FILENAME_CHARS = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")


def _pretty_json(text: str) -> str:
    """Try to pretty-print a JSON-ish blob; fall back to the raw string."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text


def _content_to_text(content: Any) -> str:
    """Flatten a multimodal content list into plain text for export."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    parts.append("> _(image attached)_")
        return "\n".join(parts)
    return "" if content is None else str(content)


def _render_tool_calls(tool_calls: list[ToolCall]) -> str:
    """Serialise an assistant ``tool_calls`` array as a fenced JSON block."""
    compact = [
        {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
    ]
    return "```json\n" + json.dumps(compact, indent=2, ensure_ascii=False) + "\n```"


def _render_message(msg: ChatMessage) -> str:
    """Render a single :class:`ChatMessage` as a Markdown section."""
    if msg.role == "system":
        # System prompt is intentionally omitted from exports — it's
        # generated at runtime and not interesting to a human reader.
        return ""
    if msg.role == "user":
        return "## You\n\n" + _content_to_text(msg.content)
    if msg.role == "assistant":
        lines: list[str] = ["## Assistant", ""]
        text = _content_to_text(msg.content)
        if text:
            lines.append(text)
        if msg.tool_calls:
            if text:
                lines.append("")
            lines.append("**Tool calls:**")
            lines.append("")
            lines.append(_render_tool_calls(msg.tool_calls))
        return "\n".join(lines)
    # Tool result.
    return "\n".join(
        [
            f"## Tool result ({msg.name or 'tool'})",
            "",
            "```json",
            _pretty_json(_content_to_text(msg.content)),
            "```",
        ]
    )


def conversation_to_markdown(conv: Conversation) -> str:
    """Render *conv* as a self-contained Markdown document."""
    sections: list[str] = []
    sections.append(f"# {conv.title or '(untitled conversation)'}")
    meta: list[str] = []
    if conv.created_at:
        meta.append(f"Created: {conv.created_at}")
    if conv.updated_at:
        meta.append(f"Updated: {conv.updated_at}")
    if conv.usage is not None and conv.usage.total_tokens:
        meta.append(f"Tokens: {conv.usage.total_tokens}")
    if meta:
        sections.append("_" + " \u2022 ".join(meta) + "_")
    for msg in conv.messages:
        rendered = _render_message(msg)
        if rendered:
            sections.append(rendered)
    return "\n\n".join(sections) + "\n"


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Strip filesystem-unsafe characters and clamp the length.

    The result is a valid filename on Windows / macOS / Linux.
    Mirrors ``sanitizeFilename`` in DataLab-Web.
    """
    cleaned = _FORBIDDEN_FILENAME_CHARS.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "conversation"
    return cleaned[:max_len].strip() if len(cleaned) > max_len else cleaned
