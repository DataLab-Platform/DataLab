# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Persistent conversation storage for the AI assistant.

Conversations are stored as JSON files (one file per conversation) in the
DataLab user configuration directory. The store keeps a configurable maximum
number of conversations and prunes the oldest ones when needed.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import os.path as osp
import uuid
from dataclasses import dataclass, field
from typing import Any

from datalab.aiassistant.providers.base import ChatMessage, ToolCall


def _now_iso() -> str:
    """Return the current local time as an ISO-8601 string (microseconds).

    Microsecond precision ensures that conversations saved in quick
    succession are still totally ordered by ``updated_at``.
    """
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _make_id() -> str:
    """Return a unique, time-sortable conversation identifier."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def derive_title(text: str, max_len: int = 60) -> str:
    """Build a short conversation title from the first user message."""
    text = " ".join(str(text).split())
    if not text:
        return "(empty)"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _message_to_dict(msg: ChatMessage) -> dict[str, Any]:
    """Serialise a :class:`ChatMessage` to a JSON-friendly dict."""
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ],
        "tool_call_id": msg.tool_call_id,
        "name": msg.name,
    }


def _message_from_dict(data: dict[str, Any]) -> ChatMessage:
    """Rebuild a :class:`ChatMessage` from its serialised dict form."""
    return ChatMessage(
        role=data.get("role", "user"),
        content=data.get("content", ""),
        tool_calls=[
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            )
            for tc in data.get("tool_calls") or []
        ],
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


@dataclass
class Conversation:
    """A persisted conversation.

    Args:
        id: Unique identifier (also used as the JSON file basename).
        title: Human-readable title (derived from the first user message).
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-modification timestamp.
        messages: Conversation messages (excluding the system prompt).
    """

    id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    messages: list[ChatMessage] = field(default_factory=list)

    @classmethod
    def new(cls) -> Conversation:
        """Return a fresh empty conversation with a new identifier."""
        now = _now_iso()
        return cls(id=_make_id(), created_at=now, updated_at=now)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [_message_to_dict(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Rebuild a :class:`Conversation` from its serialised dict form."""
        return cls(
            id=data.get("id", _make_id()),
            title=data.get("title", ""),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            messages=[_message_from_dict(m) for m in data.get("messages", [])],
        )


@dataclass
class ConversationInfo:
    """Lightweight metadata used by the conversations browser dialog."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationStore:
    """Filesystem-backed store for AI assistant conversations.

    Args:
        directory: Directory in which conversation JSON files are stored
         (created on first use).
        max_conversations: Maximum number of conversations to keep on disk.
         When the limit is exceeded after a save, the oldest conversations
         (by ``updated_at``) are pruned.
    """

    def __init__(self, directory: str, max_conversations: int = 200) -> None:
        self.directory = directory
        self.max_conversations = int(max_conversations)
        os.makedirs(directory, exist_ok=True)

    def _path(self, conv_id: str) -> str:
        return osp.join(self.directory, f"{conv_id}.json")

    def list(self) -> list[ConversationInfo]:
        """Return conversation metadata sorted from most-recent to oldest."""
        items: list[ConversationInfo] = []
        for path in glob.glob(osp.join(self.directory, "*.json")):
            try:
                with open(path, encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            items.append(
                ConversationInfo(
                    id=data.get("id", osp.splitext(osp.basename(path))[0]),
                    title=data.get("title", ""),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    message_count=len(data.get("messages", [])),
                )
            )
        items.sort(key=lambda info: info.updated_at, reverse=True)
        return items

    def load(self, conv_id: str) -> Conversation:
        """Load a conversation by identifier."""
        with open(self._path(conv_id), encoding="utf-8") as file:
            return Conversation.from_dict(json.load(file))

    def save(self, conv: Conversation) -> None:
        """Atomically persist a conversation and prune old entries."""
        conv.updated_at = _now_iso()
        path = self._path(conv.id)
        tmp = path + ".tmp"
        os.makedirs(self.directory, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(conv.to_dict(), file, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        self._prune()

    def delete(self, conv_id: str) -> None:
        """Remove a conversation file (no error if it does not exist)."""
        try:
            os.remove(self._path(conv_id))
        except FileNotFoundError:
            pass

    def _prune(self) -> None:
        items = self.list()
        for info in items[self.max_conversations :]:
            self.delete(info.id)
