# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the AI assistant conversation store and input history.
"""

from __future__ import annotations

import os.path as osp
import time

from datalab.aiassistant.conversation import (
    Conversation,
    ConversationStore,
    derive_title,
)
from datalab.aiassistant.inputhistory import InputHistory
from datalab.aiassistant.providers.base import ChatMessage, ToolCall


def test_derive_title_truncates_and_collapses_whitespace():
    """Title derivation collapses whitespace and truncates with an ellipsis."""
    assert derive_title("  hello   world  ") == "hello world"
    long = "x" * 200
    out = derive_title(long, max_len=20)
    assert len(out) == 20
    assert out.endswith("…")
    assert derive_title("") == "(empty)"


def test_conversation_roundtrip():
    """Conversation.to_dict()/from_dict() preserve all fields and tool calls."""
    conv = Conversation.new()
    conv.title = "demo"
    conv.messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="t", arguments={"a": 1})],
        ),
        ChatMessage(role="tool", content="ok", tool_call_id="1", name="t"),
    ]
    restored = Conversation.from_dict(conv.to_dict())
    assert restored.id == conv.id
    assert restored.title == "demo"
    assert [m.role for m in restored.messages] == ["user", "assistant", "tool"]
    assert restored.messages[1].tool_calls[0].arguments == {"a": 1}
    assert restored.messages[2].name == "t"


def test_conversation_store_save_load_list_delete(tmp_path):
    """ConversationStore round-trips save/load/list/delete operations."""
    store = ConversationStore(str(tmp_path))
    conv = Conversation.new()
    conv.title = "first"
    conv.messages = [ChatMessage(role="user", content="hello")]
    store.save(conv)
    assert osp.isfile(osp.join(str(tmp_path), f"{conv.id}.json"))

    items = store.list()
    assert len(items) == 1
    assert items[0].id == conv.id
    assert items[0].title == "first"
    assert items[0].message_count == 1

    loaded = store.load(conv.id)
    assert loaded.title == "first"
    assert loaded.messages[0].content == "hello"

    store.delete(conv.id)
    assert not store.list()


def test_conversation_store_prunes_oldest(tmp_path):
    """ConversationStore keeps only the most recent ``max_conversations``."""
    store = ConversationStore(str(tmp_path), max_conversations=2)
    ids = []
    for index in range(4):
        conv = Conversation.new()
        conv.title = f"c{index}"
        store.save(conv)
        ids.append(conv.id)
        # Coarse Windows clock resolution can otherwise collapse the
        # ``updated_at`` timestamps written by successive saves.
        time.sleep(0.02)
    items = store.list()
    assert len(items) == 2
    # Most recent two survive (save() refreshes updated_at to "now", so the
    # last two writes win).
    assert {info.id for info in items} == {ids[-2], ids[-1]}


def test_input_history_persistence_and_dedup(tmp_path):
    """InputHistory persists entries to disk and deduplicates them."""
    path = str(tmp_path / "hist.txt")
    hist = InputHistory(path)
    hist.add("alpha")
    hist.add("beta")
    hist.add("alpha")  # duplicate -> moved to end
    assert hist.items() == ["beta", "alpha"]

    reopened = InputHistory(path)
    assert reopened.items() == ["beta", "alpha"]


def test_input_history_navigation_preserves_draft(tmp_path):
    """InputHistory navigation captures and restores the in-progress draft."""
    hist = InputHistory(str(tmp_path / "h.txt"))
    hist.add("first")
    hist.add("second")
    # Start with a draft, then navigate up twice, then back down to draft.
    assert hist.previous("draft") == "second"
    assert hist.previous("second") == "first"
    assert hist.previous("first") == "first"  # at top, stays put
    assert hist.next("first") == "second"
    assert hist.next("second") == "draft"  # restored
    # Past the draft -> no further navigation in progress.
    assert hist.next("draft") is None


def test_input_history_handles_multiline_entries(tmp_path):
    """Multiline and backslash-containing entries round-trip through disk."""
    path = str(tmp_path / "h.txt")
    hist = InputHistory(path)
    hist.add("line1\nline2\\with-backslash")
    reopened = InputHistory(path)
    assert reopened.items() == ["line1\nline2\\with-backslash"]


def test_input_history_ignores_empty(tmp_path):
    """Empty or whitespace-only entries are silently dropped."""
    hist = InputHistory(str(tmp_path / "h.txt"))
    hist.add("   ")
    hist.add("")
    assert not hist.items()
    assert hist.previous("anything") is None
