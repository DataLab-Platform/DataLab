# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab AI Assistant
====================

This package provides an integrated AI assistant for DataLab, capable of:

- Creating, processing and analyzing signals and images.
- Writing and executing macros.
- Inspecting the current workspace.

The assistant relies on a :class:`LLMProvider` (see :mod:`.providers`) and a
:class:`ToolRegistry` (see :mod:`.tools.registry`) bridging LLM tool calls to
DataLab's :class:`~datalab.control.proxy.LocalProxy`.

User-triggered actions always require explicit confirmation through a dedicated
dialog, except for read-only inspection tools (which can be auto-approved
through preferences).
"""

from __future__ import annotations
