# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Lightweight Markdown to HTML renderer for the AI Assistant chat panel.

This module implements a minimal, dependency-free subset of Markdown
sufficient to format LLM replies inside :class:`QtWidgets.QTextBrowser`,
which only understands a limited HTML/CSS subset (Qt rich-text).

Supported features:

- Fenced code blocks (``` ... ```) with optional language hint
- Inline code (``code``)
- Bold (``**text**`` or ``__text__``)
- Italic (``*text*`` or ``_text_``)
- Strikethrough (``~~text~~``)
- ATX headings (``#``, ``##``, ``###``, …)
- Unordered lists (``-``, ``*``, ``+``)
- Ordered lists (``1.``, ``2.``, …)
- Blockquotes (``> ``)
- Horizontal rule (``---`` or ``***``)
- Inline links (``[text](url)``)
- Bare URLs (auto-linked)
- Line breaks (newlines preserved)
"""

from __future__ import annotations

import html
import re

# Order matters: code spans/fences must be extracted first so their content
# is not affected by other inline rules.

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_RE = re.compile(r"(?<![\*_\w])([*_])(?=\S)([^*_\n]+?)(?<=\S)\1(?![\*_\w])")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
_BARE_URL_RE = re.compile(r"(?<![\"'>=\(])\b(https?://[^\s<>\"']+)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s*([-*_])\s*\1\s*\1[\s\1]*$")
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")


def _render_inline(text: str) -> str:
    """Render inline Markdown constructs to HTML.

    ``text`` is assumed to be raw (un-escaped) Markdown.
    """
    placeholders: list[str] = []

    def _stash(html_fragment: str) -> str:
        placeholders.append(html_fragment)
        return f"\x00{len(placeholders) - 1}\x00"

    # 1. Inline code (escape contents, then stash).
    def _on_code(match: re.Match[str]) -> str:
        code = html.escape(match.group(1))
        return _stash(f"<code>{code}</code>")

    text = _INLINE_CODE_RE.sub(_on_code, text)

    # 2. Inline links [text](url) — stash the rendered HTML.
    def _on_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        title = match.group(3)
        title_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
        return _stash(f'<a href="{url}"{title_attr}>{label}</a>')

    text = _LINK_RE.sub(_on_link, text)

    # 3. Bare URLs — stash to avoid double-escaping.
    def _on_bare_url(match: re.Match[str]) -> str:
        url = match.group(1)
        # Trim trailing punctuation that is unlikely to be part of the URL.
        trailing = ""
        while url and url[-1] in ".,;:!?)]":
            trailing = url[-1] + trailing
            url = url[:-1]
        if not url:
            return match.group(0)
        safe = html.escape(url, quote=True)
        return _stash(f'<a href="{safe}">{safe}</a>') + trailing

    text = _BARE_URL_RE.sub(_on_bare_url, text)

    # 4. Escape remaining text.
    text = html.escape(text)

    # 5. Inline emphasis on the escaped text (safe — markers are ASCII).
    text = _BOLD_RE.sub(r"<b>\2</b>", text)
    text = _ITALIC_RE.sub(r"<i>\2</i>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)

    # 6. Restore placeholders.
    def _restore(match: re.Match[str]) -> str:
        return placeholders[int(match.group(1))]

    text = re.sub(r"\x00(\d+)\x00", _restore, text)
    return text


def _close_lists(stack: list[str], out: list[str], down_to: int = 0) -> None:
    """Close list tags down to ``down_to`` depth."""
    while len(stack) > down_to:
        out.append(f"</{stack.pop()}>")


def markdown_to_html(text: str) -> str:
    """Convert a Markdown string to a Qt-compatible HTML fragment."""
    if not text:
        return ""

    # Extract fenced code blocks first to keep their contents intact.
    code_blocks: list[str] = []

    def _on_fence(match: re.Match[str]) -> str:
        code = html.escape(match.group(2).rstrip("\n"))
        code_blocks.append(
            "<pre style='background-color:#f4f4f4;border:1px solid #ddd;"
            "padding:6px;font-family:Consolas,monospace;'>"
            f"<code>{code}</code></pre>"
        )
        return f"\x01{len(code_blocks) - 1}\x01"

    text = _FENCE_RE.sub(_on_fence, text)

    out: list[str] = []
    list_stack: list[str] = []  # "ul" or "ol"
    paragraph: list[str] = []

    def _flush_paragraph() -> None:
        if paragraph:
            joined = "<br>".join(_render_inline(line) for line in paragraph)
            out.append(f"<p style='margin:4px 0;'>{joined}</p>")
            paragraph.clear()

    for raw_line in text.splitlines():
        # Restore code-block placeholder lines as-is.
        placeholder_match = re.fullmatch(r"\x01(\d+)\x01", raw_line.strip())
        if placeholder_match:
            _flush_paragraph()
            _close_lists(list_stack, out)
            out.append(code_blocks[int(placeholder_match.group(1))])
            continue

        line = raw_line.rstrip()

        if not line.strip():
            _flush_paragraph()
            _close_lists(list_stack, out)
            continue

        if _HR_RE.match(line):
            _flush_paragraph()
            _close_lists(list_stack, out)
            out.append("<hr>")
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            _flush_paragraph()
            _close_lists(list_stack, out)
            level = len(heading_match.group(1))
            content = _render_inline(heading_match.group(2))
            out.append(f"<h{level} style='margin:8px 0 4px 0;'>{content}</h{level}>")
            continue

        ul_match = _UL_RE.match(line)
        ol_match = _OL_RE.match(line)
        if ul_match or ol_match:
            _flush_paragraph()
            _close_lists(list_stack, out)
            if ul_match:
                marker = "•"
                content = ul_match.group(2)
            else:
                marker = f"{ol_match.group(2)}."
                content = ol_match.group(3)
            out.append(
                "<p style='margin:1px 0 1px 6px;-qt-block-indent:0;"
                "text-indent:0;'>"
                f"{marker}&nbsp; {_render_inline(content)}</p>"
            )
            continue

        bq_match = _BLOCKQUOTE_RE.match(line)
        if bq_match:
            _flush_paragraph()
            _close_lists(list_stack, out)
            out.append(
                "<blockquote style='margin:4px 0 4px 8px;padding-left:8px;"
                "border-left:3px solid #ccc;color:#555;'>"
                f"{_render_inline(bq_match.group(1))}</blockquote>"
            )
            continue

        # Default: accumulate into a paragraph (close any open list first).
        if list_stack:
            _close_lists(list_stack, out)
        paragraph.append(line)

    _flush_paragraph()
    _close_lists(list_stack, out)

    return "".join(out)
