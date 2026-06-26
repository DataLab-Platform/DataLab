# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title delegate
==============

The :mod:`datalab.widgets.titledelegate` module provides a
:class:`QStyledItemDelegate` that renders object tree titles as rich text and
turns embedded short IDs (e.g. ``s001``, ``i012``) into clickable hyperlinks.

.. autoclass:: ClickableTitleDelegate
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.objectmodel import find_short_ids_in_title

if TYPE_CHECKING:
    pass


#: URL scheme used in anchors emitted by :class:`ClickableTitleDelegate`.
SHORT_ID_URL_SCHEME = "dlb-shortid"

#: Item data role holding explicit link spans for the display text, as a list of
#: ``(start, end, target_short_id)`` tuples. Used in "title" (long-name) display
#: mode, where the resolved long names no longer contain the literal short IDs:
#: the spans tell the delegate which substrings to render as clickable links and
#: which short ID each one points to. When this role is empty/absent, the
#: delegate falls back to detecting literal short IDs in the text.
LINK_SPANS_ROLE = QC.Qt.UserRole + 100


def _links_from_text(text: str) -> list[tuple[int, int, str]]:
    """Return the link spans derived from the literal short IDs found in
    ``text``, skipping the leading ``"<short_id>:"`` prefix (which would just
    re-select the current item).

    Args:
        text: item display text

    Returns:
        List of ``(start, end, target_short_id)`` tuples, sorted by ``start``.
    """
    links: list[tuple[int, int, str]] = []
    for idx, (start, end, sid) in enumerate(find_short_ids_in_title(text)):
        if idx == 0 and start == 0:
            # Leading "s001:" — keep as plain text
            continue
        links.append((start, end, sid))
    return links


def _links_for_index(index: QC.QModelIndex) -> list[tuple[int, int, str]]:
    """Return the link spans ``(start, end, target_short_id)`` to render for
    ``index``.

    Explicit spans stored under :data:`LINK_SPANS_ROLE` take precedence (used in
    long-name display mode). Otherwise, literal short IDs found in the text are
    used, skipping the leading ``"<short_id>:"`` prefix (which would just
    re-select the current item).

    Args:
        index: model index of the item

    Returns:
        List of ``(start, end, target_short_id)`` tuples, sorted by ``start``.
    """
    spans = index.data(LINK_SPANS_ROLE)
    if spans:
        return [(int(start), int(end), str(target)) for start, end, target in spans]
    text = index.data(QC.Qt.DisplayRole) or ""
    if not isinstance(text, str):
        return []
    return _links_from_text(text)


def _build_html(text: str, links: list[tuple[int, int, str]]) -> str:
    """Build the HTML representation of ``text`` with ``links`` wrapped in
    anchors.

    Args:
        text: raw item display text.
        links: list of ``(start, end, target_short_id)`` spans to turn into
         anchors (the displayed substring is the anchor text, the target short
         ID is encoded in the anchor href).

    Returns:
        HTML string.
    """
    if not links:
        return escape(text)
    out: list[str] = []
    cursor = 0
    for start, end, target in links:
        out.append(escape(text[cursor:start]))
        out.append(
            f'<a href="{SHORT_ID_URL_SCHEME}:{target}">{escape(text[start:end])}</a>'
        )
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)


def _make_text_document(
    text: str,
    option: QW.QStyleOptionViewItem,
    link_color: QG.QColor,
    links: list[tuple[int, int, str]],
    text_color: QG.QColor | None = None,
) -> QG.QTextDocument:
    """Return a :class:`QTextDocument` rendering ``text`` with the styling
    inherited from ``option``.

    ``link_color`` (and optionally ``text_color``) are baked into the
    document's default style sheet, because :class:`QTextDocument` resolves
    anchor colors at parse time — the painting palette has no effect on
    them.
    """
    doc = QG.QTextDocument()
    doc.setDefaultFont(option.font)
    doc.setDocumentMargin(0)
    css_parts = [f"a {{ color: {link_color.name()}; text-decoration: underline; }}"]
    if text_color is not None:
        css_parts.append(f"body, p, span {{ color: {text_color.name()}; }}")
    doc.setDefaultStyleSheet(" ".join(css_parts))
    doc.setHtml(_build_html(text, links))
    return doc


class ClickableTitleDelegate(QW.QStyledItemDelegate):
    """Item delegate that renders object titles with clickable short IDs.

    The delegate uses a :class:`QTextDocument` to render an HTML version of the
    item's display text in which each embedded short ID — apart from the
    leading one — is wrapped in an anchor pointing at ``dlb-shortid:<id>``.

    Hit-testing is performed by :meth:`anchor_at`, which is meant to be called
    from the host view's ``mousePressEvent`` / ``mouseMoveEvent``.
    """

    # pylint: disable=invalid-name
    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Reimplement Qt method to paint the item via a QTextDocument."""
        text = index.data(QC.Qt.DisplayRole) or ""
        links = _links_for_index(index)
        if not isinstance(text, str) or not links:
            super().paint(painter, option, index)
            return
        opt = QW.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        # Let the style draw the background, focus rect and decoration
        # (icon), but not the text.
        opt.text = ""
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        style.drawControl(QW.QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        palette = option.palette
        selected = bool(option.state & QW.QStyle.State_Selected)
        # ``QPalette.Highlight`` is the theme's accent color (vivid in both
        # light and dark modes) — much more readable than the default
        # ``QPalette.Link`` role, which many themes leave at Qt's hard-coded
        # dark blue.
        accent = palette.color(QG.QPalette.Active, QG.QPalette.Highlight)
        if selected:
            text_color = palette.color(QG.QPalette.Active, QG.QPalette.HighlightedText)
            # The selection background IS the accent colour (QPalette.Highlight),
            # so using accent directly as link colour makes links invisible on both
            # light and dark themes. Blend HighlightedText (which always contrasts
            # with the selection background) with the accent at 2:1 to get a tinted
            # colour that is both visible against the selection background and
            # visually distinct from regular selected text:
            link_color = QG.QColor(
                (text_color.red() * 2 + accent.red()) // 3,
                (text_color.green() * 2 + accent.green()) // 3,
                (text_color.blue() * 2 + accent.blue()) // 3,
            )
        else:
            text_color = palette.color(QG.QPalette.Active, QG.QPalette.Text)
            link_color = accent
        doc = _make_text_document(text, option, link_color, links, text_color)
        doc.setTextWidth(text_rect.width())
        painter.save()
        painter.translate(text_rect.topLeft())
        ctx = QG.QAbstractTextDocumentLayout.PaintContext()
        clip = QC.QRectF(0, 0, text_rect.width(), text_rect.height())
        ctx.clip = clip
        painter.setClipRect(clip)
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def anchor_at(
        self,
        index: QC.QModelIndex,
        item_rect: QC.QRect,
        pos: QC.QPoint,
        option: QW.QStyleOptionViewItem,
    ) -> str | None:
        """Return the short ID under cursor position ``pos`` (in viewport
        coordinates) for ``index``, or ``None`` if the cursor is not over any
        anchor.

        Args:
            index: model index of the item under cursor
            item_rect: visual rectangle of the item in the viewport
            pos: cursor position in viewport coordinates
            option: style option (already initialized for the item)
        """
        text = index.data(QC.Qt.DisplayRole) or ""
        links = _links_for_index(index)
        if not isinstance(text, str) or not links:
            return None
        opt = QW.QStyleOptionViewItem(option)
        opt.rect = item_rect
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        if not text_rect.contains(pos):
            return None
        # Color does not influence hit-testing — pass any value.
        doc = _make_text_document(text, option, QG.QColor("black"), links)
        doc.setTextWidth(text_rect.width())
        local = QC.QPointF(pos - text_rect.topLeft())
        href = doc.documentLayout().anchorAt(local)
        if href and href.startswith(f"{SHORT_ID_URL_SCHEME}:"):
            return href[len(SHORT_ID_URL_SCHEME) + 1 :]
        return None
