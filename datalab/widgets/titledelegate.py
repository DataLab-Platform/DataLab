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


def _build_html(text: str) -> str:
    """Build the HTML representation of ``text`` with short IDs wrapped in
    anchors.

    The first short ID occurrence is always rendered as plain text: object
    titles in DataLab tree views are formatted as ``"<short_id>: <title>"`` and
    making the leading ``s001`` clickable would just re-select the current
    item.

    Args:
        text: raw item text (e.g. ``"s003: average(s001, s002)"``).

    Returns:
        HTML string.
    """
    matches = find_short_ids_in_title(text)
    if not matches:
        return escape(text)
    out: list[str] = []
    cursor = 0
    for idx, (start, end, sid) in enumerate(matches):
        out.append(escape(text[cursor:start]))
        if idx == 0 and start == 0:
            # Leading "s001:" — keep as plain text
            out.append(escape(sid))
        else:
            out.append(f'<a href="{SHORT_ID_URL_SCHEME}:{sid}">{escape(sid)}</a>')
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)


def _make_text_document(
    text: str,
    option: QW.QStyleOptionViewItem,
    link_color: QG.QColor,
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
    doc.setHtml(_build_html(text))
    return doc


class ClickableTitleDelegate(QW.QStyledItemDelegate):
    """Item delegate that renders object titles with clickable short IDs.

    The delegate uses a :class:`QTextDocument` to render an HTML version of the
    item's display text in which each embedded short ID — apart from the
    leading one — is wrapped in an anchor pointing at ``dlb-shortid:<id>``.

    Hit-testing is performed by :meth:`anchor_at`, which is meant to be called
    from the host view's ``mousePressEvent`` / ``mouseMoveEvent``.
    """

    def __init__(self, parent: QW.QAbstractItemView) -> None:
        super().__init__(parent)

    # pylint: disable=invalid-name
    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Reimplement Qt method to paint the item via a QTextDocument."""
        text = index.data(QC.Qt.DisplayRole) or ""
        if not isinstance(text, str) or not find_short_ids_in_title(text):
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
            # On dark themes the selection background *is* the accent color,
            # so a plain accent-colored link would vanish: blend it 50/50
            # with ``HighlightedText`` (typically white) to obtain a lighter
            # tint that still reads as the same hue. On light themes the
            # accent stays distinguishable on the highlight background, so
            # we keep the unselected color for visual consistency.
            base_is_light = (
                palette.color(QG.QPalette.Active, QG.QPalette.Base).lightness() > 128
            )
            if base_is_light:
                link_color = accent
            else:
                link_color = QG.QColor(
                    (accent.red() + text_color.red()) // 2,
                    (accent.green() + text_color.green()) // 2,
                    (accent.blue() + text_color.blue()) // 2,
                )
        else:
            text_color = palette.color(QG.QPalette.Active, QG.QPalette.Text)
            link_color = accent
        doc = _make_text_document(text, option, link_color, text_color)
        doc.setTextWidth(text_rect.width())
        painter.save()
        painter.translate(text_rect.topLeft())
        ctx = QG.QAbstractTextDocumentLayout.PaintContext()
        clip = QC.QRectF(0, 0, text_rect.width(), text_rect.height())
        ctx.clip = clip
        painter.setClipRect(clip)
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    # pylint: disable=invalid-name
    def sizeHint(
        self, option: QW.QStyleOptionViewItem, index: QC.QModelIndex
    ) -> QC.QSize:
        """Reimplement Qt method to size items consistently with default text
        rendering."""
        return super().sizeHint(option, index)

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
        if not isinstance(text, str) or not find_short_ids_in_title(text):
            return None
        opt = QW.QStyleOptionViewItem(option)
        opt.rect = item_rect
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        if not text_rect.contains(pos):
            return None
        # Color does not influence hit-testing — pass any value.
        doc = _make_text_document(text, option, QG.QColor("black"))
        doc.setTextWidth(text_rect.width())
        local = QC.QPointF(pos - text_rect.topLeft())
        href = doc.documentLayout().anchorAt(local)
        if href and href.startswith(f"{SHORT_ID_URL_SCHEME}:"):
            return href[len(SHORT_ID_URL_SCHEME) + 1 :]
        return None
