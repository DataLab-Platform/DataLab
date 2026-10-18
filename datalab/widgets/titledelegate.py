# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title delegate
==============

The :mod:`datalab.widgets.titledelegate` module provides a
:class:`QStyledItemDelegate` that renders object tree titles as rich text and
turns embedded object references (e.g. ``1a2b3c4d``, ``g1a2b3c4d``) into
clickable hyperlinks.

.. autoclass:: ClickableTitleDelegate
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Callable
from html import escape
from math import ceil
from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.objectmodel import find_title_references

if TYPE_CHECKING:
    pass


#: URL scheme used in anchors emitted by :class:`ClickableTitleDelegate`.
REFERENCE_URL_SCHEME = "dlb-ref"
#: Item data role holding the item's own title reference (``<UUID8>``/``g<UUID8>``)
TITLE_REFERENCE_ROLE = QC.Qt.UserRole + 1
TITLE_META_FONT_SIZE = "90%"
TITLE_ROW_VERTICAL_PADDING = 2


def _build_html(
    text: str, reference_resolver: Callable[[str], str | None] | None = None
) -> str:
    """Build the HTML representation of ``text`` with references as anchors.

    Only references resolved by ``reference_resolver`` become links; the others
    are kept as plain text.

    Args:
        text: raw item text (e.g. ``"1a2b3c4d-5e6f7a8b"``).
        reference_resolver: callback returning the display text of a reference,
         or None when the reference cannot be resolved.

    Returns:
        HTML string.
    """
    out: list[str] = []
    cursor = 0
    for start, end, reference in find_title_references(text):
        out.append(escape(text[cursor:start]))
        display = None if reference_resolver is None else reference_resolver(reference)
        if display is None:
            out.append(escape(reference))
        else:
            out.append(
                f'<a href="{REFERENCE_URL_SCHEME}:{reference}">{escape(display)}</a>'
            )
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)


def _make_text_document(
    text: str,
    option: QW.QStyleOptionViewItem,
    link_color: QG.QColor,
    text_color: QG.QColor | None = None,
    reference_resolver: Callable[[str], str | None] | None = None,
    item_reference: str | None = None,
    meta_color: QG.QColor | None = None,
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
    if meta_color is not None:
        css_parts.append(
            ".title-meta { "
            f"color: {meta_color.name()}; font-size: {TITLE_META_FONT_SIZE}; "
            "}"
        )
    doc.setDefaultStyleSheet(" ".join(css_parts))
    title_html = _build_html(text, reference_resolver)
    if item_reference:
        own_id_html = f'<span class="title-meta">#{escape(item_reference)}</span>'
        title_html = f"{title_html}<br>{own_id_html}" if title_html else own_id_html
    doc.setHtml(title_html)
    return doc


class ClickableTitleDelegate(QW.QStyledItemDelegate):
    """Item delegate that renders object titles with clickable references.

    The delegate uses a :class:`QTextDocument` to render an HTML version of the
    item's display text in which each resolvable object or group reference is
    wrapped in an anchor pointing at ``dlb-ref:<reference>``.

    Hit-testing is performed by :meth:`anchor_at`, which is meant to be called
    from the host view's ``mousePressEvent`` / ``mouseMoveEvent``.
    """

    def __init__(
        self,
        parent: QW.QWidget,
        reference_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.reference_resolver = reference_resolver

    # pylint: disable=invalid-name
    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Reimplement Qt method to paint the item via a QTextDocument."""
        text = index.data(QC.Qt.DisplayRole) or ""
        item_reference = index.data(TITLE_REFERENCE_ROLE)
        if not isinstance(text, str) or not (
            find_title_references(text) or isinstance(item_reference, str)
        ):
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
            # The selection background uses ``Highlight`` itself, so links must
            # use the theme's contrasting text role. Underlining preserves the
            # link affordance even when link and selected text share a color.
            link_color = text_color
            meta_color = text_color
        else:
            text_color = palette.color(QG.QPalette.Active, QG.QPalette.Text)
            link_color = accent
            meta_color = palette.color(QG.QPalette.Active, QG.QPalette.Mid)
        doc = _make_text_document(
            text,
            option,
            link_color,
            text_color,
            self.reference_resolver,
            item_reference if isinstance(item_reference, str) else None,
            meta_color,
        )
        doc.setTextWidth(text_rect.width())
        painter.save()
        painter.translate(text_rect.topLeft())
        ctx = QG.QAbstractTextDocumentLayout.PaintContext()
        clip = QC.QRectF(0, 0, text_rect.width(), text_rect.height())
        ctx.clip = clip
        painter.setClipRect(clip)
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(
        self, option: QW.QStyleOptionViewItem, index: QC.QModelIndex
    ) -> QC.QSize:
        """Return a size accommodating the title and own identity line."""
        text = index.data(QC.Qt.DisplayRole) or ""
        item_reference = index.data(TITLE_REFERENCE_ROLE)
        if not isinstance(text, str) or not (
            find_title_references(text) or isinstance(item_reference, str)
        ):
            return super().sizeHint(option, index)
        opt = QW.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        doc = _make_text_document(
            text,
            opt,
            QG.QColor("black"),
            reference_resolver=self.reference_resolver,
            item_reference=item_reference if isinstance(item_reference, str) else None,
            meta_color=QG.QColor("gray"),
        )
        base_size = super().sizeHint(opt, index)
        raw_text_width = opt.fontMetrics.horizontalAdvance(text)
        horizontal_chrome = max(0, base_size.width() - raw_text_width)
        width = max(base_size.width(), ceil(doc.idealWidth()) + horizontal_chrome)
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        if text_rect.width() > 0:
            doc.setTextWidth(text_rect.width())
        height = max(
            base_size.height(), ceil(doc.size().height()) + TITLE_ROW_VERTICAL_PADDING
        )
        return QC.QSize(width, height)

    def anchor_at(
        self,
        index: QC.QModelIndex,
        item_rect: QC.QRect,
        pos: QC.QPoint,
        option: QW.QStyleOptionViewItem,
    ) -> str | None:
        """Return the reference under cursor position ``pos`` (in viewport
        coordinates) for ``index``, or ``None`` if the cursor is not over any
        anchor.

        Args:
            index: model index of the item under cursor
            item_rect: visual rectangle of the item in the viewport
            pos: cursor position in viewport coordinates
            option: style option (already initialized for the item)
        """
        text = index.data(QC.Qt.DisplayRole) or ""
        item_reference = index.data(TITLE_REFERENCE_ROLE)
        if not isinstance(text, str) or not find_title_references(text):
            return None
        opt = QW.QStyleOptionViewItem(option)
        opt.rect = item_rect
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        if not text_rect.contains(pos):
            return None
        # Color does not influence hit-testing — pass any value.
        doc = _make_text_document(
            text,
            option,
            QG.QColor("black"),
            reference_resolver=self.reference_resolver,
            item_reference=item_reference if isinstance(item_reference, str) else None,
            meta_color=QG.QColor("gray"),
        )
        doc.setTextWidth(text_rect.width())
        local = QC.QPointF(pos - text_rect.topLeft())
        href = doc.documentLayout().anchorAt(local)
        if href and href.startswith(f"{REFERENCE_URL_SCHEME}:"):
            return href[len(REFERENCE_URL_SCHEME) + 1 :]
        return None
