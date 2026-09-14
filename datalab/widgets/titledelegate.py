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

from collections.abc import Callable
from html import escape
from math import ceil
from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.objectmodel import (
    UUID_DISPLAY_LENGTH,
    find_short_ids_in_title,
    find_uuids_in_title,
)

if TYPE_CHECKING:
    pass


#: URL scheme used in anchors emitted by :class:`ClickableTitleDelegate`.
SHORT_ID_URL_SCHEME = "dlb-shortid"
UUID_URL_SCHEME = "dlb-uuid"
TITLE_UUID_ROLE = QC.Qt.UserRole + 1
TITLE_META_FONT_SIZE = "90%"
TITLE_ROW_VERTICAL_PADDING = 2


def _build_html(
    text: str, uuid_title_resolver: Callable[[str], str | None] | None = None
) -> str:
    """Build the HTML representation of ``text`` with references as anchors.

    The first short ID occurrence is always rendered as plain text: object
    titles in DataLab tree views are formatted as ``"<short_id>: <title>"`` and
    making the leading ``s001`` clickable would just re-select the current
    item.

    Args:
        text: raw item text (e.g. ``"s003: average(s001, s002)"``).

    Returns:
        HTML string.
    """
    matches: list[tuple[int, int, str, str | None, str]] = [
        (*match, SHORT_ID_URL_SCHEME, match[2])
        for match in find_short_ids_in_title(text)
    ]
    for match in find_uuids_in_title(text):
        uuid = match[2]
        display_reference = uuid[:UUID_DISPLAY_LENGTH]
        resolved_display = (
            uuid_title_resolver(uuid)
            if uuid_title_resolver is not None
            else display_reference
        )
        if resolved_display is not None:
            display_reference = resolved_display
        scheme = UUID_URL_SCHEME if resolved_display is not None else None
        matches.append((*match, scheme, display_reference))
    matches.sort(key=lambda match: match[0])
    if not matches:
        return escape(text)
    out: list[str] = []
    cursor = 0
    for start, end, reference, scheme, display_reference in matches:
        out.append(escape(text[cursor:start]))
        if scheme is None or (scheme == SHORT_ID_URL_SCHEME and start == 0):
            # Leading "s001:" — keep as plain text
            out.append(escape(display_reference))
        else:
            out.append(
                f'<a href="{scheme}:{reference}">{escape(display_reference)}</a>'
            )
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)


def _make_text_document(
    text: str,
    option: QW.QStyleOptionViewItem,
    link_color: QG.QColor,
    text_color: QG.QColor | None = None,
    uuid_title_resolver: Callable[[str], str | None] | None = None,
    item_uuid: str | None = None,
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
    title_html = _build_html(text, uuid_title_resolver)
    if item_uuid:
        short_uuid = escape(item_uuid[:UUID_DISPLAY_LENGTH])
        own_id_html = f'<span class="title-meta">#{short_uuid}</span>'
        title_html = f"{title_html}<br>{own_id_html}" if title_html else own_id_html
    doc.setHtml(title_html)
    return doc


class ClickableTitleDelegate(QW.QStyledItemDelegate):
    """Item delegate that renders object titles with clickable references.

    The delegate uses a :class:`QTextDocument` to render an HTML version of the
    item's display text in which each embedded short ID — apart from the
    leading one — is wrapped in an anchor pointing at ``dlb-shortid:<id>``.

    Hit-testing is performed by :meth:`anchor_at`, which is meant to be called
    from the host view's ``mousePressEvent`` / ``mouseMoveEvent``.
    """

    def __init__(
        self,
        parent: QW.QWidget,
        uuid_title_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.uuid_title_resolver = uuid_title_resolver

    # pylint: disable=invalid-name
    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Reimplement Qt method to paint the item via a QTextDocument."""
        text = index.data(QC.Qt.DisplayRole) or ""
        item_uuid = index.data(TITLE_UUID_ROLE)
        has_references = bool(
            find_short_ids_in_title(text) or find_uuids_in_title(text)
        )
        if not isinstance(text, str) or not (
            has_references or isinstance(item_uuid, str)
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
            self.uuid_title_resolver,
            item_uuid if isinstance(item_uuid, str) else None,
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
        """Return a size accommodating the title and own UUID metadata line."""
        text = index.data(QC.Qt.DisplayRole) or ""
        item_uuid = index.data(TITLE_UUID_ROLE)
        has_references = isinstance(text, str) and bool(
            find_short_ids_in_title(text) or find_uuids_in_title(text)
        )
        if not isinstance(text, str) or not (
            has_references or isinstance(item_uuid, str)
        ):
            return super().sizeHint(option, index)
        opt = QW.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        doc = _make_text_document(
            text,
            opt,
            QG.QColor("black"),
            uuid_title_resolver=self.uuid_title_resolver,
            item_uuid=item_uuid if isinstance(item_uuid, str) else None,
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
        item_uuid = index.data(TITLE_UUID_ROLE)
        has_references = bool(
            find_short_ids_in_title(text) or find_uuids_in_title(text)
        )
        if not isinstance(text, str) or not has_references:
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
            uuid_title_resolver=self.uuid_title_resolver,
            item_uuid=item_uuid if isinstance(item_uuid, str) else None,
            meta_color=QG.QColor("gray"),
        )
        doc.setTextWidth(text_rect.width())
        local = QC.QPointF(pos - text_rect.topLeft())
        href = doc.documentLayout().anchorAt(local)
        for scheme in (SHORT_ID_URL_SCHEME, UUID_URL_SCHEME):
            if href and href.startswith(f"{scheme}:"):
                return href[len(scheme) + 1 :]
        return None
