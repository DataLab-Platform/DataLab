# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title delegate
==============

The :mod:`datalab.widgets.titledelegate` module provides a
:class:`QStyledItemDelegate` that renders object tree titles as rich text and
turns embedded object references (e.g. ``1a2b3c4d``, ``g1a2b3c4d``) into
clickable hyperlinks. The background of items (selection, hover, focus) is drawn
by the Qt style, and the title colors are chosen to remain readable on it.

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
#: Minimum contrast ratio (WCAG) of the title colors with the item background
MIN_TITLE_CONTRAST = 4.5
#: Width of the item drawn off-screen to get the background color of the style
BACKGROUND_PROBE_WIDTH = 64


def _composite(color: QG.QColor, background: QG.QColor) -> QG.QColor:
    """Return the opaque color of ``color`` drawn over an opaque ``background``
    (palettes may have translucent colors, e.g. with the "windows11" style)"""
    alpha = color.alphaF()
    return QG.QColor.fromRgbF(
        *(
            alpha * channel + (1 - alpha) * base
            for channel, base in zip(color.getRgbF()[:3], background.getRgbF()[:3])
        )
    )


def _contrast(first: QG.QColor, second: QG.QColor) -> float:
    """Return the contrast ratio (WCAG) of two opaque colors"""

    def luminance(color: QG.QColor) -> float:
        channels = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in (color.redF(), color.greenF(), color.blueF())
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _muted_color(color: QG.QColor, background: QG.QColor) -> QG.QColor:
    """Return ``color`` blended with ``background``, as much as possible while
    keeping a readable contrast (for secondary texts)"""
    for weight in (0.6, 0.7, 0.8, 0.9):
        muted = QG.QColor(color)
        muted.setAlphaF(weight)
        muted = _composite(muted, background)
        if _contrast(muted, background) >= MIN_TITLE_CONTRAST:
            return muted
    return QG.QColor(color)


def _style_background(option: QW.QStyleOptionViewItem, style: QW.QStyle) -> QG.QColor:
    """Return the background color drawn by the style behind the text of an item

    The item is drawn off-screen without text over the base color of the view: the
    background of selected or hovered items depends on the style (e.g. a light
    selection with the "windowsvista" style, an accent color with the "windows11"
    style and alternating row colors), whatever the Highlight color of the palette.

    Args:
        option: style option of the item
        style: style drawing the item

    Returns:
        Background color (opaque).
    """
    base = option.palette.color(QG.QPalette.Base)
    if option.features & QW.QStyleOptionViewItem.Alternate:
        base = _composite(option.palette.color(QG.QPalette.AlternateBase), base)
    if not option.state & (QW.QStyle.State_Selected | QW.QStyle.State_MouseOver):
        return base
    probe = QW.QStyleOptionViewItem(option)
    probe.rect = QC.QRect(0, 0, BACKGROUND_PROBE_WIDTH, option.rect.height())
    probe.text = ""
    probe.icon = QG.QIcon()
    probe.features &= ~QW.QStyleOptionViewItem.HasDecoration
    image = QG.QImage(probe.rect.size(), QG.QImage.Format_ARGB32_Premultiplied)
    image.fill(base)
    painter = QG.QPainter(image)
    style.drawControl(QW.QStyle.CE_ItemViewItem, probe, painter, option.widget)
    painter.end()
    return image.pixelColor(probe.rect.center())


def _title_colors(
    option: QW.QStyleOptionViewItem, style: QW.QStyle
) -> tuple[QG.QColor, QG.QColor, QG.QColor]:
    """Return the colors of the title, its references and its own identity

    The colors come from the palette (text, highlighted text, link), and are chosen
    to remain readable on the background drawn by the style.

    Args:
        option: style option of the item
        style: style drawing the item

    Returns:
        Text, link and identity colors (opaque).
    """
    palette = option.palette
    if not option.state & QW.QStyle.State_Enabled:
        group = QG.QPalette.Disabled
    elif option.state & QW.QStyle.State_Active:
        group = QG.QPalette.Active
    else:
        group = QG.QPalette.Inactive
    background = _style_background(option, style)

    def contrast(color: QG.QColor) -> float:
        """Return the contrast of a color with the background"""
        return _contrast(color, background)

    text_color = _composite(palette.color(group, QG.QPalette.Text), background)
    highlighted = _composite(
        palette.color(group, QG.QPalette.HighlightedText), background
    )
    # Styles drawing a light selection keep the text color of the palette, and
    # the text color must remain readable on any background drawn by the style
    if (
        option.state & QW.QStyle.State_Selected
        or contrast(text_color) < MIN_TITLE_CONTRAST
    ) and contrast(highlighted) > contrast(text_color):
        text_color = highlighted
    if contrast(text_color) < MIN_TITLE_CONTRAST:
        # No text color of the palette fits the background (e.g. "windows11" style
        # drawing the selection with a light accent color in dark mode)
        text_color = max(QG.QColor(QC.Qt.black), QG.QColor(QC.Qt.white), key=contrast)
    link_color = _composite(palette.color(group, QG.QPalette.Link), background)
    if contrast(link_color) < MIN_TITLE_CONTRAST:
        # References remain underlined
        link_color = text_color
    return text_color, link_color, _muted_color(text_color, background)


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

    The background of items (selection, hover, focus) is drawn by the Qt style:
    the colors of the title come from the palette, and are chosen to remain
    readable on this background.
    """

    def __init__(
        self,
        parent: QW.QWidget,
        reference_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.reference_resolver = reference_resolver
        #: Optional callback returning, for a model index, the viewport X
        #: coordinate beyond which its rich title must not be drawn (e.g. to
        #: leave room for a button drawn by the view over the item), or None.
        self.text_clip_provider: Callable[[QC.QModelIndex], int | None] | None = None

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
        style = opt.widget.style() if opt.widget else QW.QApplication.style()
        text_color, link_color, meta_color = _title_colors(opt, style)
        # Let the style draw the background (selection, hover), focus rect and
        # decoration (icon), but not the text.
        opt.text = ""
        style.drawControl(QW.QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        text_rect = style.subElementRect(QW.QStyle.SE_ItemViewItemText, opt, opt.widget)
        doc = _make_text_document(
            text,
            opt,
            link_color,
            text_color,
            self.reference_resolver,
            item_reference if isinstance(item_reference, str) else None,
            meta_color,
        )
        doc.setTextWidth(text_rect.width())
        clip_width = text_rect.width()
        if self.text_clip_provider is not None:
            clip_right = self.text_clip_provider(index)
            if clip_right is not None:
                clip_width = max(0, min(clip_width, clip_right - text_rect.left()))
        painter.save()
        painter.translate(text_rect.topLeft())
        ctx = QG.QAbstractTextDocumentLayout.PaintContext()
        # Unformatted fragments do not inherit the document's body CSS.
        ctx.palette = QG.QPalette(opt.palette)
        ctx.palette.setColor(QG.QPalette.Text, text_color)
        clip = QC.QRectF(0, 0, clip_width, text_rect.height())
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
