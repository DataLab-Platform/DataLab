# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for rich object-title sizing, colors and links."""

from __future__ import annotations

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.utils.qthelpers import view_item_option
from datalab.widgets import titledelegate
from datalab.widgets.titledelegate import (
    TITLE_REFERENCE_ROLE,
    ClickableTitleDelegate,
)


def _contrast(first: QG.QColor, second: QG.QColor) -> float:
    """Return the WCAG contrast ratio of two opaque colors."""

    def luminance(color: QG.QColor) -> float:
        channels = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in (color.redF(), color.greenF(), color.blueF())
        ]
        return sum(
            value * weight for value, weight in zip(channels, (0.2126, 0.7152, 0.0722))
        )

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.mark.parametrize("style_name", QW.QStyleFactory.keys())
@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("hovered", [False, True])
def test_reference_rendering(
    style_name: str,
    dark: bool,
    selected: bool,
    active: bool,
    hovered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep actual backgrounds, text and clickable references consistent."""
    with qt_app_context():
        parent = QW.QTreeWidget()
        style = QW.QStyleFactory.create(style_name)
        parent.setStyle(style)
        palette = QG.QPalette()
        palette.setColor(QG.QPalette.Base, QG.QColor("#1f1f1f" if dark else "#ffffff"))
        palette.setColor(QG.QPalette.Text, QG.QColor("white" if dark else "black"))
        palette.setColor(
            QG.QPalette.Highlight, QG.QColor("#2a82da" if dark else "#0078d7")
        )
        palette.setColor(
            QG.QPalette.HighlightedText, QG.QColor("black" if dark else "white")
        )
        parent.setPalette(palette)
        model = QG.QStandardItemModel()
        item = QG.QStandardItem("normalize(12345678)")
        item.setData("87654321", TITLE_REFERENCE_ROLE)
        model.appendRow(item)
        delegate = ClickableTitleDelegate(parent, lambda reference: reference)
        option = QW.QStyleOptionViewItem()
        option.initFrom(parent)
        option.widget = parent
        option.rect = QC.QRect(0, 0, 400, 60)
        option.showDecorationSelected = True
        option.features |= QW.QStyleOptionViewItem.HasDisplay
        option.state = QW.QStyle.State_Enabled
        if selected:
            option.state |= QW.QStyle.State_Selected
        if active:
            option.state |= QW.QStyle.State_Active | QW.QStyle.State_HasFocus
        if hovered:
            option.state |= QW.QStyle.State_MouseOver
        documents = []
        make_document = titledelegate._make_text_document

        def record_document(*args, **kwargs):
            document = make_document(*args, **kwargs)
            documents.append((document, args[2], args[3], args[6]))
            return document

        monkeypatch.setattr(titledelegate, "_make_text_document", record_document)
        image = QG.QImage(400, 60, QG.QImage.Format_ARGB32)
        image.fill(palette.color(QG.QPalette.Base))
        painter = QG.QPainter(image)
        delegate.paint(painter, option, model.index(0, 0))
        painter.end()
        document, link, text, meta = documents[0]
        background = image.pixelColor(350, 30)
        # The background (selection, hover) is drawn by the style, as for items
        # rendered by the standard delegate
        native = QG.QImage(400, 60, QG.QImage.Format_ARGB32)
        native.fill(palette.color(QG.QPalette.Base))
        painter = QG.QPainter(native)
        QW.QStyledItemDelegate(parent).paint(painter, option, model.index(0, 0))
        painter.end()
        assert background == native.pixelColor(350, 30)
        for color in (link, text, meta):
            assert _contrast(color, background) >= 4.5
        # Inspect the rendered pixels too: unformatted title text must not use
        # QTextDocument's default (black) painting palette in dark mode.
        assert any(
            max(
                abs(image.pixelColor(x, y).red() - text.red()),
                abs(image.pixelColor(x, y).green() - text.green()),
                abs(image.pixelColor(x, y).blue() - text.blue()),
            )
            < 30
            for y in range(image.height())
            for x in range(2, 70)
        )
        cursor = QG.QTextCursor(document)
        cursor.setPosition(document.toPlainText().index("12345678"))
        cursor.movePosition(QG.QTextCursor.NextCharacter, QG.QTextCursor.KeepAnchor)
        assert cursor.charFormat().isAnchor()
        assert cursor.charFormat().fontUnderline()
        assert cursor.charFormat().foreground().color() == link
        assert "87654321" in document.toPlainText()
        monkeypatch.setattr(titledelegate, "_make_text_document", make_document)
        assert any(
            delegate.anchor_at(model.index(0, 0), option.rect, QC.QPoint(x, y), option)
            == "12345678"
            for y in range(2, image.height(), 3)
            for x in range(50, 170, 3)
        )
        parent.close()


def _title_colors_option(palette: QG.QPalette) -> QW.QStyleOptionViewItem:
    """Return the style option of an enabled, active item with ``palette``"""
    option = QW.QStyleOptionViewItem()
    option.palette = palette
    option.rect = QC.QRect(0, 0, 400, 40)
    option.state = QW.QStyle.State_Enabled | QW.QStyle.State_Active
    # As set by item views (with Qt 5, styles only fill the selected item then)
    option.showDecorationSelected = True
    return option


def test_title_colors_with_translucent_palette() -> None:
    """Translucent palette colors (e.g. "windows11" style in light mode) are
    composited over the background: the title remains dark on alternate rows."""
    with qt_app_context():
        palette = QG.QPalette()
        palette.setColor(QG.QPalette.Base, QG.QColor("#ffffff"))
        palette.setColor(QG.QPalette.AlternateBase, QG.QColor.fromRgb(0, 0, 0, 9))
        palette.setColor(QG.QPalette.Text, QG.QColor.fromRgb(0, 0, 0, 228))
        palette.setColor(QG.QPalette.HighlightedText, QG.QColor("#ffffff"))
        option = _title_colors_option(palette)
        option.features |= QW.QStyleOptionViewItem.Alternate
        style = QW.QStyleFactory.create("Fusion")
        text, link, meta = titledelegate._title_colors(option, style)  # pylint: disable=protected-access
        background = QG.QColor.fromRgb(246, 246, 246)
        assert text.lightness() < 64
        for color in (text, link, meta):
            assert color.alpha() == 255
            assert _contrast(color, background) >= 4.5


def test_title_colors_without_readable_palette_text() -> None:
    """When no text color of the palette is readable on the selection drawn by
    the style (e.g. light accent color of the "windows11" style in dark mode), the
    title is drawn in black or white."""
    with qt_app_context():
        palette = QG.QPalette()
        palette.setColor(QG.QPalette.Base, QG.QColor("#2d2d2d"))
        palette.setColor(QG.QPalette.Highlight, QG.QColor("#4cc2ff"))
        palette.setColor(QG.QPalette.Text, QG.QColor("#ffffff"))
        palette.setColor(QG.QPalette.HighlightedText, QG.QColor("#ffffff"))
        option = _title_colors_option(palette)
        option.state |= QW.QStyle.State_Selected
        style = QW.QStyleFactory.create("Fusion")
        text, link, meta = titledelegate._title_colors(option, style)  # pylint: disable=protected-access
        assert text == QG.QColor("#000000")
        for color in (link, meta):
            assert _contrast(color, QG.QColor("#4cc2ff")) >= 4.5


@pytest.mark.parametrize("with_item_uuid", [False, True])
def test_size_hint_height_tracks_painted_wrapping(with_item_uuid: bool) -> None:
    """Increase row height when the same rich title is painted more narrowly."""
    with qt_app_context():
        referenced_uuid = "12345678"
        model = QG.QStandardItemModel()
        item = QG.QStandardItem(
            f"A deliberately long computed title referencing {referenced_uuid}"
        )
        if with_item_uuid:
            item.setData("87654321", TITLE_REFERENCE_ROLE)
        model.appendRow(item)
        parent = QW.QWidget()
        delegate = ClickableTitleDelegate(parent)

        wide_option = QW.QStyleOptionViewItem()
        wide_option.widget = parent
        wide_option.rect = QC.QRect(0, 0, 500, 200)
        narrow_option = QW.QStyleOptionViewItem(wide_option)
        narrow_option.rect = QC.QRect(0, 0, 120, 200)

        wide_height = delegate.sizeHint(wide_option, model.index(0, 0)).height()
        narrow_height = delegate.sizeHint(narrow_option, model.index(0, 0)).height()

        assert narrow_height > wide_height


def test_text_clip_provider_leaves_room_for_view_buttons() -> None:
    """Do not draw rich titles beyond the X coordinate given by the view."""
    with qt_app_context():
        view = QW.QTreeView()
        model = QG.QStandardItemModel()
        item = QG.QStandardItem("Title referencing 12345678, " * 4)
        item.setData("87654321", TITLE_REFERENCE_ROLE)
        model.appendRow(item)
        view.setModel(model)
        delegate = ClickableTitleDelegate(view, lambda reference: reference)
        option = view_item_option(view)
        option.rect = QC.QRect(0, 0, 800, 40)

        def last_painted_column(clip_right: int | None) -> int:
            """Return the rightmost painted column with the given clip"""
            delegate.text_clip_provider = lambda _index: clip_right
            image = QG.QImage(800, 40, QG.QImage.Format_ARGB32)
            image.fill(QC.Qt.transparent)
            painter = QG.QPainter(image)
            delegate.paint(painter, option, model.index(0, 0))
            painter.end()
            return max(
                (
                    x
                    for x in range(image.width())
                    for y in range(image.height())
                    if image.pixelColor(x, y).alpha() > 0
                ),
                default=-1,
            )

        assert last_painted_column(None) > 300
        assert 0 < last_painted_column(300) < 300
        view.close()
