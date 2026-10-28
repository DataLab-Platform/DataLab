# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for rich object-title sizing, colors and links."""

from __future__ import annotations

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.widgets import titledelegate
from datalab.widgets.titledelegate import (
    TITLE_REFERENCE_ROLE,
    ClickableTitleDelegate,
    hover_background_color,
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
        if selected:
            assert background == QG.QColor("#204a70" if dark else "#cce8ff")
        elif hovered and not active:
            # Same discreet highlight whatever the style, distinct from the base
            # and selection colors (styles may draw their focus indicator over
            # the background of the item having the focus, e.g. Fusion)
            assert background == hover_background_color(palette)
            assert background not in (
                palette.color(QG.QPalette.Base),
                QG.QColor("#204a70" if dark else "#cce8ff"),
            )
        assert len({link.name(), text.name(), meta.name()}) == 3
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
        option = view.viewOptions()
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
