# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for rich object-title sizing."""

from __future__ import annotations

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.widgets.titledelegate import TITLE_UUID_ROLE, ClickableTitleDelegate


@pytest.mark.parametrize("with_item_uuid", [False, True])
def test_size_hint_height_tracks_painted_wrapping(with_item_uuid: bool) -> None:
    """Increase row height when the same rich title is painted more narrowly."""
    with qt_app_context():
        referenced_uuid = "12345678-1234-5678-1234-567812345678"
        model = QG.QStandardItemModel()
        item = QG.QStandardItem(
            f"A deliberately long computed title referencing {referenced_uuid}"
        )
        if with_item_uuid:
            item.setData("87654321-4321-8765-4321-876543218765", TITLE_UUID_ROLE)
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
