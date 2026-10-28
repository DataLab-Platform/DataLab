# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Delete button of the object view: geometry, tooltips, deletion and settings.

The delete button is shown at the right of the hovered signal, image or group in
the object lists of the main window (as in DataLab-Web).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.QtTest import QTest
from sigima.objects import SignalObj, create_image, create_signal

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui.objectview import DELETE_BUTTON_MARGIN, ObjectView, SimpleObjectTree
from datalab.gui.panel.base import BaseDataPanel
from datalab.objectmodel import ObjectModel, get_uuid
from datalab.tests import datalab_test_app_context


def add_signals(panel: BaseDataPanel, *titles: str) -> list[SignalObj]:
    """Add signals to the current group of a panel

    Args:
        panel: signal panel
        titles: signal titles

    Returns:
        Added signals
    """
    x = np.linspace(0.0, 1.0, 50)
    signals = [create_signal(title, x, x) for title in titles]
    for signal in signals:
        panel.add_object(signal)
    return signals


def index_of(view: ObjectView, uuid: str) -> QC.QModelIndex:
    """Return the model index of the item of an object or group"""
    return view.indexFromItem(view.get_item_from_id(uuid))


def send_hover(view: ObjectView, etype: QC.QEvent.Type, pos: QC.QPoint) -> None:
    """Send a hover event to the viewport of an object view"""
    event = QG.QHoverEvent(etype, pos, QC.QPoint(-1, -1))
    QW.QApplication.sendEvent(view.viewport(), event)


def hover_item(view: ObjectView, uuid: str) -> QC.QModelIndex:
    """Move the mouse cursor over the title of an item, and return its index"""
    index = index_of(view, uuid)
    rect = view.visualRect(index)
    send_hover(
        view, QC.QEvent.HoverMove, QC.QPoint(rect.left() + 10, rect.center().y())
    )
    return index


def delete_button_center(view: ObjectView, uuid: str) -> QC.QPoint:
    """Hover the delete button of an item and return its center"""
    rect = view.delete_button_rect(hover_item(view, uuid))
    assert rect is not None
    send_hover(view, QC.QEvent.HoverMove, rect.center())
    return rect.center()


def click_delete_button(view: ObjectView, uuid: str) -> None:
    """Click the delete button of an item"""
    pos = delete_button_center(view, uuid)
    QTest.mouseClick(view.viewport(), QC.Qt.LeftButton, QC.Qt.NoModifier, pos)


def test_delete_button_geometry() -> None:
    """The button is shown on the hovered item only, at the viewport right edge."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        view = panel.objview
        sig1, sig2 = add_signals(panel, "Signal 1", "Signal 2")
        index1, index2 = index_of(view, get_uuid(sig1)), index_of(view, get_uuid(sig2))
        assert view.delete_button_rect(index1) is None
        hover_item(view, get_uuid(sig1))
        rect = view.delete_button_rect(index1)
        assert rect is not None
        assert view.delete_button_rect(index2) is None
        viewport = view.viewport()
        assert viewport.rect().contains(rect)
        assert rect.right() == viewport.width() - DELETE_BUTTON_MARGIN - 1
        item_rect = view.visualRect(index1)
        assert abs(rect.center().y() - item_rect.center().y()) <= 1
        # Hovering another item moves the button
        hover_item(view, get_uuid(sig2))
        assert view.delete_button_rect(index1) is None
        assert view.delete_button_rect(index2) is not None
        # Leaving the view hides the button
        send_hover(view, QC.QEvent.HoverLeave, QC.QPoint(-1, -1))
        assert view.delete_button_rect(index2) is None
        # The button sticks to the visible right edge of the viewport, even when
        # long titles make the view scroll horizontally
        sig2.title = "A very long signal title " * 20
        view.update_item(get_uuid(sig2))
        view.resize_title_column()
        QW.QApplication.processEvents()  # Update the scroll bar range
        scrollbar = view.horizontalScrollBar()
        assert scrollbar.maximum() > 0
        for value in (0, scrollbar.maximum()):
            scrollbar.setValue(value)
            QW.QApplication.processEvents()
            rect = view.delete_button_rect(hover_item(view, get_uuid(sig2)))
            assert rect is not None
            assert rect.right() == viewport.width() - DELETE_BUTTON_MARGIN - 1


def test_delete_button_tooltips() -> None:
    """Tooltips tell what the button deletes."""
    with datalab_test_app_context(console=False) as win:
        panel, view = win.signalpanel, win.signalpanel.objview
        sig1, sig2 = add_signals(panel, "Signal 1", "Signal 2")
        group = panel.objmodel.get_groups()[0]
        view.select_objects([sig1])
        assert view.delete_button_tooltip(index_of(view, get_uuid(sig2))) == _(
            "Delete this signal"
        )
        assert view.delete_button_tooltip(index_of(view, get_uuid(group))) == _(
            "Delete this group"
        )
        view.select_objects([sig1, sig2])
        assert view.delete_button_tooltip(index_of(view, get_uuid(sig2))) == (
            _("Delete the %d selected objects") % 2
        )
        group2 = panel.add_group("Group 2")
        view.select_groups([group, group2])
        assert view.delete_button_tooltip(index_of(view, get_uuid(group))) == (
            _("Delete the %d selected groups") % 2
        )
        ipanel = win.imagepanel
        image = create_image("Image", np.zeros((10, 10)))
        ipanel.add_object(image)
        iview = ipanel.objview
        assert iview.delete_button_tooltip(index_of(iview, get_uuid(image))) == _(
            "Delete this image"
        )


def test_delete_unselected_object_preserves_selection() -> None:
    """Deleting an unselected object keeps the selection, and is recorded in
    history as the removal of this object only."""
    with datalab_test_app_context(console=False, history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        view = panel.objview
        sig1, sig2, sig3 = add_signals(panel, "Signal 1", "Signal 2", "Signal 3")
        history.toggle_record_mode(True)
        view.select_objects([sig1, sig3])
        click_delete_button(view, get_uuid(sig2))
        assert panel.objmodel.get_object_ids() == [get_uuid(sig1), get_uuid(sig3)]
        assert set(view.get_sel_object_uuids()) == {get_uuid(sig1), get_uuid(sig3)}
        assert view.get_current_item_id() == get_uuid(sig3)
        action = history[len(history)]
        assert action.method_name == "remove_object"
        assert action.state.selection["signal"] == [get_uuid(sig2)]


def test_delete_selected_object_deletes_selection() -> None:
    """Deleting a selected object deletes the whole selection."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        view = panel.objview
        sig1, sig2, sig3 = add_signals(panel, "Signal 1", "Signal 2", "Signal 3")
        view.select_objects([sig1, sig2])
        click_delete_button(view, get_uuid(sig2))
        assert panel.objmodel.get_object_ids() == [get_uuid(sig3)]


def test_delete_group() -> None:
    """Deleting a group deletes its objects, and keeps the selection of other
    objects."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        view = panel.objview
        add_signals(panel, "Signal 1", "Signal 2")
        group1 = panel.objmodel.get_groups()[0]
        group2 = panel.add_group("Group 2", select=True)
        (sig3,) = add_signals(panel, "Signal 3")
        view.select_objects([sig3])
        click_delete_button(view, get_uuid(group1))
        assert panel.objmodel.get_groups() == [group2]
        assert panel.objmodel.get_object_ids() == [get_uuid(sig3)]
        assert view.get_sel_object_uuids() == [get_uuid(sig3)]
        # Deleting the group of the selected objects deletes them too
        (sig4,) = add_signals(panel, "Signal 4")
        view.select_objects([sig4])
        click_delete_button(view, get_uuid(group2))
        assert len(panel.objmodel) == 0
        assert not panel.objmodel.get_groups()


def test_delete_confirmation() -> None:
    """Deletion is always confirmed, and nothing is deleted if refused."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        view = panel.objview
        sig1, sig2 = add_signals(panel, "Signal 1", "Signal 2")
        view.select_objects([sig1])
        with execenv.context(unattended=False):
            with patch.object(
                QW.QMessageBox, "warning", return_value=QW.QMessageBox.No
            ) as warning:
                click_delete_button(view, get_uuid(sig2))
            assert warning.call_count == 1
            assert '"Signal 2"' in warning.call_args.args[2]
            assert len(panel.objmodel) == 2
            assert view.get_sel_object_uuids() == [get_uuid(sig1)]
            with patch.object(
                QW.QMessageBox, "warning", return_value=QW.QMessageBox.Yes
            ):
                click_delete_button(view, get_uuid(sig2))
            assert panel.objmodel.get_object_ids() == [get_uuid(sig1)]


def test_delete_button_does_not_select_drag_or_open() -> None:
    """Pressing the button neither selects nor drags the item, releasing it
    elsewhere cancels the click, and double-clicking it opens no view."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        view = panel.objview
        sig1, sig2 = add_signals(panel, "Signal 1", "Signal 2")
        view.select_objects([sig1])
        double_clicked: list[str] = []
        view.SIG_ITEM_DOUBLECLICKED.connect(double_clicked.append)
        viewport = view.viewport()
        pos = delete_button_center(view, get_uuid(sig2))
        QTest.mousePress(viewport, QC.Qt.LeftButton, QC.Qt.NoModifier, pos)
        assert view.get_sel_object_uuids() == [get_uuid(sig1)]
        assert view.state() == QW.QAbstractItemView.NoState
        outside = QC.QPoint(10, pos.y())
        move = QG.QMouseEvent(
            QC.QEvent.MouseMove,
            QC.QPointF(outside),
            QC.Qt.NoButton,
            QC.Qt.LeftButton,
            QC.Qt.NoModifier,
        )
        QW.QApplication.sendEvent(viewport, move)
        QTest.mouseRelease(viewport, QC.Qt.LeftButton, QC.Qt.NoModifier, outside)
        assert len(panel.objmodel) == 2
        assert view.get_sel_object_uuids() == [get_uuid(sig1)]
        pos = delete_button_center(view, get_uuid(sig2))
        event = QG.QMouseEvent(
            QC.QEvent.MouseButtonDblClick,
            QC.QPointF(pos),
            QC.Qt.LeftButton,
            QC.Qt.LeftButton,
            QC.Qt.NoModifier,
        )
        QW.QApplication.sendEvent(viewport, event)
        assert not double_clicked
        assert len(panel.objmodel) == 2


def test_delete_button_setting() -> None:
    """The option hides the button: clicking its place selects the item."""
    with Conf.show_delete_button_on_hover.context(False):
        with datalab_test_app_context(console=False) as win:
            view = win.signalpanel.objview
            assert not view.is_delete_button_enabled()
            sig1, sig2 = add_signals(win.signalpanel, "Signal 1", "Signal 2")
            view.select_objects([sig1])
            index = hover_item(view, get_uuid(sig2))
            assert view.delete_button_rect(index) is None
            view.set_delete_button_enabled(True)
            pos = view.delete_button_rect(index).center()
            view.set_delete_button_enabled(False)
            QTest.mouseClick(view.viewport(), QC.Qt.LeftButton, QC.Qt.NoModifier, pos)
            assert len(win.signalpanel.objmodel) == 2
            assert view.get_sel_object_uuids() == [get_uuid(sig2)]


def test_hover_events_survive_style_changes() -> None:
    """Hovered items stay highlighted after switching the Qt style (some styles
    disable hover events of item views when unpolishing them)."""
    with qt_app_context():
        app = QW.QApplication.instance()
        initial_style = app.style().objectName()
        tree = SimpleObjectTree(None, ObjectModel("gs"))
        tree.show()
        try:
            for style_name in QW.QStyleFactory.keys() + [initial_style]:
                app.setStyle(QW.QStyleFactory.create(style_name))
                QW.QApplication.processEvents()
                assert tree.viewport().testAttribute(QC.Qt.WA_Hover)
        finally:
            app.setStyle(QW.QStyleFactory.create(initial_style))
            tree.close()
