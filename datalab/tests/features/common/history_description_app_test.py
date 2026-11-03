# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel rows stay compact for voluminous parameters (issue #369)."""

from __future__ import annotations

import numpy as np
import sigima.params
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import create_image

from datalab.tests import datalab_test_app_context


def test_recorded_brightness_contrast_row_is_compact() -> None:
    """The row of an action holding a histogram stays one line high."""
    with datalab_test_app_context(console=False, history=True) as win:
        panel = win.imagepanel
        history = win.historypanel
        rng = np.random.default_rng(0)
        source = create_image(
            "Source", rng.integers(0, 4000, (64, 64)).astype(np.uint16)
        )
        panel.add_object(source)
        history.toggle_record_mode(True)
        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(source)
        param.minimum, param.maximum = 100.0, 3000.0
        feature = panel.processor.get_feature("adjust_brightness_contrast")
        panel.processor.run_feature(feature, param, edit=False)
        history.toggle_record_mode(False)

        tree = history.tree
        assert tree.verticalScrollMode() == QW.QAbstractItemView.ScrollPerPixel
        action = next(
            action
            for session in history.history_sessions
            for action in session.actions
            if action.func_name == "adjust_brightness_contrast"
        )
        item = None
        iterator = QW.QTreeWidgetItemIterator(tree)
        while iterator.value():
            if iterator.value().data(0, QC.Qt.UserRole) == action.uuid:
                item = iterator.value()
                break
            iterator += 1
        assert item is not None
        widget = tree.itemWidget(item, tree.DESCRIPTION_COLUMN)
        QW.QApplication.processEvents()
        tree.doItemsLayout()
        collapsed_height = tree.visualItemRect(item).height()
        assert collapsed_height < 3 * tree.fontMetrics().lineSpacing()

        widget.set_expanded(True)
        QW.QApplication.processEvents()
        tree.doItemsLayout()
        expanded_height = tree.visualItemRect(item).height()
        assert collapsed_height < expanded_height <= widget.details_max_height()
