# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Curve style unit tests
----------------------

Automatic curve styles depend on the position of the curves on the plot, are not
stored in signal metadata, and are not shifted by dialogs showing signals. Curve
styles chosen by the user are kept.
"""

# guitest: show

from __future__ import annotations

import os.path as osp

import numpy as np
from plotpy.items import CurveItem
from qtpy import QtGui as QG
from sigima.objects import SignalObj, create_signal

from datalab.adapters_plotpy import (
    CURVESTYLES,
    LINE_STYLE_KEYS,
    CurveStyles,
    create_adapter_from_object,
)
from datalab.gui.panel.signal import SignalPanel
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers


def _create_signals(count: int) -> list[SignalObj]:
    """Create test signals"""
    x = np.linspace(0.0, 1.0, 50)
    return [create_signal(f"Signal {i}", x, x + i) for i in range(count)]


def _item(panel: SignalPanel, obj: SignalObj) -> CurveItem:
    """Return the curve item of a signal"""
    return panel.plothandler[get_uuid(obj)]


def _style(item: CurveItem) -> tuple[str, str]:
    """Return the color and the line style of a curve item"""
    return QG.QColor(item.param.line.color).name(), item.param.line.style


def _set_user_style(panel: SignalPanel, item: CurveItem, color: str) -> None:
    """Set a curve color as the user does with the curve parameters dialog"""
    item.param.line.color = color
    item.param.update_item(item)
    panel.plot_item_parameters_changed(item)


def test_styles_follow_curve_positions() -> None:
    """Visible curves are given the automatic styles of their positions"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals(12)
        panel.add_objects(signals, set_current=False)
        panel.objview.select_objects(signals)
        for index, obj in enumerate(signals):
            assert _style(_item(panel, obj)) == CurveStyles.style_at(index)
        assert _style(_item(panel, signals[11]))[1] == "DashLine"

        # Whatever the signals shown, the first curves get the first styles
        panel.objview.select_objects([signals[7], signals[4]])
        assert _style(_item(panel, signals[4])) == CurveStyles.style_at(0)
        assert _style(_item(panel, signals[7])) == CurveStyles.style_at(1)


def test_style_cycle_not_used_by_panel_and_dialogs() -> None:
    """Showing signals, on the panel or in dialogs, does not shift the style cycle"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals(3)
        panel.add_objects(signals, set_current=False)
        CURVESTYLES.reset_styles()
        panel.objview.select_objects(signals)
        # As done by the ROI editor or "View in a new window":
        for obj in signals:
            item = _item(panel, obj)
            copy = create_adapter_from_object(obj).make_item(update_from=item)
            assert _style(copy) == _style(item)
        new_item = create_adapter_from_object(signals[0]).make_item()
        assert _style(new_item) == CurveStyles.style_at(0)


def test_user_styles_are_kept() -> None:
    """Curve styles chosen by the user are kept and stored in metadata"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            signals = _create_signals(3)
            panel.add_objects(signals, set_current=False)
            panel.objview.select_objects(signals)
            _set_user_style(panel, _item(panel, signals[1]), "#000000")
            assert signals[1].metadata["color"] == "#000000"
            for obj in (signals[0], signals[2]):
                for key in LINE_STYLE_KEYS:
                    assert key not in obj.metadata

            # The curve keeps its color and its position in the style cycle
            panel.objview.select_objects([signals[2], signals[1]])
            assert _style(_item(panel, signals[1]))[0] == "#000000"
            assert _style(_item(panel, signals[2])) == CurveStyles.style_at(1)

            # Workspace round trip: only the user-defined style is stored
            fname = osp.join(tmpdir, "curve_styles.h5")
            win.save_h5_workspace(fname)
            panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)
            loaded = panel.objmodel.get_all_objects()
            assert loaded[1].metadata["color"] == "#000000"
            for obj in (loaded[0], loaded[2]):
                for key in LINE_STYLE_KEYS:
                    assert key not in obj.metadata
            panel.objview.select_objects(loaded)
            assert _style(_item(panel, loaded[0])) == CurveStyles.style_at(0)
            assert _style(_item(panel, loaded[1]))[0] == "#000000"
            assert _style(_item(panel, loaded[2])) == CurveStyles.style_at(2)


def test_legacy_styles_are_automatic() -> None:
    """Styles stored by earlier versions from their style cycle are automatic"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            signals = _create_signals(2)
            for obj in signals:
                obj.metadata["color"] = CurveStyles.LEGACY_COLORS[2]
                obj.metadata["style"] = "DashDotDotLine"
            panel.add_objects(signals, set_current=False)
            panel.objview.select_objects(signals)
            for index, obj in enumerate(signals):
                assert _style(_item(panel, obj)) == CurveStyles.style_at(index)
            win.save_h5_workspace(osp.join(tmpdir, "legacy_styles.h5"))
            for obj in signals:
                for key in LINE_STYLE_KEYS:
                    assert key not in obj.metadata


def test_reset_curve_styles() -> None:
    """Resetting curve styles gives the selected curves automatic styles again"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals(3)
        panel.add_objects(signals, set_current=False)
        panel.objview.select_objects(signals)
        for obj in signals:
            _set_user_style(panel, _item(panel, obj), "#000000")
        signals[0].metadata["style"] = "DotLine"

        panel.objview.select_objects(signals[:2])
        panel.reset_curve_styles()
        for index, obj in enumerate(signals[:2]):
            for key in LINE_STYLE_KEYS:
                assert key not in obj.metadata
            assert _style(_item(panel, obj)) == CurveStyles.style_at(index)
        # Signals which are not selected keep their style
        assert signals[2].metadata["color"] == "#000000"
        assert _style(_item(panel, signals[2]))[0] == "#000000"


if __name__ == "__main__":
    test_styles_follow_curve_positions()
    test_style_cycle_not_used_by_panel_and_dialogs()
    test_user_styles_are_kept()
    test_legacy_styles_are_automatic()
    test_reset_curve_styles()
