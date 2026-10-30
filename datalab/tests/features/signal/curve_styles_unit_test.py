# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Curve style unit tests
----------------------

Automatic curve styles depend on the position of the curves on the plot, are not
stored in signal metadata, and are not shifted by dialogs showing signals. Curve
styles chosen by the user are kept. Automatic styles stored by DataLab 1.3 or
earlier are removed when loading files saved by these versions.
"""

# guitest: show

from __future__ import annotations

import os.path as osp

import h5py
import numpy as np
import pytest
from plotpy.items import CurveItem
from qtpy import QtGui as QG
from sigima.objects import SignalObj, create_signal

from datalab.adapters_plotpy import (
    CURVESTYLES,
    LINE_STYLE_KEYS,
    CurveStyles,
    create_adapter_from_object,
)
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.signal import SignalPanel
from datalab.h5.native import DATALAB_VERSION_NAME, is_legacy_style_version
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers

#: Style of the automatic style cycle of DataLab 1.3 or earlier
LEGACY_STYLE = (CurveStyles.LEGACY_COLORS[2], "DashDotDotLine")


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


def _stored_style(obj: SignalObj) -> tuple[str | None, str | None]:
    """Return the color and the line style stored in signal metadata"""
    return obj.metadata.get("color"), obj.metadata.get("style")


def _set_user_style(
    panel: SignalPanel, item: CurveItem, color: str, linestyle: str | None = None
) -> None:
    """Set a curve style as the user does with the curve parameters dialog"""
    item.param.line.color = color
    if linestyle is not None:
        item.param.line.style = linestyle
    item.param.update_item(item)
    panel.plot_item_parameters_changed(item)


def _set_saved_version(filename: str, version: str) -> None:
    """Set the DataLab version that saved a native HDF5 file"""
    with h5py.File(filename, "r+") as h5file:
        del h5file[DATALAB_VERSION_NAME]
        h5file[DATALAB_VERSION_NAME] = version


def _reload_workspace(win: DLMainWindow, filename: str) -> list[SignalObj]:
    """Reload a workspace and show all its signals"""
    panel = win.signalpanel
    panel.remove_all_objects()
    win.load_h5_workspace([filename], reset_all=True)
    loaded = panel.objmodel.get_all_objects()
    panel.objview.select_objects(loaded)
    return loaded


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.3.0", True),
        ("1.3.10", True),
        ("0.20.1", True),
        ("1.4.0", False),
        ("1.4.0rc1", False),  # Pre-releases of 1.4 do not store automatic styles
        ("1.4.0.dev0", False),
        ("1.10.0", False),
        ("2.0.0", False),
        ("invalid", False),
        ("", False),
        (None, False),
    ],
)
def test_legacy_style_version(version: str | None, expected: bool) -> None:
    """Versions which stored automatic curve styles are DataLab 1.3 or earlier"""
    assert is_legacy_style_version(version) is expected


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


def test_user_style_from_legacy_cycle_is_kept() -> None:
    """A style chosen by the user is kept over several workspace round trips, even
    if it matches a style of the automatic style cycle of DataLab 1.3 or earlier"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            signals = _create_signals(2)
            panel.add_objects(signals, set_current=False)
            panel.objview.select_objects(signals)
            _set_user_style(panel, _item(panel, signals[1]), "#1f77b4", "SolidLine")
            assert CurveStyles.is_legacy_automatic("#1f77b4", "SolidLine")
            fname = osp.join(tmpdir, "user_styles.h5")
            for _cycle in range(2):
                win.save_h5_workspace(fname)
                loaded = _reload_workspace(win, fname)
                assert _stored_style(loaded[1]) == ("#1f77b4", "SolidLine")
                assert _style(_item(panel, loaded[1])) == ("#1f77b4", "SolidLine")
                assert _stored_style(loaded[0]) == (None, None)
                assert _style(_item(panel, loaded[0])) == CurveStyles.style_at(0)


def test_legacy_styles_converted_on_load() -> None:
    """Automatic styles stored by DataLab 1.3 or earlier are removed on load"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            signals = _create_signals(3)
            signals[0].metadata["color"], signals[0].metadata["style"] = LEGACY_STYLE
            signals[1].metadata["color"], signals[1].metadata["style"] = LEGACY_STYLE
            # A legacy color with a line style which is not a legacy automatic one:
            signals[2].metadata["color"] = LEGACY_STYLE[0]
            signals[2].metadata["style"] = "DotLine"
            panel.add_objects(signals, set_current=False)
            panel.objview.select_objects(signals)
            fname = osp.join(tmpdir, "legacy_styles.h5")
            win.save_h5_workspace(fname)

            # Saved by the current version: stored styles are kept
            loaded = _reload_workspace(win, fname)
            assert _stored_style(loaded[0]) == LEGACY_STYLE
            assert _style(_item(panel, loaded[0])) == LEGACY_STYLE

            # Saved by DataLab 1.3: legacy automatic styles are removed
            _set_saved_version(fname, "1.3.0")
            loaded = _reload_workspace(win, fname)
            for index in (0, 1):
                assert _stored_style(loaded[index]) == (None, None)
                assert _style(_item(panel, loaded[index])) == CurveStyles.style_at(
                    index
                )
            assert _stored_style(loaded[2]) == (LEGACY_STYLE[0], "DotLine")
            assert _style(_item(panel, loaded[2])) == (LEGACY_STYLE[0], "DotLine")


def test_legacy_styles_converted_on_dlhist_import() -> None:
    """Automatic styles stored by DataLab 1.3 or earlier are removed when importing
    a history file in a workspace in use"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            signals = _create_signals(1)
            signals[0].metadata["color"], signals[0].metadata["style"] = LEGACY_STYLE
            panel.add_objects(signals, set_current=False)
            fname = osp.join(tmpdir, "legacy_styles.dlhist")
            assert win.historypanel.save_to_dlhist_file(fname)
            _set_saved_version(fname, "1.3.0")
            existing = panel.objmodel.get_all_objects()
            assert win.historypanel.open_dlhist_file(fname)
            imported = [
                obj for obj in panel.objmodel.get_all_objects() if obj not in existing
            ]
            assert len(imported) == 1
            assert _stored_style(imported[0]) == (None, None)
            assert _stored_style(existing[0]) == LEGACY_STYLE


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
    test_user_style_from_legacy_cycle_is_kept()
    test_legacy_styles_converted_on_load()
    test_legacy_styles_converted_on_dlhist_import()
    test_reset_curve_styles()
