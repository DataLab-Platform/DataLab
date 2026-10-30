# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal auto downsampling unit tests
-----------------------------------

Signal display settings (auto downsampling, line width) are applied to the curves of
all signals when they are changed in the Settings dialog, and not only to the curves
of the selected signals, including the curves of a reloaded workspace.
"""

# guitest: show

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest
from plotpy.items import CurveItem
from sigima.objects import SignalObj, create_signal

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import Conf
from datalab.gui import main as main_module
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.signal import SignalPanel
from datalab.gui.settings import ViewSettings
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers

#: Number of points of the test signals
SIZES = (1_000, 30_000, 300_000)


def _create_signals(sizes: tuple[int, ...] = SIZES) -> list[SignalObj]:
    """Create test signals with given numbers of points"""
    signals = []
    for size in sizes:
        x = np.linspace(0.0, 1.0, size)
        signals.append(create_signal(f"{size} points", x, np.sin(2 * np.pi * 50 * x)))
    return signals


def _add_signals(panel: SignalPanel, signals: list[SignalObj]) -> None:
    """Add signals to the panel, create their curves, then select the first one,
    so that the curves of the other signals exist but are hidden"""
    panel.add_objects(signals, set_current=False)
    panel.objview.select_objects(signals)
    panel.objview.select_objects(signals[:1])


def _item(panel: SignalPanel, obj: SignalObj) -> CurveItem:
    """Return the curve item of a signal"""
    return panel.plothandler[get_uuid(obj)]


def _edit_settings(
    win: DLMainWindow, monkeypatch: pytest.MonkeyPatch, **options
) -> None:
    """Change settings as the user does with the Settings dialog

    Args:
        win: DataLab main window
        monkeypatch: pytest monkeypatch fixture
        options: option names and their new values
    """

    def fake_edit_settings(_parent) -> list[str]:
        for name, value in options.items():
            getattr(Conf, name).set(value)
        return list(options)

    monkeypatch.setattr(main_module, "edit_settings", fake_edit_settings)
    win.settings_action.trigger()


def _check_downsampling(
    panel: SignalPanel, signals: list[SignalObj], maxpoints: int | None
) -> None:
    """Check the number of points drawn for each signal

    Args:
        panel: signal panel
        signals: signals to check
        maxpoints: maximum number of points to draw (None if auto downsampling is
         disabled)
    """
    for obj in signals:
        item = _item(panel, obj)
        size = obj.x.size
        if maxpoints is not None and size > maxpoints:
            assert item.param.use_dsamp, obj.title
            # At most 2 more points with min/max downsampling (PlotPy >= 2.12)
            assert item.dataSize() <= maxpoints + 2, obj.title
        else:
            assert item.dataSize() == size, obj.title


def _visibility(panel: SignalPanel, signals: list[SignalObj]) -> list[bool]:
    """Return the visibility of the curves of signals"""
    return [_item(panel, obj).isVisible() for obj in signals]


def test_settings_default_maxpoints() -> None:
    """The Settings dialog shows the actual default maximum number of points"""
    assert ViewSettings().sig_autodownsampling_maxpoints == (
        Conf.sig_autodownsampling_maxpoints.get()
    )


def test_downsampling_settings_apply_to_all_curves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto downsampling settings are applied to the curves of all signals"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals()
        _add_signals(panel, signals)
        maxpoints = Conf.sig_autodownsampling_maxpoints.get()
        _check_downsampling(panel, signals, maxpoints)
        visibility = _visibility(panel, signals)
        assert visibility == [True, False, False]

        _edit_settings(win, monkeypatch, sig_autodownsampling_maxpoints=10_000)
        _check_downsampling(panel, signals, 10_000)
        assert _visibility(panel, signals) == visibility

        _edit_settings(win, monkeypatch, sig_autodownsampling=False)
        _check_downsampling(panel, signals, None)

        _edit_settings(win, monkeypatch, sig_autodownsampling=True)
        _check_downsampling(panel, signals, 10_000)
        assert _visibility(panel, signals) == visibility


def test_line_width_applies_to_all_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Line width settings are applied to the curves of all signals"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals((100, 200, 300))
        _add_signals(panel, signals)
        _edit_settings(win, monkeypatch, sig_linewidth=2.0)
        for obj in signals:
            item = _item(panel, obj)
            assert item.param.line.width == 2.0, obj.title
            assert item.pen().widthF() == 2.0, obj.title


def test_line_width_applies_to_reloaded_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Line width settings are applied to the curves of a reloaded workspace, and
    are still clamped for curves with many points"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            panel = win.signalpanel
            threshold = Conf.sig_linewidth_perfs_threshold.get()
            _add_signals(panel, _create_signals((100, 200, threshold + 1)))
            fname = osp.join(tmpdir, "line_width.h5")
            win.save_h5_workspace(fname)
            panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)
            loaded = panel.objmodel.get_all_objects()
            panel.objview.select_objects(loaded)
            _edit_settings(win, monkeypatch, sig_linewidth=2.0)
            for obj in loaded:
                assert "width" not in obj.metadata, obj.title
                expected = 1.0 if obj.x.size > threshold else 2.0
                assert _item(panel, obj).param.line.width == expected, obj.title


def test_settings_update_keeps_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying signal display settings neither duplicates nor shows shape items"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals()
        _add_signals(panel, signals)
        with Conf.show_result_dialog.context(False):
            panel.processor.run_feature("stats")
        plot = panel.plothandler.plot
        nb_items = len(plot.get_items())
        visibility = _visibility(panel, signals)
        _edit_settings(win, monkeypatch, sig_autodownsampling_maxpoints=10_000)
        assert len(plot.get_items()) == nb_items
        assert _visibility(panel, signals) == visibility


def test_dialog_items_use_current_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot items made from existing curves (e.g. for dialogs) use the current
    auto downsampling settings"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = _create_signals()
        _add_signals(panel, signals)
        _edit_settings(win, monkeypatch, sig_autodownsampling_maxpoints=10_000)
        obj = signals[-1]
        item = create_adapter_from_object(obj).make_item(update_from=_item(panel, obj))
        assert item.param.use_dsamp
        assert item.dataSize() <= 10_000 + 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
