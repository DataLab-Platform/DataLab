# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration test

Checking .ini configuration file management.
"""

# guitest: show

import os
import os.path as osp

import pytest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigimax.utils import conf as confmod

from datalab import app
from datalab.config import Conf
from datalab.config.persistence import (
    CONF_VERSION,
    DataLabUserConfig,
    load_options_from_ini,
)
from datalab.env import execenv
from datalab.tests import helpers
from datalab.utils.qthelpers import datalab_app_context

SEC_MAIN = "main"
OPT_MAX = "window_maximized"
OPT_POS = "window_position"
OPT_SIZ = "window_size"
OPT_DIR = "base_dir"

SEC_CONS = "console"
OPT_CON = "console_enabled"

CONFIGS = (
    {
        SEC_MAIN: {
            OPT_MAX: False,
            OPT_POS: (250, 250),
            OPT_SIZ: (1300, 700),
            OPT_DIR: "",
        },
        SEC_CONS: {
            OPT_CON: False,
        },
    },
    {
        SEC_MAIN: {
            OPT_MAX: False,
            OPT_POS: (100, 100),
            OPT_SIZ: (810, 600),
            OPT_DIR: osp.dirname(__file__),
        },
        SEC_CONS: {
            OPT_CON: False,
        },
    },
    {
        SEC_MAIN: {
            OPT_MAX: True,
            OPT_POS: (10, 10),
            OPT_SIZ: (810, 600),
            OPT_DIR: "",
        },
        SEC_CONS: {
            OPT_CON: True,
        },
    },
)


def apply_conf(backend, settings, name):
    """Apply configuration"""
    execenv.print(f"  Applying configuration {name}:")
    fname = backend.filename()
    try:
        os.remove(fname)
        execenv.print(f"    Removed configuration file {fname}")
    except FileNotFoundError:
        execenv.print(f"    Configuration file {fname} was not found")
    for section, section_settings in settings.items():
        for option, value in section_settings.items():
            execenv.print(f"    Writing [{section}][{option}] = {value}")
            backend.set(section, option, value)
    backend.save()
    load_options_from_ini(Conf, backend)


def is_wsl() -> bool:
    """Return True if running on Windows Subsystem for Linux (WSL)"""
    if os.name == "posix":
        return "WSL" in os.uname().release  # pylint: disable=no-member
    return False


def is_offscreen() -> bool:
    """Return True if running in offscreen mode (e.g. in CI or pytest session)"""
    return os.environ.get("QT_QPA_PLATFORM", "") == "offscreen"


def assert_in_interval(val1, val2, interval, context):
    """Raise an AssertionError if val1 is in [val2-interval/2,val2+interval/2]"""
    itv1, itv2 = val2 - 0.5 * interval, val2 + 0.5 * interval
    try:
        assert itv1 <= val1 <= itv2
    except AssertionError as exc:
        if is_wsl() or is_offscreen():
            # Ignore this assertion error in two cases:
            # 1. If running on WSL, as the position of windows is not reliable
            # 2. If running in offscreen mode (e.g. in CI or pytest session),
            #    as the position of windows may not be set correctly in offscreen mode
            pass
        else:
            raise AssertionError(f"{context}: {itv1} <= {val1} <= {itv2}") from exc


def check_conf(conf, name, win: QW.QMainWindow, h5files):
    """Check configuration"""
    execenv.print(f"  Checking configuration {name}: ")
    sec_main_name = SEC_MAIN
    sec_cons_name = SEC_CONS
    sec_main = conf[sec_main_name]
    sec_cons = conf[sec_cons_name]
    execenv.print(f"    Checking [{sec_main_name}][{OPT_MAX}]: ", end="")
    assert sec_main[OPT_MAX] == (win.windowState() == QC.Qt.WindowState.WindowMaximized)
    execenv.print("OK")
    execenv.print(f"    Checking [{sec_main_name}][{OPT_POS}]: ", end="")
    if not sec_main[OPT_MAX]:  # Check position/size only when not maximized
        #  Check position, taking into account screen offset (e.g. Linux/Gnome)
        conf_x, conf_y = sec_main[OPT_POS]
        conf_w, conf_h = sec_main[OPT_SIZ]
        available_go = QW.QApplication.primaryScreen().availableGeometry()
        x_offset, y_offset = available_go.x(), available_go.y()
        assert_in_interval(win.x(), conf_x - x_offset, 2, "X position")
        assert_in_interval(win.y(), conf_y - y_offset / 2, 15 + y_offset, "Y position")
        #  Check size
        assert_in_interval(win.width(), conf_w, 5, "Width")
        assert_in_interval(win.height(), conf_h, 5, "Height")
        execenv.print(f"OK {win.x(), win.y(), win.width(), win.height()}")
    else:
        execenv.print("Passed (maximized)")
    execenv.print(f"    Checking [{sec_cons_name}][{OPT_CON}]: ", end="")
    assert sec_cons[OPT_CON] == (win.console is not None)
    execenv.print("OK")
    execenv.print(f"    Checking [{sec_main_name}][{OPT_DIR}]: ", end="")
    if h5files is None:
        assert conf[SEC_MAIN][OPT_DIR] == Conf.base_dir.get()
        execenv.print("OK (written in conf file)")
    else:
        assert Conf.base_dir.get() == osp.dirname(h5files[0])
        execenv.print("OK (changed to HDF5 file path)")


def test_config(tmp_path, monkeypatch):
    """Testing DataLab configuration file"""
    backend = DataLabUserConfig({})
    monkeypatch.setattr(backend, "get_path", lambda basename: str(tmp_path / basename))
    backend.set_application("DataLab_pytest", CONF_VERSION, load=False)
    monkeypatch.setattr(confmod, "CONF", backend)
    original_options = Conf.to_dict()
    try:
        with execenv.context(unattended=True):
            h5files = [helpers.get_test_fnames("*.h5")[1]]
            execenv.print("Testing DataLab configuration settings:")
            for index, settings in enumerate(CONFIGS):
                name = f"CONFIG{index}"
                apply_conf(backend, settings, name)
                with datalab_app_context(exec_loop=True) as qapp:
                    win = app.create(splash=False, h5files=h5files)
                    qapp.processEvents()
                    check_conf(settings, name, win, h5files)
                h5files = None
            execenv.print("=> Everything is OK")
    finally:
        Conf.from_dict(original_options)


if __name__ == "__main__":
    pytest.main([__file__])
