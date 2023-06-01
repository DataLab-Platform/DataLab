# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab settings
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
from qtpy import QtWidgets as QW

from cdl.config import Conf, _


class MainSettings(gdt.DataSet):
    """DataLab main settings"""

    g0 = gdt.BeginGroup(_("Settings for main window and general features"))
    process_isolation_enabled = gdi.BoolItem(
        "",
        _("Process isolation"),
        help=_(
            "With process isolation, each computation is run in a separate process,"
            "<br>which prevents the application from freezing during long computations."
        ),
    )
    rpc_server_enabled = gdi.BoolItem(
        "",
        _("RPC server"),
        help=_(
            "RPC server is used to communicate with external applications,"
            "<br>like your own scripts (e.g. from Spyder or Jupyter) or other software."
        ),
    )
    available_memory_threshold = gdi.IntItem(
        _("Available memory threshold"),
        unit=_("MB"),
        help=_(
            "Threshold below which a warning is displayed before loading any new data"
        ),
    )
    ignore_dependency_check = gdi.BoolItem(
        "",
        _("Ignore dependency check"),
        help=_("Disable the dependency check at startup for critical packages"),
    )
    plugins_enabled = gdi.BoolItem(
        "",
        _("Third-party plugins"),
        help=_("Disable third-party plugins at startup"),
    )
    _g0 = gdt.EndGroup("")


class ConsoleSettings(gdt.DataSet):
    """DataLab console settings"""

    g0 = gdt.BeginGroup(
        _("Settings for internal console, used for debugging or advanced users")
    )
    console_enabled = gdi.BoolItem("", _("Console enabled"))
    external_editor_path = gdi.StringItem(_("External editor path"))
    external_editor_args = gdi.StringItem(_("External editor arguments"))
    _g0 = gdt.EndGroup("")


class IOSettings(gdt.DataSet):
    """DataLab I/O settings"""

    g0 = gdt.BeginGroup(_("Settings for I/O operations"))
    h5_fullpath_in_title = gdi.BoolItem("", _("HDF5 full path in title"))
    h5_fname_in_title = gdi.BoolItem("", _("HDF5 file name in title"))
    _g0 = gdt.EndGroup("")


class ProcSettings(gdt.DataSet):
    """DataLab processing settings"""

    g0 = gdt.BeginGroup(_("Settings for computations"))
    fft_shift_enabled = gdi.BoolItem("", _("FFT shift"))
    extract_roi_singleobj = gdi.BoolItem("", _("Extract ROI in single object"))
    ignore_warnings = gdi.BoolItem("", _("Ignore warnings"))
    _g0 = gdt.EndGroup("")


# Generator yielding (param, section, option) tuples from configuration dictionary
def _iter_conf(paramdict: dict[str, gdt.DataSet]) -> tuple[gdt.DataSet, str, str]:
    """Iterate over configuration parameters"""
    confdict = Conf.to_dict()
    for section in confdict:
        if section in paramdict:
            for option in confdict[section]:
                param = paramdict[section]
                if hasattr(param, option):
                    yield param, section, option


def conf_to_datasets(paramdict: dict[str, gdt.DataSet]) -> None:
    """Convert DataLab configuration to datasets"""
    for param, section, option in _iter_conf(paramdict):
        value = getattr(getattr(Conf, section), option).get()
        setattr(param, option, value)


def datasets_to_conf(paramdict: dict[str, gdt.DataSet]) -> None:
    """Convert datasets to DataLab configuration"""
    for param, section, option in _iter_conf(paramdict):
        value = getattr(param, option)
        getattr(getattr(Conf, section), option).set(value)


RESTART_OPTIONS = (
    ("process_isolation_enabled", _("Process isolation enable status")),
    ("rpc_server_enabled", _("RPC server enable status")),
    ("console_enabled", _("Console enable status")),
    ("plugins_enabled", _("Third-party plugins support")),
)


def get_restart_items_values(paramdict: dict[str, gdt.DataSet]) -> list:
    """Get restart items values"""
    values = []
    for option, _name in RESTART_OPTIONS:
        for param, _section, _option in _iter_conf(paramdict):
            if option == _option:
                values.append(getattr(param, option))
    return values


def edit_settings(parent: QW.QWidget | None = None):
    """Edit DataLab settings"""
    paramdict = {
        "main": MainSettings(_("General features")),
        "console": ConsoleSettings(_("Console")),
        "io": IOSettings(_("I/O")),
        "proc": ProcSettings(_("Processing")),
    }
    conf_to_datasets(paramdict)
    before = get_restart_items_values(paramdict)
    params = gdt.DataSetGroup(paramdict.values(), title=_("Settings"))
    if params.edit(parent=parent):
        after = get_restart_items_values(paramdict)
        if before != after:
            # List the options that were changed
            changed = [
                item[1]
                for item, value in zip(RESTART_OPTIONS, before)
                if value != after[RESTART_OPTIONS.index(item)]
            ]
            # Show a message box to inform the user that a restart is required
            QW.QMessageBox.information(
                parent,
                _("Restart required"),
                _(
                    "The following options have been changed:\n\n"
                    "- %s\n\n"
                    "A restart is required for these changes to take effect."
                )
                % "\n- ".join(changed),
            )
        datasets_to_conf(paramdict)
