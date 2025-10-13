# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Settings
========

The :mod:`datalab.gui.settings` module provides the DataLab settings dialog
and related classes.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Any, Generator

import guidata.dataset as gds
from guidata.dataset import restore_dataset, update_dataset
from plotpy.styles import BaseImageParam
from qtpy import QtWidgets as QW

from datalab.config import Conf, _


class MainSettings(gds.DataSet):
    """DataLab main settings"""

    g0 = gds.BeginGroup(_("Settings for main window and general features"))
    color_mode = gds.ChoiceItem(
        _("Color mode"),
        zip(Conf.main.color_mode.values, Conf.main.color_mode.values),
        help=_("Color mode for the application"),
    )
    process_isolation_enabled = gds.BoolItem(
        "",
        _("Process isolation"),
        help=_(
            "With process isolation, each computation is run in a separate process,"
            "<br>which prevents the application from freezing during long computations."
        ),
    )
    rpc_server_enabled = gds.BoolItem(
        "",
        _("RPC server"),
        help=_(
            "RPC server is used to communicate with external applications,"
            "<br>like your own scripts (e.g. from Spyder or Jupyter) or other software."
        ),
    )
    available_memory_threshold = gds.IntItem(
        _("Available memory threshold"),
        default=0,
        unit=_("MB"),
        help=_(
            "Threshold below which a warning is displayed before loading any new data"
        ),
    )
    plugins_enabled = gds.BoolItem(
        "",
        _("Third-party plugins"),
        help=_("Disable third-party plugins at startup"),
    )
    plugins_path = gds.DirectoryItem(
        _("Plugins path"),
        allow_none=True,
        help=_(
            "Path to third-party plugins.<br><br>"
            "DataLab will discover plugins in this path, "
            "as well as in your PYTHONPATH."
        ),
    )
    _g0 = gds.EndGroup("")


class ConsoleSettings(gds.DataSet):
    """DataLab console settings"""

    g0 = gds.BeginGroup(
        _("Settings for internal console, used for debugging or advanced users")
    )
    console_enabled = gds.BoolItem("", _("Console enabled"))
    show_console_on_error = gds.BoolItem(
        "",
        _("Show console on error"),
        help=_(
            "If enabled, the console will be shown when an error occurs in the "
            "application. This is useful for debugging purposes as it allows to "
            "see the error traceback."
        ),
    )
    external_editor_path = gds.StringItem(_("External editor path"), "")
    external_editor_args = gds.StringItem(_("External editor arguments"), "")
    _g0 = gds.EndGroup("")


class IOSettings(gds.DataSet):
    """DataLab I/O settings"""

    g0 = gds.BeginGroup(_("Settings for I/O operations"))
    h5_clear_workspace = gds.BoolItem("", _("Clear workspace before loading HDF5 file"))
    h5_clear_workspace_ask = gds.BoolItem("", _("Ask before clearing workspace"))
    h5_fullpath_in_title = gds.BoolItem(
        "",
        _("HDF5 full path in title"),
        help=_(
            "If enabled, the full path of the HDF5 data set will be used as the title "
            "for the signal/image object.<br>"
            "If disabled, only the name of the data set will be used as the title."
        ),
    )
    h5_fname_in_title = gds.BoolItem(
        "",
        _("HDF5 file name in title"),
        help=_(
            "If enabled, the name of the HDF5 file will be used as a suffix in the "
            "title of the signal/image object."
        ),
    )
    _g0 = gds.EndGroup("")


class ProcSettings(gds.DataSet):
    """DataLab processing settings"""

    g0 = gds.BeginGroup(_("Settings for computations"))
    operation_mode = gds.ChoiceItem(
        _("Operation mode"),
        zip(Conf.proc.operation_mode.values, Conf.proc.operation_mode.values),
        help=_(
            "Operation mode for computations taking <i>N</i> inputs:"
            "<ul><li><b>single</b>: single operand mode</li>"
            "<li><b>pairwise</b>: pairwise operation mode</li></ul>"
            "<br>Computations taking <i>N</i> inputs are the ones where:"
            "<ul><li>N(>=2) objects in %s 1 object out</li>"
            "<li>N(>=1) objects + 1 object in %s N objects out</li></ul>"
        )
        % ("→", "→"),
    )
    use_signal_bounds = gds.BoolItem(
        "",
        _("Use signal bounds for new signals"),
        help=_(
            "If enabled, when creating a new signal, the xmin and xmax values will be "
            "initialized from the current signal's bounds.<br>"
            "If disabled, the default values from the New Signal dialog will be used."
        ),
    )
    use_image_dims = gds.BoolItem(
        "",
        _("Use image dimensions for new images"),
        help=_(
            "If enabled, when creating a new image, the width and height values will "
            "be initialized from the current image's dimensions.<br>"
            "If disabled, the default values from the New Image dialog will be used."
        ),
    )
    fft_shift_enabled = gds.BoolItem(
        "",
        _("FFT shift"),
        help=_(
            "Enable FFT shift to center the zero-frequency component in the frequency "
            "spectrum for easier visualization and analysis."
        ),
    )
    extract_roi_singleobj = gds.BoolItem(
        "",
        _("Extract ROI in single object"),
        help=_(
            "If enabled, multiple ROIs will be extracted into a single object.<br>"
            "If disabled, each ROI will be extracted into a separate object."
        ),
    )
    keep_results = gds.BoolItem(
        "",
        _("Keep results after computation"),
        help=_(
            "If enabled, the results of a previous analysis will be kept in object's "
            "metadata after the computation.<br>"
            "If disabled, the results will be removed from the object's metadata."
            "<br><br>"
            "This option is disabled by default because keeping analysis results may "
            "be confusing as those results could be outdated following the "
            "computation."
        ),
    )
    ignore_warnings = gds.BoolItem(
        "", _("Ignore warnings"), help=_("Ignore warnings during computations")
    )
    xarray_compat_behavior = gds.ChoiceItem(
        _("X-array compatibility behavior"),
        zip(Conf.proc.xarray_compat_behavior.values, [_("Ask"), _("Interpolate")]),
        help=_(
            "Behavior when X arrays are incompatible in multi-signal computations:"
            "<ul><li><b>Ask</b>: display confirmation dialog (default)</li>"
            "<li><b>Interpolate</b>: automatically interpolate signals</li></ul>"
        ),
    )
    _g0 = gds.EndGroup("")


class ImageDefaultSettings(BaseImageParam):
    """Image visualization default settings"""

    _multiselection = True  # Hide label (not clean because it's not the intended use)


#  pylint:disable=unused-argument
def edit_default_image_settings(
    dataset: gds.DataSet, item: gds.DataItem, value: Any, parent: QW.QWidget
) -> bool:
    """Edit default image settings

    Args:
        dataset: dataset
        item: Data item
        value: Value
        parent: Parent widget

    Returns:
        True if the settings were edited
    """
    param = ImageDefaultSettings(_("Default image visualization settings"))
    ima_def_dict = Conf.view.get_def_dict("ima")
    update_dataset(param, ima_def_dict)
    if param.edit(parent=parent):
        restore_dataset(param, ima_def_dict)
        Conf.view.set_def_dict("ima", ima_def_dict)
        return True
    return False


class ViewSettings(gds.DataSet):
    """DataLab Visualization settings"""

    g0 = gds.BeginGroup(_("Common"))
    plot_toolbar_position = gds.ImageChoiceItem(
        _("Plot toolbar position"),
        (
            ("top", _("Top (above plot)"), "libre-gui-arrow-up.svg"),
            ("bottom", _("Bottom (below plot)"), "libre-gui-arrow-down.svg"),
            ("left", _("Left (of plot)"), "libre-gui-arrow-left.svg"),
            ("right", _("Right (of plot)"), "libre-gui-arrow-right.svg"),
        ),
    )
    ignore_title_insertion_msg = gds.BoolItem(
        "",
        _("Ignore title insertion message"),
        help=_(
            "Ignore information message when inserting object title as annotation label"
        ),
        default=False,
    )
    _g0 = gds.EndGroup("")

    g1 = gds.BeginGroup(_("Signal"))
    _prop_ads = gds.ValueProp(False)
    sig_autodownsampling = gds.BoolItem(
        "",
        _("Use auto downsampling"),
        help=_("Use auto downsampling for large signals"),
    ).set_prop("display", store=_prop_ads)
    sig_autodownsampling_maxpoints = gds.IntItem(
        _("Downsampling max points"),
        default=10000,
        min=1000,
        help=_("Maximum number of points for downsampling"),
    ).set_prop("display", active=_prop_ads)
    sig_autoscale_margin_percent = gds.FloatItem(
        _("Autoscale margin"),
        unit=_("%"),
        default=0.2,
        min=0.0,
        max=50.0,
        help=_(
            "Percentage of margin to add around data when auto-scaling signal "
            "plots.<br>A value of 0.2&#37; adds a small margin for better "
            "visualization.<br>Set to 0&#37; for no margin (exact data bounds)."
        ),
    )
    sig_datetime_format_s = gds.StringItem(
        _("DateTime format (s/min/h)"),
        default="%H:%M:%S",
        help=_(
            "Format string for datetime X-axis labels when using standard time "
            "units (s, min, h).<br>Uses Python's strftime format codes "
            "(e.g., %H:%M:%S for hours:minutes:seconds)."
        ),
    )
    sig_datetime_format_ms = gds.StringItem(
        _("DateTime format (ms/μs/ns)"),
        default="%H:%M:%S.%f",
        help=_(
            "Format string for datetime X-axis labels when using sub-second time "
            "units (ms, μs, ns).<br>Uses Python's strftime format codes "
            "(e.g., %H:%M:%S.%f for hours:minutes:seconds.microseconds)."
        ),
    )
    _g1 = gds.EndGroup("")

    g2 = gds.BeginGroup(_("Image"))
    ima_aspect_ratio_1_1 = gds.BoolItem(
        "",
        _("Lock aspect ratio to 1:1"),
        help=_(
            "If enabled, the aspect ratio of the image will be locked to 1:1.<br>"
            "If disabled, the aspect ratio will be determined by the physical pixel "
            "size (this is the default and recommended setting)."
        ),
    )
    ima_ref_lut_range = gds.BoolItem(
        "",
        _("Use reference image LUT range"),
        help=_(
            "If this setting is enabled, images are shown<br>"
            "with the same LUT range as the first selected image"
        ),
    )
    ima_eliminate_outliers = gds.FloatItem(
        _("Eliminate outliers"),
        unit=_("%"),
        default=2.0,
        min=0.0,
        max=100.0,
        help=_(
            "Eliminate a percentage of the highest and lowest values<br>"
            "of the image histogram - <i>recommanded values are below 1%</i>"
        ),
    )
    ima_autoscale_margin_percent = gds.FloatItem(
        _("Autoscale margin"),
        unit=_("%"),
        default=0.2,
        min=0.0,
        max=50.0,
        help=_(
            "Percentage of margin to add around data when auto-scaling image "
            "plots.<br>A value of 0.2&#37; adds a small margin for better "
            "visualization.<br>Set to 0&#37; for no margin (exact data bounds)."
        ),
    )
    ima_defaults = gds.ButtonItem(
        _("Default image visualization settings"),
        edit_default_image_settings,
        icon="image.svg",
        default=False,
    )
    _g2 = gds.EndGroup("")


# Generator yielding (param, section, option) tuples from configuration dictionary
def _iter_conf(
    paramdict: dict[str, gds.DataSet],
) -> Generator[tuple[gds.DataSet, str, str], None, None]:
    """Iterate over configuration parameters"""
    confdict = Conf.to_dict()
    for section_name, section in confdict.items():
        if section_name in paramdict:
            for option in section:
                param = paramdict[section_name]
                if hasattr(param, option):
                    yield param, section_name, option


def conf_to_datasets(paramdict: dict[str, gds.DataSet]) -> None:
    """Convert DataLab configuration to datasets"""
    for param, section, option in _iter_conf(paramdict):
        value = getattr(getattr(Conf, section), option).get()
        # ConfigParser automatically unescapes %% to % when reading, but to be safe
        # we ensure datetime format strings are properly unescaped for display
        if option in ("sig_datetime_format_s", "sig_datetime_format_ms"):
            value = value.replace("%%", "%")
        setattr(param, option, value)


def datasets_to_conf(paramdict: dict[str, gds.DataSet]) -> None:
    """Convert datasets to DataLab configuration"""
    for param, section, option in _iter_conf(paramdict):
        value = getattr(param, option)
        # Escape % characters for datetime format strings (ConfigParser requirement)
        if option in ("sig_datetime_format_s", "sig_datetime_format_ms"):
            value = value.replace("%", "%%")
        getattr(getattr(Conf, section), option).set(value)


RESTART_OPTIONS = (
    ("process_isolation_enabled", _("Process isolation enable status")),
    ("rpc_server_enabled", _("RPC server enable status")),
    ("console_enabled", _("Console enable status")),
    ("plugins_enabled", _("Third-party plugins support")),
    ("plugins_path", _("Third-party plugins path")),
)


def get_restart_items_values(paramdict: dict[str, gds.DataSet]) -> list:
    """Get restart items values"""
    values = []
    for option, _name in RESTART_OPTIONS:
        for param, _section, _option in _iter_conf(paramdict):
            if option == _option:
                values.append(getattr(param, option))
    return values


def get_all_values(paramdict: dict[str, gds.DataSet]) -> list:
    """Get all values"""
    values = []
    for param, _section, _option in _iter_conf(paramdict):
        values.append(getattr(param, _option))
    return values


def get_all_options(paramdict: dict[str, gds.DataSet]) -> list:
    """Get all options"""
    options = []
    for _param, _section, _option in _iter_conf(paramdict):
        options.append(_option)
    return options


def edit_settings(parent: QW.QWidget) -> None:
    """Edit DataLab settings

    Args:
        parent: Parent widget
    """
    paramdict = {
        "main": MainSettings(_("General")),
        "proc": ProcSettings(_("Processing")),
        "view": ViewSettings(_("Visualization")),
        "io": IOSettings(_("I/O")),
        "console": ConsoleSettings(_("Console")),
    }
    conf_to_datasets(paramdict)
    before = get_restart_items_values(paramdict)
    all_values_before = get_all_values(paramdict)
    params = gds.DataSetGroup(paramdict.values(), title=_("Settings"))
    changed_options = []
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
        all_values_after = get_all_values(paramdict)
        all_options = get_all_options(paramdict)
        if all_values_before != all_values_after:
            changed_options = [
                option
                for option, value in zip(all_options, all_values_before)
                if value != all_values_after[all_options.index(option)]
            ]
        for vis_defaults in ("ima_defaults",):
            if getattr(paramdict["view"], vis_defaults):
                changed_options.append(vis_defaults)

    return changed_options
