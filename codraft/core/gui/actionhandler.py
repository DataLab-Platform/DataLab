# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Action handler module

This module handles all application actions (menus, toolbars, context menu).
These actions point to CodraFT panels, processors, objectlist, ...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc
import enum

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from codraft.config import _
from codraft.widgets import fitdialog


class ActionCategory(enum.Enum):
    """Action categories"""

    FILE = enum.auto()
    EDIT = enum.auto()
    VIEW = enum.auto()
    OPERATION = enum.auto()
    PROCESSING = enum.auto()
    COMPUTING = enum.auto()


class BaseActionHandler(metaclass=abc.ABCMeta):
    """Object handling panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = ""  # e.g. "signal"

    def __init__(self, panel, objlist, itmlist, processor, toolbar):
        self.panel = panel
        self.objlist = objlist
        self.itmlist = itmlist
        self.processor = processor
        self.feature_actions = {}
        self.operation_end_actions = None
        # Object selection dependent actions
        self.actlist_1more = []
        self.actlist_2more = []
        self.actlist_1 = []
        self.actlist_2 = []
        self.actlist_cmenu = []  # Context menu
        if self.__class__ is not BaseActionHandler:
            self.create_all_actions(toolbar)

    def get_context_menu_actions(self):
        """Return context menu action list"""
        return self.actlist_cmenu

    def selection_rows_changed(self):
        """Number of selected rows has changed"""
        nbrows = len(self.objlist.get_selected_rows())
        for act in self.actlist_1more:
            act.setEnabled(nbrows >= 1)
        for act in self.actlist_2more:
            act.setEnabled(nbrows >= 2)
        for act in self.actlist_1:
            act.setEnabled(nbrows == 1)
        for act in self.actlist_2:
            act.setEnabled(nbrows == 2)

    def create_all_actions(self, toolbar):
        """Setup actions, menus, toolbar"""
        featact = self.feature_actions
        featact[ActionCategory.FILE] = file_act = self.create_file_actions()
        featact[ActionCategory.EDIT] = edit_act = self.create_edit_actions()
        featact[ActionCategory.VIEW] = view_act = self.create_view_actions()
        featact[ActionCategory.OPERATION] = self.create_operation_actions()
        featact[ActionCategory.PROCESSING] = self.create_processing_actions()
        featact[ActionCategory.COMPUTING] = self.create_computing_actions()
        add_actions(toolbar, file_act + [None] + edit_act + [None] + view_act)

    def cra(
        self, title, triggered=None, toggled=None, shortcut=None, icon=None, tip=None
    ):
        """Create action convenience method"""
        return create_action(self.panel, title, triggered, toggled, shortcut, icon, tip)

    def create_file_actions(self):
        """Create file actions"""
        new_act = self.cra(
            _("New %s...") % self.OBJECT_STR,
            icon=get_icon(f"new_{self.OBJECT_STR}.svg"),
            tip=_("Create new %s") % self.OBJECT_STR,
            triggered=self.panel.new_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.New),
        )
        open_act = self.cra(
            _("Open %s...") % self.OBJECT_STR,
            icon=get_icon("libre-gui-import.svg"),
            tip=_("Open %s") % self.OBJECT_STR,
            triggered=self.panel.open_objects,
            shortcut=QG.QKeySequence(QG.QKeySequence.Open),
        )
        save_act = self.cra(
            _("Save %s...") % self.OBJECT_STR,
            icon=get_icon("libre-gui-export.svg"),
            tip=_("Save selected %s") % self.OBJECT_STR,
            triggered=self.panel.save_objects,
            shortcut=QG.QKeySequence(QG.QKeySequence.Save),
        )
        importmd_act = self.cra(
            _("Import metadata into %s...") % self.OBJECT_STR,
            icon=get_icon("metadata_import.svg"),
            tip=_("Import metadata into %s") % self.OBJECT_STR,
            triggered=self.panel.import_metadata_from_file,
        )
        exportmd_act = self.cra(
            _("Export metadata from %s...") % self.OBJECT_STR,
            icon=get_icon("metadata_export.svg"),
            tip=_("Export selected %s metadata") % self.OBJECT_STR,
            triggered=self.panel.export_metadata_from_file,
        )
        self.actlist_1more += [save_act]
        self.actlist_cmenu += [save_act]
        self.actlist_1 += [importmd_act, exportmd_act]
        return [new_act, open_act, save_act, None, importmd_act, exportmd_act]

    def create_edit_actions(self):
        """Create edit actions"""
        dup_action = self.cra(
            _("Duplicate"),
            icon=get_icon("libre-gui-copy.svg"),
            triggered=self.panel.duplicate_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.Copy),
        )
        cpymeta_action = self.cra(
            _("Copy metadata"),
            icon=get_icon("metadata_copy.svg"),
            triggered=self.panel.copy_metadata,
        )
        pstmeta_action = self.cra(
            _("Paste metadata"),
            icon=get_icon("metadata_paste.svg"),
            triggered=self.panel.paste_metadata,
        )
        cleanup_action = self.cra(
            _("Clean up data view"),
            icon=get_icon("libre-tools-vacuum-cleaner.svg"),
            tip=_("Clean up data view before updating plotting panels"),
            toggled=self.itmlist.toggle_cleanup_dataview,
        )
        cleanup_action.setChecked(True)
        delm_action = self.cra(
            _("Delete object metadata"),
            icon=get_icon("metadata_delete.svg"),
            tip=_("Delete all that is contained in object metadata"),
            triggered=self.panel.delete_metadata,
        )
        delall_action = self.cra(
            _("Delete all"),
            shortcut="Shift+Ctrl+Suppr",
            icon=get_icon("delete_all.svg"),
            triggered=self.panel.delete_all_objects,
        )
        del_action = self.cra(
            _("Remove"),
            icon=get_icon("delete.svg"),
            triggered=self.panel.remove_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.Delete),
        )
        self.actlist_1more += [
            dup_action,
            del_action,
            delm_action,
            pstmeta_action,
            delall_action,
        ]
        self.actlist_cmenu += [dup_action, del_action]
        self.actlist_1 += [cpymeta_action]
        return [
            dup_action,
            del_action,
            delall_action,
            None,
            cpymeta_action,
            pstmeta_action,
            delm_action,
        ]

    def create_view_actions(self):
        """Create view actions"""
        view_action = self.cra(
            _("View in a new window"),
            icon=get_icon("libre-gui-binoculars.svg"),
            triggered=self.panel.open_separate_view,
        )
        showlabel_action = self.cra(
            _("Graphical object titles"),
            icon=get_icon("show_titles.svg"),
            tip=_("Show or hide ROI and other graphical object titles or subtitles"),
            toggled=self.panel.toggle_show_titles,
        )
        showlabel_action.setChecked(True)
        self.actlist_1more += [view_action]
        self.actlist_cmenu = [view_action, None] + self.actlist_cmenu
        return [view_action, showlabel_action]

    def create_operation_actions(self):
        """Create operation actions"""
        proc = self.processor
        sum_action = self.cra(_("Sum"), proc.compute_sum)
        average_action = self.cra(_("Average"), proc.compute_average)
        diff_action = self.cra(_("Difference"), lambda: proc.compute_difference(False))
        qdiff_action = self.cra(
            _("Quadratic difference"), lambda: proc.compute_difference(True)
        )
        prod_action = self.cra(_("Product"), proc.compute_product)
        div_action = self.cra(_("Division"), proc.compute_division)
        roi_action = self.cra(
            _("ROI extraction"),
            proc.extract_roi,
            icon=get_icon(f"{self.OBJECT_STR}_roi.svg"),
        )
        swapaxes_action = self.cra(_("Swap X/Y axes"), proc.swap_axes)
        abs_action = self.cra(_("Absolute value"), proc.compute_abs)
        log_action = self.cra("Log10(y)", proc.compute_log10)
        self.actlist_1more += [roi_action, swapaxes_action, abs_action, log_action]
        self.actlist_2more += [sum_action, average_action, prod_action]
        self.actlist_2 += [diff_action, qdiff_action, div_action]
        self.operation_end_actions = [roi_action, swapaxes_action]
        return [
            sum_action,
            average_action,
            diff_action,
            qdiff_action,
            prod_action,
            div_action,
            None,
            abs_action,
            log_action,
        ]

    def create_processing_actions(self):
        """Create processing actions"""
        proc = self.processor
        threshold_action = self.cra(_("Thresholding"), proc.compute_threshold)
        clip_action = self.cra(_("Clipping"), proc.compute_clip)
        lincal_action = self.cra(_("Linear calibration"), proc.calibrate)
        gauss_action = self.cra(_("Gaussian filter"), proc.compute_gaussian)
        movavg_action = self.cra(_("Moving average"), proc.compute_moving_average)
        movmed_action = self.cra(_("Moving median"), proc.compute_moving_median)
        wiener_action = self.cra(_("Wiener filter"), proc.compute_wiener)
        fft_action = self.cra(_("FFT"), proc.compute_fft)
        ifft_action = self.cra(_("Inverse FFT"), proc.compute_ifft)
        for act in (fft_action, ifft_action):
            act.setToolTip(_("Warning: only real part is plotted"))
        actions = [
            threshold_action,
            clip_action,
            lincal_action,
            gauss_action,
            movavg_action,
            movmed_action,
            wiener_action,
            fft_action,
            ifft_action,
        ]
        self.actlist_1more += actions
        return actions

    @abc.abstractmethod
    def create_computing_actions(self):
        """Create computing actions"""
        proc = self.processor
        defineroi_action = self.cra(
            _("Regions of interest..."),
            triggered=proc.edit_regions_of_interest,
            icon=get_icon("roi.svg"),
        )
        stats_action = self.cra(
            _("Statistics") + "...",
            triggered=proc.compute_stats,
            icon=get_icon("stats.svg"),
        )
        self.actlist_1 += [defineroi_action, stats_action]
        self.actlist_cmenu += [None, defineroi_action, stats_action]
        return [defineroi_action, None, stats_action]


class SignalActionHandler(BaseActionHandler):
    """Object handling signal panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("signal")

    def create_operation_actions(self):
        """Create operation actions"""
        base_actions = super().create_operation_actions()
        proc = self.processor
        peakdetect_action = self.cra(
            _("Peak detection"),
            proc.detect_peaks,
            icon=get_icon("peak_detect.svg"),
        )
        self.actlist_1more += [peakdetect_action]
        roi_actions = self.operation_end_actions
        return base_actions + [None, peakdetect_action, None] + roi_actions

    def create_processing_actions(self):
        """Create processing actions"""
        base_actions = super().create_processing_actions()
        proc = self.processor
        normalize_action = self.cra(_("Normalize"), proc.normalize)
        deriv_action = self.cra(_("Derivative"), proc.compute_derivative)
        integ_action = self.cra(_("Integral"), proc.compute_integral)
        polyfit_action = self.cra(_("Polynomial fit"), proc.compute_polyfit)
        mgfit_action = self.cra(_("Multi-Gaussian fit"), proc.compute_multigaussianfit)

        def cra_fit(title, fitdlgfunc):
            """Create curve fitting action"""
            return self.cra(title, lambda: proc.compute_fit(title, fitdlgfunc))

        gaussfit_action = cra_fit(_("Gaussian fit"), fitdialog.gaussianfit)
        lorentzfit_action = cra_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
        voigtfit_action = cra_fit(_("Voigt fit"), fitdialog.voigtfit)
        actions1 = [normalize_action, deriv_action, integ_action]
        actions2 = [
            gaussfit_action,
            lorentzfit_action,
            voigtfit_action,
            polyfit_action,
            mgfit_action,
        ]
        self.actlist_1more += actions1 + actions2
        return actions1 + [None] + base_actions + [None] + actions2

    def create_computing_actions(self):
        """Create computing actions"""
        base_actions = super().create_computing_actions()
        proc = self.processor
        fwhm_action = self.cra(
            _("Full width at half-maximum"),
            triggered=proc.compute_fwhm,
            tip=_("Compute Full Width at Half-Maximum (FWHM)"),
        )
        fw1e2_action = self.cra(
            _("Full width at") + " 1/e²",
            triggered=proc.compute_fw1e2,
            tip=_("Compute Full Width at Maximum") + "/e²",
        )
        self.actlist_1more += [fwhm_action, fw1e2_action]
        return base_actions + [fwhm_action, fw1e2_action]


class ImageActionHandler(BaseActionHandler):
    """Object handling image panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("image")

    def create_operation_actions(self):
        """Create operation actions"""
        base_actions = super().create_operation_actions()
        proc = self.processor
        rotate_menu = QW.QMenu(_("Rotation"), self.panel)
        hflip_act = self.cra(
            _("Flip horizontally"),
            triggered=proc.flip_horizontally,
            icon=get_icon("flip_horizontally.svg"),
        )
        vflip_act = self.cra(
            _("Flip vertically"),
            triggered=proc.flip_vertically,
            icon=get_icon("flip_vertically.svg"),
        )
        rot90_act = self.cra(
            _("Rotate %s right") % "90°",  # pylint: disable=consider-using-f-string
            triggered=proc.rotate_270,
            icon=get_icon("rotate_right.svg"),
        )
        rot270_act = self.cra(
            _("Rotate %s left") % "90°",  # pylint: disable=consider-using-f-string
            triggered=proc.rotate_90,
            icon=get_icon("rotate_left.svg"),
        )
        rotate_act = self.cra(
            _("Rotate arbitrarily..."), triggered=proc.rotate_arbitrarily
        )
        resize_act = self.cra(_("Resize"), triggered=proc.resize_image)
        logp1_act = self.cra("Log10(z+n)", triggered=proc.compute_logp1)
        flatfield_act = self.cra(
            _("Flat-field correction"), triggered=proc.flat_field_correction
        )
        self.actlist_2 += [flatfield_act]
        self.actlist_1more += [
            resize_act,
            hflip_act,
            vflip_act,
            logp1_act,
            rot90_act,
            rot270_act,
            rotate_act,
        ]
        self.actlist_cmenu += [None, hflip_act, vflip_act, rot90_act, rot270_act]
        add_actions(
            rotate_menu, [hflip_act, vflip_act, rot90_act, rot270_act, rotate_act]
        )
        roi_actions = self.operation_end_actions
        actions = [
            logp1_act,
            flatfield_act,
            None,
            rotate_menu,
            None,
            resize_act,
        ]
        return base_actions + actions + roi_actions

    def create_computing_actions(self):
        """Create computing actions"""
        base_actions = super().create_computing_actions()
        proc = self.processor
        # TODO: [P3] Add "Create ROI grid..." action to create a regular grid or ROIs
        cent_act = self.cra(
            _("Centroid"), proc.compute_centroid, tip=_("Compute image centroid")
        )
        encl_act = self.cra(
            _("Minimum enclosing circle center"),
            proc.compute_enclosing_circle,
            tip=_("Compute smallest enclosing circle center"),
        )
        peak_act = self.cra(
            _("2D peak detection"),
            proc.compute_peak_detection,
            tip=_("Compute automatic 2D peak detection"),
        )
        contour_act = self.cra(
            _("Contour detection"),
            proc.compute_contour_shape,
            tip=_("Compute contour shape fit"),
        )
        self.actlist_1more += [cent_act, encl_act, peak_act, contour_act]
        return base_actions + [cent_act, encl_act, peak_act, contour_act]
