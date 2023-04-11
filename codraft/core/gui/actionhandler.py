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

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
import enum
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, List

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from codraft.config import _
from codraft.widgets import fitdialog

if TYPE_CHECKING:
    from codraft.core.gui.panel import BaseDataPanel
    from codraft.core.gui.plotitemlist import BaseItemList
    from codraft.core.gui.processor.base import BaseProcessor
    from codraft.core.model.base import ObjectItf


class SelectCond:
    """Signal or image select conditions"""

    @staticmethod
    def always(selected_objects: List[ObjectItf]) -> bool:
        """Always true"""
        return True

    @staticmethod
    def exactly_one(selected_objects: List[ObjectItf]) -> bool:
        """Exactly one signal or image is selected"""
        return len(selected_objects) == 1

    @staticmethod
    def exactly_two(selected_objects: List[ObjectItf]) -> bool:
        """Exactly two signals or images are selected"""
        return len(selected_objects) == 2

    @staticmethod
    def at_least_one(selected_objects: List[ObjectItf]) -> bool:
        """At least one signal or image is selected"""
        return len(selected_objects) >= 1

    @staticmethod
    def at_least_two(selected_objects: List[ObjectItf]) -> bool:
        """At least two signals or images are selected"""
        return len(selected_objects) >= 2

    @staticmethod
    def with_roi(selected_objects: List[ObjectItf]) -> bool:
        """At least one signal or image has a ROI"""
        return any(obj.roi is not None for obj in selected_objects)


class ActionCategory(enum.Enum):
    """Action categories"""

    FILE = enum.auto()
    EDIT = enum.auto()
    VIEW = enum.auto()
    OPERATION = enum.auto()
    PROCESSING = enum.auto()
    COMPUTING = enum.auto()
    CONTEXT_MENU = enum.auto()
    TOOLBAR = enum.auto()
    SUBMENU = enum.auto()  # temporary
    PLUGINS = enum.auto()  # for plugins actions


class BaseActionHandler(metaclass=abc.ABCMeta):
    """Object handling panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = ""  # e.g. "signal"

    def __init__(
        self,
        panel: BaseDataPanel,
        itmlist: BaseItemList,
        proc: BaseProcessor,
        toolbar: QW.QToolBar,
    ):
        self.panel = panel
        self.itmlist = itmlist
        self.proc = proc
        self.toolbar = toolbar
        self.feature_actions = {}
        self.operation_end_actions = None
        self.__category_in_progress: ActionCategory = None
        self.__submenu_in_progress = False
        self.__actions = {}

    @contextmanager
    def new_category(self, category: ActionCategory):
        """Context manager for creating a new menu"""
        self.__category_in_progress = category
        try:
            yield
        finally:
            self.__category_in_progress = None

    @contextmanager
    def new_menu(self, title: str):
        """Context manager for creating a new menu"""
        menu = QW.QMenu(title)
        self.__submenu_in_progress = True
        try:
            yield
        finally:
            self.__submenu_in_progress = False
            add_actions(menu, self.feature_actions.pop(ActionCategory.SUBMENU))
            self.add_to_action_list(menu)

    def new_action(
        self,
        title: str,
        position: int = None,
        separator: bool = False,
        triggered: Callable = None,
        toggled: Callable = None,
        shortcut: QW.QShortcut = None,
        icon: QG.QIcon = None,
        tip: str = None,
        select_condition: Callable = None,
        context_menu_pos: int = None,
        context_menu_sep: bool = False,
        toolbar_pos: int = None,
        toolbar_sep: bool = False,
    ):
        """Create new action and add it to list of actions

        param title: action title
        param position: add action to menu at this position
        param separator: add separator before action in menu
        (or after if pos is positive)
        param triggered: triggered callback
        param toggled: toggled callback
        param shortcut: shortcut
        param icon: icon
        param tip: tooltip
        param select_condition: condition to enable action
        param context_menu_pos: add action to context menu at this position
        param context_menu_sep: add separator before action in context menu
        (or after if context_menu_pos is positive)
        param toolbar_pos: add action to toolbar at this position
        param toolbar_sep: add separator before action in toolbar
        (or after if toolbar_pos is positive)

        return: new action

        If select_condition is None, action is enabled if at least one object
        is selected.
        """
        action = create_action(None, title, triggered, toggled, shortcut, icon, tip)
        self.add_action(action, select_condition)
        self.add_to_action_list(action, None, position, separator)
        if context_menu_pos is not None:
            self.add_to_action_list(
                action, ActionCategory.CONTEXT_MENU, context_menu_pos, context_menu_sep
            )
        if toolbar_pos is not None:
            self.add_to_action_list(
                action, ActionCategory.TOOLBAR, toolbar_pos, toolbar_sep
            )
        return action

    def add_to_action_list(
        self,
        action: QW.QAction,
        category: ActionCategory = None,
        pos: int = None,
        sep: bool = False,
    ):
        if category is None:
            if self.__submenu_in_progress:
                category = ActionCategory.SUBMENU
            elif self.__category_in_progress is not None:
                category = self.__category_in_progress
            else:
                raise ValueError("No category specified")
        if pos is None:
            pos = -1
        actionlist = self.feature_actions.setdefault(category, [])
        add_separator_after = pos >= 0
        if pos < 0:
            pos = len(actionlist) + pos + 1
        actionlist.insert(pos, action)
        if sep:
            if add_separator_after:
                pos += 1
            actionlist.insert(pos, None)

    def add_action(
        self,
        action: QW.QAction,
        select_condition: Callable = None,
    ):
        """Add action to list of actions

        param action: action to add
        param select_condition: condition to enable action

        If select_condition is None, action is enabled if at least one object
        is selected.
        """
        if select_condition is None:
            select_condition = SelectCond.at_least_one
        self.__actions.setdefault(select_condition, []).append(action)

    def selection_rows_changed(self, selected_objects: List[ObjectItf]):
        """Number of selected rows has changed"""
        for cond, actlist in self.__actions.items():
            if cond is not None:
                for act in actlist:
                    act.setEnabled(cond(selected_objects))

    def create_all_actions(self):
        """Create all actions"""
        self.create_first_actions()
        self.create_last_actions()
        add_actions(self.toolbar, self.feature_actions.pop(ActionCategory.TOOLBAR))

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        with self.new_category(ActionCategory.FILE):
            self.new_action(
                _("New %s...") % self.OBJECT_STR,
                icon=get_icon(f"new_{self.OBJECT_STR}.svg"),
                tip=_("Create new %s") % self.OBJECT_STR,
                triggered=self.panel.new_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.New),
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Open %s...") % self.OBJECT_STR,
                icon=get_icon("libre-gui-import.svg"),
                tip=_("Open %s") % self.OBJECT_STR,
                triggered=self.panel.open_objects,
                shortcut=QG.QKeySequence(QG.QKeySequence.Open),
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Save %s...") % self.OBJECT_STR,
                icon=get_icon("libre-gui-export.svg"),
                tip=_("Save selected %s") % self.OBJECT_STR,
                triggered=self.panel.save_objects,
                shortcut=QG.QKeySequence(QG.QKeySequence.Save),
                select_condition=SelectCond.at_least_one,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Import metadata into %s...") % self.OBJECT_STR,
                separator=True,
                icon=get_icon("metadata_import.svg"),
                tip=_("Import metadata into %s") % self.OBJECT_STR,
                triggered=self.panel.import_metadata_from_file,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Export metadata from %s...") % self.OBJECT_STR,
                icon=get_icon("metadata_export.svg"),
                tip=_("Export selected %s metadata") % self.OBJECT_STR,
                triggered=self.panel.export_metadata_from_file,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
                toolbar_sep=True,
            )

        with self.new_category(ActionCategory.EDIT):
            self.new_action(
                _("Duplicate"),
                icon=get_icon("libre-gui-copy.svg"),
                triggered=self.panel.duplicate_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.Copy),
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Remove"),
                icon=get_icon("delete.svg"),
                triggered=self.panel.remove_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.Delete),
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Delete all"),
                shortcut="Shift+Ctrl+Suppr",
                icon=get_icon("delete_all.svg"),
                triggered=self.panel.delete_all_objects,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Copy metadata"),
                separator=True,
                icon=get_icon("metadata_copy.svg"),
                triggered=self.panel.copy_metadata,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Paste metadata"),
                icon=get_icon("metadata_paste.svg"),
                triggered=self.panel.paste_metadata,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Delete object metadata"),
                icon=get_icon("metadata_delete.svg"),
                tip=_("Delete all that is contained in object metadata"),
                triggered=self.panel.delete_metadata,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Copy titles to clipboard"),
                separator=True,
                icon=get_icon("copy_titles.svg"),
                triggered=self.panel.copy_titles_to_clipboard,
                toolbar_pos=-1,
                toolbar_sep=True,
            )
        # TODO: Add cleanup_action to edit menu???
        # cleanup_action = self.new_action(
        #     _("Clean up data view"),
        #     icon=get_icon("libre-tools-vacuum-cleaner.svg"),
        #     tip=_("Clean up data view before updating plotting panels"),
        #     toggled=self.itmlist.toggle_cleanup_dataview,
        #     select_condition=SelectCond.always,
        # )
        # cleanup_action.setChecked(True)

        with self.new_category(ActionCategory.VIEW):
            self.new_action(
                _("View in a new window"),
                icon=get_icon("libre-gui-binoculars.svg"),
                triggered=self.panel.open_separate_view,
                context_menu_pos=0,
                context_menu_sep=True,
                toolbar_pos=-1,
            )
            showlabel_action = self.new_action(
                _("Show graphical object titles"),
                icon=get_icon("show_titles.svg"),
                tip=_(
                    "Show or hide ROI and other graphical object titles or subtitles"
                ),
                toggled=self.panel.toggle_show_titles,
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            showlabel_action.setChecked(False)

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("Sum"),
                triggered=self.proc.compute_sum,
                select_condition=SelectCond.at_least_two,
            )
            self.new_action(
                _("Average"),
                triggered=self.proc.compute_average,
                select_condition=SelectCond.at_least_two,
            )
            self.new_action(
                _("Difference"),
                triggered=lambda: self.proc.compute_difference(False),
                select_condition=SelectCond.exactly_two,
            )
            self.new_action(
                _("Quadratic difference"),
                triggered=lambda: self.proc.compute_difference(True),
                select_condition=SelectCond.exactly_two,
            )
            self.new_action(
                _("Product"),
                triggered=self.proc.compute_product,
                select_condition=SelectCond.at_least_two,
            )
            self.new_action(
                _("Division"),
                triggered=self.proc.compute_division,
                select_condition=SelectCond.exactly_two,
            )
            self.new_action(_("Absolute value"), triggered=self.proc.compute_abs)
            self.new_action("Log10(y)", triggered=self.proc.compute_log10)

        with self.new_category(ActionCategory.PROCESSING):
            self.new_action(
                _("Thresholding"),
                triggered=self.proc.compute_threshold,
            )
            self.new_action(
                _("Clipping"),
                triggered=self.proc.compute_clip,
            )
            self.new_action(
                _("Linear calibration"),
                triggered=self.proc.calibrate,
            )
            self.new_action(
                _("Gaussian filter"),
                triggered=self.proc.compute_gaussian,
            )
            self.new_action(
                _("Moving average"),
                triggered=self.proc.compute_moving_average,
            )
            self.new_action(
                _("Moving median"),
                triggered=self.proc.compute_moving_median,
            )
            self.new_action(
                _("Wiener filter"),
                triggered=self.proc.compute_wiener,
            )
            self.new_action(
                _("FFT"),
                triggered=self.proc.compute_fft,
                tip=_("Warning: only real part is plotted"),
            )
            self.new_action(
                _("Inverse FFT"),
                triggered=self.proc.compute_ifft,
                tip=_("Warning: only real part is plotted"),
            )

        with self.new_category(ActionCategory.COMPUTING):
            self.new_action(
                _("Edit regions of interest..."),
                triggered=self.proc.edit_regions_of_interest,
                icon=get_icon("roi.svg"),
                select_condition=SelectCond.exactly_one,
                context_menu_pos=-1,
                context_menu_sep=True,
            )
            self.new_action(
                _("Remove regions of interest"),
                triggered=self.proc.delete_regions_of_interest,
                icon=get_icon("roi_delete.svg"),
                select_condition=SelectCond.with_roi,
                context_menu_pos=-1,
            )
            self.new_action(
                _("Statistics") + "...",
                separator=True,
                triggered=self.proc.compute_stats,
                icon=get_icon("stats.svg"),
                select_condition=SelectCond.exactly_one,
                context_menu_pos=-1,
                context_menu_sep=True,
            )

    def create_last_actions(self):
        """Create actions that are added to the menus in the end"""
        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("ROI extraction"),
                triggered=self.proc.extract_roi,
                icon=get_icon(f"{self.OBJECT_STR}_roi.svg"),
            )
            self.new_action(_("Swap X/Y axes"), triggered=self.proc.swap_axes)


class SignalActionHandler(BaseActionHandler):
    """Object handling signal panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("signal")

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        with self.new_category(ActionCategory.PROCESSING):
            self.new_action(_("Normalize"), triggered=self.proc.normalize)
            self.new_action(_("Derivative"), triggered=self.proc.compute_derivative)
            self.new_action(_("Integral"), triggered=self.proc.compute_integral)

        super().create_first_actions()

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("Peak detection"),
                triggered=self.proc.detect_peaks,
                icon=get_icon("peak_detect.svg"),
            )

        def cra_fit(title, fitdlgfunc):
            """Create curve fitting action"""
            return self.new_action(
                title,
                triggered=lambda: self.proc.compute_fit(title, fitdlgfunc),
            )

        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Fitting")):
                cra_fit(_("Gaussian fit"), fitdialog.gaussianfit)
                cra_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
                cra_fit(_("Voigt fit"), fitdialog.voigtfit)
                self.new_action(
                    _("Polynomial fit"),
                    triggered=self.proc.compute_polyfit,
                )
                self.new_action(
                    _("Multi-Gaussian fit"),
                    triggered=self.proc.compute_multigaussianfit,
                )

        with self.new_category(ActionCategory.COMPUTING):
            self.new_action(
                _("Full width at half-maximum"),
                triggered=self.proc.compute_fwhm,
                tip=_("Compute Full Width at Half-Maximum (FWHM)"),
            )
            self.new_action(
                _("Full width at") + " 1/e²",
                triggered=self.proc.compute_fw1e2,
                tip=_("Compute Full Width at Maximum") + "/e²",
            )


class ImageActionHandler(BaseActionHandler):
    """Object handling image panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("image")

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        super().create_first_actions()

        with self.new_category(ActionCategory.VIEW):
            showcontrast_action = self.new_action(
                _("Show contrast panel"),
                icon=get_icon("contrast.png"),
                tip=_("Show or hide contrast adjustment panel"),
                toggled=self.panel.toggle_show_contrast,
                toolbar_pos=-1,
            )
            showcontrast_action.setChecked(True)

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                "Log10(z+n)",
                triggered=self.proc.compute_logp1,
            )
            self.new_action(
                _("Flat-field correction"),
                triggered=self.proc.flat_field_correction,
                select_condition=SelectCond.exactly_two,
            )

            with self.new_menu(_("Rotation")):
                self.new_action(
                    _("Flip horizontally"),
                    triggered=self.proc.flip_horizontally,
                    icon=get_icon("flip_horizontally.svg"),
                    context_menu_pos=-1,
                    context_menu_sep=True,
                )
                self.new_action(
                    _("Flip vertically"),
                    triggered=self.proc.flip_vertically,
                    icon=get_icon("flip_vertically.svg"),
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate %s right")
                    % "90°",  # pylint: disable=consider-using-f-string
                    triggered=self.proc.rotate_270,
                    icon=get_icon("rotate_right.svg"),
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate %s left")
                    % "90°",  # pylint: disable=consider-using-f-string
                    triggered=self.proc.rotate_90,
                    icon=get_icon("rotate_left.svg"),
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate arbitrarily..."),
                    triggered=self.proc.rotate_arbitrarily,
                )

            self.new_action(
                _("Distribute on a grid..."),
                triggered=self.proc.distribute_on_grid,
                select_condition=SelectCond.at_least_two,
            )
            self.new_action(
                _("Reset image positions"),
                triggered=self.proc.reset_positions,
                select_condition=SelectCond.at_least_two,
            )

            self.new_action(
                _("Resize"),
                triggered=self.proc.resize,
                separator=True,
            )
            self.new_action(_("Pixel binning"), triggered=self.proc.rebin)

        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Exposure")):
                self.new_action(
                    _("Intensity rescaling"),
                    triggered=self.proc.rescale_intensity,
                )
                self.new_action(
                    _("Histogram equalization"),
                    triggered=self.proc.equalize_hist,
                )
                self.new_action(
                    _("Adaptive histogram equalization"),
                    triggered=self.proc.equalize_adapthist,
                )
            with self.new_menu(_("Restoration")):
                self.new_action(
                    _("Total variation denoising"),
                    triggered=self.proc.compute_denoise_tv,
                )
                self.new_action(
                    _("Bilateral filter denoising"),
                    triggered=self.proc.compute_denoise_bilateral,
                )
                self.new_action(
                    _("Wavelet denoising"),
                    triggered=self.proc.compute_denoise_wavelet,
                )
                self.new_action(
                    _("White Top-Hat denoising"),
                    triggered=self.proc.compute_denoise_tophat,
                )
            with self.new_menu(_("Morphology")):
                self.new_action(
                    _("White Top-Hat (disk)"),
                    triggered=self.proc.compute_white_tophat,
                )
                self.new_action(
                    _("Black Top-Hat (disk)"),
                    triggered=self.proc.compute_black_tophat,
                )
                self.new_action(
                    _("Erosion (disk)"),
                    triggered=self.proc.compute_erosion,
                )
                self.new_action(
                    _("Dilation (disk)"),
                    triggered=self.proc.compute_dilation,
                )
                self.new_action(
                    _("Opening (disk)"),
                    triggered=self.proc.compute_opening,
                )
                self.new_action(
                    _("Closing (disk)"),
                    triggered=self.proc.compute_closing,
                )

            self.new_action(_("Canny filter"), triggered=self.proc.compute_canny)

        with self.new_category(ActionCategory.COMPUTING):
            # TODO: [P3] Add "Create ROI grid..." action to create a regular grid
            # or ROIs (maybe reuse/derive from `core.gui.processor.image.GridParam`)
            self.new_action(
                _("Centroid"),
                triggered=self.proc.compute_centroid,
                tip=_("Compute image centroid"),
            )
            self.new_action(
                _("Minimum enclosing circle center"),
                triggered=self.proc.compute_enclosing_circle,
                tip=_("Compute smallest enclosing circle center"),
            )
            self.new_action(
                _("2D peak detection"),
                separator=True,
                triggered=self.proc.compute_peak_detection,
                tip=_("Compute automatic 2D peak detection"),
            )
            self.new_action(
                _("Contour detection"),
                triggered=self.proc.compute_contour_shape,
                tip=_("Compute contour shape fit"),
            )
            self.new_action(
                _("Circle Hough transform"),
                triggered=self.proc.compute_hough_circle_peaks,
                tip=_("Detect circular shapes using circle Hough transform"),
            )

            with self.new_menu(_("Blob detection")):
                self.new_action(
                    _("Blob detection (DOG)"),
                    triggered=self.proc.compute_blob_dog,
                    tip=_("Detect blobs using Difference of Gaussian (DOG) method"),
                )
                self.new_action(
                    _("Blob detection (DOH)"),
                    triggered=self.proc.compute_blob_doh,
                    tip=_("Detect blobs using Determinant of Hessian (DOH) method"),
                )
                self.new_action(
                    _("Blob detection (LOG)"),
                    triggered=self.proc.compute_blob_log,
                    tip=_("Detect blobs using Laplacian of Gaussian (LOG) method"),
                )
                self.new_action(
                    _("Blob detection (OpenCV)"),
                    triggered=self.proc.compute_blob_opencv,
                    tip=_("Detect blobs using OpenCV SimpleBlobDetector"),
                )
