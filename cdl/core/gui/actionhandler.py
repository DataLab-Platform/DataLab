# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Action handler
==============

The :mod:`cdl.core.gui.actionhandler` module handles all application actions
(menus, toolbars, context menu). These actions point to DataLab panels, processors,
objecthandler, ...

Utility classes
---------------

.. autoclass:: SelectCond
    :members:

.. autoclass:: ActionCategory
    :members:

Handler classes
---------------

.. autoclass:: SignalActionHandler
    :members:
    :inherited-members:

.. autoclass:: ImageActionHandler
    :members:
    :inherited-members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import enum
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import Conf, _
from cdl.widgets import fitdialog

if TYPE_CHECKING:
    from cdl.core.gui.objectmodel import ObjectGroup
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj


class SelectCond:
    """Signal or image select conditions"""

    @staticmethod
    def __compat_groups(selected_groups: list[ObjectGroup], min_len: int = 1) -> bool:
        """Check if groups are compatible"""
        return (
            len(selected_groups) >= min_len
            and all(len(group) == len(selected_groups[0]) for group in selected_groups)
            and all(len(group) > 0 for group in selected_groups)
        )

    @staticmethod
    # pylint: disable=unused-argument
    def always(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """Always true"""
        return True

    @staticmethod
    def exactly_one(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """Exactly one signal or image is selected"""
        return len(selected_groups) == 0 and len(selected_objects) == 1

    @staticmethod
    # pylint: disable=unused-argument
    def exactly_one_group(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """Exactly one group is selected"""
        return len(selected_groups) == 1

    @staticmethod
    # pylint: disable=unused-argument
    def at_least_one_group_or_one_object(
        sel_groups: list[ObjectGroup],
        sel_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """At least one group or one signal or image is selected"""
        return len(sel_objects) >= 1 or len(sel_groups) >= 1

    @staticmethod
    # pylint: disable=unused-argument
    def at_least_one(
        sel_groups: list[ObjectGroup],
        sel_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """At least one signal or image is selected"""
        return len(sel_objects) >= 1 or SelectCond.__compat_groups(sel_groups, 1)

    @staticmethod
    def at_least_two(
        sel_groups: list[ObjectGroup],
        sel_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """At least two signals or images are selected"""
        return len(sel_objects) >= 2 or SelectCond.__compat_groups(sel_groups, 2)

    @staticmethod
    # pylint: disable=unused-argument
    def with_roi(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """At least one signal or image has a ROI"""
        return any(obj.roi is not None for obj in selected_objects)


class ActionCategory(enum.Enum):
    """Action categories"""

    FILE = enum.auto()
    EDIT = enum.auto()
    VIEW = enum.auto()
    OPERATION = enum.auto()
    PROCESSING = enum.auto()
    ANALYSIS = enum.auto()
    CONTEXT_MENU = enum.auto()
    PANEL_TOOLBAR = enum.auto()
    VIEW_TOOLBAR = enum.auto()
    SUBMENU = enum.auto()  # temporary
    PLUGINS = enum.auto()  # for plugins actions


class BaseActionHandler(metaclass=abc.ABCMeta):
    """Object handling panel GUI interactions: actions, menus, ...

    Args:
        panel: Panel to handle
        panel_toolbar: Panel toolbar (actions related to the panel objects management)
        view_toolbar: View toolbar (actions related to the panel view, i.e. plot)
    """

    OBJECT_STR = ""  # e.g. "signal"

    def __init__(
        self,
        panel: SignalPanel | ImagePanel,
        panel_toolbar: QW.QToolBar,
        view_toolbar: QW.QToolBar,
    ):
        self.panel = panel
        self.panel_toolbar = panel_toolbar
        self.view_toolbar = view_toolbar
        self.feature_actions = {}
        self.operation_end_actions = None
        self.__category_in_progress: ActionCategory = None
        self.__submenu_in_progress = False
        self.__actions: dict[Callable, list[QW.QAction]] = {}
        self.__submenus: dict[str, QW.QMenu] = {}

    @contextmanager
    def new_category(self, category: ActionCategory) -> Generator[None, None, None]:
        """Context manager for creating a new menu.

        Args:
            category: Action category

        Yields:
            None
        """
        self.__category_in_progress = category
        try:
            yield
        finally:
            self.__category_in_progress = None

    @contextmanager
    def new_menu(
        self, title: str, icon_name: str | None = None
    ) -> Generator[None, None, None]:
        """Context manager for creating a new menu.

        Args:
            title: Menu title
            icon_name: Menu icon name. Defaults to None.

        Yields:
            None
        """
        key = self.__category_in_progress.name + "/" + title
        is_new = key not in self.__submenus
        if is_new:
            self.__submenus[key] = menu = QW.QMenu(title)
            if icon_name:
                menu.setIcon(get_icon(icon_name))
        else:
            menu = self.__submenus[key]
        self.__submenu_in_progress = True
        try:
            yield
        finally:
            self.__submenu_in_progress = False
            add_actions(menu, self.feature_actions.pop(ActionCategory.SUBMENU))
            if is_new:
                self.add_to_action_list(menu)

    def new_action(
        self,
        title: str,
        position: int | None = None,
        separator: bool = False,
        triggered: Callable | None = None,
        toggled: Callable | None = None,
        shortcut: QW.QShortcut | None = None,
        icon_name: str | None = None,
        tip: str | None = None,
        select_condition: Callable | str | None = None,
        context_menu_pos: int | None = None,
        context_menu_sep: bool = False,
        toolbar_pos: int | None = None,
        toolbar_sep: bool = False,
        toolbar_category: ActionCategory | None = None,
    ) -> QW.QAction:
        """Create new action and add it to list of actions.

        Args:
            title: action title
            position: add action to menu at this position. Defaults to None.
            separator: add separator before action in menu
             (or after if pos is positive). Defaults to False.
            triggered: triggered callback. Defaults to None.
            toggled: toggled callback. Defaults to None.
            shortcut: shortcut. Defaults to None.
            icon_name: icon name. Defaults to None.
            tip: tooltip. Defaults to None.
            select_condition: selection condition. Defaults to None.
             If str, must be the name of a method of SelectCond, i.e. one of
             "always", "exactly_one", "exactly_one_group",
             "at_least_one_group_or_one_object", "at_least_one",
             "at_least_two", "with_roi".
            context_menu_pos: add action to context menu at this position.
             Defaults to None.
            context_menu_sep: add separator before action in context menu
             (or after if context_menu_pos is positive). Defaults to False.
            toolbar_pos: add action to toolbar at this position. Defaults to None.
            toolbar_sep: add separator before action in toolbar
             (or after if toolbar_pos is positive). Defaults to False.
            toolbar_category: toolbar category. Defaults to None.
             If toolbar_pos is not None, this specifies the category of the toolbar.
             If None, defaults to ActionCategory.VIEW_TOOLBAR if the current category
             is ActionCategory.VIEW, else to ActionCategory.PANEL_TOOLBAR.

        Returns:
            New action
        """
        if isinstance(select_condition, str):
            assert select_condition in SelectCond.__dict__
            select_condition = getattr(SelectCond, select_condition)

        action = create_action(
            parent=self.panel,
            title=title,
            triggered=triggered,
            toggled=toggled,
            shortcut=shortcut,
            icon=get_icon(icon_name) if icon_name else None,
            tip=tip,
            context=QC.Qt.WidgetWithChildrenShortcut,  # [1]
        )
        self.panel.addAction(action)  # [1]
        # [1] This is needed to make actions work with shortcuts for active panel,
        # because some of the shortcuts are using the same keybindings for both panels.
        # (Fixes #10)

        self.add_action(action, select_condition)
        self.add_to_action_list(action, None, position, separator)
        if context_menu_pos is not None:
            self.add_to_action_list(
                action, ActionCategory.CONTEXT_MENU, context_menu_pos, context_menu_sep
            )
        if toolbar_pos is not None:
            if toolbar_category is None:
                if self.__category_in_progress is ActionCategory.VIEW:
                    toolbar_category = ActionCategory.VIEW_TOOLBAR
                else:
                    toolbar_category = ActionCategory.PANEL_TOOLBAR
            self.add_to_action_list(action, toolbar_category, toolbar_pos, toolbar_sep)
        return action

    def add_to_action_list(
        self,
        action: QW.QAction,
        category: ActionCategory | None = None,
        pos: int | None = None,
        sep: bool = False,
    ) -> None:
        """Add action to list of actions.

        Args:
            action: action to add
            category: action category. Defaults to None.
             If None, action is added to the current category.
            pos: add action to menu at this position. Defaults to None.
             If None, action is added at the end of the list.
            sep: add separator before action in menu
             (or after if pos is positive). Defaults to False.
        """
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
        self, action: QW.QAction, select_condition: Callable | None = None
    ) -> None:
        """Add action to list of actions.

        Args:
            action: action to add
            select_condition: condition to enable action. Defaults to None.
             If None, action is enabled if at least one object is selected.
        """
        if select_condition is None:
            select_condition = SelectCond.at_least_one
        self.__actions.setdefault(select_condition, []).append(action)

    def selected_objects_changed(
        self,
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> None:
        """Update actions based on selected objects.

        Args:
            selected_groups: selected groups
            selected_objects: selected objects
        """
        for cond, actlist in self.__actions.items():
            if cond is not None:
                for act in actlist:
                    act.setEnabled(cond(selected_groups, selected_objects))

    def create_all_actions(self):
        """Create all actions"""
        self.create_first_actions()
        self.create_last_actions()
        add_actions(
            self.panel_toolbar, self.feature_actions.pop(ActionCategory.PANEL_TOOLBAR)
        )
        # For the view toolbar, we add the actions to the beginning of the toolbar:
        before = self.view_toolbar.actions()[0]
        for action in self.feature_actions.pop(ActionCategory.VIEW_TOOLBAR):
            if action is None:
                self.view_toolbar.insertSeparator(before)
            else:
                self.view_toolbar.insertAction(before, action)
        self.view_toolbar.insertSeparator(before)

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        with self.new_category(ActionCategory.FILE):
            self.new_action(
                _("New %s...") % self.OBJECT_STR,
                icon_name=f"new_{self.OBJECT_STR}.svg",
                tip=_("Create new %s") % self.OBJECT_STR,
                triggered=self.panel.new_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.New),
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Open %s...") % self.OBJECT_STR,
                # icon: fileopen_signal.svg or fileopen_image.svg
                icon_name=f"fileopen_{self.__class__.__name__[:3].lower()}.svg",
                tip=_("Open %s") % self.OBJECT_STR,
                triggered=self.panel.load_from_files,
                shortcut=QG.QKeySequence(QG.QKeySequence.Open),
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Save %s...") % self.OBJECT_STR,
                # icon: filesave_signal.svg or filesave_image.svg
                icon_name=f"filesave_{self.__class__.__name__[:3].lower()}.svg",
                tip=_("Save selected %s") % self.OBJECT_STR,
                triggered=self.panel.save_to_files,
                shortcut=QG.QKeySequence(QG.QKeySequence.Save),
                select_condition=SelectCond.at_least_one,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Import text file..."),
                icon_name="import_text.svg",
                triggered=self.panel.exec_import_wizard,
                select_condition=SelectCond.always,
            )

        with self.new_category(ActionCategory.EDIT):
            self.new_action(
                _("New group..."),
                icon_name="new_group.svg",
                tip=_("Create a new group"),
                triggered=self.panel.new_group,
                select_condition=SelectCond.always,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Rename group..."),
                icon_name="rename_group.svg",
                tip=_("Rename selected group"),
                triggered=self.panel.rename_group,
                select_condition=SelectCond.exactly_one_group,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Move up"),
                icon_name="move_up.svg",
                tip=_("Move up selection (groups or objects)"),
                triggered=self.panel.objview.move_up,
                select_condition=SelectCond.at_least_one_group_or_one_object,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Move down"),
                icon_name="move_down.svg",
                tip=_("Move down selection (groups or objects)"),
                triggered=self.panel.objview.move_down,
                select_condition=SelectCond.at_least_one_group_or_one_object,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Duplicate"),
                icon_name="duplicate.svg",
                tip=_("Duplicate selected %s") % self.OBJECT_STR,
                separator=True,
                triggered=self.panel.duplicate_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.Copy),
                select_condition=SelectCond.at_least_one_group_or_one_object,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Remove"),
                icon_name="delete.svg",
                tip=_("Remove selected %s") % self.OBJECT_STR,
                triggered=self.panel.remove_object,
                shortcut=QG.QKeySequence(QG.QKeySequence.Delete),
                select_condition=SelectCond.at_least_one_group_or_one_object,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Delete all"),
                select_condition=SelectCond.always,
                shortcut="Shift+Ctrl+Suppr",
                tip=_("Delete all groups and objects"),
                icon_name="delete_all.svg",
                triggered=self.panel.delete_all_objects,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Copy metadata"),
                separator=True,
                icon_name="metadata_copy.svg",
                tip=_("Copy metadata from selected %s") % self.OBJECT_STR,
                triggered=self.panel.copy_metadata,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Paste metadata"),
                icon_name="metadata_paste.svg",
                tip=_("Paste metadata into selected %s") % self.OBJECT_STR,
                triggered=self.panel.paste_metadata,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Import metadata") + "...",
                icon_name="metadata_import.svg",
                tip=_("Import metadata into %s") % self.OBJECT_STR,
                triggered=self.panel.import_metadata_from_file,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Export metadata") + "...",
                icon_name="metadata_export.svg",
                tip=_("Export selected %s metadata") % self.OBJECT_STR,
                triggered=self.panel.export_metadata_from_file,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Delete object metadata"),
                icon_name="metadata_delete.svg",
                tip=_("Delete all that is contained in object metadata"),
                triggered=self.panel.delete_metadata,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Add object title to plot"),
                separator=True,
                triggered=self.panel.add_label_with_title,
                tip=_("Add object title as a label to the plot"),
            )
            self.new_action(
                _("Copy titles to clipboard"),
                icon_name="copy_titles.svg",
                tip=_("Copy titles of selected objects to clipboard"),
                triggered=self.panel.copy_titles_to_clipboard,
            )
            self.new_action(
                _("Edit regions of interest..."),
                separator=True,
                triggered=self.panel.processor.edit_regions_of_interest,
                icon_name="roi.svg",
                context_menu_pos=-1,
                context_menu_sep=True,
                toolbar_pos=-1,
                toolbar_category=ActionCategory.VIEW_TOOLBAR,
            )
            self.new_action(
                _("Remove regions of interest"),
                triggered=self.panel.processor.delete_regions_of_interest,
                icon_name="roi_delete.svg",
                select_condition=SelectCond.with_roi,
                context_menu_pos=-1,
            )

        with self.new_category(ActionCategory.VIEW):
            self.new_action(
                _("View in a new window") + "...",
                icon_name="new_window.svg",
                tip=_("View selected %s in a new window") % self.OBJECT_STR,
                triggered=self.panel.open_separate_view,
                context_menu_pos=0,
                context_menu_sep=True,
                toolbar_pos=0,
            )
            self.new_action(
                _("Edit annotations") + "...",
                icon_name="annotations.svg",
                tip=_("Edit annotations of selected %s") % self.OBJECT_STR,
                triggered=lambda: self.panel.open_separate_view(edit_annotations=True),
                context_menu_pos=1,
                toolbar_pos=-1,
            )
            main = self.panel.mainwindow
            for cat in (ActionCategory.VIEW, ActionCategory.VIEW_TOOLBAR):
                for act in (main.autorefresh_action, main.showfirstonly_action):
                    self.add_to_action_list(act, cat, -1)
            self.new_action(
                _("Refresh manually"),
                icon_name="refresh-manual.svg",
                tip=_("Refresh plot, even if auto-refresh is enabled"),
                shortcut=QG.QKeySequence(QG.QKeySequence.Refresh),
                triggered=self.panel.manual_refresh,
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            for cat in (ActionCategory.VIEW, ActionCategory.VIEW_TOOLBAR):
                self.add_to_action_list(main.showlabel_action, cat, -1)

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("Sum"),
                triggered=self.panel.processor.compute_sum,
                select_condition=SelectCond.at_least_two,
                icon_name="sum.svg",
            )
            self.new_action(
                _("Average"),
                triggered=self.panel.processor.compute_average,
                select_condition=SelectCond.at_least_two,
                icon_name="average.svg",
            )
            self.new_action(
                _("Difference"),
                triggered=self.panel.processor.compute_difference,
                select_condition=SelectCond.at_least_one,
                icon_name="difference.svg",
            )
            self.new_action(
                _("Quadratic difference"),
                triggered=self.panel.processor.compute_quadratic_difference,
                select_condition=SelectCond.at_least_one,
                icon_name="quadratic_difference.svg",
            )
            self.new_action(
                _("Product"),
                triggered=self.panel.processor.compute_product,
                select_condition=SelectCond.at_least_two,
                icon_name="product.svg",
            )
            self.new_action(
                _("Division"),
                triggered=self.panel.processor.compute_division,
                select_condition=SelectCond.at_least_one,
                icon_name="division.svg",
            )
            self.new_action(
                _("Arithmetic operation") + "...",
                triggered=self.panel.processor.compute_arithmetic,
                select_condition=SelectCond.at_least_one,
                icon_name="arithmetic.svg",
            )
            with self.new_menu(_("Constant Operations"), icon_name="constant.svg"):
                self.new_action(
                    _("Add constant"),
                    triggered=self.panel.processor.compute_addition_constant,
                    select_condition=SelectCond.at_least_one,
                    icon_name="constant_add.svg",
                )
                self.new_action(
                    _("Substract constant"),
                    triggered=self.panel.processor.compute_difference_constant,
                    select_condition=SelectCond.at_least_one,
                    icon_name="constant_substract.svg",
                )
                self.new_action(
                    _("Multiply by constant"),
                    triggered=self.panel.processor.compute_product_constant,
                    select_condition=SelectCond.at_least_one,
                    icon_name="constant_multiply.svg",
                )
                self.new_action(
                    _("Divide by constant"),
                    triggered=self.panel.processor.compute_division_constant,
                    select_condition=SelectCond.at_least_one,
                    icon_name="constant_divide.svg",
                )
            self.new_action(
                _("Absolute value"),
                triggered=self.panel.processor.compute_abs,
                separator=True,
                icon_name="abs.svg",
            )
            self.new_action(
                _("Real part"),
                triggered=self.panel.processor.compute_re,
                icon_name="re.svg",
            )
            self.new_action(
                _("Imaginary part"),
                triggered=self.panel.processor.compute_im,
                icon_name="im.svg",
            )
            self.new_action(
                _("Convert data type"),
                triggered=self.panel.processor.compute_astype,
                separator=True,
                icon_name="convert_dtype.svg",
            )
            self.new_action(
                _("Exponential"),
                triggered=self.panel.processor.compute_exp,
                separator=True,
                icon_name="exp.svg",
            )
            self.new_action(
                _("Logarithm (base 10)"),
                triggered=self.panel.processor.compute_log10,
                separator=False,
                icon_name="log10.svg",
            )

        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(
                _("Axis transformation"), icon_name="axis_transform.svg"
            ):
                self.new_action(
                    _("Linear calibration"),
                    triggered=self.panel.processor.compute_calibration,
                )
                self.new_action(
                    _("Swap X/Y axes"),
                    triggered=self.panel.processor.compute_swap_axes,
                    icon_name="swap_x_y.svg",
                )
            with self.new_menu(_("Level adjustment"), icon_name="level_adjustment.svg"):
                self.new_action(
                    _("Normalize"),
                    triggered=self.panel.processor.compute_normalize,
                    icon_name="normalize.svg",
                )
                self.new_action(
                    _("Clipping"),
                    triggered=self.panel.processor.compute_clip,
                    icon_name="clip.svg",
                )
                self.new_action(
                    _("Offset correction"),
                    triggered=self.panel.processor.compute_offset_correction,
                    icon_name="offset_correction.svg",
                    tip=_("Evaluate and subtract the offset value from the data"),
                )
            with self.new_menu(_("Noise reduction"), icon_name="noise_reduction.svg"):
                self.new_action(
                    _("Gaussian filter"),
                    triggered=self.panel.processor.compute_gaussian_filter,
                )
                self.new_action(
                    _("Moving average"),
                    triggered=self.panel.processor.compute_moving_average,
                )
                self.new_action(
                    _("Moving median"),
                    triggered=self.panel.processor.compute_moving_median,
                )
                self.new_action(
                    _("Wiener filter"),
                    triggered=self.panel.processor.compute_wiener,
                )
            with self.new_menu(_("Fourier analysis"), icon_name="fourier.svg"):
                self.new_action(
                    _("FFT"),
                    triggered=self.panel.processor.compute_fft,
                    tip=_("Warning: only real part is plotted"),
                )
                self.new_action(
                    _("Inverse FFT"),
                    triggered=self.panel.processor.compute_ifft,
                    tip=_("Warning: only real part is plotted"),
                )
                self.new_action(
                    _("Magnitude spectrum"),
                    triggered=self.panel.processor.compute_magnitude_spectrum,
                )
                self.new_action(
                    _("Phase spectrum"),
                    triggered=self.panel.processor.compute_phase_spectrum,
                )
                self.new_action(
                    _("Power spectral density"),
                    triggered=self.panel.processor.compute_psd,
                )

        with self.new_category(ActionCategory.ANALYSIS):
            self.new_action(
                _("Statistics") + "...",
                triggered=self.panel.processor.compute_stats,
                icon_name="stats.svg",
                context_menu_pos=-1,
                context_menu_sep=True,
            )
            self.new_action(
                _("Histogram") + "...",
                triggered=self.panel.processor.compute_histogram,
                icon_name="histogram.svg",
                context_menu_pos=-1,
            )

    def create_last_actions(self):
        """Create actions that are added to the menus in the end"""
        with self.new_category(ActionCategory.PROCESSING):
            self.new_action(
                _("ROI extraction"),
                triggered=self.panel.processor.compute_roi_extraction,
                # Icon name is 'signal_roi.svg' or 'image_roi.svg':
                icon_name=f"{self.OBJECT_STR}_roi.svg",
                separator=True,
            )

        with self.new_category(ActionCategory.ANALYSIS):
            self.new_action(
                _("Show results") + "...",
                triggered=self.panel.show_results,
                icon_name="show_results.svg",
                separator=True,
                select_condition=SelectCond.at_least_one_group_or_one_object,
            )
            self.new_action(
                _("Plot results") + "...",
                triggered=self.panel.plot_results,
                icon_name="plot_results.svg",
                select_condition=SelectCond.at_least_one_group_or_one_object,
            )
            self.new_action(
                _("Delete results") + "...",
                triggered=self.panel.delete_results,
                icon_name="delete_results.svg",
                select_condition=SelectCond.at_least_one_group_or_one_object,
            )


class SignalActionHandler(BaseActionHandler):
    """Object handling signal panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("signal")

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        super().create_first_actions()

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("Power"),
                triggered=self.panel.processor.compute_power,
                separator=True,
                icon_name="power.svg",
            )
            self.new_action(
                _("Square root"),
                triggered=self.panel.processor.compute_sqrt,
                separator=False,
                icon_name="sqrt.svg",
            )
            self.new_action(
                _("Derivative"),
                triggered=self.panel.processor.compute_derivative,
                separator=True,
                icon_name="derivative.svg",
            )
            self.new_action(
                _("Integral"),
                triggered=self.panel.processor.compute_integral,
                icon_name="integral.svg",
            )

        def cra_fit(title, fitdlgfunc, iconname, tip: str | None = None):
            """Create curve fitting action"""
            return self.new_action(
                title,
                triggered=lambda: self.panel.processor.compute_fit(title, fitdlgfunc),
                icon_name=iconname,
                tip=tip,
            )

        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Axis transformation")):
                self.new_action(
                    _("Reverse X-axis"),
                    triggered=self.panel.processor.compute_reverse_x,
                    icon_name="reverse_signal_x.svg",
                )
                self.new_action(
                    _("Convert to Cartesian coordinates"),
                    triggered=self.panel.processor.compute_polar2cartesian,
                )
                self.new_action(
                    _("Convert to polar coordinates"),
                    triggered=self.panel.processor.compute_cartesian2polar,
                )

            with self.new_menu(_("Frequency filters"), icon_name="highpass.svg"):
                self.new_action(
                    _("Low-pass filter"),
                    triggered=self.panel.processor.compute_lowpass,
                    icon_name="lowpass.svg",
                )
                self.new_action(
                    _("High-pass filter"),
                    triggered=self.panel.processor.compute_highpass,
                    icon_name="highpass.svg",
                )
                self.new_action(
                    _("Band-pass filter"),
                    triggered=self.panel.processor.compute_bandpass,
                    icon_name="bandpass.svg",
                )
                self.new_action(
                    _("Band-stop filter"),
                    triggered=self.panel.processor.compute_bandstop,
                    icon_name="bandstop.svg",
                )
            with self.new_menu(_("Fitting"), icon_name="expfit.svg"):
                cra_fit(_("Linear fit"), fitdialog.linearfit, "linearfit.svg")
                self.new_action(
                    _("Polynomial fit"),
                    triggered=self.panel.processor.compute_polyfit,
                    icon_name="polyfit.svg",
                )
                cra_fit(_("Gaussian fit"), fitdialog.gaussianfit, "gaussfit.svg")
                cra_fit(_("Lorentzian fit"), fitdialog.lorentzianfit, "lorentzfit.svg")
                cra_fit(_("Voigt fit"), fitdialog.voigtfit, "voigtfit.svg")
                self.new_action(
                    _("Multi-Gaussian fit"),
                    triggered=self.panel.processor.compute_multigaussianfit,
                    icon_name="multigaussfit.svg",
                )
                cra_fit(_("Exponential fit"), fitdialog.exponentialfit, "expfit.svg")
                cra_fit(_("Sinusoidal fit"), fitdialog.sinusoidalfit, "sinfit.svg")
                cra_fit(
                    _("CDF fit"),
                    fitdialog.cdffit,
                    "cdffit.svg",
                    tip=_(
                        "Cumulative distribution function fit, "
                        "related to Error function (erf)"
                    ),
                )
            self.new_action(
                _("Windowing"),
                triggered=self.panel.processor.compute_windowing,
                icon_name="windowing.svg",
                tip=_(
                    "Apply a window function (or apodization): Hanning, Hamming, ..."
                ),
            )
            self.new_action(
                _("Detrending"),
                triggered=self.panel.processor.compute_detrending,
                icon_name="detrending.svg",
            )
            self.new_action(
                _("Interpolation"),
                triggered=self.panel.processor.compute_interpolation,
                icon_name="interpolation.svg",
            )
            self.new_action(
                _("Resampling"),
                triggered=self.panel.processor.compute_resampling,
                icon_name="resampling.svg",
            )
            with self.new_menu(_("Stability analysis"), icon_name="stability.svg"):
                self.new_action(
                    _("Allan variance"),
                    triggered=self.panel.processor.compute_allan_variance,
                )
                self.new_action(
                    _("Allan deviation"),
                    triggered=self.panel.processor.compute_allan_deviation,
                )
                self.new_action(
                    _("Modified Allan deviation"),
                    triggered=self.panel.processor.compute_modified_allan_variance,
                )
                self.new_action(
                    _("Hadamard variance"),
                    triggered=self.panel.processor.compute_hadamard_variance,
                )
                self.new_action(
                    _("Total variance"),
                    triggered=self.panel.processor.compute_total_variance,
                )
                self.new_action(
                    _("Time deviation"),
                    triggered=self.panel.processor.compute_time_deviation,
                )
                self.new_action(
                    _("All stability features") + "...",
                    triggered=self.panel.processor.compute_all_stability,
                    separator=True,
                    tip=_("Compute all stability features"),
                )

        with self.new_category(ActionCategory.ANALYSIS):
            self.new_action(
                _("Full width at half-maximum"),
                triggered=self.panel.processor.compute_fwhm,
                separator=True,
                tip=_("Compute Full Width at Half-Maximum (FWHM)"),
                icon_name="fwhm.svg",
            )
            self.new_action(
                _("Full width at") + " 1/e²",
                triggered=self.panel.processor.compute_fw1e2,
                tip=_("Compute Full Width at Maximum") + "/e²",
                icon_name="fw1e2.svg",
            )
            self.new_action(
                _("Abscissa of the minimum and maximum"),
                triggered=self.panel.processor.compute_x_at_minmax,
                tip=_(
                    "Compute the smallest argument of the minima and the smallest "
                    "argument of the maxima"
                ),
            )
            self.new_action(
                _("Abscissa at y=..."),
                triggered=self.panel.processor.compute_x_at_y,
            )
            self.new_action(
                _("Peak detection"),
                separator=True,
                triggered=self.panel.processor.compute_peak_detection,
                icon_name="peak_detect.svg",
            )
            self.new_action(
                _("Sampling rate and period") + "...",
                separator=True,
                triggered=self.panel.processor.compute_sampling_rate_period,
                tip=_(
                    "Compute sampling rate and period for a constant sampling signal"
                ),
            )
            self.new_action(
                _("Dynamic parameters") + "...",
                triggered=self.panel.processor.compute_dynamic_parameters,
                context_menu_pos=-1,
                tip=_("Compute dynamic parameters: ENOB, SNR, SINAD, THD, ..."),
            )
            self.new_action(
                _("Bandwidth at -3dB") + "...",
                triggered=self.panel.processor.compute_bandwidth_3db,
                context_menu_pos=-1,
                tip=_(
                    "Compute bandwidth at -3dB assuming a low-pass filter "
                    "already expressed in dB"
                ),
            )
            self.new_action(
                _("Contrast"),
                triggered=self.panel.processor.compute_contrast,
                tip=_(
                    "Compute contrast of a signal, i.e. (max-min)/(max+min), "
                    "e.g. for an image profile"
                ),
            )

        with self.new_category(ActionCategory.VIEW):
            antialiasing_action = self.new_action(
                _("Curve anti-aliasing"),
                icon_name="curve_antialiasing.svg",
                toggled=self.panel.toggle_anti_aliasing,
                tip=_("Toggle curve anti-aliasing on/off (may slow down plotting)"),
                toolbar_pos=-1,
            )
            antialiasing_action.setChecked(Conf.view.sig_antialiasing.get(True))
            self.new_action(
                _("Reset curve styles"),
                select_condition=SelectCond.always,
                icon_name="reset_curve_styles.svg",
                triggered=self.panel.reset_curve_styles,
                tip=_(
                    "Curve styles are looped over a list of predefined styles.\n"
                    "This action resets the list to its initial state."
                ),
                toolbar_pos=-1,
            )

    def create_last_actions(self):
        """Create actions that are added to the menus in the end"""
        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                _("Convolution"),
                triggered=self.panel.processor.compute_convolution,
                separator=True,
                icon_name="convolution.svg",
            )
        super().create_last_actions()


class ImageActionHandler(BaseActionHandler):
    """Object handling image panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("image")

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        super().create_first_actions()

        with self.new_category(ActionCategory.VIEW):
            showcontrast_action = self.new_action(
                _("Show contrast panel"),
                icon_name="contrast.png",
                tip=_("Show or hide contrast adjustment panel"),
                select_condition=SelectCond.always,
                toggled=self.panel.toggle_show_contrast,
                toolbar_pos=-1,
            )
            showcontrast_action.setChecked(Conf.view.show_contrast.get(True))

        with self.new_category(ActionCategory.OPERATION):
            self.new_action(
                "Log10(z+n)",
                triggered=self.panel.processor.compute_logp1,
            )
            self.new_action(
                _("Flat-field correction"),
                separator=True,
                triggered=self.panel.processor.compute_flatfield,
                select_condition=SelectCond.at_least_one,
            )

            with self.new_menu(_("Flip or rotation"), icon_name="rotate_right.svg"):
                self.new_action(
                    _("Flip horizontally"),
                    triggered=self.panel.processor.compute_fliph,
                    icon_name="flip_horizontally.svg",
                    context_menu_pos=-1,
                    context_menu_sep=True,
                )
                self.new_action(
                    _("Flip diagonally"),
                    triggered=self.panel.processor.compute_swap_axes,
                    icon_name="swap_x_y.svg",
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Flip vertically"),
                    triggered=self.panel.processor.compute_flipv,
                    icon_name="flip_vertically.svg",
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate %s right") % "90°",  # pylint: disable=consider-using-f-string
                    separator=True,
                    triggered=self.panel.processor.compute_rotate270,
                    icon_name="rotate_right.svg",
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate %s left") % "90°",  # pylint: disable=consider-using-f-string
                    triggered=self.panel.processor.compute_rotate90,
                    icon_name="rotate_left.svg",
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Rotate by..."),
                    triggered=self.panel.processor.compute_rotate,
                )

            with self.new_menu(_("Intensity profiles"), icon_name="profile.svg"):
                self.new_action(
                    _("Line profile..."),
                    triggered=self.panel.processor.compute_line_profile,
                    icon_name="profile.svg",
                    tip=_("Extract horizontal or vertical profile"),
                    context_menu_pos=-1,
                    context_menu_sep=True,
                )
                self.new_action(
                    _("Segment profile..."),
                    triggered=self.panel.processor.compute_segment_profile,
                    icon_name="profile_segment.svg",
                    tip=_("Extract profile along a segment"),
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Average profile..."),
                    triggered=self.panel.processor.compute_average_profile,
                    icon_name="profile_average.svg",
                    tip=_("Extract average horizontal or vertical profile"),
                    context_menu_pos=-1,
                )
                self.new_action(
                    _("Radial profile extraction..."),
                    triggered=self.panel.processor.compute_radial_profile,
                    icon_name="profile_radial.svg",
                    tip=_("Radial profile extraction around image centroid"),
                )

            self.new_action(
                _("Distribute on a grid..."),
                triggered=self.panel.processor.distribute_on_grid,
                icon_name="distribute_on_grid.svg",
                select_condition=SelectCond.at_least_two,
            )
            self.new_action(
                _("Reset image positions"),
                triggered=self.panel.processor.reset_positions,
                icon_name="reset_positions.svg",
                select_condition=SelectCond.at_least_two,
            )

        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Thresholding"), icon_name="thresholding.svg"):
                self.new_action(
                    _("Parametric thresholding"),
                    triggered=self.panel.processor.compute_threshold,
                )
                self.new_action(
                    _("ISODATA thresholding"),
                    triggered=self.panel.processor.compute_threshold_isodata,
                )
                self.new_action(
                    _("Li thresholding"),
                    triggered=self.panel.processor.compute_threshold_li,
                )
                self.new_action(
                    _("Mean thresholding"),
                    triggered=self.panel.processor.compute_threshold_mean,
                )
                self.new_action(
                    _("Minimum thresholding"),
                    triggered=self.panel.processor.compute_threshold_minimum,
                )
                self.new_action(
                    _("Otsu thresholding"),
                    triggered=self.panel.processor.compute_threshold_otsu,
                )
                self.new_action(
                    _("Triangle thresholding"),
                    triggered=self.panel.processor.compute_threshold_triangle,
                )
                self.new_action(
                    _("Yen thresholding"),
                    triggered=self.panel.processor.compute_threshold_yen,
                )
                self.new_action(
                    _("All thresholding methods") + "...",
                    triggered=self.panel.processor.compute_all_threshold,
                    separator=True,
                    tip=_("Apply all thresholding methods"),
                )
            with self.new_menu(_("Exposure"), icon_name="exposure.svg"):
                self.new_action(
                    _("Gamma correction"),
                    triggered=self.panel.processor.compute_adjust_gamma,
                )
                self.new_action(
                    _("Logarithmic correction"),
                    triggered=self.panel.processor.compute_adjust_log,
                )
                self.new_action(
                    _("Sigmoid correction"),
                    triggered=self.panel.processor.compute_adjust_sigmoid,
                )
                self.new_action(
                    _("Histogram equalization"),
                    triggered=self.panel.processor.compute_equalize_hist,
                )
                self.new_action(
                    _("Adaptive histogram equalization"),
                    triggered=self.panel.processor.compute_equalize_adapthist,
                )
                self.new_action(
                    _("Intensity rescaling"),
                    triggered=self.panel.processor.compute_rescale_intensity,
                )
            with self.new_menu(_("Restoration"), icon_name="noise_reduction.svg"):
                self.new_action(
                    _("Total variation denoising"),
                    triggered=self.panel.processor.compute_denoise_tv,
                )
                self.new_action(
                    _("Bilateral filter denoising"),
                    triggered=self.panel.processor.compute_denoise_bilateral,
                )
                self.new_action(
                    _("Wavelet denoising"),
                    triggered=self.panel.processor.compute_denoise_wavelet,
                )
                self.new_action(
                    _("White Top-Hat denoising"),
                    triggered=self.panel.processor.compute_denoise_tophat,
                )
                self.new_action(
                    _("All denoising methods") + "...",
                    triggered=self.panel.processor.compute_all_denoise,
                    separator=True,
                    tip=_("Apply all denoising methods"),
                )
            with self.new_menu(_("Morphology"), icon_name="morphology.svg"):
                self.new_action(
                    _("White Top-Hat (disk)"),
                    triggered=self.panel.processor.compute_white_tophat,
                )
                self.new_action(
                    _("Black Top-Hat (disk)"),
                    triggered=self.panel.processor.compute_black_tophat,
                )
                self.new_action(
                    _("Erosion (disk)"),
                    triggered=self.panel.processor.compute_erosion,
                )
                self.new_action(
                    _("Dilation (disk)"),
                    triggered=self.panel.processor.compute_dilation,
                )
                self.new_action(
                    _("Opening (disk)"),
                    triggered=self.panel.processor.compute_opening,
                )
                self.new_action(
                    _("Closing (disk)"),
                    triggered=self.panel.processor.compute_closing,
                )
                self.new_action(
                    _("All morphological operations") + "...",
                    triggered=self.panel.processor.compute_all_morphology,
                    separator=True,
                    tip=_("Apply all morphological operations"),
                )
            with self.new_menu(_("Edges"), icon_name="edges.svg"):
                self.new_action(
                    _("Roberts filter"), triggered=self.panel.processor.compute_roberts
                )
                self.new_action(
                    _("Prewitt filter"),
                    triggered=self.panel.processor.compute_prewitt,
                    separator=True,
                )
                self.new_action(
                    _("Prewitt filter (horizontal)"),
                    triggered=self.panel.processor.compute_prewitt_h,
                )
                self.new_action(
                    _("Prewitt filter (vertical)"),
                    triggered=self.panel.processor.compute_prewitt_v,
                )
                self.new_action(
                    _("Sobel filter"),
                    triggered=self.panel.processor.compute_sobel,
                    separator=True,
                )
                self.new_action(
                    _("Sobel filter (horizontal)"),
                    triggered=self.panel.processor.compute_sobel_h,
                )
                self.new_action(
                    _("Sobel filter (vertical)"),
                    triggered=self.panel.processor.compute_sobel_v,
                )
                self.new_action(
                    _("Scharr filter"),
                    triggered=self.panel.processor.compute_scharr,
                    separator=True,
                )
                self.new_action(
                    _("Scharr filter (horizontal)"),
                    triggered=self.panel.processor.compute_scharr_h,
                )
                self.new_action(
                    _("Scharr filter (vertical)"),
                    triggered=self.panel.processor.compute_scharr_v,
                )
                self.new_action(
                    _("Farid filter"),
                    triggered=self.panel.processor.compute_farid,
                    separator=True,
                )
                self.new_action(
                    _("Farid filter (horizontal)"),
                    triggered=self.panel.processor.compute_farid_h,
                )
                self.new_action(
                    _("Farid filter (vertical)"),
                    triggered=self.panel.processor.compute_farid_v,
                )
                self.new_action(
                    _("Laplace filter"),
                    triggered=self.panel.processor.compute_laplace,
                    separator=True,
                )
                self.new_action(
                    _("All edges filters") + "...",
                    triggered=self.panel.processor.compute_all_edges,
                    separator=True,
                    tip=_("Compute all edges filters"),
                )
                self.new_action(
                    _("Canny filter"), triggered=self.panel.processor.compute_canny
                )
            self.new_action(
                _("Butterworth filter"),
                triggered=self.panel.processor.compute_butterworth,
            )

        with self.new_category(ActionCategory.ANALYSIS):
            # TODO: [P3] Add "Create ROI grid..." action to create a regular grid
            # or ROIs (maybe reuse/derive from `core.gui.processor.image.GridParam`)
            self.new_action(
                _("Centroid"),
                separator=True,
                triggered=self.panel.processor.compute_centroid,
                tip=_("Compute image centroid"),
            )
            self.new_action(
                _("Minimum enclosing circle center"),
                triggered=self.panel.processor.compute_enclosing_circle,
                tip=_("Compute smallest enclosing circle center"),
            )
            self.new_action(
                _("2D peak detection"),
                separator=True,
                triggered=self.panel.processor.compute_peak_detection,
                tip=_("Compute automatic 2D peak detection"),
            )
            self.new_action(
                _("Contour detection"),
                triggered=self.panel.processor.compute_contour_shape,
                tip=_("Compute contour shape fit"),
            )
            self.new_action(
                _("Circle Hough transform"),
                triggered=self.panel.processor.compute_hough_circle_peaks,
                tip=_("Detect circular shapes using circle Hough transform"),
            )

            with self.new_menu(_("Blob detection")):
                self.new_action(
                    _("Blob detection (DOG)"),
                    triggered=self.panel.processor.compute_blob_dog,
                    tip=_("Detect blobs using Difference of Gaussian (DOG) method"),
                )
                self.new_action(
                    _("Blob detection (DOH)"),
                    triggered=self.panel.processor.compute_blob_doh,
                    tip=_("Detect blobs using Determinant of Hessian (DOH) method"),
                )
                self.new_action(
                    _("Blob detection (LOG)"),
                    triggered=self.panel.processor.compute_blob_log,
                    tip=_("Detect blobs using Laplacian of Gaussian (LOG) method"),
                )
                self.new_action(
                    _("Blob detection (OpenCV)"),
                    triggered=self.panel.processor.compute_blob_opencv,
                    tip=_("Detect blobs using OpenCV SimpleBlobDetector"),
                )

    def create_last_actions(self):
        """Create actions that are added to the menus in the end"""
        with self.new_category(ActionCategory.PROCESSING):
            self.new_action(
                _("Resize"),
                triggered=self.panel.processor.compute_resize,
                icon_name="resize.svg",
                separator=True,
            )
            self.new_action(
                _("Pixel binning"),
                triggered=self.panel.processor.compute_binning,
                icon_name="binning.svg",
            )
        super().create_last_actions()
