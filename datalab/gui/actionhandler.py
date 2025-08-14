# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Action handler
==============

The :mod:`datalab.gui.actionhandler` module handles all application actions
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

import sigima.objects as sio
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.gui import newobject
from datalab.widgets import fitdialog

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    from datalab.gui.panel.image import ImagePanel
    from datalab.gui.panel.signal import SignalPanel
    from datalab.objectmodel import ObjectGroup


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
    def exactly_one_group_or_one_object(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """Exactly one group or one signal or image is selected"""
        return len(selected_groups) == 1 or len(selected_objects) == 1

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

    @staticmethod
    # pylint: disable=unused-argument
    def exactly_one_with_roi(
        selected_groups: list[ObjectGroup],
        selected_objects: list[SignalObj | ImageObj],
    ) -> bool:
        """Exactly one signal or image has a ROI"""
        return (
            len(selected_groups) == 0
            and len(selected_objects) == 1
            and selected_objects[0].roi is not None
        )


class ActionCategory(enum.Enum):
    """Action categories"""

    FILE = enum.auto()
    EDIT = enum.auto()
    VIEW = enum.auto()
    ROI = enum.auto()
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

    # pylint: disable=too-many-positional-arguments
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

    def action_for(
        self,
        function_or_name: Callable | str,
        position: int | None = None,
        separator: bool = False,
        context_menu_pos: int | None = None,
        context_menu_sep: bool = False,
        toolbar_pos: int | None = None,
        toolbar_sep: bool = False,
        toolbar_category: ActionCategory | None = None,
    ) -> QW.QAction:
        """Create action for a feature.

        Args:
            function_or_name: function or name of the feature
            position: add action to menu at this position. Defaults to None.
            separator: add separator before action in menu
            context_menu_pos: add action to context menu at this position.
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
        feature = self.panel.processor.get_feature(function_or_name)
        if feature.pattern == "n_to_1":
            condition = SelectCond.at_least_two
        else:
            condition = SelectCond.at_least_one
        return self.new_action(
            feature.action_title,
            position=position,
            separator=separator,
            triggered=lambda: self.panel.processor.run_feature(feature.function),
            select_condition=condition,
            icon_name=feature.icon_name,
            tip=feature.comment,
            context_menu_pos=context_menu_pos,
            context_menu_sep=context_menu_sep,
            toolbar_pos=toolbar_pos,
            toolbar_sep=toolbar_sep,
            toolbar_category=toolbar_category,
        )

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

    @abc.abstractmethod
    def create_new_object_actions(self):
        """Create actions for creating new objects"""

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        with self.new_category(ActionCategory.FILE):
            with self.new_menu(
                _("New %s") % self.OBJECT_STR, icon_name=f"new_{self.OBJECT_STR}.svg"
            ):
                self.create_new_object_actions()
            self.new_action(
                _("Open %s...") % self.OBJECT_STR,
                # icon: fileopen_signal.svg or fileopen_image.svg
                icon_name=f"fileopen_{self.__class__.__name__[:3].lower()}.svg",
                tip=_("Open one or more %s files") % self.OBJECT_STR,
                triggered=self.panel.load_from_files,
                shortcut=QG.QKeySequence(QG.QKeySequence.Open),
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Open from directory..."),
                icon_name="fileopen_directory.svg",
                tip=_("Open all %s files from directory") % self.OBJECT_STR,
                triggered=self.panel.load_from_directory,
                select_condition=SelectCond.always,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Save %s...") % self.OBJECT_STR,
                # icon: filesave_signal.svg or filesave_image.svg
                icon_name=f"filesave_{self.__class__.__name__[:3].lower()}.svg",
                tip=_("Save selected %ss") % self.OBJECT_STR,
                triggered=self.panel.save_to_files,
                shortcut=QG.QKeySequence(QG.QKeySequence.Save),
                select_condition=SelectCond.at_least_one,
                context_menu_pos=-1,
                toolbar_pos=-1,
            )
            # New bulk save action (multiple objects with pattern)
            self.new_action(
                _("Bulk save..."),
                icon_name=f"filesave_{self.__class__.__name__[:3].lower()}.svg",
                tip=_("Bulk save selected %s with naming pattern") % self.OBJECT_STR,
                triggered=self.panel.bulk_save_dialog,
                select_condition=SelectCond.at_least_one,
            )
            self.new_action(
                _("Import text file..."),
                icon_name="import_text.svg",
                triggered=self.panel.exec_import_wizard,
                select_condition=SelectCond.always,
            )

        with self.new_category(ActionCategory.EDIT):
            self.new_action(
                _("Rename"),
                icon_name="rename.svg",
                shortcut="F2",
                tip=_("Edit title of selected %s or group") % self.OBJECT_STR,
                triggered=self.panel.rename_selected_object_or_group,
                select_condition=SelectCond.exactly_one_group_or_one_object,
                context_menu_pos=-1,
            )
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

        with self.new_category(ActionCategory.ROI):
            self.new_action(
                _("Edit graphically") + "...",
                triggered=self.panel.processor.edit_roi_graphically,
                icon_name="roi.svg",
                context_menu_pos=-1,
                context_menu_sep=True,
                toolbar_pos=-1,
                toolbar_category=ActionCategory.VIEW_TOOLBAR,
                tip=_("Edit regions of interest graphically"),
            )
            self.new_action(
                _("Edit numerically") + "...",
                triggered=self.panel.processor.edit_roi_numerically,
                select_condition=SelectCond.exactly_one_with_roi,
                tip=_("Edit regions of interest numerically"),
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

        # MARK: OPERATION
        with self.new_category(ActionCategory.OPERATION):
            self.action_for("addition")
            self.action_for("average")
            self.action_for("standard_deviation")
            self.action_for("difference")
            self.action_for("quadratic_difference")
            self.action_for("product")
            self.action_for("division")
            self.action_for("inverse")
            self.action_for("arithmetic")
            with self.new_menu(_("Constant Operations"), icon_name="constant.svg"):
                self.action_for("addition_constant")
                self.action_for("difference_constant")
                self.action_for("product_constant")
                self.action_for("division_constant")
            self.action_for("absolute", separator=True)
            self.action_for("phase")
            self.action_for("complex_from_magnitude_phase")
            self.action_for("real", separator=True)
            self.action_for("imag")
            self.action_for("complex_from_real_imag")
            self.action_for("astype", separator=True)
            self.action_for("exp", separator=True)
            self.action_for("log10", separator=False)

        # MARK: PROCESSING
        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(
                _("Axis transformation"), icon_name="axis_transform.svg"
            ):
                self.action_for("calibration")
                self.action_for("transpose")
            with self.new_menu(_("Level adjustment"), icon_name="level_adjustment.svg"):
                self.action_for("normalize")
                self.action_for("clip")
                self.new_action(
                    _("Offset correction"),
                    triggered=self.panel.processor.compute_offset_correction,
                    icon_name="offset_correction.svg",
                    tip=_("Evaluate and subtract the offset value from the data"),
                )
            with self.new_menu(_("Noise addition"), icon_name="noise_addition.svg"):
                self.action_for("add_gaussian_noise")
                self.action_for("add_poisson_noise")
                self.action_for("add_uniform_noise")
            with self.new_menu(_("Noise reduction"), icon_name="noise_reduction.svg"):
                self.action_for("gaussian_filter")
                self.action_for("moving_average")
                self.action_for("moving_median")
                self.action_for("wiener")
            with self.new_menu(_("Fourier analysis"), icon_name="fourier.svg"):
                self.action_for("zero_padding")
                self.action_for("fft")
                self.action_for("ifft")
                self.action_for("magnitude_spectrum")
                self.action_for("phase_spectrum")
                self.action_for("psd")

        # MARK: ANALYSIS
        with self.new_category(ActionCategory.ANALYSIS):
            self.action_for("stats", context_menu_pos=-1, context_menu_sep=True)
            self.action_for("histogram", context_menu_pos=-1)

    def create_last_actions(self):
        """Create actions that are added to the menus in the end"""
        with self.new_category(ActionCategory.ROI):
            self.new_action(
                _("Extract") + "...",
                triggered=self.panel.processor.compute_roi_extraction,
                # Icon name is 'signal_roi.svg' or 'image_roi.svg':
                icon_name=f"{self.OBJECT_STR}_roi.svg",
                separator=True,
            )
            self.new_action(
                _("Copy"),
                separator=True,
                icon_name="roi_copy.svg",
                tip=_("Copy regions of interest from selected %s") % self.OBJECT_STR,
                triggered=self.panel.copy_roi,
                select_condition=SelectCond.exactly_one_with_roi,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Paste"),
                icon_name="roi_paste.svg",
                tip=_("Paste regions of interest into selected %s") % self.OBJECT_STR,
                triggered=self.panel.paste_roi,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Import") + "...",
                icon_name="roi_import.svg",
                tip=_("Import regions of interest into %s") % self.OBJECT_STR,
                triggered=self.panel.import_roi_from_file,
                select_condition=SelectCond.exactly_one,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Export") + "...",
                icon_name="roi_export.svg",
                tip=_("Export selected %s regions of interest") % self.OBJECT_STR,
                triggered=self.panel.export_roi_to_file,
                select_condition=SelectCond.exactly_one_with_roi,
                toolbar_pos=-1,
            )
            self.new_action(
                _("Remove all"),
                separator=True,
                triggered=self.panel.processor.delete_regions_of_interest,
                icon_name="roi_delete.svg",
                select_condition=SelectCond.with_roi,
                context_menu_pos=-1,
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

    def create_new_object_actions(self):
        """Create actions for creating new objects"""
        for label, pclass in (
            (_("Zeros"), sio.ZerosParam),
            (_("Normal distribution"), sio.NormalDistribution1DParam),
            (_("Poisson distribution"), sio.PoissonDistribution1DParam),
            (_("Uniform distribution"), sio.UniformDistribution1DParam),
            (_("Gaussian"), sio.GaussParam),
            (_("Lorentzian"), sio.LorentzParam),
            (_("Voigt"), sio.VoigtParam),
            (_("Blackbody (Planck's law)"), sio.PlanckParam),
            (_("Sinus"), sio.SinusParam),
            (_("Cosinus"), sio.CosinusParam),
            (_("Sawtooth"), sio.SawtoothParam),
            (_("Triangle"), sio.TriangleParam),
            (_("Square"), sio.SquareParam),
            (_("Cardinal sine"), sio.SincParam),
            (_("Linear chirp"), sio.LinearChirpParam),
            (_("Step"), sio.StepParam),
            (_("Exponential"), sio.ExponentialParam),
            (_("Logistic"), sio.LogisticParam),
            (_("Pulse"), sio.PulseParam),
            (_("Polynomial"), sio.PolyParam),
            (_("Custom"), newobject.CustomSignalParam),
        ):
            self.new_action(
                label,
                tip=_("Create new %s") % label,
                triggered=lambda pclass=pclass: self.panel.new_object(pclass()),
                select_condition=SelectCond.always,
            )

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        super().create_first_actions()

        # MARK: OPERATION
        with self.new_category(ActionCategory.OPERATION):
            self.action_for("power", separator=True)
            self.action_for("sqrt")
            self.action_for("derivative", separator=True)
            self.action_for("integral")

        def cra_fit(title, fitdlgfunc, iconname, tip: str | None = None):
            """Create curve fitting action"""
            return self.new_action(
                title,
                triggered=lambda: self.panel.processor.compute_fit(title, fitdlgfunc),
                icon_name=iconname,
                tip=tip,
            )

        # MARK: PROCESSING
        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Axis transformation")):
                self.action_for("reverse_x")
                self.action_for("to_cartesian")
                self.action_for("to_polar")
            with self.new_menu(_("Frequency filters"), icon_name="highpass.svg"):
                self.action_for("lowpass")
                self.action_for("highpass")
                self.action_for("bandpass")
                self.action_for("bandstop")
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
            self.action_for("apply_window")
            self.action_for("detrending")
            self.action_for("interpolate")
            self.action_for("resampling")
            with self.new_menu(_("Stability analysis"), icon_name="stability.svg"):
                self.action_for("allan_variance")
                self.action_for("allan_deviation")
                self.action_for("modified_allan_variance")
                self.action_for("hadamard_variance")
                self.action_for("total_variance")
                self.action_for("time_deviation")
                self.new_action(
                    _("All stability features") + "...",
                    triggered=self.panel.processor.compute_all_stability,
                    separator=True,
                    tip=_("Compute all stability features"),
                )
            self.action_for("xy_mode", separator=True)

        # MARK: ANALYSIS
        with self.new_category(ActionCategory.ANALYSIS):
            self.action_for("fwhm")
            self.action_for("fw1e2")
            self.new_action(
                _("Full width at y=..."),
                triggered=self.panel.processor.compute_full_width_at_y,
                tip=_("Compute the full width at a given y value"),
            )
            self.action_for("x_at_minmax")
            self.new_action(
                _("First abscissa at y=..."),
                triggered=self.panel.processor.compute_x_at_y,
                tip=_(
                    "Compute the first abscissa at a given y value "
                    "(linear interpolation)"
                ),
            )
            self.new_action(
                _("Ordinate at x=..."),
                triggered=self.panel.processor.compute_y_at_x,
                tip=_("Compute the ordinate at a given x value (linear interpolation)"),
            )
            self.new_action(
                _("Peak detection"),
                separator=True,
                triggered=self.panel.processor.compute_peak_detection,
                icon_name="peak_detect.svg",
            )
            self.action_for("sampling_rate_period", separator=True)
            self.action_for("dynamic_parameters", context_menu_pos=-1)
            self.action_for("bandwidth_3db", context_menu_pos=-1)
            self.action_for("contrast")

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
            self.action_for("convolution", separator=True)
        super().create_last_actions()


class ImageActionHandler(BaseActionHandler):
    """Object handling image panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("image")

    def create_new_object_actions(self):
        """Create actions for creating new objects"""
        for label, pclass in (
            (_("Zeros"), sio.Zeros2DParam),
            (_("Normal distribution"), sio.NormalDistribution2DParam),
            (_("Poisson distribution"), sio.PoissonDistribution2DParam),
            (_("Uniform distribution"), sio.UniformDistribution2DParam),
            (_("Gaussian"), sio.Gauss2DParam),
            (_("Ramp"), sio.Ramp2DParam),
        ):
            self.new_action(
                label,
                tip=_("Create new %s") % label,
                triggered=lambda pclass=pclass: self.panel.new_object(pclass()),
                select_condition=SelectCond.always,
            )

    def create_first_actions(self):
        """Create actions that are added to the menus in the first place"""
        super().create_first_actions()

        with self.new_category(ActionCategory.ROI):
            self.new_action(
                _("Create ROI grid") + "...",
                triggered=self.panel.processor.create_roi_grid,
                icon_name="roi_grid.svg",
                tip=_("Create a grid of regions of interest"),
            )

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

        # MARK: OPERATION
        with self.new_category(ActionCategory.OPERATION):
            self.action_for("logp1")
            self.action_for("flatfield", separator=True)

            with self.new_menu(_("Flip or rotation"), icon_name="rotate_right.svg"):
                self.action_for("fliph", context_menu_pos=-1, context_menu_sep=True)
                self.action_for("transpose", context_menu_pos=-1)
                self.action_for("flipv", context_menu_pos=-1)
                self.action_for("rotate270", context_menu_pos=-1)
                self.action_for("rotate90", context_menu_pos=-1)
                self.action_for("rotate")

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

        # MARK: PROCESSING
        with self.new_category(ActionCategory.PROCESSING):
            with self.new_menu(_("Frequency filters"), icon_name="noise_reduction.svg"):
                self.action_for("butterworth")
                self.action_for("gaussian_freq_filter")
            with self.new_menu(_("Thresholding"), icon_name="thresholding.svg"):
                self.action_for("threshold")
                self.action_for("threshold_isodata")
                self.action_for("threshold_li")
                self.action_for("threshold_mean")
                self.action_for("threshold_minimum")
                self.action_for("threshold_otsu")
                self.action_for("threshold_triangle")
                self.action_for("threshold_yen")
                self.new_action(
                    _("All thresholding methods") + "...",
                    triggered=self.panel.processor.compute_all_threshold,
                    separator=True,
                    tip=_("Apply all thresholding methods"),
                )
            with self.new_menu(_("Exposure"), icon_name="exposure.svg"):
                self.action_for("adjust_gamma")
                self.action_for("adjust_log")
                self.action_for("adjust_sigmoid")
                self.action_for("equalize_hist")
                self.action_for("equalize_adapthist")
                self.action_for("rescale_intensity")
            with self.new_menu(_("Restoration"), icon_name="noise_reduction.svg"):
                self.action_for("denoise_tv")
                self.action_for("denoise_bilateral")
                self.action_for("denoise_wavelet")
                self.action_for("denoise_tophat")
                self.new_action(
                    _("All denoising methods") + "...",
                    triggered=self.panel.processor.compute_all_denoise,
                    separator=True,
                    tip=_("Apply all denoising methods"),
                )
            with self.new_menu(_("Morphology"), icon_name="morphology.svg"):
                self.action_for("white_tophat")
                self.action_for("black_tophat")
                self.action_for("erosion")
                self.action_for("dilation")
                self.action_for("opening")
                self.action_for("closing")
                self.new_action(
                    _("All morphological operations") + "...",
                    triggered=self.panel.processor.compute_all_morphology,
                    separator=True,
                    tip=_("Apply all morphological operations"),
                )
            with self.new_menu(_("Edge detection"), icon_name="edge_detection.svg"):
                self.action_for("canny")
                self.action_for("farid", separator=True)
                self.action_for("farid_h")
                self.action_for("farid_v")
                self.action_for("laplace", separator=True)
                self.action_for("prewitt", separator=True)
                self.action_for("prewitt_h")
                self.action_for("prewitt_v")
                self.action_for("roberts", separator=True)
                self.action_for("scharr", separator=True)
                self.action_for("scharr_h")
                self.action_for("scharr_v")
                self.action_for("sobel", separator=True)
                self.action_for("sobel_h")
                self.action_for("sobel_v")
                self.action_for("scharr", separator=True)
                self.action_for("scharr_h")
                self.action_for("scharr_v")
                self.action_for("farid", separator=True)
                self.action_for("farid_h")
                self.action_for("farid_v")
                self.action_for("laplace", separator=True)
                self.new_action(
                    _("All edge detection filters..."),
                    triggered=self.panel.processor.compute_all_edges,
                    separator=True,
                    tip=_("Compute all edge detection filters"),
                )
                self.action_for("canny")
            self.action_for("butterworth")
            self.new_action(
                _("Erase area") + "...",
                triggered=self.panel.processor.compute_erase,
                icon_name="erase.svg",
                separator=True,
                tip=_("Erase area in the image as defined by a region of interest"),
            )

        # MARK: ANALYSIS
        with self.new_category(ActionCategory.ANALYSIS):
            # TODO: [P3] Add "Create ROI grid..." action to create a regular grid
            # or ROIs (maybe reuse/derive from `sigima.params.GridParam`)
            self.action_for("centroid", separator=True)
            self.action_for("enclosing_circle")
            self.new_action(
                _("2D peak detection"),
                separator=True,
                triggered=self.panel.processor.compute_peak_detection,
                tip=_("Compute automatic 2D peak detection"),
            )
            self.action_for("contour_shape")
            self.action_for("hough_circle_peaks")

            with self.new_menu(_("Blob detection")):
                self.action_for("blob_dog")
                self.action_for("blob_doh")
                self.action_for("blob_log")
                self.action_for("blob_opencv")

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
