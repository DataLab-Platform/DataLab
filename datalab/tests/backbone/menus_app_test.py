# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Menu contents test
------------------

Testing that the processing features are reachable from the menus of the panels.
"""

# guitest: show

from __future__ import annotations

from qtpy import QtWidgets as QW

from datalab.gui.actionhandler import ActionCategory
from datalab.gui.panel.base import BaseDataPanel
from datalab.tests import datalab_test_app_context


def _action_titles(actions: list[QW.QAction | QW.QMenu | None]) -> list[str]:
    """Return the titles of actions, including the actions of their submenus

    Args:
        actions: actions, submenus or separators (None)

    Returns:
        Titles of the actions, submenu titles excluded
    """
    titles = []
    for action in actions:
        if action is None:
            continue
        if isinstance(action, QW.QMenu):
            titles.extend(_action_titles(action.actions()))
        elif action.menu() is not None:
            titles.extend(_action_titles(action.menu().actions()))
        elif not action.isSeparator():
            titles.append(action.text())
    return titles


def _check_features_in_menu(
    panel: BaseDataPanel, category: ActionCategory, feature_ids: tuple[str, ...]
) -> None:
    """Check that features have exactly one action in a menu of a panel

    Args:
        panel: signal or image panel
        category: action category of the menu
        feature_ids: identifiers of the features
    """
    titles = _action_titles(panel.get_category_actions(category))
    for feature_id in feature_ids:
        title = panel.processor.get_feature(feature_id).action_title
        count = titles.count(title)
        assert count == 1, f"{feature_id} ({title!r}) found {count} times in {category}"


def test_shared_submenus_keep_all_actions() -> None:
    """Submenus filled by both a panel and the base action handler keep all actions

    The base action handler creates the "Axis transformation" submenu with
    "Polynomial calibration": the image action handler must add "Set uniform
    coordinates" without losing it, and the signal action handler must not
    duplicate it.
    """
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        _check_features_in_menu(
            win.imagepanel,
            ActionCategory.PROCESSING,
            ("set_uniform_coords", "calibration"),
        )
        _check_features_in_menu(
            win.signalpanel,
            ActionCategory.PROCESSING,
            ("transpose", "calibration", "to_cartesian"),
        )


if __name__ == "__main__":
    test_shared_submenus_keep_all_actions()
