# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Print available DataLab menus and their actions."""

from __future__ import annotations

import sigima.objects

from datalab.gui.main import DLMainWindow
from datalab.tests import datalab_test_app_context


def parse_menu_as_text(win: DLMainWindow) -> str:
    """Recursively parse a menu and its actions into a text representation.

    Args:
        menu: The menu to parse
        indent: Current indentation level

    Returns:
        A string representation of the menu and its actions
    """

    def parse_actions(actions, indent_level: int) -> list[str]:
        """Recursively parse menu actions including submenus"""
        lines = []
        indent = "  " * indent_level
        for action in actions:
            if action.isSeparator():
                # Skip separators
                continue
            if action.menu() is not None:
                # Submenu: recursively parse its actions
                lines.append(f"{indent}{action.text()}")
                submenu_actions = action.menu().actions()
                lines.extend(parse_actions(submenu_actions, indent_level + 1))
            else:
                lines.append(f"{indent}{action.text()}")
        return lines

    txtlist = []
    for panel_name in ("signal", "image"):
        win.set_current_panel(panel_name)
        txtlist.append(f"Menus for {panel_name} panel:")
        for name in (
            "file",
            "create",
            "edit",
            "roi",
            "view",
            "operation",
            "processing",
            "analysis",
            "help",
        ):
            menu = getattr(win, f"{name}_menu")
            # Update menu content before parsing
            if name == "file":
                win._DLMainWindow__update_file_menu()
            elif name == "view":
                win._DLMainWindow__update_view_menu()
            elif name != "help":
                win._DLMainWindow__update_generic_menu(menu)
            txtlist.append(f"  {menu.title().replace('&', '')}:")
            txtlist.extend(parse_actions(menu.actions(), 2))
        txtlist.append("")
    return "\n".join(txtlist)


def print_datalab_menus() -> None:
    """Print available DataLab menus and their actions."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        # Add a signal and an image to have more actions in the menus
        sig = sigima.objects.create_signal_from_param(sigima.objects.LorentzParam())
        win.signalpanel.add_object(sig)
        param = sigima.objects.Gauss2DParam.create(height=100, width=100)
        ima = sigima.objects.create_image_from_param(param)
        win.imagepanel.add_object(ima)
        print(parse_menu_as_text(win))


if __name__ == "__main__":
    print_datalab_menus()
