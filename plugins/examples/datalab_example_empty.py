# -*- coding: utf-8 -*-

"""
Empty plugin example
====================

This is an empty example of a DataLab plugin.

It adds a new menu entry in "Plugins" menu, with a sub-menu "Empty plugin (example)".
This sub-menu contains one action, "Do nothing".

It also demonstrates how to create nested submenus.

Usage
-----

1. Copy this file to your DataLab plugins directory
2. Restart DataLab or use "Plugins > Reload plugins"
3. The plugin menu will appear under "Plugins"

Key Concepts
------------

**Plugin Structure:**

- Inherit from ``PluginBase``
- Define ``PLUGIN_INFO`` class attribute
- Implement ``create_actions()`` to register menu entries
- Use ``acth.new_menu()`` context manager for submenus
- Use ``acth.new_action()`` to register actions

**Action Conditions:**

- ``select_condition=None`` (default): Action enabled when ≥1 object selected
- ``select_condition="always"``: Action always enabled
- ``select_condition="single"``: Action enabled when exactly 1 object selected
- See ActionHandler documentation for more options

**Dialog Methods:**

Plugins have access to dialog convenience methods:

- ``self.show_info(message)``: Information dialog
- ``self.show_warning(message)``: Warning dialog
- ``self.show_error(message)``: Error dialog
- ``self.ask_yesno(question)``: Yes/No confirmation

See Also
--------

- DataLab plugin documentation: https://datalab-platform.com/en/features/advanced/plugins.html
- ``datalab_example_nested_menus.py``: Advanced nested menu structures
- ``datalab_example_dialogs.py``: Dialog methods showcase
- ``datalab_example_imageproc.py``: Image processing example
"""

import datalab.plugins


class EmptyPlugin(datalab.plugins.PluginBase):
    """DataLab Example Plugin"""

    PLUGIN_INFO = datalab.plugins.PluginInfo(
        name="Empty plugin (example)",
        version="1.0.0",
        description="This is an empty example plugin",
    )

    def do_nothing(self) -> None:
        """Do nothing"""
        self.show_info("Do nothing")

    def do_something_simple(self) -> None:
        """Do something simple in submenu"""
        self.show_info("Simple action in submenu")

    def do_something_advanced(self) -> None:
        """Do something advanced in nested submenu"""
        self.show_info("Advanced action in nested submenu")

    def create_actions(self) -> None:
        """Create actions

        This method is called once during plugin registration. Use the action
        handler (acthandler) to create menus and actions.

        Pattern:
            with acth.new_menu("Menu Name"):
                acth.new_action("Action Name", triggered=callback_method)
        """
        acth = self.imagepanel.acthandler

        # Main plugin menu - creates top-level entry under "Plugins"
        with acth.new_menu(self.PLUGIN_INFO.name):
            # Action with default select_condition (enabled when ≥1 image selected)
            # Note: select_condition=None means "at least one object selected"
            acth.new_action("Do nothing", triggered=self.do_nothing)

            # Example of a submenu (Level 1)
            with acth.new_menu("Submenu Example"):
                acth.new_action(
                    "Simple action",
                    triggered=self.do_something_simple,
                    select_condition="always",
                )

                # Example of a nested submenu (Level 2)
                with acth.new_menu("Advanced Submenu"):
                    acth.new_action(
                        "Advanced action",
                        triggered=self.do_something_advanced,
                        select_condition="always",
                    )
