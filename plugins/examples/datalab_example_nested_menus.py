# -*- coding: utf-8 -*-

"""
Nested menus plugin example
============================

This example demonstrates how to create nested submenus in a DataLab plugin.

It shows:
- Simple submenu (1 level)
- Nested submenu (2 levels)
- Deep nested submenu (3 levels)
- Multiple submenus at the same level
- Creating similar menus for both Signal and Image panels

Usage
-----

1. Copy this file to your DataLab plugins directory
2. Restart DataLab or use "Plugins > Reload plugins"
3. Navigate through the nested menu structure

Menu Structure Created
----------------------

For **Signal Panel**::

    Nested Menus (example)
    ├── Basic action
    ├── Submenu 1
    │   └── Action in Submenu 1
    └── Submenu 2 (with nesting)
        ├── Action in Submenu 2
        └── Nested Level 2
            ├── Action at Level 2
            └── Deeply Nested Level 3
                └── Action at Level 3

For **Image Panel**: Similar structure with adapted action names.

Key Concepts
------------

**Nested Context Managers:**

Use nested ``with acth.new_menu()`` statements to create deep hierarchies::

    with acth.new_menu("Level 1"):
        acth.new_action("Action at Level 1", ...)

        with acth.new_menu("Level 2"):
            acth.new_action("Action at Level 2", ...)

            with acth.new_menu("Level 3"):
                acth.new_action("Action at Level 3", ...)

**Best Practices:**

- Keep nesting to 2-3 levels maximum for usability
- Use descriptive menu names that indicate hierarchy
- Group related actions in submenus
- Avoid too many items in one submenu (aim for 5-10 max)

**Multiple Panels:**

You can create different menu structures for Signal and Image panels using
``self.signalpanel.acthandler`` and ``self.imagepanel.acthandler`` respectively.

See Also
--------

- DataLab plugin documentation: https://datalab-platform.com/en/features/advanced/plugins.html
- ``datalab_example_empty.py``: Basic plugin structure
- ``datalab_example_dialogs.py``: Dialog methods
"""

import datalab.plugins


class NestedMenusExample(datalab.plugins.PluginBase):
    """DataLab Nested Menus Example Plugin"""

    PLUGIN_INFO = datalab.plugins.PluginInfo(
        name="Nested Menus (example)",
        version="1.0.0",
        description="Example plugin demonstrating nested menu structures",
    )

    def action_basic(self) -> None:
        """Basic action at root level"""
        self.show_info("Basic action at root level")

    def action_submenu_1(self) -> None:
        """Action in first submenu"""
        self.show_info("Action in Submenu 1")

    def action_submenu_2(self) -> None:
        """Action in second submenu"""
        self.show_info("Action in Submenu 2")

    def action_nested_level_2(self) -> None:
        """Action in nested submenu (level 2)"""
        self.show_info("Action in nested submenu (Level 2)")

    def action_nested_level_3(self) -> None:
        """Action in deeply nested submenu (level 3)"""
        self.show_info("Action in deeply nested submenu (Level 3)")

    def create_actions(self) -> None:
        """Create actions with various nested menu structures

        This demonstrates:
        - Multiple submenus at same level (Submenu 1, Submenu 2)
        - Nested submenus up to 3 levels deep
        - Different action handlers for Signal and Image panels
        """
        # Example for Signal Panel
        sah = self.signalpanel.acthandler
        with sah.new_menu(self.PLUGIN_INFO.name):
            # Basic action at root level (always enabled)
            sah.new_action(
                "Basic action", triggered=self.action_basic, select_condition="always"
            )

            # First submenu (Level 1)
            with sah.new_menu("Submenu 1"):
                sah.new_action(
                    "Action in Submenu 1",
                    triggered=self.action_submenu_1,
                    select_condition="always",
                )

            # Second submenu (Level 1) with nesting
            with sah.new_menu("Submenu 2 (with nesting)"):
                sah.new_action(
                    "Action in Submenu 2",
                    triggered=self.action_submenu_2,
                    select_condition="always",
                )

                # Nested submenu (Level 2)
                with sah.new_menu("Nested Level 2"):
                    sah.new_action(
                        "Action at Level 2",
                        triggered=self.action_nested_level_2,
                        select_condition="always",
                    )

                    # Deeply nested submenu (Level 3)
                    with sah.new_menu("Deeply Nested Level 3"):
                        sah.new_action(
                            "Action at Level 3",
                            triggered=self.action_nested_level_3,
                            select_condition="always",
                        )

        # Same structure for Image Panel
        iah = self.imagepanel.acthandler
        with iah.new_menu(self.PLUGIN_INFO.name):
            iah.new_action(
                "Basic action (Image)",
                triggered=self.action_basic,
                select_condition="always",
            )

            with iah.new_menu("Image Processing"):
                iah.new_action(
                    "Process image",
                    triggered=self.action_submenu_1,
                    select_condition="always",
                )

                with iah.new_menu("Advanced Processing"):
                    iah.new_action(
                        "Advanced process",
                        triggered=self.action_nested_level_2,
                        select_condition="always",
                    )
