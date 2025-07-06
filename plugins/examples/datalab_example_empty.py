# -*- coding: utf-8 -*-

"""
Empty plugin example
====================

This is an empty example of a DataLab plugin.

It adds a new menu entry in "Plugins" menu, with a sub-menu "Empty plugin (example)".
This sub-menu contains one action, "Do nothing".
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

    def create_actions(self) -> None:
        """Create actions"""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            # Note: in the following call, `select_condition` is by default `None`,
            # so the action is enabled only if at least one image is selected.
            acth.new_action("Do nothing", triggered=self.do_nothing)
