from datalab.plugins import PluginBase, PluginInfo
from qtpy import QtWidgets as QW

class {class_name}(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="{plugin_name}",
        version="1.0.0",
        description="Plugin with multiple widgets in dropdown menu",
    )

    def action_1(self):
        {test_code_1}

    def action_2(self):
        {test_code_2}

    def action_3(self):
        {test_code_3}

    def action_4(self):
        {test_code_4}

    def action_5(self):
        {test_code_5}

    def create_actions(self):
        acth = self.signalpanel.acthandler
        action_prefix = "{action_prefix}"

        # Create a menu with many actions to test dropdown
        with acth.new_menu("{menu_name}"):
            for i in range(1, 6):
                acth.new_action(
                    f"{action_prefix} {i}",
                    triggered=getattr(self, f"action_{i}"),
                    select_condition="always"
                )
