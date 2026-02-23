from datalab.plugins import PluginBase, PluginInfo

class {class_name}(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="{plugin_name}",
        version="1.0.0",
        description="Test plugin with nested menus",
    )

    def action_level_1(self):
        {test_code_1}

    def action_level_2(self):
        {test_code_2}

    def action_level_3(self):
        {test_code_3}

    def create_actions(self):
        # Test nested menus (2 and 3 levels deep)
        acth = self.signalpanel.acthandler

        # Level 1 menu
        with acth.new_menu("{menu_level_1}"):
            acth.new_action(
                "{action_level_1}",
                triggered=self.action_level_1,
                select_condition="always"
            )

            # Level 2 submenu (nested in Level 1)
            with acth.new_menu("{menu_level_2}"):
                acth.new_action(
                    "{action_level_2}",
                    triggered=self.action_level_2,
                    select_condition="always"
                )

                # Level 3 submenu (nested in Level 2)
                with acth.new_menu("{menu_level_3}"):
                    acth.new_action(
                        "{action_level_3}",
                        triggered=self.action_level_3,
                        select_condition="always"
                    )
