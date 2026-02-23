from datalab.plugins import PluginBase, PluginInfo

class {class_name}(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="{plugin_name}",
        version="1.0.0",
        description="Test plugin with dialogs",
    )

    def test_show_warning(self):
        self.show_warning("Test warning message")
        {test_code}

    def test_show_error(self):
        self.show_error("Test error message")
        {test_code}

    def test_show_info(self):
        self.show_info("Test info message")
        {test_code}

    def test_ask_yesno(self):
        # In test mode, this should be mocked
        result = self.ask_yesno("Test question?")
        {test_code}

    def create_actions(self):
        acth = self.signalpanel.acthandler
        with acth.new_menu("{menu_name}"):
            acth.new_action(
                "Test show_warning",
                triggered=self.test_show_warning,
                select_condition="always"
            )
            acth.new_action(
                "Test show_error",
                triggered=self.test_show_error,
                select_condition="always"
            )
            acth.new_action(
                "Test show_info",
                triggered=self.test_show_info,
                select_condition="always"
            )
            acth.new_action(
                "Test ask_yesno",
                triggered=self.test_ask_yesno,
                select_condition="always"
            )
