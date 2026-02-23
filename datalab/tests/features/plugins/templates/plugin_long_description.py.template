from datalab.plugins import PluginBase, PluginInfo

class {class_name}(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="{plugin_name}",
        version="1.0.0",
        description=(
            "{long_description}"
        ),
    )

    def test_action(self):
        {test_code}

    def create_actions(self):
        acth = self.signalpanel.acthandler
        acth.new_action(
            "{action_name}",
            triggered=self.test_action,
            select_condition="always"
        )
