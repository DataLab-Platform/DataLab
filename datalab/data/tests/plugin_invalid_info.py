from datalab.plugins import PluginBase


class InvalidInfoPlugin(PluginBase):
    PLUGIN_INFO = None  # Invalid: should raise ValueError

    def create_actions(self):
        pass
