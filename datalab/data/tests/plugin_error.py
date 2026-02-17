from datalab.plugins import PluginBase, PluginInfo


class BrokenPlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="Broken Plugin",
        version="1.0.0",
        description="This plugin raises an error on init",
    )

    def __init__(self):
        super().__init__()
        raise RuntimeError("Planned failure")

    def create_actions(self):
        pass
