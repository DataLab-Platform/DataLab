from datalab.plugins import PluginBase, PluginInfo


class NoCreateActionsPlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="No Create Actions Plugin",
        version="1.0.0",
        description="Plugin without create_actions method",
    )

    # Missing create_actions() method - should raise error
