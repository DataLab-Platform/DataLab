import abc

from datalab.plugins import PluginBase, PluginInfo


class AbstractPlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="Abstract Plugin",
        version="1.0.0",
        description="Abstract plugin class",
    )

    @abc.abstractmethod
    def some_abstract_method(self):
        """This should prevent instantiation"""
        pass

    def create_actions(self):
        pass
