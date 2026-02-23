
from datalab.plugins import PluginBase, PluginInfo
from datalab.config import _
from qtpy import QtWidgets as QW

class {class_name}(PluginBase):
    PLUGIN_INFO = PluginInfo(
        name="{plugin_name}",
        version="1.0.0",
        description="Test plugin",
    )

    def create_actions(self):
        # Action common to both
        act_common = QW.QAction("{action_name}_common", self.main)
        act_common.setObjectName("{action_object_name}_common")
        act_common.triggered.connect(self.run_test_action)

        # Action specific to signal
        act_signal = QW.QAction("{action_name}_signal", self.main)
        act_signal.setObjectName("{action_object_name}_signal")
        act_signal.triggered.connect(self.run_test_action)

        # Action specific to image
        act_image = QW.QAction("{action_name}_image", self.main)
        act_image.setObjectName("{action_object_name}_image")
        act_image.triggered.connect(self.run_test_action)

        # Add to signal panel (common + signal)
        self.signalpanel.acthandler.add_action(act_common)
        self.signalpanel.acthandler.add_to_action_list(act_common)
        self.signalpanel.acthandler.add_action(act_signal)
        self.signalpanel.acthandler.add_to_action_list(act_signal)

        # Add to image panel (common + image)
        self.imagepanel.acthandler.add_action(act_common)
        self.imagepanel.acthandler.add_to_action_list(act_common)
        self.imagepanel.acthandler.add_action(act_image)
        self.imagepanel.acthandler.add_to_action_list(act_image)

    def run_test_action(self):
        {test_code}
