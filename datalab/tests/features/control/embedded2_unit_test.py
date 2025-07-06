# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Application embedded test 2

DataLab main window is simply hidden when closing application.
It is shown and raised above other windows when reopening application.
"""

# guitest: show

from datalab.gui.main import CDLMainWindow
from datalab.tests.features.control import embedded1_unit_test


class HostWindow(embedded1_unit_test.AbstractHostWindow):
    """Test main view"""

    def init_cdl(self) -> None:
        """Open DataLab test"""
        if self.datalab is None:
            self.datalab = CDLMainWindow(console=False, hide_on_close=True)
            self.host.log("✨ Initialized DataLab window")
            self.datalab.show()
        else:
            self.datalab.show()
            self.datalab.raise_window()
        self.host.log("=> Shown DataLab window")

    def close_cdl(self) -> None:
        """Close DataLab window"""
        if self.datalab is not None:
            self.host.log("=> Closed DataLab")
            self.datalab.close()

    def closeEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Close event

        Reimplemented from QWidget.closeEvent"""
        if self.datalab is None or self.datalab.close_properly():
            event.accept()
        else:
            event.ignore()


def test_embedded_feature():
    """Testing embedded feature"""
    embedded1_unit_test.run_host_window(HostWindow)


if __name__ == "__main__":
    test_embedded_feature()
