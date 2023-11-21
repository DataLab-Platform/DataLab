# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Application embedded test 2

DataLab main window is simply hidden when closing application.
It is shown and raised above other windows when reopening application.
"""

# guitest: show

from cdl.core.gui.main import CDLMainWindow
from cdl.tests.features import embedded1_unit


class HostWindow(embedded1_unit.AbstractHostWindow):
    """Test main view"""

    def init_cdl(self) -> None:
        """Open DataLab test"""
        if self.cdl is None:
            self.cdl = CDLMainWindow(console=False, hide_on_close=True)
            self.host.log("✨ Initialized DataLab window")
            self.cdl.show()
        else:
            self.cdl.show()
            self.cdl.raise_window()
        self.host.log("=> Shown DataLab window")

    def close_cdl(self) -> None:
        """Close DataLab window"""
        if self.cdl is not None:
            self.host.log("=> Closed DataLab")
            self.cdl.close()

    def closeEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Close event

        Reimplemented from QWidget.closeEvent"""
        if self.cdl is None or self.cdl.close_properly():
            event.accept()
        else:
            event.ignore()


def test_embedded_feature():
    """Testing embedded feature"""
    embedded1_unit.run_host_window(HostWindow)


if __name__ == "__main__":
    test_embedded_feature()
