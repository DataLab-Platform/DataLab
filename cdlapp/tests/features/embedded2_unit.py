# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Application embedded test 2

DataLab main window is simply hidden when closing application.
It is shown and raised above other windows when reopening application.
"""

# guitest: show

from cdlapp.core.gui.main import CDLMainWindow
from cdlapp.tests.features import embedded1_unit


class HostWindow(embedded1_unit.AbstractHostWindow):
    """Test main view"""

    def init_cdl(self) -> None:
        """Open DataLab test"""
        if self.cdlapp is None:
            self.cdlapp = CDLMainWindow(console=False, hide_on_close=True)
            self.host.log("✨ Initialized DataLab window")
            self.cdlapp.show()
        else:
            self.cdlapp.show()
            self.cdlapp.raise_()
        self.host.log("=> Shown DataLab window")

    def close_cdl(self) -> None:
        """Close DataLab window"""
        if self.cdlapp is not None:
            self.host.log("=> Closed DataLab")
            self.cdlapp.close()

    def closeEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Close event

        Reimplemented from QWidget.closeEvent"""
        if self.cdlapp is None or self.cdlapp.close_properly():
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    embedded1_unit.test_embedded_feature(HostWindow)
