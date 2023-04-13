# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Application embedded test 2

CobraDataLab main window is simply hidden when closing application.
It is shown and raised above other windows when reopening application.
"""

from cdl.core.gui.main import CDLMainWindow
from cdl.tests import embedded1_unit

SHOW = True  # Show test in GUI-based test launcher


class HostWindow(embedded1_unit.BaseHostWindow):
    """Test main view"""

    def init_cdl(self):
        """Open CobraDataLab test"""
        if self.cdl is None:
            self.cdl = CDLMainWindow(console=False, hide_on_close=True)
            self.host.log("✨ Initialized CobraDataLab window")
            self.cdl.show()
        else:
            self.cdl.show()
            self.cdl.raise_()
        self.host.log("=> Shown CobraDataLab window")

    def close_cdl(self):
        """Close CobraDataLab window"""
        if self.cdl is not None:
            self.host.log("=> Closed CobraDataLab")
            self.cdl.close()


if __name__ == "__main__":
    embedded1_unit.test_embedded_feature(HostWindow)
