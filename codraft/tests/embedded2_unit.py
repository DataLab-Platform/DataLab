# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Application embedded test 2

CodraFT main window is simply hidden when closing application.
It is shown and raised above other windows when reopening application.
"""

from codraft.core.gui.main import CodraFTMainWindow
from codraft.tests import embedded1_unit

SHOW = True  # Show test in GUI-based test launcher


class HostWindow(embedded1_unit.BaseHostWindow):
    """Test main view"""

    def open_codraft(self):
        """Open CodraFT test"""
        if self.codraft is None:
            self.codraft = CodraFTMainWindow(console=False, hide_on_close=True)
            self.host.log("✨ Initialized CodraFT window")
            self.codraft.show()
        else:
            self.codraft.show()
            self.codraft.raise_()
        self.host.log("=> Shown CodraFT window")

    def close_codraft(self):
        """Close CodraFT window"""
        if self.codraft is not None:
            self.host.log("=> Closed CodraFT")
            self.codraft.close()


if __name__ == "__main__":
    embedded1_unit.test_embedded_feature(HostWindow)
