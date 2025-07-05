# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image tools application test:

  - Testing `ZAxisLogTool`
  - Testing `profile_to_signal` function (image cross section -> curve)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import os.path as osp

from plotpy.coords import axes_to_canvas
from plotpy.tools import CrossSectionTool
from qtpy import QtCore as QC
from sigima.obj import NewImageParam
from sigima.tests.data import create_multigauss_image

from cdl.gui.docks import profile_to_signal
from cdl.tests import cdltest_app_context


def test_image_tools_app():
    """Run image tools test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        newparam = NewImageParam.create(height=200, width=200)
        ima = create_multigauss_image(newparam)
        panel.add_object(ima)
        panel.set_current_object_title(f"Test image for {osp.basename(__file__)}")
        plotwidget = panel.plothandler.plotwidget
        mgr = plotwidget.get_manager()
        plot = plotwidget.get_plot()

        # === Testing "profile_to_signal" ----------------------------------------------
        cstool: CrossSectionTool = mgr.get_tool(CrossSectionTool)
        xcs_panel, ycs_panel = mgr.get_xcs_panel(), mgr.get_ycs_panel()
        xcs_panel.setVisible(True)
        ycs_panel.setVisible(True)
        x, y = newparam.width // 2, newparam.height // 2
        active_item = plot.get_active_item()
        pos = QC.QPointF(*axes_to_canvas(active_item, x, y))
        cstool.add_shape_to_plot(plot, pos, pos)
        for panel in (xcs_panel, ycs_panel):
            profile_to_signal(plot, panel)


if __name__ == "__main__":
    test_image_tools_app()
