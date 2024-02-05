# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image tools application test:

  - Testing `ZAxisLogTool`
  - Testing `profile_to_signal` function (image cross section -> curve)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import os.path as osp

from guidata.qthelpers import qt_wait
from plotpy.coords import axes_to_canvas
from plotpy.tools import CrossSectionTool
from qtpy import QtCore as QC

import cdl.obj
from cdl.patch import ZAxisLogTool, profile_to_signal
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_multigauss_image


def test_image_tools_app():
    """Run image tools test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        newparam = cdl.obj.new_image_param(height=200, width=200)
        ima = create_multigauss_image(newparam)
        panel.add_object(ima)
        panel.set_current_object_title(f"Test image for {osp.basename(__file__)}")
        plotwidget = panel.plothandler.plotwidget
        plotmanager = plotwidget.get_manager()
        plot = plotwidget.get_plot()

        # === Testing "ZAxisLogTool" ---------------------------------------------------
        lstool = plotmanager.get_tool(ZAxisLogTool)
        qt_wait(1, except_unattended=True)
        for _index in range(2):
            lstool.activate()
            qt_wait(1, except_unattended=True)

        # === Testing "profile_to_signal" ----------------------------------------------
        cstool: CrossSectionTool = plotmanager.get_tool(CrossSectionTool)
        for panel in (plotmanager.get_xcs_panel(), plotmanager.get_ycs_panel()):
            panel.setVisible(True)
        x, y = newparam.width // 2, newparam.height // 2
        active_item = plot.get_active_item()
        pos = QC.QPointF(*axes_to_canvas(active_item, x, y))
        cstool.add_shape_to_plot(plot, pos, pos)
        for csw in (plotwidget.xcsw, plotwidget.ycsw):
            profile_to_signal(csw.cs_plot)


if __name__ == "__main__":
    test_image_tools_app()
