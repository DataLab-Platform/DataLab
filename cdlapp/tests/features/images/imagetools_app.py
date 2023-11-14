# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Image tools application test:

  - Testing `ZAxisLogTool`
  - Testing `to_cdl` function (image cross section -> curve)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import os.path as osp

from guidata.qthelpers import qt_wait
from plotpy.coords import axes_to_canvas
from plotpy.tools import CrossSectionTool
from qtpy import QtCore as QC

import cdlapp.obj
from cdlapp.patch import ZAxisLogTool, to_cdl
from cdlapp.tests import test_cdl_app_context
from cdlapp.tests.data import create_multigauss_image


def test():
    """Run image tools test scenario"""
    with test_cdl_app_context() as win:
        panel = win.imagepanel
        newparam = cdlapp.obj.new_image_param(height=200, width=200)
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

        # === Testing "to_cdl" -----------------------------------------------------
        cstool = plotmanager.get_tool(CrossSectionTool)
        cstool.activate()
        x, y = newparam.width // 2, newparam.height // 2
        active_item = plot.get_active_item()
        pos = QC.QPointF(*axes_to_canvas(active_item, x, y))
        cstool.end_rect(plot.filter, pos, pos)
        for csw in (plotwidget.xcsw, plotwidget.ycsw):
            to_cdl(csw.cs_plot)


if __name__ == "__main__":
    test()
