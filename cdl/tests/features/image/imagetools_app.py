# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image tools application test:

  - Testing `ZAxisLogTool`
  - Testing `to_cdl` function (image cross section -> curve)
"""

import os.path as osp

from guiqwt.baseplot import axes_to_canvas
from guiqwt.tools import CrossSectionTool
from qtpy import QtCore as QC

import cdl.obj
from cdl.patch import ZAxisLogTool, to_cdl
from cdl.tests import cdl_app_context
from cdl.tests.data import create_multigauss_image
from cdl.utils.qthelpers import qt_wait

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        newparam = cdl.obj.new_image_param(height=200, width=200)
        ima = create_multigauss_image(newparam)
        panel.add_object(ima)
        panel.set_current_object_title(f"Test image for {osp.basename(__file__)}")
        plotwidget = panel.plothandler.plotwidget

        # === Testing "ZAxisLogTool" ---------------------------------------------------
        lstool = plotwidget.get_tool(ZAxisLogTool)
        qt_wait(1, except_unattended=True)
        for _index in range(2):
            lstool.activate()
            qt_wait(1, except_unattended=True)

        # === Testing "to_cdl" -----------------------------------------------------
        plot = plotwidget.plot
        cstool = plotwidget.get_tool(CrossSectionTool)
        cstool.activate()
        x, y = newparam.width // 2, newparam.height // 2
        pos = QC.QPointF(*axes_to_canvas(plot.active_item, x, y))
        cstool.end_rect(plotwidget.plot.filter, pos, pos)
        for csw in (plotwidget.xcsw, plotwidget.ycsw):
            to_cdl(csw.cs_plot)


if __name__ == "__main__":
    test()
