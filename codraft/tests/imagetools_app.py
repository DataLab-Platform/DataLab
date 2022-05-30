# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Image tools application test:

  - Testing `ZAxisLogTool`
  - Testing `to_codraft` function (image cross section -> curve)
"""

from guiqwt.baseplot import axes_to_canvas
from guiqwt.tools import CrossSectionTool
from qtpy import QtCore as QC

from codraft.patch import ZAxisLogTool, to_codraft
from codraft.tests import codraft_app_context
from codraft.tests.data import create_test_image3
from codraft.utils.qthelpers import qt_wait

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    with codraft_app_context() as win:
        panel = win.imagepanel
        size = 200
        ima = create_test_image3(size)
        panel.add_object(ima)
        plotwidget = panel.itmlist.plotwidget

        # === Testing "ZAxisLogTool" ---------------------------------------------------
        lstool = plotwidget.get_tool(ZAxisLogTool)
        qt_wait(1)
        for _index in range(2):
            lstool.activate()
            qt_wait(1)

        # === Testing "to_codraft" -----------------------------------------------------
        plot = plotwidget.plot
        cstool = plotwidget.get_tool(CrossSectionTool)
        cstool.activate()
        pos = QC.QPointF(*axes_to_canvas(plot.active_item, size // 2, size // 2))
        cstool.end_rect(plotwidget.plot.filter, pos, pos)
        for csw in (plotwidget.xcsw, plotwidget.ycsw):
            to_codraft(csw.cs_plot)


if __name__ == "__main__":
    test()
