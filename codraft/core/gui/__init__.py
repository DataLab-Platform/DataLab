# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT core.gui module

This module handles all GUI features which are specific to CodraFT:

  * core.gui.main: handles CodraFT main window which relies on signal and image panels
  * core.gui.panel: handles CodraFT signal and image panels, relying on:
    * core.gui.actionhandler
    * core.gui.objectlist
    * core.gui.plotitemlist
    * core.gui.roieditor
    * core.gui.processor
  * core.gui.docks: handles CodraFT dockwidgets
  * core.gui.h5io: handles HDF5 browser widget and related features
"""
