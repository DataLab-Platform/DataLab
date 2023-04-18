# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab core.gui module

This module handles all GUI features which are specific to CobraDataLab (CDL):

  * core.gui.main: handles CDL main window which relies on signal and image panels
  * core.gui.panel: handles CDL signal and image panels, relying on:
    * core.gui.actionhandler
    * core.gui.objectlist
    * core.gui.plothandler
    * core.gui.roieditor
    * core.gui.processor
  * core.gui.docks: handles CDL dockwidgets
  * core.gui.h5io: handles HDF5 browser widget and related features
"""

import abc

from guidata.userconfigio import BaseIOHandler


class ObjItf:
    """Interface for objects handled by panels"""

    @property
    @abc.abstractmethod
    def title(self) -> str:
        """Object title"""

    @abc.abstractmethod
    def serialize(self, writer: BaseIOHandler):
        """Serialize this object"""

    @abc.abstractmethod
    def deserialize(self, reader: BaseIOHandler):
        """Deserialize this object"""
