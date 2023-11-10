# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab core.gui module

This module handles all GUI features which are specific to DataLab (CDL):

  * core.gui.main: handles CDL main window which relies on signal and image panels
  * core.gui.panel: handles CDL signal and image panels, relying on:
    * core.gui.actionhandler
    * core.gui.objecthandler
    * core.gui.plothandler
    * core.gui.roieditor
    * core.gui.processor
  * core.gui.docks: handles CDL dockwidgets
  * core.gui.h5io: handles HDF5 browser widget and related features
"""

import abc

from guidata.dataset.io import BaseIOHandler


class ObjItf(abc.ABC):
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
