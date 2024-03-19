# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
GUI
===

The :mod:`cdl.core.gui` package contains functionnalities related to the graphical
user interface (GUI) of the DataLab (CDL) project. Those features are mostly specific
to DataLab and are not intended to be used independently.

The purpose of this section of the documentation is to provide an overview of the
DataLab GUI architecture and to describe the main features of the modules contained
in this package. It is not intended to provide a detailed description of the GUI
features, but rather to provide a starting point for the reader who wants to
understand the DataLab internal architecture.

DataLab's main window is composed of several parts, each of them being handled by a
specific module of this package:

- **Signal and image panels**: those panels are used to display signals and images
  and to provide a set of tools to manipulate them. Each data panel relies on a set
  of modules to handle the GUI features (:mod:`cdl.core.gui.actionhandler` and
  :mod:`cdl.core.gui.objectview`), the data model (:mod:`cdl.core.gui.objectmodel`),
  the data visualization (:mod:`cdl.core.gui.plothandler`),
  and the data processing (:mod:`cdl.core.gui.processor`).

- **Macro panel**: this panel is used to display and run macros. It relies on the
  :mod:`cdl.core.gui.macroeditor` module to handle the macro edition and execution.

- **Specialized widgets**: those widgets are used to handle specific features such as
  ROI edition (:mod:`cdl.core.gui.roieditor`),
  Intensity profile edition (:mod:`cdl.core.gui.profiledialog`), etc.

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`cdl.core.gui.main`
      - DataLab main window and application

    * - :mod:`cdl.core.gui.panel`
      - Signal, image and macro panels

    * - :mod:`cdl.core.gui.actionhandler`
      - Application actions (menus, toolbars, context menu)

    * - :mod:`cdl.core.gui.objectview`
      - Widgets to display object (signal/image) trees

    * - :mod:`cdl.core.gui.plothandler`
      - `PlotPy` plot items for representing signals and images

    * - :mod:`cdl.core.gui.roieditor`
      - ROI editor

    * - :mod:`cdl.core.gui.processor`
      - Processor

    * - :mod:`cdl.core.gui.docks`
      - Dock widgets

    * - :mod:`cdl.core.gui.h5io`
      - HDF5 input/output

"""

import abc

from guidata.io import BaseIOHandler


class ObjItf(abc.ABC):
    """Interface for objects handled by panels"""

    @property
    @abc.abstractmethod
    def title(self) -> str:
        """Object title"""

    @abc.abstractmethod
    def regenerate_uuid(self):
        """Regenerate UUID

        This method is used to regenerate UUID after loading the object from a file.
        This is required to avoid UUID conflicts when loading objects from file
        without clearing the workspace first.
        """

    @abc.abstractmethod
    def serialize(self, writer: BaseIOHandler):
        """Serialize this object"""

    @abc.abstractmethod
    def deserialize(self, reader: BaseIOHandler):
        """Deserialize this object"""
