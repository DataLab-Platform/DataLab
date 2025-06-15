# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
GUI
===

The :mod:`cdl.gui` package contains functionnalities related to the graphical
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
  of modules to handle the GUI features (:mod:`cdl.gui.actionhandler` and
  :mod:`cdl.gui.objectview`), the data model (:mod:`cdl.gui.objectmodel`),
  the data visualization (:mod:`cdl.gui.plothandler`),
  and the data processing (:mod:`cdl.gui.processor`).

- **Macro panel**: this panel is used to display and run macros. It relies on the
  :mod:`cdl.gui.macroeditor` module to handle the macro edition and execution.

- **Specialized widgets**: those widgets are used to handle specific features such as
  ROI edition (:mod:`cdl.gui.roieditor`),
  Intensity profile edition (:mod:`cdl.gui.profiledialog`), etc.

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`cdl.gui.main`
      - DataLab main window and application

    * - :mod:`cdl.gui.panel`
      - Signal, image and macro panels

    * - :mod:`cdl.gui.actionhandler`
      - Application actions (menus, toolbars, context menu)

    * - :mod:`cdl.gui.objectview`
      - Widgets to display object (signal/image) trees

    * - :mod:`cdl.gui.plothandler`
      - `PlotPy` plot items for representing signals and images

    * - :mod:`cdl.gui.roieditor`
      - ROI editor

    * - :mod:`cdl.gui.processor`
      - Processor

    * - :mod:`cdl.gui.docks`
      - Dock widgets

    * - :mod:`cdl.gui.h5io`
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
    def serialize(self, writer: BaseIOHandler):
        """Serialize this object"""

    @abc.abstractmethod
    def deserialize(self, reader: BaseIOHandler):
        """Deserialize this object"""
