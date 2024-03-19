"""
Panel
=====

The :mod:`cdl.core.gui.panel` package provides the **panel objects**
for signals and images.

Three types of panels are available:

- :class:`cdl.core.gui.panel.signal.SignalPanel`: Signal panel
- :class:`cdl.core.gui.panel.image.ImagePanel`: Image panel
- :class:`cdl.core.gui.panel.macro.MacroPanel`: Macro panel

Signal and Image Panels are called **Data Panels** and are used to display and
handle signals and images in the main window of DataLab.

Data Panels rely on the :class:`cdl.core.gui.panel.base.ObjectProp` class (managing
the object properties) and a set of modules to handle the GUI features:

- :mod:`cdl.core.gui.actionhandler`: Application actions (menus, toolbars, context menu)
- :mod:`cdl.core.gui.objectview`: Widgets to display object (signal/image) trees
- :mod:`cdl.core.gui.plothandler`: `PlotPy` plot items for representing signals and images
- :mod:`cdl.core.gui.processor`: Processor (computation)
- :mod:`cdl.core.gui.panel.roieditor`: ROI editor

The Macro Panel is used to display and run macros. It relies on the
:mod:`cdl.core.gui.macroeditor` module to handle the macro edition and execution.

Base features
-------------

.. automodule:: cdl.core.gui.panel.base

Signal panel
------------

.. automodule:: cdl.core.gui.panel.signal

Image panel
-----------

.. automodule:: cdl.core.gui.panel.image

Macro panel
-----------

.. automodule:: cdl.core.gui.panel.macro
"""
