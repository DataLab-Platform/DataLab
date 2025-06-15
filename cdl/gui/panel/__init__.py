"""
Panel
=====

The :mod:`cdl.gui.panel` package provides the **panel objects**
for signals and images.

Three types of panels are available:

- :class:`cdl.gui.panel.signal.SignalPanel`: Signal panel
- :class:`cdl.gui.panel.image.ImagePanel`: Image panel
- :class:`cdl.gui.panel.macro.MacroPanel`: Macro panel

Signal and Image Panels are called **Data Panels** and are used to display and
handle signals and images in the main window of DataLab.

Data Panels rely on the :class:`cdl.gui.panel.base.ObjectProp` class (managing
the object properties) and a set of modules to handle the GUI features:

- :mod:`cdl.gui.actionhandler`: Application actions (menus, toolbars, context menu)
- :mod:`cdl.gui.objectview`: Widgets to display object (signal/image) trees
- :mod:`cdl.gui.plothandler`: `PlotPy` items for representing signals and images
- :mod:`cdl.gui.processor`: Processor (computation)
- :mod:`cdl.gui.panel.roieditor`: ROI editor

The Macro Panel is used to display and run macros. It relies on the
:mod:`cdl.gui.macroeditor` module to handle the macro edition and execution.

Base features
-------------

.. automodule:: cdl.gui.panel.base

Signal panel
------------

.. automodule:: cdl.gui.panel.signal

Image panel
-----------

.. automodule:: cdl.gui.panel.image

Macro panel
-----------

.. automodule:: cdl.gui.panel.macro
"""
