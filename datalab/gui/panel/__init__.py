"""
Panel
=====

The :mod:`datalab.gui.panel` package provides the **panel objects**
for signals and images.

Three types of panels are available:

- :class:`datalab.gui.panel.signal.SignalPanel`: Signal panel
- :class:`datalab.gui.panel.image.ImagePanel`: Image panel
- :class:`datalab.gui.panel.macro.MacroPanel`: Macro panel

Signal and Image Panels are called **Data Panels** and are used to display and
handle signals and images in the main window of DataLab.

Data Panels rely on the :class:`datalab.gui.panel.base.ObjectProp` class (managing
the object properties) and a set of modules to handle the GUI features:

- :mod:`datalab.gui.actionhandler`: Application actions (menus, toolbars, context menu)
- :mod:`datalab.gui.objectview`: Widgets to display object (signal/image) trees
- :mod:`datalab.gui.plothandler`: `PlotPy` items for representing signals and images
- :mod:`datalab.gui.processor`: Processor (computation)
- :mod:`datalab.gui.panel.roieditor`: ROI editor

The Macro Panel is used to display and run macros. It relies on the
:mod:`datalab.gui.macroeditor` module to handle the macro edition and execution.

Base features
-------------

.. automodule:: datalab.gui.panel.base

Signal panel
------------

.. automodule:: datalab.gui.panel.signal

Image panel
-----------

.. automodule:: datalab.gui.panel.image

Macro panel
-----------

.. automodule:: datalab.gui.panel.macro
"""
