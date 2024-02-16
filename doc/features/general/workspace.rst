.. _workspace:

Workspace
=========

.. meta::
    :description: Workspace and HDF5 Browser in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, workspace, HDF5, browser, scientific, data, analysis, visualization, platform

Basic concepts
--------------

Working with DataLab is very easy. The user interface is intuitive and
self-explanatory. The main window is divided into two main areas:

- The right area shows the list of data sets which are currently loaded in
  DataLab, distibuted over two tabs: **Signals** and **Images**. The user can
  switch between the two tabs by clicking on the corresponding tab: this
  switches the main window to the corresponding panel, as well as the menu
  and toolbar contents. Below the list of data sets, a **Properties** view
  shows information about the currently selected data set.

- The left area shows the visualization of the currently selected data set.
  The visualization is updated automatically when the user selects a new data
  set in the list of data sets.

.. figure:: /images/shots/s_app_at_startup.png

    DataLab main window, at startup.

DataLab has its own internal data model, in which data sets are organized around
a tree structure. Each panel in the main window corresponds to a branch of the
tree. Each data set shown in the panels corresponds to a leaf of the tree. Inside
the data set, the data is organized in an object-oriented way, with a set of
attributes and methods. The data model is described in more details in the
API section (see :mod:`cdl.obj`).

For each data set (1D signal or 2D image), not only the data itself is stored,
but also a set of metadata, which describes the data or the way it has to be
displayed. The metadata is stored in a dictionary, which is accessible through
the ``metadata`` attribute of the data set (and may also be browsed in the
**Properties** view, with the **Metadata** button).

The DataLab **Workspace** is defined as the collection of all data sets which
are currently loaded in DataLab, in both the **Signals** and **Images** panels.
The workspace may be saved to an HDF5 file, and reloaded later. It is also
possible to import data sets from an HDF5 file into the workspace, using the
:ref:`h5browser`.

.. note::

    Data sets may also be saved or loaded individually, using data formats
    such as `.txt` or `.npy` for 1D signals (see :ref:`open_signal` for the
    list of supported formats), , or `.tiff` or `.dcm` for 2D images
    (see :ref:`open_image` for the list of supported formats).

.. _h5browser:

HDF5 Browser
------------

The "HDF5 Browser" is a modal dialog box allowing to import almost any
1D and 2D data into DataLab workspace (and eventually metadata).

.. image:: /images/h5browser.png

Compatible curve or image data are displayed in a hierarchical view
on the left panel, as well as other scalar data (scalar values are just
shown for context purpose and may not be imported into DataLab workspace).

The HDF5 browser is fairly simple to use:
  * On the left panel, select the curve or image data you want to import
  * Selected data is plotted on the right panel
  * Click on "Check all" if you want to import all compatible data
  * Then validate by clicking on "OK"
