.. _overview:

Overview
========

.. meta::
    :description: Overview and basic concepts of DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, workspace, scientific, data, analysis, visualization, platform

Basic concepts
--------------

Working with DataLab is very easy. The user interface is intuitive and
self-explanatory. The main window is divided into two main areas:

- The left area shows the list of data sets which are currently loaded in
  DataLab, distibuted over two tabs: **Signals** and **Images**. The user can
  switch between the two tabs by clicking on the corresponding tab: this
  switches the main window to the corresponding panel, as well as the menu
  and toolbar contents. Below the list of data sets, a **Properties** view
  shows information about the currently selected data set.

- The right area shows the visualization of the currently selected data set.
  The visualization is updated automatically when the user selects a new data
  set in the list of data sets.

.. figure:: /images/shots/s_app_at_startup.png

    DataLab main window, at startup.

Internal data model and workspace
---------------------------------

DataLab has its own internal data model, in which data sets are organized around
a tree structure. Each panel in the main window corresponds to a branch of the
tree. Each data set shown in the panels corresponds to a leaf of the tree. Inside
the data set, the data is organized in an object-oriented way, with a set of
attributes and methods. The data model is described in more details in the
API section (see :mod:`sigima.objects`).

For each data set (1D signal or 2D image), not only the data itself is stored,
but also a set of metadata, which describes the data or the way it has to be
displayed. The metadata is stored in a dictionary, which is accessible through
the ``metadata`` attribute of the data set (and may also be browsed in the
**Properties** view, with the **Metadata** button).

The DataLab **Workspace** is defined as the collection of all data sets which
are currently loaded in DataLab, in both the **Signals** and **Images** panels.

Loading and saving the workspace
--------------------------------

The following actions are available to manage the workspace from the **File** menu:

- **Open HDF5 file**: load a workspace from an HDF5 file.

- **Save to HDF5 file**: save the current workspace to an HDF5 file.

- **Browse HDF5 file**: open the :ref:`h5browser` to explore the content of an
  HDF5 file and import data sets into the workspace.

.. note::

    Data sets may also be saved or loaded individually, using data formats
    such as `.txt` or `.npy` for 1D signals (see :ref:`open_signal` for the
    list of supported formats), , or `.tiff` or `.dcm` for 2D images
    (see :ref:`open_image` for the list of supported formats).

Interactive object creation and processing
------------------------------------------

DataLab provides an interactive workflow for creating objects and adjusting processing
parameters, allowing you to fine-tune results without creating multiple objects.

Interactive object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new signal or image using the creation functions (e.g., Gaussian signal,
2D peak image, etc.), DataLab stores the creation parameters in the object's metadata.
This enables interactive parameter adjustment after creation:

1. Create a signal or image using **Operations** > **Create** menu
2. Select the created object in the list
3. A **Creation** tab appears in the Properties panel (bottom-left)
4. Modify any creation parameter (amplitude, frequency, size, etc.)
5. Click **Apply** to regenerate the object with new parameters

The object is updated in-place, preserving any subsequent processing or analysis results.
This is particularly useful for:

- Exploring different parameter values without cluttering the workspace
- Fine-tuning object characteristics while observing the results
- Educational purposes to demonstrate parameter effects

.. note::

    Interactive creation is only available for objects created with parameter classes
    (not for imported data from files).

Interactive 1-to-1 processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When applying a 1-to-1 processing operation that has configurable parameters (e.g.,
Gaussian filter, threshold, morphological operations), DataLab stores the processing
metadata, enabling parameter adjustment and re-processing:

1. Apply a processing operation with parameters (e.g., **Processing** > **Filtering** > **Gaussian filter**)
2. The result object contains processing metadata (parameters, source object, function name)
3. Select the processed object in the list
4. A **Processing** tab appears in the Properties panel
5. Modify processing parameters (e.g., filter sigma value)
6. Click **Apply** to re-process with updated parameters

The processed object is updated in-place with the new results. This workflow is ideal for:

- Iteratively tuning filter parameters while observing results in real-time
- Adjusting threshold values without creating multiple intermediate objects
- Experimenting with different morphological structure element sizes
- Educational demonstrations of parameter effects on processing results

.. note::

    **Processing metadata requirements:**

    - Only works for 1-to-1 processing functions with parameter classes
    - Processing functions without parameters (e.g., absolute value, inverse) work as before
    - Source object must still exist for re-processing (error shown if deleted)

    **Not supported for:**

    - 2-to-1 processing (operations combining two objects)
    - n-to-1 processing (operations aggregating multiple objects)
    - These patterns don't benefit significantly from interactive parameter adjustment

Example workflow
^^^^^^^^^^^^^^^^

Here's a typical workflow using interactive processing:

1. **Create a test signal**: Operations > Create > Gaussian signal

   - Initial parameters: amplitude=1.0, mu=50, sigma=10

2. **Adjust creation parameters**: In the Creation tab, change sigma to 20, click Apply

   - Signal is regenerated with new width

3. **Apply Gaussian filter**: Processing > Filtering > Gaussian filter

   - Initial sigma=2.0

4. **Fine-tune filtering**: In the Processing tab, try sigma=1.0, then sigma=5.0

   - Each Apply updates the filtered result
   - Compare different smoothing levels without creating multiple objects

This interactive approach provides immediate visual feedback and keeps your workspace
organized by avoiding proliferation of test objects with slightly different parameters.
