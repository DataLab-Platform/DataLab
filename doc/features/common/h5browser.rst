.. _h5browser:

HDF5 Browser
============

.. meta::
    :description: HDF5 Browser in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, workspace, HDF5, browser, scientific, data, analysis, visualization, platform

The "HDF5 Browser" is a modal dialog box allowing to import almost any
1D and 2D data into DataLab workspace (and eventually metadata).

.. image:: /images/h5browser.png

Compatible curve or image data are displayed in a hierarchical view
on the left panel, as well as other scalar data (scalar values are just
shown for context purpose and may not be imported into DataLab workspace).

The right panel displays the selected curve or image data. It also shows information
on "Group" (path, description, etc.) and "Attributes" (names and values).

The HDF5 browser is fairly simple to use:
  * On the left panel, select the curve or image data you want to import
  * Selected data is plotted on the right panel
  * Click on "Check all" if you want to import all compatible data
  * Then validate by clicking on "OK"

.. note::
    The HDF5 browser may be used to explore multiple HDF5 files at once, thus allowing
    to import data from different files into the same DataLab workspace.

NeXus files and physical axes
----------------------------

Open ``.nxs`` and ``.nx5`` files with **Browse HDF5 file**, just like ``.h5``
files. A NeXus ``NXdata`` group designates a measured dataset and its axes.
Select the group to import the complete scientific object, rather than selecting
its individual datasets: a 1D group produces a signal, and a 2D group an image.
The raw datasets remain available separately in the tree.

The import preserves axis values, labels and units, including non-uniform image
coordinates. Signal Y uncertainties are imported when supplied; image
uncertainties are not imported. Histogram axes supplied as bin edges are
converted to bin centers, and this conversion is recorded.

An axis with an invalid shape or length, an ``axes`` attribute that does not
give one axis per dimension, or an axis that does not hold numbers is reported
in the group description and prevents importing the group. An undeclared axis
uses point or pixel indices; the importer does not infer a physical axis from
neighbouring datasets. Images need increasing coordinates: a decreasing image
axis is reversed together with the image, and this conversion is recorded,
while an image axis that is not strictly monotonic prevents the import.

Files with several counters or incomplete NeXus descriptions can be interpreted
by a plugin-provided data mapping. Each mapped object appears as an additional
item with its own axes, labels and units. Mappings must be registered before
opening the file; DataLab does not automatically discover JSON files beside it.
Plugin authors can follow the
`SigimaX HDF5 import guide <https://sigimax.readthedocs.io/en/latest/user_guide/hdf5_workspace.html>`_.

Large datasets and previews
--------------------------

Listing numeric datasets uses their shape and type without loading all their
values. Textual value previews read only the first elements, and attributes
containing more than 10 000 values are not copied to imported object metadata.
Graphical previews are read in a background thread; only the preview of the
last selected item is displayed, and items selected meanwhile are not read.
Selecting an item that cannot be previewed, such as a group, clears the
preview. Broken links, such as external links to missing detector files, are
skipped when the file is opened.

Large ordinary 1D arrays and 2D images, NeXus and mapped objects are previewed
with a sampling step above ``2**24`` elements (about 16.8 million). Import still
reads every value, not the sampled preview. This is not a universal memory
limit: compound datasets and signals stored as 2D tables still use complete
objects for their graphical previews. Closing a file or the browser waits for
its preview read to end.

Datasets with three or more dimensions, such as image stacks and spectral
cubes, are listed but cannot be imported as complete objects. There is no
graphical slice selector yet. Developers can read 1D or 2D views without loading
the whole cube using the selection API described in the SigimaX guide.

Numerical values and experimental origin
---------------------------------------

Unsigned 32-bit detector counts above 2 147 483 647 are imported as ``float64``
images without clipping or rounding. Other integer arrays are also converted
to an exact supported type when possible. If conversion to floating point
cannot preserve every integer (in particular beyond ``+/-2**53`` for ``float64``),
Sigima emits a warning instead of silently promising exact values. Complex
signals keep their imaginary part, including single-precision ones.

NeXus and mapped objects carry an ``experimental_source`` metadata entry
describing the source file, group, measured dataset, axes (``"."`` for a
dimension without axis) and recorded conversions. Mapped objects also identify
their mapping. This entry describes
where the measurement came from; it is not the processing provenance shown in
the Properties panel and does not make the experiment reproducible.

Save the imported objects in a DataLab workspace to keep both their data and
this metadata. Reopening that workspace does not require the original NeXus
file or the mapping plugin. Generic raw dataset imports do not automatically
receive this structured experimental source.
