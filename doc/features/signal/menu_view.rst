.. _sig-menu-view:

View options for Signals
========================

.. figure:: /images/shots/s_view.png

    Screenshot of the "View" menu.

When the "Signal Panel" is selected, the menus and toolbars are updated to
provide signal-related actions.

The "View" menu allows you to visualize the current signal or group of signals.
It also allows you to show/hide titles, to enable/disable anti-aliasing, or to
refresh the visualization.

View in a new window
^^^^^^^^^^^^^^^^^^^^

Open a new window to visualize the selected signals.

This option allows you to visualize the selected signals in a separate window,
in which you can visualize the data more comfortably (e.g., by maximizing the
window) and you can also annotate the data.

When you click on the button "Annotations" in the toolbar of the new window,
the annotation editing mode is activated (see the section "Edit annotations" below).

Edit annotations
^^^^^^^^^^^^^^^^

Open a new window to edit annotations.

This option allows you to show the selected signals in a separate window,
in which you can visualize the data more comfortably (e.g., by maximizing the
window) as well as to add or edit annotations.

Annotations are used to add comments or labels to the data, and they can be used
to highlight specific events or regions of interest.

.. note::
    The annotations are saved in the metadata of the signal, so they are
    persistent and will be displayed every time you visualize the signal.

.. figure:: /images/annotations/signal_annotations1.png

    Annotations may be added in the separate view.

The typical workflow to edit annotations is as follows:

1. Select the "Edit annotations" option.
2. In the new window, add annotations by clicking on the corresponding buttons
   at the bottom of the window.
3. Eventually, customize the annotations by changing their properties (e.g.,
   the text, the color, the position, etc.) using the "Parameters" option in the
   context menu of the annotations.
4. When you are done, click on the "OK" button to save the annotations. This will
   close the window and the annotations will be saved in the metadata of the signal
   and will be displayed in the main window.

.. figure:: /images/annotations/signal_annotations2.png

    Annotations are now part of the signal metadata.

.. note::
    Annotations may be copied from a signal to another by using the
    "copy/paste metadata" features.

Auto-refresh
^^^^^^^^^^^^

Automatically refresh the visualization when the data changes:

- When enabled (default), the plot view is automatically refreshed when the
  data changes.

- When disabled, the plot view is not refreshed until you manually refresh it
  by using the "Refresh manually" menu entry.

Even though the refresh algorithm is optimized, it may still take some time to
refresh the plot view when the data changes, especially when the data set is
large. Therefore, you may want to disable the auto-refresh feature when you are
working with large data sets, and enable it again when you are done. This will
avoid unnecessary refreshes.

Show only first selected signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Toggle between showing all selected signals or only the first one.

When this option is enabled, only the first selected signal is displayed in the plot
view. This may be useful when multiple signals are selected and focusing on a single
signal is (temporarily) preferred.

Refresh manually
^^^^^^^^^^^^^^^^

Refresh the visualization manually.

This triggers a refresh of the plot view. It is useful when the auto-refresh
feature is disabled, or when you want to force a refresh of the plot view.

Show graphical object titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show/hide titles of analysis results or annotations.

This option allows you to show or hide the titles of the graphical objects
(e.g., the titles of the analysis results or annotations). Hiding the titles
can be useful when you want to visualize the data without any distractions,
or if there are too many titles and they are overlapping.

Curve anti-aliasing
^^^^^^^^^^^^^^^^^^^

Enable/disable anti-aliasing of curves.

Anti-aliasing makes the curves look smoother, but it may also make them look less sharp.

.. note::
    Anti-aliasing is enabled by default.

.. warning::
    Anti-aliasing may slow down the visualization, especially when
    working with large data sets.

Reset curve styles
^^^^^^^^^^^^^^^^^^

Give the selected signals automatic curve styles again.

When plotting curves, DataLab automatically assigns a color and a line style to
each curve, according to its position among the curves shown on the plot (in the
order of the signals in the panel): the first curves are drawn with solid lines
in distinct colors, readable on both light and dark backgrounds, and the next
ones with dashed lines in the same colors. The first curve shown is thus always
assigned the first color, whatever the signals shown.

Automatic styles are not saved with the signals. If you choose another color or
line style for a curve (with the "Parameters" option of its context menu), this
style is saved in the metadata of the signal and is kept afterwards.

This menu entry removes the colors and line styles chosen for the selected signals,
so that their curves are assigned automatic styles again.

.. note::
    Workspaces and history files saved with DataLab 1.3 or earlier store the
    automatic style of each curve: when opening them, these styles are removed
    from the metadata of the signals, whose curves are assigned the current
    automatic styles. A color and a line style chosen at that time cannot be told
    apart from an automatic style when they are identical to one of them: they
    are removed as well. Files saved with later versions, as well as metadata
    imported from a file, are not converted.