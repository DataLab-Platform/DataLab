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

Open a new window to visualize and the selected signals.

In the separate window, you may visualize your data more comfortably
(e.g., by maximizing the window) and you may also annotate the data.

Edit annotations
^^^^^^^^^^^^^^^^

Open a new window to edit annotations.

.. seealso::
    See :ref:`ref-to-signal-annotations` for more details on annotations.

Auto-refresh
^^^^^^^^^^^^

Automatically refresh the visualization when the data changes.
When enabled (default), the plot view is automatically refreshed when the
data changes. When disabled, the plot view is not refreshed until you
manually refresh it by clicking the "Refresh manually" button in the
toolbar. Even though the refresh algorithm is optimized, it may still
take some time to refresh the plot view when the data changes, especially
when the data set is large. Therefore, you may want to disable the
auto-refresh feature when you are working with large data sets,
and enable it again when you are done. This will avoid unnecessary
refreshes.

Refresh manually
^^^^^^^^^^^^^^^^

Refresh the visualization manually. This triggers a refresh of the plot
view, even if the auto-refresh feature is disabled.

Show graphical object titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show/hide titles of computing results or annotations.

Curve anti-aliasing
^^^^^^^^^^^^^^^^^^^

Enable/disable anti-aliasing of curves. Anti-aliasing makes the curves
look smoother, but it may also make them look less sharp.

.. note::
    Anti-aliasing is enabled by default.

.. warning::
    Anti-aliasing may slow down the visualization, especially when
    working with large data sets.

Reset curve styles
^^^^^^^^^^^^^^^^^^

When plotting curves, DataLab automatically assigns a color and a line style to
each curve. Both parameters are chosen from a predefined list of colors and
line styles, and are assigned in a round-robin fashion.

This menu entry allows you to reset the curve styles, so that the next time
you plot curves, the first curve will be assigned the first color and the first
line style of the predefined lists, and the loop will start again from there.

Other menu entries
^^^^^^^^^^^^^^^^^^

Show/hide panels or toolbars.
