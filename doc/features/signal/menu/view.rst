"View" menu
===========

.. image:: /images/shots/s_view.png

View in a new window
    Open a new window to visualize and the selected signals.

    In the separate window, you may visualize your data more comfortably
    (e.g., by maximizing the window) and you may also annotate the data.

    .. seealso::
        See :ref:`ref-to-signal-annotations` for more details on annotations.

Show graphical object titles
    Show/hide titles of computing results or annotations.

Auto-refresh
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
    Refresh the visualization manually. This triggers a refresh of the plot
    view, even if the auto-refresh feature is disabled.

Curve anti-aliasing
    Enable/disable anti-aliasing of curves. Anti-aliasing makes the curves
    look smoother, but it may also make them look less sharp.

    .. note::
        Anti-aliasing is enabled by default.

    .. warning::
        Anti-aliasing may slow down the visualization, especially when
        working with large data sets.

Other menu entries
    Show/hide panels or toolbars.
