"Computing" menu
================

.. image:: /images/shots/s_computing.png

Regions of interest
    Open a dialog box to setup multiple Region Of Interests (ROI).
    ROI are stored as metadata, and thus attached to signal.

    ROI definition dialog is exactly the same as ROI extraction (see above):
    the ROI is defined by moving the position and adjusting the width of an
    horizontal range.

    .. figure:: /images/shots/s_roi_signal.png

        A signal with an ROI.

Statistics
    Compute statistics on selected signal and show a summary table.

    .. figure:: /images/shots/s_stats.png

        Example of statistical summary table: each row is associated to an ROI
        (the first row gives the statistics for the whole data).

Full width at half-maximum
    Fit data to a Gaussian, Lorentzian or Voigt model using
    least-square method.
    Then, compute the full width at half-maximum value.

    .. figure:: /images/shots/s_fwhm.png

        The computed result is displayed as an annotated segment.

Full width at 1/e²
    Fit data to a Gaussian model using least-square method.
    Then, compute the full width at 1/e².

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to signal and serialized with it when exporting
    current session in a HDF5 file.
