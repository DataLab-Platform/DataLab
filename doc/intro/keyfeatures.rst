.. _key_features:

Key features
============

.. meta::
    :description: Key features of DataLab, the open-source data visualization and processing platform for scientists and engineers
    :keywords: DataLab, key features, signal processing, image processing, data visualization

This page presents briefly DataLab key features.

.. figure:: ../images/DataLab-Screenshot-Theme.png

    DataLab supports dark and light mode depending on your platform settings (this
    is handled by the `guidata`_ package, and may be overridden by setting the
    `QT_COLOR_MODE` environment variable to `dark` or `light`).

.. _guidata: https://pypi.python.org/pypi/guidata

Data visualization
^^^^^^^^^^^^^^^^^^

====== ====== ====================================
Signal Image  Feature
====== ====== ====================================
✓      ✓      Screenshots (save, copy)
✓      Z-axis Lin/log scales
✓      ✓      Data table editing
✓      ✓      Statistics on user-defined ROI
✓      ✓      Markers
..     ✓      Aspect ratio (1:1, custom)
..     ✓      50+ available colormaps (customizable)
..     ✓      Intensity profiles (line, average, radial)
✓      ✓      Annotations
✓      ✓      Persistance of settings in workspace
..     ✓      Distribute images on a grid
✓      ✓      Single or superimposed views
====== ====== ====================================

Data processing
^^^^^^^^^^^^^^^

====== ====== ===================================================
Signal Image  Feature
====== ====== ===================================================
✓      ✓      Process isolation for running computations
✓      ✓      Remote control from Jupyter, Spyder or any IDE
✓      ✓      Remote control from a third-party application
✓      ✓      Sum, average, difference, product...
✓      ✓      Operations with a constant
✓      ✓      Square root, power, logarithm, exponential...
✓      ✓      Average, standard deviation
✓      ..     Derivative, integral
✓      ✓      ROI extraction, Swap X/Y axes
✓      ..     Semi-automatic multi-peak detection
✓      ✓      Convolution, deconvolution
..     ✓      Flat-field correction
..     ✓      Flip, rotation, scaling...
..     ✓      Intensity profiles (line, average, radial)
..     ✓      Pixel binning
✓      ✓      Linear calibration
✓      ✓      Normalization, Clipping, Offset correction
✓      ..     Reverse X-axis
..     ✓      Thresholding (manual, Otsu...)
✓      ✓      Gaussian filter, Wiener filter
✓      ✓      Moving average, moving median
✓      ✓      FFT, inverse FFT, Power/Phase/Magnitude spectrum, Power Spectral Density
✓      ..     Interpolation, resampling
✓      ..     Detrending
✓      ..     X-Y mode
✓      ..     Interactive fit: Gauss, Lorentz, Voigt, polynomial, CDF...
✓      ..     Interactive multigaussian fit
✓      ..     Frequency filters (low-pass, high-pass, band-pass, band-stop)
✓      ..     Windowing (Hann, Hamming...)
..     ✓      Butterworth filter
..     ✓      Exposure correction (gamma, log...)
..     ✓      Restauration (Total Variation, Bilateral...)
..     ✓      Morphology (erosion, dilation...)
..     ✓      Edges detection (Roberts, Sobel...)
✓      ✓      Analysis on custom ROI
✓      ..     FWHM, FW @ 1/e²
✓      ..     Dynamic parameters (ENOB, SNR...), Sampling period/Rate
..     ✓      Centroid (robust method w/r noise)
..     ✓      Minimum enclosing circle center
..     ✓      2D peak detection
..     ✓      Contour detection
..     ✓      Circle Hough transform
..     ✓      Blob detection (OpenCV, Laplacian of Gaussian...)
====== ====== ===================================================
