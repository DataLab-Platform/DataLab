"Processing" menu
=================

.. image:: /images/shots/i_processing.png

Linear calibration
    Create a new image which is a linear calibration
    of each selected image with respect to Z axis:

    .. list-table::
        :header-rows: 1
        :widths: 40, 60

        * - Parameter
          - Linear calibration
        * - Z-axis
          - :math:`z_{1} = a.z_{0} + b`

Thresholding
    Apply the thresholding to each selected image.

Clipping
    Apply the clipping to each selected image.

Moving average
    Compute moving average of each selected image
    (implementation based on `scipy.ndimage.uniform_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html>`_).

Moving median
    Compute moving median of each selected image
    (implementation based on `scipy.signal.medfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_).

Wiener filter
    Compute Wiener filter of each selected image
    (implementation based on `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_).

FFT
    Create a new image which is the Fast Fourier Transform (FFT)
    of each selected image.

Inverse FFT
    Create a new image which is the inverse FFT of each selected image.

Exposure
    Gamma correction
        Apply gamma correction to each selected image
        (implementation based on `skimage.exposure.adjust_gamma <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma>`_)

    Logarithmic correction
        Apply logarithmic correction to each selected image
        (implementation based on `skimage.exposure.adjust_log <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log>`_)

    Sigmoid correction
        Apply sigmoid correction to each selected image
        (implementation based on `skimage.exposure.adjust_sigmoid <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid>`_)

    Histogram equalization
        Equalize image histogram levels
        (implementation based on `skimage.exposure.equalize_hist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist>`_)

    Adaptive histogram equalization
        Equalize image histogram levels using Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm
        (implementation based on `skimage.exposure.equalize_adapthist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>`_)

    Intensity rescaling
        Stretch or shrink image intensity levels
        (implementation based on `skimage.exposure.rescale_intensity <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity>`_)

Restoration
    Total variation denoising
        Denoise image using Total Variation algorithm
        (implementation based on `skimage.restoration.denoise_tv_chambolle <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-tv-chambolle>`_)

    Bilateral filter denoising
        Denoise image using bilateral filter
        (implementation based on `skimage.restoration.denoise_bilateral <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-bilateral>`_)

    Wavelet denoising
        Perform wavelet denoising on image
        (implementation based on `skimage.restoration.denoise_wavelet <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-wavelet>`_)

    White Top-Hat denoising
        Denoise image by subtracting its white top hat transform
        (using a disk footprint)

Morphology
    White Top-Hat (disk)
        Perform white top hat transform of an image, using a disk footprint
        (implementation based on `skimage.morphology.white_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat>`_)

    Black Top-Hat (disk)
        Perform black top hat transform of an image, using a disk footprint
        (implementation based on `skimage.morphology.black_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.black_tophat>`_)

    Erosion (disk)
        Perform morphological erosion on an image, using a disk footprint
        (implementation based on `skimage.morphology.erosion <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion>`_)

    Dilation (disk)
        Perform morphological dilation on an image, using a disk footprint
        (implementation based on `skimage.morphology.dilation <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation>`_)

    Opening (disk)
        Perform morphological opening on an image, using a disk footprint
        (implementation based on `skimage.morphology.opening <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening>`_)

    Closing (disk)
        Perform morphological closing on an image, using a disk footprint
        (implementation based on `skimage.morphology.closing <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing>`_)

Canny filter
    Perform edge filtering on an image, using the Canny algorithm
    (implementation based on `skimage.feature.canny <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_)
