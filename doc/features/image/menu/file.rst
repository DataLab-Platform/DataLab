"File" menu
===========

.. image:: /images/shots/i_file.png

New image
    Create a new image from various models
    (supported datatypes: uint8, uint16, int16, float32, float64):

    .. list-table::
        :header-rows: 1
        :widths: 20, 80

        * - Model
          - Equation
        * - Zeros
          - :math:`z[i] = 0`
        * - Empty
          - Data is directly taken from memory as it is
        * - Random
          - :math:`z[i] \in [0, z_{max})` where :math:`z_{max}` is the datatype maximum value
        * - 2D Gaussian
          - :math:`z = A.exp(-\dfrac{(\sqrt{(x-x0)^2+(y-y0)^2}-\mu)^2}{2\sigma^2})`

Open image
    Create a new image from the following supported filetypes:

    .. list-table::
        :header-rows: 1

        * - File type
          - Extensions
        * - PNG files
          - .png
        * - TIFF files
          - .tif, .tiff
        * - 8-bit images
          - .jpg, .gif
        * - NumPy arrays
          - .npy
        * - Text files
          - .txt, .csv, .asc
        * - Andor SIF files
          - .sif
        * - SPIRICON files
          - .scor-data
        * - FXD files
          - .fxd
        * - Bitmap images
          - .bmp

Save image
    Save current image (see "Open image" supported filetypes).

Import metadata into image
    Import metadata from a JSON text file.

Export metadata from image
    Export metadata to a JSON text file.

Open HDF5 file
    Import data from a HDF5 file.

Save to HDF5 file
    Export the whole CodraFT session (all signals and images) into a HDF5 file.

Browse HDF5 file
    Open the :ref:`h5browser` in a new window to browse and import data
    from HDF5 file.