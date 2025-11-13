.. _ima-menu-create:

Create Images
=============

This section describes how to create new images from various mathematical models.

.. figure:: /images/shots/i_create.png

    Screenshot of the "Create" menu.

When the "Image Panel" is selected, the menus and toolbars are updated to
provide image-related actions.

The "Create" menu allows you to create new images from various models (see below).

New image
^^^^^^^^^

Create a new image from various models
(supported datatypes: uint8, uint16, int16, float32, float64):

.. list-table::
    :header-rows: 1
    :widths: 10, 20, 70
    :class: longtable

    * - Icon
      - Model
      - Equation
    * - .. image:: ../../../datalab/data/icons/create/2d-zero.svg
           :width: 30px
      - Zero
      - :math:`z[i] = 0`
    * - .. image:: ../../../datalab/data/icons/create/2d-normal.svg
           :width: 30px
      - Normal distribution
      - :math:`z[i]` is normally distributed with configurable mean and standard deviation
    * - .. image:: ../../../datalab/data/icons/create/2d-poisson.svg
           :width: 30px
      - Poisson distribution
      - :math:`z[i]` is Poisson distributed with configurable mean
    * - .. image:: ../../../datalab/data/icons/create/2d-uniform.svg
           :width: 30px
      - Uniform distribution
      - :math:`z[i]` is uniformly distributed between two configurable bounds
    * - .. image:: ../../../datalab/data/icons/create/2d-gaussian.svg
           :width: 30px
      - 2D Gaussian
      - :math:`z = A \cdot \exp\left(-\dfrac{\left(\sqrt{\left(x-x_0\right)^2+\left(y-y_0\right)^2}-\mu\right)^2}{2\sigma^2}\right)`
    * - .. image:: ../../../datalab/data/icons/create/2d-ramp.svg
           :width: 30px
      - 2D Ramp
      - :math:`z = A (x - x_0) + B (y - y_0) + C`
    * - .. image:: ../../../datalab/data/icons/create/checkerboard.svg
           :width: 30px
      - Checkerboard
      - Alternating square pattern for calibration and spatial frequency analysis
    * - .. image:: ../../../datalab/data/icons/create/grating.svg
           :width: 30px
      - Sinusoidal grating
      - :math:`z = A \sin(2\pi(f_x \cdot x + f_y \cdot y) + \varphi) + C`
    * - .. image:: ../../../datalab/data/icons/create/ring.svg
           :width: 30px
      - Ring pattern
      - Concentric circular rings for radial analysis
    * - .. image:: ../../../datalab/data/icons/create/siemens.svg
           :width: 30px
      - Siemens star
      - Radial spoke pattern for resolution testing
    * - .. image:: ../../../datalab/data/icons/create/2d-sinc.svg
           :width: 30px
      - 2D sinc
      - :math:`z = A \cdot \mathrm{sinc}\left(\dfrac{\sqrt{(x-x_0)^2+(y-y_0)^2}}{\sigma}\right) + C`
