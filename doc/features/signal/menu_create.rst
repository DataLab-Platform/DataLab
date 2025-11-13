.. _sig-menu-create:

Create Signals
==============

This section describes how to create new signals from various mathematical models.

.. figure:: /images/shots/s_create.png

    Screenshot of the "Create" menu.

When the "Signal Panel" is selected, the menus and toolbars are updated to
provide signal-related actions.

The "Create" menu allows you to create new signals from various models (see below).

New signal
^^^^^^^^^^

Create a new signal from various models:

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Model
      - Equation
    * - Zero
      - :math:`y[i] = 0`
    * - Normal distribution
      - :math:`y[i]` is normally distributed with configurable mean and standard deviation
    * - Poisson distribution
      - :math:`y[i]` is Poisson distributed with configurable mean
    * - Uniform distribution
      - :math:`y[i]` is uniformly distributed between two configurable bounds
    * - Gaussian
      - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi} \cdot \sigma} \cdot \exp\left(-\dfrac{1}{2} \cdot \left(\dfrac{x-x_{0}}{\sigma}\right)^2\right)`
    * - Lorentzian
      - :math:`y = y_{0}+\dfrac{A}{\sigma \cdot \pi} \cdot \dfrac{1}{1+\left(\dfrac{x-x_{0}}{\sigma}\right)^2}`
    * - Voigt
      - :math:`y = y_{0}+A \cdot \dfrac{\Re\left(\exp\left(-z^2\right) \cdot \erfc(-j \cdot z)\right)}{\sqrt{2\pi} \cdot \sigma}` with :math:`z = \dfrac{x-x_{0}-j \cdot \sigma}{\sqrt{2} \cdot \sigma}`
    * - Blackbody (Planck's law)
      - :math:`y = \dfrac{2 h c^2}{\lambda^5 \left(\exp\left(\dfrac{h c}{\lambda k T}\right)-1\right)}`
    * - Sine
      - :math:`y = y_{0}+A\sin\left(2\pi \cdot f \cdot x+\phi\right)`
    * - Cosine
      - :math:`y = y_{0}+A\cos\left(2\pi \cdot f \cdot x+\phi\right)`
    * - Sawtooth
      - :math:`y = y_{0}+A \left( 2 \left( f x + \frac{\phi}{2\pi} - \left\lfloor f x + \frac{\phi}{2\pi} + \frac{1}{2} \right\rfloor \right) \right)`
    * - Triangle
      - :math:`y = y_{0}+A \sawtooth\left(2 \pi f x + \phi, \text{width} = 0.5\right)`
    * - Square
      - :math:`y = y_0 + A \sgn\left( \sin\left( 2\pi f x + \phi \right) \right)`
    * - Cardinal sine
      - :math:`y = y_0 + A \sinc\left(2\pi f x + \phi\right)`
    * - Linear chirp
      - :math:`y = y_{0} + A \sin\left(\phi_{0} + 2\pi \left(f_{0}\, x +  \frac{1}{2} c\, x^{2}\right)\right)`
    * - Step
      - :math:`y = y_{0}+A \left\{\begin{array}{ll}1 & \text{if } x > x_{0} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Exponential
      - :math:`y = y_{0}+A \exp\left(B \cdot x\right)`
    * - Logistic
      - :math:`y = y_{0} + \dfrac{A}{1 + \exp\left(-k \left(x - x_{0}\right)\right)}`
    * - Pulse
      - :math:`y = y_{0}+A \left\{\begin{array}{ll}1 & \text{if } x_{0} < x < x_{1} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Step Pulse
      - | :math:`y = \left( \begin{cases} y_0 & \text{if } x < t_0 \\ y_0 + A \cdot \dfrac{x - t_0}{t_r} & \text{if } t_0 \leq x < t_0 + t_r \\ y_0 + A & \text{if } x \geq t_0 + t_r \end{cases} \right) + \mathcal{N}\left(0, \sigma_n\right)`

        | where:

        * :math:`t_0` is the pulse start time,
        * :math:`t_r` is the rise time,
        * :math:`\sigma_n` is the noise amplitude

    * - Square Pulse
      - | :math:`y(x) = \left(\begin{cases} y_0 & \text{if } x < t_0 \\ y_0 + A \cdot \dfrac{x - t_0}{t_r} & \text{if } t_0 \leq x < t_0 + t_r \\ y_0 + A & \text{if } t_0 + t_r \leq x < t_1 \\ y_0 + A - A \cdot \dfrac{x - t_1}{t_f} & \text{if } t_1 \leq x < t_1 + t_f \\ y_0 & \text{if } x \geq t_1 + t_f \end{cases} \right) + \mathcal{N}(0, \sigma_n)`

        | where:

        * :math:`t_0` is the pulse start time,
        * :math:`t_r` is the rise time,
        * :math:`t_f` is the fall time,
        * :math:`t_1 = t_0 + t_r + d` is the time at which the decay starts,
        * :math:`\sigma_n` is the noise amplitude
        * the duration of the plateau :math:`d` is computed as :math:`d = t_{\mathrm{FWHM}} - \dfrac{t_r + t_f}{2}` from the full width at half maximum :math:`t_{\mathrm{FWHM}}`

        .. warning::

            The duration of the plateau :math:`d` should not be negative.

    * - Polynomial
      - :math:`y = y_{0}+A_{0}+A_{1} \cdot x+A_{2} \cdot x^2+\ldots+A_{n} \cdot x^n`
    * - Custom
      - Manual input of X and Y values
