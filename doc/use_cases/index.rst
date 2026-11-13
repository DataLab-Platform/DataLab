.. _use_cases_index:

Use cases
=========

.. meta::
    :description: What DataLab is used for: spectroscopy, laser beam analysis and photonics, non-destructive testing, scientific camera characterization, transient and pulsed signals, instrument control and data acquisition. Real-world signal and image processing workflows, without writing an application.
    :keywords: DataLab, use cases, spectroscopy, laser beam profiling, photonics, non-destructive testing, defect detection, X-ray radiography, camera characterization, photon transfer curve, pulse analysis, transient signals, instrument control, data acquisition, TANGO, signal processing, image processing

DataLab is used in research and industry to analyze data produced by scientific
instruments -- cameras, digitizers, spectrometers -- in a single workspace, without
having to develop a custom application. It was initially created for the plasma
diagnostics of `CEA <https://www.cea.fr>`_'s Laser Megajoule facility, where it is
used in production, and has since been applied to many other fields.

The pages below show how DataLab addresses a concrete problem in each domain,
with demonstration data, step-by-step tutorials and, for complete workflows,
application plugins of the `DataLab plugin catalog
<https://datalab-platform.com/plugins/>`_.

.. only:: html and not latex

    .. grid:: 1 1 3 3
        :gutter: 1 2 3 4

        .. grid-item-card:: :octicon:`pulse;1em;sd-text-info`  Spectroscopy
            :link: spectroscopy
            :link-type: doc

            Denoising, baseline correction and peak fitting on spectra

        .. grid-item-card:: :octicon:`sun;1em;sd-text-info`  Photonics & lasers
            :link: photonics
            :link-type: doc

            Beam profiling and interferogram analysis from camera images

        .. grid-item-card:: :octicon:`search;1em;sd-text-info`  Non-destructive testing
            :link: ndt
            :link-type: doc

            Automated detection of defects and particles on images, X-ray radiography

        .. grid-item-card:: :octicon:`device-camera;1em;sd-text-info`  Camera characterization
            :link: camera
            :link-type: doc

            Response, noise, photon transfer curve and dark current of scientific cameras

        .. grid-item-card:: :octicon:`zap;1em;sd-text-info`  Transient & pulsed signals
            :link: pulse
            :link-type: doc

            Pulse features, shot-to-shot stability, step response and timing

        .. grid-item-card:: :octicon:`plug;1em;sd-text-info`  Instruments & acquisition
            :link: instruments
            :link-type: doc

            Live view and acquisition of instrument data, e.g. TANGO devices

.. toctree::
   :maxdepth: 1
   :hidden:

   spectroscopy
   photonics
   ndt
   camera
   pulse
   instruments
