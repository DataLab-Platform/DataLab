.. _use_case_ndt:

DataLab for non-destructive testing
===================================

.. meta::
    :description: Automatically detect defects, particles or indications on inspection images (radiography, thermography, C-scan): denoising, blob detection, regions of interest and batch processing, plus X-ray radiography features with the X-ray Radiography & NDT plugin -- with DataLab, the open-source signal and image analysis platform, extensible with your own algorithms.
    :keywords: non-destructive testing software, NDT image analysis, defect detection, automated inspection, blob detection, radiography, X-ray radiography software, image quality indicator, DICONDE, thermography, C-scan, image denoising, plugin, open-source, DataLab

The problem
-----------

Inspection produces images -- radiographs, thermograms, ultrasonic C-scans,
surface pictures -- on which defects or indications must be found, counted and
measured, often on whole series of parts. Manual review does not scale and is
hard to make reproducible; fully custom software is expensive to build and
maintain for each inspection bench.

What DataLab does
-----------------

DataLab provides the image-analysis building blocks and keeps the whole chain
traceable:

- **denoise** inspection images (median, binning, and other filters),
- **detect blob-like features** -- defects, particles, spots -- automatically,
  with several detection algorithms,
- restrict the analysis to **regions of interest**, get measurable results
  (positions, sizes) exportable to your reports,
- **save the workspace** (images + processing history + results) to a single
  HDF5 file for traceability,
- and when the built-in algorithms are not enough, **plug in your own
  Python processing** with the plugin system -- your operators get a new menu
  entry, not a new software.

.. figure:: ../images/tutorials/blobs/13.png

    Automatic blob detection on a denoised test image in DataLab.

X-ray radiography
-----------------

For X-ray radiography, the `X-ray Radiography & NDT
<https://github.com/DataLab-Platform/datalab-radiography>`_ plugin adds
dedicated features to DataLab:

- **radiographic display**: gray scale and automatic window, film convention,
  windows from regions of interest or DICOM attributes,
- **calibrated measurements**: several named magnifications per image, one per
  object plane, and distances, angles, diameters and areas measured in the
  detector plane and in each object plane at once,
- **detector corrections** (offset, gain, defective pixels) and **physical
  quantities**: attenuation, equivalent thickness, step wedge thickness
  calibration with beam hardening,
- **enhancement** filters with sizes in mm, and **image quality**: SNR,
  normalized SNR, CNR, basic spatial resolution with a duplex image quality
  indicator,
- **NDT measurements**: wall thickness in tangential radiography, detection
  and sizing of indications,
- pixel spacing, magnification and window of **DICOM and DICONDE** files,
- a live **X-ray radiography simulator** with known truth, and simulated and
  real-data examples.

The plugin is an official plugin of the `DataLab plugin catalog
<https://datalab-platform.com/plugins/>`_, at the Alpha stage: its methods are
validated on synthetic data with known truth and checked on open real data.
They take inspiration from radiography standards (ISO 17636-2, ISO 19232,
ASTM E2597, ISO 20769) without claiming compliance.

Proof in production
-------------------

In the field of non-destructive testing, `CEA <https://www.cea.fr>`_ entrusted
`CODRA <https://codra.net/>`_ with X-GRID, a software for the automatic
reconstruction of radiographic scenes from partial X-ray images -- with no
prior metadata on position, orientation or magnification. Its processing
pipeline (denoising, robust blob detection, homography estimation, image
fusion) was prototyped interactively with DataLab before being integrated into
the production tool. This work was presented at
:doc:`EuroSciPy 2026 <../outreach/euroscipy2026>`.

Try it
------

- :octicon:`rocket;1em;sd-text-info` **No install:** open
  `DataLab-Web with a demo inspection image already loaded
  <https://datalab-platform.com/web/?preload=demos/ndt.h5>`_ in your
  browser -- your data never leaves your machine.
- :octicon:`book;1em;sd-text-info` **Step-by-step tutorial:**
  :doc:`Automated detection of defects and particles on images <../intro/tutorials/blobs>`.
- :octicon:`package;1em;sd-text-info` **X-ray radiography:** install the
  X-ray Radiography & NDT plugin from the **Catalog** tab of
  **Plugins > Configure plugins...** (see :ref:`plugin_catalog`), then open
  its simulated weld example from **Plugins > X-ray Radiography & NDT >
  Examples**; see its `documentation
  <https://github.com/DataLab-Platform/datalab-radiography/blob/develop/doc/README.md>`_.
- :octicon:`download;1em;sd-text-info` Or :ref:`install the desktop application <installation>`.
