PyData Paris 2024
=================

.. meta::
    :description: DataLab presentation at PyData Paris 2024 conference
    :keywords: DataLab, PyData, PyData Paris, Python, data science, conference, Paris

Conference Overview
-------------------

In September 2024, DataLab was presented at `PyData Paris 2024 <https://pydata.org/paris2024/>`_, a conference dedicated to Python in data science, machine learning, and analytics. This French conference provided an opportunity to present DataLab in depth to the local scientific Python community.

DataLab Presentation
--------------------

**Talk Title:** *DataLab: Bridging Scientific and Industrial Worlds for Advanced Signal and Image Processing*

**Presenter:** Pierre Raybaut (Executive VP, Engineering, CODRA)

This presentation offered a comprehensive look at DataLab's capabilities through practical demonstrations and use cases relevant to both scientific research and industrial applications.

Presentation Structure
^^^^^^^^^^^^^^^^^^^^^^

**Introduction**
    DataLab as a tool merging scientific research and industrial applications.

**Live Demo**
    The integrated demo showcasing:

    - Signal processing: basic operations, peak detection, curve fitting, FWHM measurements
    - Image processing: histogram computation, rotation, ROI management, centroid computation, contour detection
    - Advanced features: intensity profiles, restoration filters, morphological filters, edge detection

**Getting Started**
    - Comprehensive documentation (tutorials, API, contribution guidelines)
    - Multiple installation methods: pip, conda, Windows installer
    - Wide distribution channels

Four Key Use Cases
^^^^^^^^^^^^^^^^^^

The presentation detailed DataLab's versatility through four distinct use cases:

**1. Analyze Signals and Images**
    Using DataLab as a standalone application - a Swiss Army knife for data analysis with ready-to-use features and plugin extensibility.

**2. Prototype Processing Pipelines**
    Mixing Python code with DataLab's features by exchanging data between your IDE/notebook and DataLab, benefiting from both worlds.

**3. Debug Processing Applications**
    Establishing a connection between your application and DataLab to inspect data at different pipeline stages with visual feedback.

    *Example*: Development of an automatic image stitching software for CEA, using DataLab to visualize images and results at each algorithm step.

**4. Enhance Applications**
    Using DataLab as a library or companion application to add advanced processing features.

    *Example*: Plasma diagnostic control system for CEA - the application sends images to DataLab for visualization and computation, receiving back processed results and parameters.

Validation Approach
^^^^^^^^^^^^^^^^^^^

The presentation highlighted DataLab's two-tier validation process:

**Functional Validation**
    Classic automated testing (TDD approach, CI/CD workflows) achieving 90% code coverage - exceptional for a GUI application.

**Technical (Scientific) Validation**
    Ensuring result accuracy with 84% coverage of scientific features, with all validation status tracked and automatically documented.

Watch the Full Presentation
---------------------------

.. raw:: html

   <div style="max-width: 800px; margin: 2em auto;">
   <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
   <iframe src="https://www.youtube.com/embed/yn1bR-BVfn8"
   style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
   frameborder="0" allowfullscreen></iframe>
   </div>
   </div>

Resources
---------

- `Video recording on YouTube <https://www.youtube.com/watch?v=yn1bR-BVfn8&list=PLGVZCDnMOq0pKya8gksd00ennKuyoH7v7>`_
- `PyData Paris 2024 conference website <https://pydata.org/paris2024/>`_

Key Takeaways
-------------

The PyData Paris presentation emphasized several critical aspects:

**Companion Tool Philosophy**
    DataLab doesn't replace your IDE or Jupyter notebook - it complements them by providing:

    - Ready-to-use features for data reading, editing, and visualization
    - Fine-tuning capabilities for algorithm development
    - Visual debugging support

**Real-World Applications**
    Concrete examples from CEA projects demonstrated DataLab's practical value in production environments.

**Extensibility**
    The ability to customize DataLab through plugins and macros while maintaining industrial-grade reliability.

**Documentation Excellence**
    Automatically generated validation status documentation building trust with users.

Impact on DataLab
-----------------

The PyData Paris presentation and feedback contributed to:

- Increased focus on use case documentation
- Enhanced emphasis on the "companion tool" positioning
- Debugged issues with Conda package installation
