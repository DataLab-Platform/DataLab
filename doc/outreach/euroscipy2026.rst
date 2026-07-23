EuroSciPy 2026
==============

.. meta::
    :description: DataLab presentation at EuroSciPy 2026 in Kraków, Poland - automatic reconstruction of X-ray scenes with Python and DataLab
    :keywords: DataLab, EuroSciPy, EuroSciPy 2026, Kraków, Poland, X-ray, scene reconstruction, homography, blob detection, scientific Python, CEA, Codra, non-destructive testing

Conference Overview
-------------------

In July 2026, DataLab will be presented at `EuroSciPy 2026 <https://euroscipy.org/>`_, the 18th European Conference on Python in Science, held at AGH University of Kraków, Poland. EuroSciPy is the leading European gathering dedicated to the use and development of the Python language in scientific research, bringing together users and developers of scientific tools from both academia and industry. The 2026 edition is co-located with EuroPython.

**Event Details:**

:Date: July 21, 2026
:Time: 09:30 - 10:00
:Location: AGH University of Kraków, Poland
:Room: Room 1.19 (Ground Floor, Shannon)
:Track: Computational Tools and Scientific Python Infrastructure
:Format: Talk (25 minutes + Q&A)

Presentation
------------

**Title:** *Automatic Reconstruction of X-ray Scenes with Python and DataLab*

**Speaker:** Pierre Raybaut

Abstract
--------

In the field of non-destructive testing, the French Alternative Energies and Atomic Energy Commission (CEA) entrusted CODRA with the specification and development of software for the automatic reconstruction of radiographic scenes from partial X-ray images.

The challenge was to assemble a full-field X-ray scene from multiple acquisitions obtained by moving a detector or juxtaposing imaging plates, without any prior metadata regarding position, orientation, or magnification.

The reconstruction processing pipeline includes several key steps:

- image pre-processing and denoising,
- robust blob detection,
- homography estimation for geometric correction,
- fusion of corrected sub-images into a coherent global scene.

The entire workflow was developed using open-source scientific Python libraries (NumPy, SciPy, scikit-image, OpenCV) and prototyped interactively with `DataLab <https://datalab-platform.com>`_, an open-source platform for signal and image processing. DataLab made it possible to inspect intermediate results, adjust parameters, and validate geometric transformations step by step.

This project illustrates how the scientific Python ecosystem enables the development of industrial-grade imaging software, from interactive prototyping to automated deployment, using 100% open-source components.

Why This Matters for DataLab
----------------------------

This presentation is significant for several reasons:

**Real-World Industrial Application**
    Showcases DataLab in a high-stakes industrial use case developed by CODRA for CEA, demonstrating the platform's value beyond research and prototyping.

**Content-Based Reconstruction**
    Unlike traditional image stitching techniques that rely on overlapping textures or acquisition metadata, this method is entirely driven by content-based detection of patterns embedded in the scene, with each sub-image aligned through homography estimation in a global coordinate system.

**Interactive Prototyping**
    Highlights how DataLab's remote control and interactive visualization accelerated the development and debugging of a highly parameterized image processing pipeline, enabling rapid iteration before integration into a production tool.

**European Scientific Python Community**
    Brings DataLab to EuroSciPy, the main European conference for scientific Python, reinforcing the platform's place in the open-source ecosystem.

Presentation Highlights
-----------------------

The talk will include:

- a walkthrough of the reconstruction strategy,
- a discussion of the image processing challenges involved,
- a live or recorded demonstration of the interactive prototyping environment with DataLab,
- reflections on software architecture and reproducibility in scientific imaging workflows.

.. note::

   **Upcoming Event**: This presentation is scheduled for **July 21, 2026**. Presentation materials and additional information may be added here after the event.

Resources
---------

- `EuroSciPy 2026 website <https://euroscipy.org/>`_
- `Talk page on the conference schedule <https://pretalx.com/euroscipy-2026/talk/E3X9EX/>`_

.. seealso::

   For more information about DataLab's use in CEA projects, see:

   - :doc:`../features/index` - DataLab's validation approach
   - :doc:`../intro/introduction` - DataLab's operating modes, including remote control
   - :doc:`osxp2025` - The same X-ray reconstruction case study at OSXP 2025
