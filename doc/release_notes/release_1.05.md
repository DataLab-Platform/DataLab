# Version 1.5 #

## DataLab Version 1.5.0 ##

### Requirements ###

* **DataLab-Platform**:
  * Sigima ≥ 1.5.0 is required: it reads NIfTI images.
* **Third-party**:
  * NiBabel ≥ 5.3 is an optional dependency, installed with the new `nifti` extra (`pip install datalab-platform[nifti]`) and included in the standalone version: it is required to open NIfTI images.

### ✨ New Features ###

* **NIfTI images**: **File > Open** of the Image panel now opens NIfTI images (`.nii` and `.nii.gz`), e.g. from medical imaging or neuroimaging, when NiBabel is installed. A 2D image opens directly. For a 3D volume, a dialog shows its dimensions, data type and spatial unit, and selects a slice or a range of slices (the central slice by default) with their indicative memory size: only these slices are read, each one becoming an image in a group named after the file. Pixel values, sizes and units are those of the file, and its header information is kept in the `nifti` metadata of each image. Replaying an import from the history opens the same slices without asking again, and the remote control API can open volumes without any dialog. Opening a NIfTI file does not de-identify it (closes [Issue #590](https://github.com/DataLab-Platform/DataLab/issues/590)).
* **Opening files**: canceling the reading of a file now also skips the remaining files of the selection, instead of reading them one after the other; the images already read are kept.
