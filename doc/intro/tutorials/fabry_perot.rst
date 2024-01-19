Measuring Fabry-Perot fringes
=============================

This example shows how to measure Fabry-Perot fringes using the image processing
features of DataLab:

- Load an image of a Fabry-Perot interferometer
- Define a circular region of interest (ROI) around the central fringe
- Detect contours in the ROI and fit them to circles
- Show the radius of the circles
- Annotate the image
- Copy/paste the ROI to another image
- Save the workspace

First, we open DataLab and load the images:

.. figure:: ../../images/tutorials/fabry_perot/01.png

   Open the image files with "File > Open...", or with the |fileopen_ima| button in
   the toolbar, or by dragging and dropping the files into DataLab (on the panel on
   the right).

.. |fileopen_ima| image:: ../../../cdl/data/icons/fileopen_ima.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/fabry_perot/02.png

    Select the test images "fabry_perot1.jpg" and "fabry_perot2.jpg" and click "Open".

The selected image is displayed in the main window. We can zoom in and out by pressing
the right mouse button and dragging the mouse up and down. We can also pan the image
by pressing the middle mouse button and dragging the mouse.

.. figure:: ../../images/tutorials/fabry_perot/03.png

   Zoom in and out with the right mouse button. Pan the image with the middle mouse
   button.

Then, let's define a circular region of interest (ROI) around the central fringe.

.. figure:: ../../images/tutorials/fabry_perot/04.png

   Select the "Edit regions of interest" tool in the "Computing" menu.

.. figure:: ../../images/tutorials/fabry_perot/05.png

   The "Regions of interest" dialog opens. Click "Add region of interest" and select
   a circular ROI. Resize the predefined ROI by dragging the handles. Note that you
   may change the ROI radius while keeping its center fixed by pressing the "Ctrl" key.
   Click "OK" to close the dialog.

.. figure:: ../../images/tutorials/fabry_perot/06.png

   Another dialog box opens, and asks you to confirm the ROI parameters. Click "OK".

.. figure:: ../../images/tutorials/fabry_perot/07.png

   The ROI is displayed on the image: masked pixels are grayed out, and the ROI
   boundary is displayed in blue (note that, internally, the ROI is defined by a
   binary mask, i.e. image data is represented as a NumPy masked array).

Now, let's detect the contours in the ROI and fit them to circles.

.. figure:: ../../images/tutorials/fabry_perot/08.png

   Select the "Contour detection" tool in the "Computing" menu.

.. figure:: ../../images/tutorials/fabry_perot/09.png

    The "Contour" parameters dialog opens. Select the shape "Circle" and click "OK".

.. figure:: ../../images/tutorials/fabry_perot/10.png

    The "Results" dialog opens, and displays the fitted circle parameters. Click "OK".

.. figure:: ../../images/tutorials/fabry_perot/11.png

    The fitted circles are displayed on the image.

.. note::

    If you want to show the results again, click on the "Show results" button, below
    the image list:

    .. image:: ../../images/tutorials/fabry_perot/12.png

The images (or signals) can also be displayed in a separate window, by clicking on
the "View in a new window" entry in the "View" menu (or the |new_window| button in
the toolbar). This is useful to compare side by side images or signals.

.. |new_window| image:: ../../../cdl/data/icons/new_window.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/fabry_perot/13.png

   The image is displayed in a separate window. The ROI and the fitted circles are
   also displayed. Annotations can be added to the image by clicking on the buttons
   at the bottom of the window. The annotations are stored in the metadata of the
   image, and together with the image data when the workspace is saved.
   Click on "OK" to close the window.

.. figure:: ../../images/tutorials/fabry_perot/14.png

   The image is displayed in the main window, together with the annotations.

If you want to take a closer look at the metadata, you can open the "Metadata" dialog.

.. figure:: ../../images/tutorials/fabry_perot/15.png

    The "Metadata" button is located below the image list.

.. figure:: ../../images/tutorials/fabry_perot/16.png

    The "Metadata" dialog opens. Among other information, it displays the annotations
    (in a JSON format), some style information (e.g. the colormap), and the ROI.

Now, let's delete the image metadata (including the annotations) to clean up the image.

.. figure:: ../../images/tutorials/fabry_perot/17.png

   Select the "Delete metadata" entry in the "Edit" menu, or the |metadata_delete|
   button in the toolbar.

.. |metadata_delete| image:: ../../../cdl/data/icons/metadata_delete.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/fabry_perot/18.png

    The "Delete metadata" dialog opens. Click "No" to keep the ROI and delete the
    rest of the metadata.

If we want to define the exact same ROI on the second image, we can copy/paste the
ROI from the first image to the second image, using the metadata.

.. figure:: ../../images/tutorials/fabry_perot/19.png

    Select the "Copy metadata" entry in the "Edit" menu, or the |metadata_copy|
    button in the toolbar.

.. |metadata_copy| image:: ../../../cdl/data/icons/metadata_copy.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/fabry_perot/20.png

    Select the second image in the "Images" panel, then select the "Paste metadata"
    entry in the "Edit" menu, or the |metadata_paste| button in the toolbar.

.. |metadata_paste| image:: ../../../cdl/data/icons/metadata_paste.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/fabry_perot/21.png

    The ROI is added to the second image.

.. figure:: ../../images/tutorials/fabry_perot/22.png

    Select the "Contour detection" tool in the "Computing" menu, with the same
    parameters as before (shape "Circle"). On this image, there are two fringes,
    so four circles are fitted. The "Results" dialog opens, and displays the
    fitted circle parameters. Click "OK".

.. figure:: ../../images/tutorials/fabry_perot/23.png

    The fitted circles are displayed on the image.

Finally, we can save the workspace to a file. The workspace contains all the images
that were loaded in DataLab, as well as the processing results. It also contains the
visualization settings (colormaps, contrast, etc.), the metadata, and the annotations.

.. figure:: ../../images/tutorials/fabry_perot/24.png

    Save the workspace to a file with "File > Save to HDF5 file...",
    or the |filesave_h5| button in the toolbar.

.. |filesave_h5| image:: ../../../cdl/data/icons/filesave_h5.svg
    :width: 24px
    :height: 24px

If you want to load the workspace again, you can use the "File > Open HDF5 file..."
(or the |fileopen_h5| button in the toolbar) to load the whole workspace, or the
"File > Browse HDF5 file..." (or the |h5browser| button in the toolbar) to load
only a selection of data sets from the workspace.

.. |fileopen_h5| image:: ../../../cdl/data/icons/fileopen_h5.svg
    :width: 24px
    :height: 24px

.. |h5browser| image:: ../../../cdl/data/icons/h5browser.svg
    :width: 24px
    :height: 24px