# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Blob detection application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from skimage.data import human_mitosis  # pylint: disable=no-name-in-module

import cdl.param
from cdl.obj import create_image
from cdl.tests import cdl_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run ROI unit test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel
        proc = panel.processor
        for paramclass, compute_method, name in (
            (cdl.param.BlobDOGParam, proc.compute_blob_dog, "BlobDOG"),
            (cdl.param.BlobDOHParam, proc.compute_blob_doh, "BlobDOH"),
            (cdl.param.BlobLOGParam, proc.compute_blob_log, "BlobLOG"),
            (cdl.param.BlobOpenCVParam, proc.compute_blob_opencv, "BlobOpenCV"),
        ):
            image = create_image(title, human_mitosis())
            panel.add_object(image)
            param = paramclass()
            title = f"Testing {name} with default parameters"
            if isinstance(param, cdl.param.BlobOpenCVParam):
                param.filter_by_color = False
                title = f"Testing {name} with filter_by_color={param.filter_by_color}"
            compute_method(param)


if __name__ == "__main__":
    test()
