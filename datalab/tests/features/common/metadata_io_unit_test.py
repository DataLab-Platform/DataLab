# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Metadata import/export unit test:

  - Create an image with annotations and result shapes
  - Add the image to DataLab
  - Export image metadata to file (JSON)
  - Delete image metadata
  - Import image metadata from previous file
  - Check if image metadata is the same as the original image
"""

# guitest: show

import os.path as osp

import numpy as np
from sigima.objects.scalar import KindShape
from sigima.tests import data as test_data
from sigima.tests import helpers
from sigima.tests.helpers import compare_metadata

from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_metadata_io_unit():
    """Run image tools test scenario"""
    with execenv.context(unattended=True):
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            fname = osp.join(tmpdir, "test.dlabmeta")
            with datalab_test_app_context() as win:
                panel = win.imagepanel
                ima = test_data.create_annotated_image()
                # Create geometry results for testing

                # Create a point geometry directly
                from sigima.objects.scalar import GeometryResult

                from datalab.adapters_metadata.geometry_adapter import GeometryAdapter

                point_geom = GeometryResult(
                    title="Point Test",
                    kind=KindShape.POINT,
                    coords=np.array([[10.0, 20.0]]),
                    roi_indices=np.array([0], dtype=int),
                    attrs={},
                )

                GeometryAdapter(point_geom).add_to(ima)
                panel.add_object(ima)
                orig_metadata = ima.metadata.copy()
                panel.export_metadata_from_file(fname)
                panel.delete_metadata()

                # The +1 is for the "number" metadata option which has no default:
                assert len(ima.metadata) == len(ima.get_metadata_options_defaults()) + 1

                panel.import_metadata_from_file(fname)
                execenv.print("Check metadata export <--> import features:")
                compare_metadata(orig_metadata, ima.metadata)


if __name__ == "__main__":
    test_metadata_io_unit()
