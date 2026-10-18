# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Metadata import/export unit test:

  - Create an image with annotations, geometry result and table results
  - Add the image to DataLab
  - Export image metadata to file (JSON)
  - Delete image metadata
  - Import image metadata from previous file
  - Check if image metadata is the same as the original image
"""

# guitest: show

import os.path as osp

from sigima.tests import data as test_data

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers


def test_metadata_io_unit():
    """Run image tools test scenario"""
    with execenv.context(unattended=True):
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            fname = osp.join(tmpdir, "test.dlabmeta")
            with datalab_test_app_context() as win:
                panel = win.imagepanel

                # Create a test image with annotations
                ima = test_data.create_annotated_image()

                # Add geometry results to test their serialization
                for geometry in test_data.generate_geometry_results():
                    GeometryAdapter(geometry).add_to(ima)

                # Add table results to test their serialization
                for table in test_data.generate_table_results():
                    TableAdapter(table).add_to(ima)

                panel.add_object(ima)
                orig_uuid = get_uuid(ima)
                orig_metadata = ima.metadata.copy()
                panel.export_metadata_from_file(fname)
                panel.delete_metadata()

                assert get_uuid(ima) == orig_uuid
                assert panel.objmodel[orig_uuid] is ima
                assert ima.metadata["__number"] == panel.objmodel.get_number(ima)
                expected_options = {
                    f"__{name}" for name in ima.get_metadata_options_defaults()
                }
                expected_options.update({"__uuid", "__number"})
                assert set(ima.metadata) == expected_options

                panel.import_metadata_from_file(fname)
                execenv.print("Check metadata export <--> import features:")
                helpers.compare_metadata(
                    orig_metadata, ima.metadata, raise_on_diff=True
                )


def test_metadata_import_keeps_identity():
    """Metadata exported from an object, imported into another: the latter keeps
    its own UUID and number, so that provenance and titles still refer to it."""
    with execenv.context(unattended=True):
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            fname = osp.join(tmpdir, "other.dlabmeta")
            with datalab_test_app_context() as win:
                panel = win.signalpanel
                for _index in range(2):
                    panel.new_object(edit=False)
                first, second = panel.objmodel.get_all_objects()
                first.metadata["custom"] = "value"
                panel.objview.set_current_object(first)
                panel.export_metadata_from_file(fname)
                panel.objview.set_current_object(second)
                uuid, number = get_uuid(second), second.metadata["__number"]
                panel.import_metadata_from_file(fname)
                assert second.metadata["custom"] == "value"
                assert (get_uuid(second), second.metadata["__number"]) == (
                    uuid,
                    number,
                )
                assert get_uuid(first) != uuid
                assert panel.objmodel[uuid] is second


if __name__ == "__main__":
    test_metadata_io_unit()
