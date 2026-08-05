# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration option fields
-----------------------------------

The generic option field types (config path, working directory, font, format
string, DataSet) live in :mod:`sigimax.config`. Only the DataLab compatibility
layer remains here:

- :class:`DataSetOptionField`: remaps the module path of DataSet options
  persisted by DataLab <= 1.2, when ``datalab.config`` was a module rather than
  a package.
"""

from __future__ import annotations

import json

from sigimax.config import DataSetOptionField as _BaseDataSetOptionField


class DataSetOptionField(_BaseDataSetOptionField):
    """DataSet option field handling the DataLab <= 1.2 module path.

    Configurations written before ``datalab.config`` became a package refer to
    the DataSet classes as ``datalab.config.<ClassName>``; they now live in
    ``datalab.config.config``.
    """

    def from_json(self, json_str: str) -> None:
        """Deserialize a DataSet from a JSON string, remapping legacy modules.

        Args:
            json_str: The JSON string to deserialize.
        """
        data = json.loads(json_str)
        if data.get("class_module") == "datalab.config":
            data["class_module"] = "datalab.config.config"
            json_str = json.dumps(data)
        super().from_json(json_str)
