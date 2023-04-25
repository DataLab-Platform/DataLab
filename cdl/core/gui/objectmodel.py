# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Object (signal/image/group) data model
--------------------------------------

This module defines the object data model used by the GUI to store signals,
images and groups.

The model is based on a hierarchical tree of objects, with two levels:

- The top level is a list of groups (ObjectGroup instances)
- The second level is a list of objects (SignalParam or ImageParam instances)

The model is implemented by the ObjectModel class.

The ObjectGroup class represents a group of objects. It is a container for
SignalParam and ImageParam instances.

The ObjectModel class is a container for ObjectGroup instances, as well as
a container for SignalParam and ImageParam instances.

.. autosummary::
    :toctree:

    ObjectModel
    ObjectGroup

.. autoclass:: ObjectModel
    :members:

.. autoclass:: ObjectGroup
    :members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import re
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from cdl.core.model.image import ImageParam
    from cdl.core.model.signal import SignalParam


class ObjectGroup:
    """Represents a DataLab object group"""

    PREFIX = "g"

    def __init__(self, title: str, model: ObjectModel) -> None:
        self.model = model
        self.uuid: str = str(uuid4())  # Group uuid
        self._objects: List[str] = []  # List of object uuids
        self._title: str = title
        self.__gnb = 0

    def set_group_number(self, gnb: int):
        """Set group number (used for short ID)"""
        self.__gnb = gnb

    @property
    def short_id(self):
        """Short group ID"""
        return f"{self.PREFIX}{self.__gnb:03d}"

    @property
    def title(self) -> str:
        """Return group title"""
        return self._title

    def set_title(self, title: str) -> None:
        """Set group title"""
        self._title = title

    def __iter__(self) -> Iterator[SignalParam | ImageParam]:
        """Iterate over objects in group"""
        return iter(self.model.get_objects(self._objects))

    def __len__(self) -> int:
        """Return number of objects in group"""
        return len(self._objects)

    def __getitem__(self, index: int) -> SignalParam | ImageParam:
        """Return object at index"""
        return self.model.get_object(self._objects[index])

    def __contains__(self, obj: SignalParam | ImageParam) -> bool:
        """Return True if obj is in group"""
        return obj.uuid in self._objects

    def append(self, obj: SignalParam | ImageParam) -> None:
        """Append object to group"""
        self._objects.append(obj.uuid)

    def insert(self, index: int, obj: SignalParam | ImageParam) -> None:
        """Insert object at index"""
        self._objects.insert(index, obj.uuid)

    def remove(self, obj: SignalParam | ImageParam) -> None:
        """Remove object from group"""
        self._objects.remove(obj.uuid)

    def clear(self) -> None:
        """Clear group"""
        self._objects.clear()

    def get_objects(self) -> List[SignalParam | ImageParam]:
        """Return objects in group"""
        return self.model.get_objects(self._objects)

    def get_object_ids(self) -> List[str]:
        """Return object ids in group"""
        return self._objects


class ObjectModel:
    """Represents a DataLab object model (groups of signals/images)"""

    def __init__(self) -> None:
        # Dict of objects, key is object uuid:
        self._objects: Dict[str, SignalParam | ImageParam] = {}
        # List of groups:
        self._groups: List[ObjectGroup] = []

    def refresh_short_ids(self) -> None:
        """Refresh short ids of objects"""
        gnb = onb = 1
        for group in self._groups:
            group.set_group_number(gnb)
            gnb += 1
            for obj in group:
                obj.set_object_number(onb)
                onb += 1

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self._objects)

    def __getitem__(self, uuid: str) -> SignalParam | ImageParam:
        """Return object with uuid"""
        return self._objects[uuid]

    def __repr__(self) -> str:
        """Return object representation"""
        return repr(self._objects)

    def __str__(self) -> str:
        """Return object string representation"""
        return str(self._objects)

    def __bool__(self) -> bool:
        """Return True if model is not empty"""
        return bool(self._objects)

    def clear(self) -> None:
        """Clear model"""
        self._objects.clear()
        self._groups.clear()

    def get_object_or_group(self, uuid: str) -> SignalParam | ImageParam | ObjectGroup:
        """Return object or group with uuid"""
        if uuid in self._objects:
            return self._objects[uuid]
        for group in self._groups:
            if group.uuid == uuid:
                return group
        raise KeyError(f"Object or group with uuid {uuid} not found")

    def get_group(self, uuid: str) -> ObjectGroup:
        """Return group with uuid"""
        for group in self._groups:
            if group.uuid == uuid:
                return group
        raise KeyError(f"Group with uuid {uuid} not found")

    def get_groups(self, uuids: Optional[List[str]] = None) -> List[ObjectGroup]:
        """Return groups"""
        if uuids is None:
            return self._groups
        return [group for group in self._groups if group.uuid in uuids]

    def add_group(self, title: str) -> ObjectGroup:
        """Add group to model"""
        group = ObjectGroup(title, self)
        self._groups.append(group)
        return group

    def get_object_group_id(self, obj: SignalParam | ImageParam) -> Optional[str]:
        """Return group id of object"""
        for group in self._groups:
            if obj in group:
                return group.uuid
        return None

    def get_group_object_ids(self, group_id: str) -> List[str]:
        """Return object ids in group"""
        for group in self._groups:
            if group.uuid == group_id:
                return group._objects
        raise KeyError(f"Group with uuid '{group_id}' not found")

    def remove_group(self, group: ObjectGroup) -> None:
        """Remove group from model"""
        self._groups.remove(group)
        for obj in group:
            remove_obj = True
            for other_group in self._groups:
                if obj in other_group:
                    remove_obj = False
                    break
            if remove_obj:
                del self._objects[obj.uuid]

    def add_object(self, obj: SignalParam | ImageParam, group_id: str) -> None:
        """Add object to model"""
        self._objects[obj.uuid] = obj
        for group in self._groups:
            if group.uuid == group_id:
                group.append(obj)
                break
        else:
            raise KeyError(f"Group with uuid '{group_id}' not found")

    def remove_object(self, obj: SignalParam | ImageParam) -> None:
        """Remove object from model"""
        del self._objects[obj.uuid]
        for group in self._groups:
            if obj in group:
                group.remove(obj)

    def get_objects(self, uuids: List[str]) -> List[SignalParam | ImageParam]:
        """Return objects with uuids"""
        return [self._objects[uuid] for uuid in uuids]

    def get_object_ids(self) -> List[str]:
        """Return object ids"""
        return list(self._objects.keys())

    def get_object_titles(self) -> List[str]:
        """Return object titles"""
        return [obj.title for obj in self._objects.values()]

    def get_object_by_title(self, title: str) -> SignalParam | ImageParam:
        """Return object with title"""
        for obj in self._objects.values():
            if obj.title == title:
                return obj
        raise KeyError(f"Object with title '{title}' not found")
