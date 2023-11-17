# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Object (signal/image/group) data model
--------------------------------------

This module defines the object data model used by the GUI to store signals,
images and groups.

The model is based on a hierarchical tree of objects, with two levels:

- The top level is a list of groups (ObjectGroup instances)
- The second level is a list of objects (SignalObj or ImageObj instances)

The model is implemented by the ObjectModel class.

The ObjectGroup class represents a group of objects. It is a container for
SignalObj and ImageObj instances.

The ObjectModel class is a container for ObjectGroup instances, as well as
a container for SignalObj and ImageObj instances.

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

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj


def fix_titles(
    objlist: list[ObjectGroup | SignalObj | ImageObj],
    obj: ObjectGroup | SignalObj | ImageObj,
    operation: str,
) -> None:
    """Fix all object/group titles before adding or removing an object/group

    Args:
        objlist (list[ObjectGroup | SignalObj | ImageObj]): list of objects/groups
        obj (ObjectGroup | SignalObj | ImageObj): object/group to be added or removed
        operation (str): operation to be performed ("add" or "remove")
    """
    assert len(objlist) > 0
    assert operation in ("add", "remove")
    sign = 1 if operation == "add" else -1
    onb = obj.number
    pfx = obj.PREFIX
    oname = f"{pfx}%03d"
    for obj_i in objlist:
        for match in re.finditer(pfx + "[0-9]{3}", obj_i.title):
            before = match.group()
            i_match = int(before[1:])
            if sign == -1 and i_match == onb:
                after = f"{pfx}xxx"
            elif (sign == -1 and i_match > onb) or (sign == 1 and i_match >= onb):
                after = oname % (i_match + sign)
            else:
                continue
            obj_i.title = obj_i.title.replace(before, after)


class ObjectGroup:
    """Represents a DataLab object group"""

    PREFIX = "g"

    def __init__(self, title: str, model: ObjectModel) -> None:
        self.model = model
        self.uuid: str = str(uuid4())  # Group uuid
        self.__objects: list[str] = []  # list of object uuids
        self.__title: str = title
        self.__gnb = 0

    @property
    def number(self) -> int:
        """Return group number (used for short ID)"""
        return self.__gnb

    @number.setter
    def number(self, gnb: int):
        """Set group number (used for short ID)"""
        self.__gnb = gnb

    @property
    def short_id(self):
        """Short group ID"""
        return f"{self.PREFIX}{self.__gnb:03d}"

    @property
    def title(self) -> str:
        """Return group title"""
        return self.__title

    @title.setter
    def title(self, title: str) -> None:
        """Set group title"""
        self.__title = title

    def __iter__(self) -> Iterator[SignalObj | ImageObj]:
        """Iterate over objects in group"""
        return iter(self.model.get_objects(self.__objects))

    def __len__(self) -> int:
        """Return number of objects in group"""
        return len(self.__objects)

    def __getitem__(self, index: int) -> SignalObj | ImageObj:
        """Return object at index"""
        return self.model[self.__objects[index]]

    def __contains__(self, obj: SignalObj | ImageObj) -> bool:
        """Return True if obj is in group"""
        return obj.uuid in self.__objects

    def append(self, obj: SignalObj | ImageObj) -> None:
        """Append object to group"""
        self.__objects.append(obj.uuid)

    def insert(self, index: int, obj: SignalObj | ImageObj) -> None:
        """Insert object at index"""
        fix_titles(self.model.get_all_objects(), obj, "add")
        self.__objects.insert(index, obj.uuid)

    def remove(self, obj: SignalObj | ImageObj) -> None:
        """Remove object from group"""
        fix_titles(self.model.get_all_objects(), obj, "remove")
        self.__objects.remove(obj.uuid)

    def clear(self) -> None:
        """Clear group"""
        self.__objects.clear()

    def get_objects(self) -> list[SignalObj | ImageObj]:
        """Return objects in group"""
        return self.model.get_objects(self.__objects)

    def get_object_ids(self) -> list[str]:
        """Return object ids in group"""
        return self.__objects


class ObjectModel:
    """Represents a DataLab object model (groups of signals/images)"""

    def __init__(self) -> None:
        # dict of objects, key is object uuid:
        self._objects: dict[str, SignalObj | ImageObj] = {}
        # list of groups:
        self._groups: list[ObjectGroup] = []

    def reset_short_ids(self) -> None:
        """Reset short IDs (used for object numbering)

        This method is called when an object was removed from a group."""
        gnb = onb = 1
        for group in self._groups:
            group.number = gnb
            gnb += 1
            for obj in group:
                obj.number = onb
                onb += 1

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self._objects)

    def __getitem__(self, uuid: str) -> SignalObj | ImageObj:
        """Return object with uuid"""
        return self._objects[uuid]

    def __iter__(self) -> Iterator[SignalObj | ImageObj]:
        """Iterate over objects"""
        return iter(self._objects.values())

    def __repr__(self) -> str:
        """Return object representation"""
        return repr(self._objects)

    def __str__(self) -> str:
        """Return object string representation"""
        return str(self._objects)

    def __bool__(self) -> bool:
        """Return True if model is not empty"""
        return bool(self._objects)

    def __contains__(self, obj: SignalObj | ImageObj) -> bool:
        """Return True if obj is in model"""
        return obj.uuid in self._objects

    def clear(self) -> None:
        """Clear model"""
        self._objects.clear()
        self._groups.clear()

    def get_all_objects(self) -> list[SignalObj | ImageObj]:
        """Return all objects, in order of appearance in groups"""
        objects = []
        for group in self._groups:
            objects.extend(group.get_objects())
        return objects

    def get_object_or_group(self, uuid: str) -> SignalObj | ImageObj | ObjectGroup:
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

    def get_groups(self, uuids: list[str] | None = None) -> list[ObjectGroup]:
        """Return groups"""
        if uuids is None:
            return self._groups
        return [group for group in self._groups if group.uuid in uuids]

    def add_group(self, title: str) -> ObjectGroup:
        """Add group to model"""
        group = ObjectGroup(title, self)
        gnb = 1
        if self._groups:
            gnb += self._groups[-1].number
        group.number = gnb
        self._groups.append(group)
        return group

    def get_object_group_id(self, obj: SignalObj | ImageObj) -> str | None:
        """Return group id of object"""
        for group in self._groups:
            if obj in group:
                return group.uuid
        return None

    def get_group_object_ids(self, group_id: str) -> list[str]:
        """Return object ids in group"""
        for group in self._groups:
            if group.uuid == group_id:
                return group.get_object_ids()
        raise KeyError(f"Group with uuid '{group_id}' not found")

    def remove_group(self, group: ObjectGroup) -> None:
        """Remove group from model"""
        fix_titles(self.get_groups(), group, "remove")
        self._groups.remove(group)
        for obj in group:
            remove_obj = True
            for other_group in self._groups:
                if obj in other_group:
                    remove_obj = False
                    break
            if remove_obj:
                del self._objects[obj.uuid]
        self.reset_short_ids()

    def add_object(self, obj: SignalObj | ImageObj, group_id: str) -> None:
        """Add object to model"""
        self._objects[obj.uuid] = obj
        onb = 0
        for group in self._groups:
            onb += len(group)
            if group.uuid == group_id:
                obj.number = onb + 1
                group.append(obj)
                break
        else:
            raise KeyError(f"Group with uuid '{group_id}' not found")
        self.reset_short_ids()

    def remove_object(self, obj: SignalObj | ImageObj) -> None:
        """Remove object from model"""
        for group in self._groups:
            if obj in group:
                group.remove(obj)
        del self._objects[obj.uuid]
        self.reset_short_ids()

    def get_object(self, index: int, group_index: int = 0) -> SignalObj | ImageObj:
        """Return object with index.

        Args:
            index (int): object index
            group_index (int | None): group index. Defaults to 0.

        Returns:
            object with index

        Raises:
            IndexError: if object with index not found
        """
        try:
            return self._groups[group_index][index]
        except IndexError as exc:
            raise IndexError(
                f"Object with index {index} (group {group_index}) not found"
            ) from exc

    def get_objects(self, uuids: list[str]) -> list[SignalObj | ImageObj]:
        """Return objects with uuids"""
        return [self._objects[uuid] for uuid in uuids]

    def get_object_ids(self) -> list[str]:
        """Return object ids, in order of appearance in groups"""
        return [obj.uuid for obj in self.get_all_objects()]

    def get_object_titles(self) -> list[str]:
        """Return object titles, in order of appearance in groups"""
        return [obj.title for obj in self.get_all_objects()]

    def get_object_from_title(self, title: str) -> SignalObj | ImageObj:
        """Return object with title.

        Args:
            title: object title

        Returns:
            object with title

        Raises:
            KeyError: if object with title not found
        """
        for obj in self._objects.values():
            if obj.title == title:
                return obj
        raise KeyError(f"Object with title '{title}' not found")
