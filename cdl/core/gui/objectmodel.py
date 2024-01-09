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

    def get_group_from_number(self, number: int) -> ObjectGroup:
        """Return group from its number.

        Args:
            number: group number (unique in model)

        Returns:
            Group

        Raises:
            IndexError: if group with number not found
        """
        for group in self._groups:
            if group.number == number:
                return group
        raise IndexError(f"Group with number {number} not found")

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

    def get_object_from_number(self, number: int) -> SignalObj | ImageObj:
        """Return object from its number.

        Args:
            number: object number (unique in model)

        Returns:
            Object

        Raises:
            IndexError: if object with number not found
        """
        for obj in self._objects.values():
            if obj.number == number:
                return obj
        raise IndexError(f"Object with number {number} not found")

    def get_objects(self, uuids: list[str]) -> list[SignalObj | ImageObj]:
        """Return objects with uuids"""
        return [self._objects[uuid] for uuid in uuids]

    def get_object_ids(self) -> list[str]:
        """Return object ids, in order of appearance in groups"""
        return [obj.uuid for obj in self.get_all_objects()]

    def get_group_titles_with_object_infos(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Tuple: groups titles, lists of inner objects uuids and titles
        """
        grp_titles = []
        obj_uuids = []
        obj_titles = []
        for group in self._groups:
            grp_titles.append(group.title)
            obj_uuids.append(group.get_object_ids())
            obj_titles.append([obj.title for obj in group])
        return grp_titles, obj_uuids, obj_titles

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

    def __get_group_object_mapping_to_shortid(self) -> dict[str, str]:
        """Return dictionary mapping group/object uuids to their short ID"""
        mapping = {}
        for group in self._groups:
            mapping[group.uuid] = group.short_id
            for obj in group:
                mapping[obj.uuid] = obj.short_id
        return mapping

    def __replace_short_ids_by_uuids_in_titles(self) -> None:
        """Replace short IDs by uuids in titles

        This method is called before reorganizing groups or objects. It replaces
        the short IDs in titles by the uuids. This is needed because the short IDs
        are used to reflect in the title the operation performed on the object/group,
        e.g. "fft(s001)" or "g001 + g002". But when reorganizing groups or objects,
        the short IDs may change, so we need to replace them by the uuids, which
        are stable. Once the reorganization is done, we will replace the uuids by the
        new short IDs thanks to the `__replace_uuids_by_short_ids_in_titles` method.
        """
        mapping = self.__get_group_object_mapping_to_shortid()
        for obj in self._objects.values():
            for obj_uuid, short_id in mapping.items():
                obj.title = obj.title.replace(short_id, obj_uuid)
        for group in self._groups:
            for grp_uuid, short_id in mapping.items():
                group.title = group.title.replace(short_id, grp_uuid)

    def __replace_uuids_by_short_ids_in_titles(self) -> None:
        """Replace uuids by short IDs in titles

        This method is called after reorganizing groups or objects. It replaces
        the uuids in titles by the short IDs.
        """
        mapping = self.__get_group_object_mapping_to_shortid()
        for obj in self._objects.values():
            for obj_uuid, short_id in mapping.items():
                obj.title = obj.title.replace(obj_uuid, short_id)
        for group in self._groups:
            for grp_uuid, short_id in mapping.items():
                group.title = group.title.replace(grp_uuid, short_id)
        # Replace remaining UUIDs with f"{obj.PREFIX}xxx"
        # (this may happen if groups or objects were removed in the meantime):
        pattern = re.compile(r"\b[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}\b")
        for obj in self._objects.values():
            obj.title = pattern.sub(f"{obj.PREFIX}xxx", obj.title)
        for group in self._groups:
            for obj in group:
                obj.title = pattern.sub(f"{obj.PREFIX}xxx", obj.title)

    def reorder_groups(self, group_ids: list[str]) -> None:
        """Reorder groups.

        Args:
            group_ids: list of group uuids
        """
        # Replace short IDs by uuids in titles:
        self.__replace_short_ids_by_uuids_in_titles()
        # Reordering groups:
        self._groups = [self.get_group(group_id) for group_id in group_ids]
        # Reset short IDs:
        self.reset_short_ids()
        # Replace uuids by short IDs in titles:
        self.__replace_uuids_by_short_ids_in_titles()

    def reorder_objects(self, obj_ids: dict[str, list[str]]) -> None:
        """Reorder objects in groups.

        Args:
            obj_ids: dict of group uuids and list of object uuids
        """
        # Replace short IDs by uuids in titles:
        self.__replace_short_ids_by_uuids_in_titles()
        # Reordering objects in groups:
        for group_id, obj_uuids in obj_ids.items():
            group = self.get_group(group_id)
            group.clear()
            for obj_uuid in obj_uuids:
                group.append(self._objects[obj_uuid])
        # Reset short IDs:
        self.reset_short_ids()
        # Replace uuids by short IDs in titles:
        self.__replace_uuids_by_short_ids_in_titles()
