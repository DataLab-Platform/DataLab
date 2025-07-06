# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Object model
============

The :mod:`datalab.gui.objectmodel` module defines the object data model used by the
GUI to store signals, images and groups.

The model is based on a hierarchical tree of objects, with two levels:

- The top level is a list of groups (`ObjectGroup` instances)
- The second level is a list of objects (`SignalObj` or `ImageObj` instances)

The model is implemented by the `ObjectModel` class.

Object group
------------

The `ObjectGroup` class represents a group of objects. It is a container for
`SignalObj` and `ImageObj` instances.

.. autoclass:: ObjectGroup

Object model
------------

The `ObjectModel` class is a container for ObjectGroup instances, as well as
a container for `SignalObj` and `ImageObj` instances.

.. autoclass:: ObjectModel
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Callable
from uuid import uuid4

from sigima import ImageObj, SignalObj


def get_number(obj: SignalObj | ImageObj | ObjectGroup) -> int:
    """Get object number from metadata"""
    if isinstance(obj, ObjectGroup):
        return obj.number
    number = obj.get_metadata_option("number")
    assert isinstance(number, int)
    return number


def set_number(obj: SignalObj | ImageObj | ObjectGroup, number: int) -> None:
    """Set object number in metadata"""
    assert isinstance(number, int)
    if isinstance(obj, ObjectGroup):
        obj.number = number
    else:
        obj.set_metadata_option("number", number)


def get_uuid(obj: SignalObj | ImageObj | ObjectGroup) -> str:
    """Get object UUID"""
    if isinstance(obj, ObjectGroup):
        return obj.uuid
    return obj.get_metadata_option("uuid", str(uuid4()))


def set_uuid(obj: SignalObj | ImageObj | ObjectGroup) -> None:
    """Set object UUID"""
    if isinstance(obj, ObjectGroup):
        obj.uuid = str(uuid4())
    else:
        obj.set_metadata_option("uuid", str(uuid4()))


def get_short_id(obj: SignalObj | ImageObj | ObjectGroup) -> str:
    """Short object ID"""
    return f"{obj.PREFIX}{get_number(obj):03d}"


def patch_title_with_ids(
    dst_obj: SignalObj | ImageObj,
    src_objs: list[SignalObj] | list[ImageObj],
    id_func: Callable,
) -> None:
    """Patch object title with short IDs of source objects

    Destination object's title has been set to a string containing placeholders
    (e.g. "integral({0})"), by `sigima` computation function using a generic mecanism
    (see `sigima.base.dst_1_to_1` for example).

    Args:
        dst_obj: destination object
        src_objs: list of source objects
        id_func: function to get ID from object (e.g. `short_id` or `get_uuid`)
    """
    ids = [id_func(obj) for obj in src_objs]
    title = dst_obj.title
    assert isinstance(title, str), "Title must be a string"
    try:
        dst_obj.title = title.format(*ids)
    except IndexError as exc:
        raise ValueError(
            f"Not enough source objects to fill title placeholders: {title}"
        ) from exc


class ObjectGroup:
    """Represents a DataLab object group

    Args:
        title: group title
        model: object model
    """

    PREFIX = "g"

    def __init__(self, title: str, model: ObjectModel) -> None:
        self.model = model
        self.uuid: str = str(uuid4())  # Group uuid
        self.number: int = 0  # Group number (used for short ID)
        self.__objects: list[str] = []  # list of object uuids
        self.__title: str = title

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

    def __getitem__(
        self, index: int | slice
    ) -> SignalObj | ImageObj | list[SignalObj | ImageObj]:
        """Return object at index"""
        if isinstance(index, slice):
            return [
                self.model[self.__objects[i]]
                for i in range(*index.indices(len(self)))
                if i < len(self)
            ]
        return self.model[self.__objects[index]]

    def __contains__(self, obj: SignalObj | ImageObj) -> bool:
        """Return True if obj is in group"""
        return get_uuid(obj) in self.__objects

    def append(self, obj: SignalObj | ImageObj) -> None:
        """Append object to group"""
        self.__objects.append(get_uuid(obj))

    def insert(self, index: int, obj: SignalObj | ImageObj) -> None:
        """Insert object at index"""
        self.model.replace_short_ids_by_uuids_in_titles([obj])
        self.__objects.insert(index, get_uuid(obj))
        self.model.reset_short_ids()
        self.model.replace_uuids_by_short_ids_in_titles()

    def remove(self, obj: SignalObj | ImageObj) -> None:
        """Remove object from group"""
        self.model.replace_short_ids_by_uuids_in_titles()
        self.__objects.remove(get_uuid(obj))
        self.model.reset_short_ids()
        self.model.replace_uuids_by_short_ids_in_titles()

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
            set_number(group, gnb)
            gnb += 1
            for obj in group:
                set_number(obj, onb)
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
        return get_uuid(obj) in self._objects

    def clear(self) -> None:
        """Clear model"""
        self._objects.clear()
        self._groups.clear()

    def get_all_objects(
        self, flatten: bool = True
    ) -> list[SignalObj | ImageObj] | list[list[SignalObj | ImageObj]]:
        """Return all objects, in order of appearance in groups

        Args:
            flatten: if True, return a flat list of objects, otherwise return
             a list of lists (one list per group)

        Returns:
            List of objects in order of appearance in groups.
        """
        objects = []
        for group in self._groups:
            if flatten:
                objects.extend(group.get_objects())
            else:
                objects.append(group.get_objects())
        return objects

    def get_object_or_group(self, uuid: str) -> SignalObj | ImageObj | ObjectGroup:
        """Return object or group with uuid"""
        if uuid in self._objects:
            return self._objects[uuid]
        for group in self._groups:
            if get_uuid(group) == uuid:
                return group
        raise KeyError(f"Object or group with uuid {uuid} not found")

    def get_group(self, uuid: str) -> ObjectGroup:
        """Return group with uuid"""
        for group in self._groups:
            if get_uuid(group) == uuid:
                return group
        raise KeyError(f"Group with uuid {uuid} not found")

    def get_number(self, obj_or_group: SignalObj | ImageObj | ObjectGroup) -> int:
        """Return number of object or group"""
        if isinstance(obj_or_group, ObjectGroup):
            try:
                return self._groups.index(obj_or_group) + 1
            except ValueError as exc:
                raise KeyError(
                    f"Group {get_uuid(obj_or_group)} not found in model"
                ) from exc
        if isinstance(obj_or_group, (SignalObj, ImageObj)):
            objs = self.get_all_objects()
            try:
                return objs.index(obj_or_group) + 1
            except ValueError as exc:
                raise KeyError(
                    f"Object {get_uuid(obj_or_group)} not found in model"
                ) from exc
        raise KeyError(f"Object or group {get_uuid(obj_or_group)} not found in model")

    def get_group_from_number(self, number: int) -> ObjectGroup:
        """Return group from its number.

        Args:
            number: group number (starts with 1)

        Returns:
            Group

        Raises:
            IndexError: if group with number not found
        """
        if number < 1:
            raise IndexError(f"Group number {number} is out of range (must be >= 1)")
        if number > len(self._groups):
            raise IndexError(
                f"Group number {number} is out of range (max is {len(self._groups)})"
            )
        return self._groups[number - 1]

    def get_group_from_title(self, title: str) -> ObjectGroup:
        """Return group from its title.

        Args:
            title: group title

        Returns:
            Group

        Raises:
            KeyError: if group with title not found
        """
        for group in self._groups:
            if group.title == title:
                return group
        raise KeyError(f"Group with title '{title}' not found")

    def get_group_from_object(self, obj: SignalObj | ImageObj) -> ObjectGroup:
        """Return group containing object

        Args:
            obj: object to find group for

        Returns:
            Group

        Raises:
            KeyError: if object not found in any group
        """
        for group in self._groups:
            if obj in group:
                return group
        raise KeyError(f"Object with uuid '{get_uuid(obj)}' not found in any group")

    def get_groups(self, uuids: list[str] | None = None) -> list[ObjectGroup]:
        """Return groups"""
        if uuids is None:
            return self._groups
        return [group for group in self._groups if get_uuid(group) in uuids]

    def add_group(self, title: str) -> ObjectGroup:
        """Add group to model"""
        group = ObjectGroup(title, self)
        self._groups.append(group)
        return group

    def get_object_group_id(self, obj: SignalObj | ImageObj) -> str | None:
        """Return group id of object

        Args:
            obj: object to get group id from

        Returns:
            group id or None if object is not in any group
        """
        try:
            return get_uuid(self.get_group_from_object(obj))
        except KeyError:
            return None

    def get_group_object_ids(self, group_id: str) -> list[str]:
        """Return object ids in group"""
        for group in self._groups:
            if get_uuid(group) == group_id:
                return group.get_object_ids()
        raise KeyError(f"Group with uuid '{group_id}' not found")

    def remove_group(self, group: ObjectGroup) -> None:
        """Remove group from model"""
        self.replace_short_ids_by_uuids_in_titles()
        self._groups.remove(group)
        for obj in group:
            remove_obj = True
            for other_group in self._groups:
                if obj in other_group:
                    remove_obj = False
                    break
            if remove_obj:
                del self._objects[get_uuid(obj)]
        self.reset_short_ids()
        self.replace_uuids_by_short_ids_in_titles()

    def add_object(self, obj: SignalObj | ImageObj, group_id: str) -> None:
        """Add object to model"""
        self.replace_short_ids_by_uuids_in_titles([obj])
        self._objects[get_uuid(obj)] = obj
        onb = 0
        for group in self._groups:
            onb += len(group)
            if get_uuid(group) == group_id:
                set_number(obj, onb + 1)
                group.append(obj)
                break
        else:
            raise KeyError(f"Group with uuid '{group_id}' not found")
        self.reset_short_ids()
        self.replace_uuids_by_short_ids_in_titles()

    def remove_object(self, obj: SignalObj | ImageObj) -> None:
        """Remove object from model"""
        for group in self._groups:
            if obj in group:
                group.remove(obj)
        del self._objects[get_uuid(obj)]
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
        if number < 1:
            raise IndexError(f"Object number {number} is out of range (must be >= 1)")
        objs = self.get_all_objects()
        if number > len(objs):
            raise IndexError(
                f"Object number {number} is out of range (max is {len(objs)})"
            )
        return objs[number - 1]

    def get_objects(self, uuids: list[str]) -> list[SignalObj | ImageObj]:
        """Return objects with uuids"""
        return [self._objects[uuid] for uuid in uuids]

    def get_object_ids(self, flatten: bool = True) -> list[str] | list[list[str]]:
        """Return object ids, in order of appearance in groups

        Args:
            flatten: if True, return a flat list of object ids, otherwise return
             a list of lists (one list per group)

        Returns:
            List of object ids in order of appearance in groups.
        """
        ids = []
        for group in self._groups:
            if flatten:
                ids.extend(group.get_object_ids())
            else:
                ids.append(group.get_object_ids())
        return ids

    def get_group_titles_with_object_info(
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

    def get_object_titles(self, flatten: bool = True) -> list[str] | list[list[str]]:
        """Return object titles, in order of appearance in groups

        Args:
            flatten: if True, return a flat list of object titles, otherwise return
             a list of lists (one list per group)

        Returns:
            List of object titles in order of appearance in groups.
        """
        if flatten:
            return [obj.title for obj in self.get_all_objects()]
        return [[obj.title for obj in group] for group in self._groups]

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
            mapping[get_uuid(group)] = get_short_id(group)
            for obj in group:
                mapping[get_uuid(obj)] = get_short_id(obj)
        return mapping

    def replace_short_ids_by_uuids_in_titles(
        self, other_objects: tuple[SignalObj | ImageObj] | None = None
    ) -> None:
        """Replace short IDs by uuids in titles

        Args:
            other_objects: tuple of other objects to consider for short ID replacement

        .. note::

            This method is called before reorganizing groups or objects. It replaces the
            short IDs in titles by the uuids. This is needed because the short IDs are
            used to reflect in the title the operation performed on the object/group,
            e.g. "fft(s001)" or "g001 + g002". But when reorganizing groups or objects,
            the short IDs may change, so we need to replace them by the uuids, which are
            stable. Once the reorganization is done, we will replace the uuids by the
            new short IDs thanks to the `__replace_uuids_by_short_ids_in_titles` method.
        """
        mapping = self.__get_group_object_mapping_to_shortid()
        objs = self._objects.values()
        if other_objects is not None:
            objs = list(objs) + list(other_objects)
        for obj in objs:
            for obj_uuid, short_id in mapping.items():
                obj.title = obj.title.replace(short_id, obj_uuid)
        for group in self._groups:
            for grp_uuid, short_id in mapping.items():
                group.title = group.title.replace(short_id, grp_uuid)

    def replace_uuids_by_short_ids_in_titles(self) -> None:
        """Replace uuids by short IDs in titles

        .. note::

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
        self.replace_short_ids_by_uuids_in_titles()
        # Reordering groups:
        self._groups = [self.get_group(group_id) for group_id in group_ids]
        # Reset short IDs:
        self.reset_short_ids()
        # Replace uuids by short IDs in titles:
        self.replace_uuids_by_short_ids_in_titles()

    def reorder_objects(self, obj_ids: dict[str, list[str]]) -> None:
        """Reorder objects in groups.

        Args:
            obj_ids: dict of group uuids and list of object uuids
        """
        # Replace short IDs by uuids in titles:
        self.replace_short_ids_by_uuids_in_titles()
        # Reordering objects in groups:
        for group_id, obj_uuids in obj_ids.items():
            group = self.get_group(group_id)
            group.clear()
            for obj_uuid in obj_uuids:
                group.append(self._objects[obj_uuid])
        # Reset short IDs:
        self.reset_short_ids()
        # Replace uuids by short IDs in titles:
        self.replace_uuids_by_short_ids_in_titles()
