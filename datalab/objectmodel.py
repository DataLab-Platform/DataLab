# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Object model
============

The :mod:`datalab.objectmodel` module defines the object data model used by the
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
from collections.abc import Callable, Iterator, Sequence
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
    """Get object UUID.

    For data objects (signals/images), the UUID is stored in metadata under
    the ``__uuid`` option. It is materialized on first access (via
    :func:`set_uuid`) so that the returned value is stable across calls and
    survives serialization.
    """
    if isinstance(obj, ObjectGroup):
        return obj.uuid
    uuid = obj.metadata.get("__uuid")
    if not uuid:
        set_uuid(obj)
        uuid = obj.metadata["__uuid"]
    return uuid


def set_uuid(obj: SignalObj | ImageObj | ObjectGroup) -> None:
    """Set object UUID"""
    if isinstance(obj, ObjectGroup):
        obj.uuid = str(uuid4())
    else:
        obj.set_metadata_option("uuid", str(uuid4()))


def get_short_id(obj: SignalObj | ImageObj | ObjectGroup) -> str:
    """Short object ID"""
    if isinstance(obj, ObjectGroup):
        return f"{obj.prefix}{get_number(obj):03d}"
    return f"{obj.PREFIX}{get_number(obj):03d}"


UUID_DISPLAY_LENGTH = 8
UUID_REGEX = re.compile(
    r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", re.IGNORECASE
)
#: Title references: ``1a2b3c4d`` for objects, ``g1a2b3c4d`` for groups
TITLE_REFERENCE_REGEX = re.compile(r"\bg?[0-9a-f]{8}\b")


def get_short_uuid(obj_or_uuid: SignalObj | ImageObj | ObjectGroup | str) -> str:
    """Return the eight-character display prefix of an UUID."""
    uuid = obj_or_uuid if isinstance(obj_or_uuid, str) else get_uuid(obj_or_uuid)
    return uuid[:UUID_DISPLAY_LENGTH]


def get_title_reference(obj: SignalObj | ImageObj | ObjectGroup) -> str:
    """Return the reference embedded in titles: ``<UUID8>`` or ``g<UUID8>``."""
    prefix = "g" if isinstance(obj, ObjectGroup) else ""
    return f"{prefix}{get_short_uuid(obj)}"


def get_uuid_display_id(obj: SignalObj | ImageObj | ObjectGroup) -> str:
    """Return an object's own display identity in ``#<reference>`` form."""
    return f"#{get_title_reference(obj)}"


def find_title_references(title: str) -> list[tuple[int, int, str]]:
    """Return object and group references found in a title.

    Args:
        title: Title string to scan

    Returns:
        List of ``(start, end, reference)`` tuples sorted by ``start``.
    """
    return [
        (match.start(), match.end(), match.group(0))
        for match in TITLE_REFERENCE_REGEX.finditer(title)
    ]


def render_title(title: str, resolver: Callable[[str], str | None]) -> str:
    """Replace the title references that ``resolver`` resolves.

    Args:
        title: Title containing object and group references
        resolver: Callback returning a replacement for a reference, or None

    Returns:
        Title with resolved references replaced, others left unchanged.
    """
    for start, end, reference in reversed(find_title_references(title)):
        replacement = resolver(reference)
        if replacement:
            title = title[:start] + replacement + title[end:]
    return title


def resolve_title_references(
    title: str,
    source_title: Callable[[str], str | None],
    _visited: frozenset[str] = frozenset(),
) -> str:
    """Replace references by source titles, resolving nested references too.

    Args:
        title: Title containing object and group references
        source_title: Callback returning the title of a referenced source, or None

    Returns:
        Title with every resolvable reference replaced, at any nesting depth.
    """

    def resolve(reference: str) -> str | None:
        if reference in _visited:
            return None
        source = source_title(reference)
        if source is None:
            return None
        return resolve_title_references(source, source_title, _visited | {reference})

    return render_title(title, resolve)


def patch_title_with_ids(
    dst_obj: SignalObj | ImageObj,
    src_objs: list[SignalObj] | list[ImageObj],
) -> None:
    """Patch an object title with source references.

    Destination object's title has been set to a string containing placeholders
    (e.g. "integral({0})"), by `sigima` computation function using a generic mecanism
    (see `sigima.base.dst_1_to_1` for example).

    Args:
        dst_obj: destination object
        src_objs: list of source objects
    """
    ids = [get_title_reference(obj) for obj in src_objs]
    title = dst_obj.title
    assert isinstance(title, str), "Title must be a string"
    try:
        dst_obj.title = title.format(*ids)
    except IndexError as exc:
        raise ValueError(
            f"Not enough source objects to fill title placeholders: {title}"
        ) from exc


def remap_title_references(title: str, reference_remap: dict[str, str]) -> str:
    """Replace the title references listed in ``reference_remap``."""
    return render_title(title, reference_remap.get)


class ObjectGroup:
    """Represents a DataLab object group

    Args:
        title: group title
        model: object model
        prefix: prefix for short ID ("gs" for signal groups, "gi" for image groups)
        group_uuid: optional group UUID. If None, a new UUID is generated.
    """

    def __init__(
        self,
        title: str,
        model: ObjectModel,
        prefix: str,
        group_uuid: str | None = None,
    ) -> None:
        self.model = model
        self.prefix = prefix  # Instance-specific prefix
        self.uuid: str = group_uuid or str(uuid4())  # Group uuid
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
        self.__objects.insert(index, get_uuid(obj))
        self.model.reset_short_ids()

    def remove(self, obj: SignalObj | ImageObj) -> None:
        """Remove object from group"""
        self.__objects.remove(get_uuid(obj))
        self.model.reset_short_ids()

    def clear(self) -> None:
        """Clear group"""
        self.__objects.clear()

    def get_objects(self) -> list[SignalObj | ImageObj]:
        """Return objects in group"""
        return self.model.get_objects(self.__objects)

    def get_object_ids(self) -> list[str]:
        """Return object ids in group"""
        return self.__objects.copy()


class ObjectModel:
    """Represents a DataLab object model (groups of signals/images)"""

    def __init__(self, group_prefix: str) -> None:
        """Initialize object model

        Args:
            group_prefix: prefix for group short IDs ("gs" for signal, "gi" for image)
        """
        self._group_prefix = group_prefix
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

    def has_uuid(self, uuid: str) -> bool:
        """Check if an object or group with the given UUID exists in the model.

        Args:
            uuid: UUID string to check

        Returns:
            True if an object or group with this UUID exists, False otherwise
        """
        return uuid in self._objects or any(
            group.uuid == uuid for group in self._groups
        )

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

    def find_by_title_reference(
        self, reference: str
    ) -> list[SignalObj | ImageObj | ObjectGroup]:
        """Return the objects or groups matching a title reference.

        Args:
            reference: ``<UUID8>`` (object) or ``g<UUID8>`` (group) reference

        Returns:
            Matching objects or groups (more than one only on prefix collision).
        """
        if reference.startswith("g"):
            return [g for g in self._groups if get_title_reference(g) == reference]
        # Objects are keyed by their UUID, whose prefix is their title reference
        return [
            obj
            for uuid, obj in self._objects.items()
            if uuid[:UUID_DISPLAY_LENGTH] == reference
        ]

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
            ValueError: if multiple groups have the same title
        """
        matches = [group for group in self._groups if group.title == title]
        if not matches:
            raise KeyError(f"Group with title '{title}' not found")
        if len(matches) > 1:
            match_ids = ", ".join(get_uuid_display_id(group) for group in matches)
            raise ValueError(
                f"Group title '{title}' is ambiguous; matches: {match_ids}"
            )
        return matches[0]

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

    def add_group(self, title: str, group_uuid: str | None = None) -> ObjectGroup:
        """Add group to model

        Args:
            title: group title
            group_uuid: optional group UUID. If None, a new UUID is generated.

        Returns:
            Created group object
        """
        group = ObjectGroup(title, self, self._group_prefix, group_uuid)
        self._groups.append(group)
        self.reset_short_ids()
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

    def add_object(self, obj: SignalObj | ImageObj, group_id: str) -> None:
        """Add object to model"""
        self.add_objects((obj,), group_id)

    def __numbering_is_current(self) -> bool:
        """Return True if groups and objects hold the numbers of their positions"""
        onb = 1
        for gnb, group in enumerate(self._groups, start=1):
            if group.number != gnb:
                return False
            for obj in group:
                if obj.metadata.get("__number") != onb:
                    return False
                onb += 1
        return True

    def add_objects(
        self,
        objects: Sequence[SignalObj | ImageObj],
        group_id: str,
    ) -> bool:
        """Add objects to the model atomically.

        Args:
            objects: Objects to add, in insertion order
            group_id: UUID of the target group

        Returns:
            True if the short IDs of objects or groups already in the model
            changed; False when the objects were added after every existing
            object (the usual case), which leaves them unchanged.

        Raises:
            KeyError: If the target group does not exist
            ValueError: If an object UUID is duplicated or already exists
        """
        objects = tuple(objects)
        if not objects:
            return False
        group = self.get_group(group_id)
        object_ids = [get_uuid(obj) for obj in objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("The same object cannot be added twice")
        existing_ids = set(self._objects).intersection(object_ids)
        if existing_ids:
            raise ValueError(
                f"Object UUID already exists in model: {sorted(existing_ids)[0]}"
            )
        group_index = self._groups.index(group)
        appended = (
            all(len(other) == 0 for other in self._groups[group_index + 1 :])
            and self.__numbering_is_current()
        )
        number = sum(len(other) for other in self._groups)
        for obj, obj_uuid in zip(objects, object_ids):
            self._objects[obj_uuid] = obj
            group.append(obj)
        if appended:
            for obj in objects:
                number += 1
                set_number(obj, number)
        else:
            self.reset_short_ids()
        return not appended

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
            ValueError: if multiple objects have the same title
        """
        matches = [obj for obj in self._objects.values() if obj.title == title]
        if not matches:
            raise KeyError(f"Object with title '{title}' not found")
        if len(matches) > 1:
            match_ids = ", ".join(get_uuid_display_id(obj) for obj in matches)
            raise ValueError(
                f"Object title '{title}' is ambiguous; matches: {match_ids}"
            )
        return matches[0]

    def get_title_reference_index(
        self,
    ) -> dict[str, list[SignalObj | ImageObj | ObjectGroup]]:
        """Return a dictionary mapping title references to groups and objects.

        Each value is the list :meth:`find_by_title_reference` returns for the
        reference. The index must be built again after objects or groups are
        added or removed.
        """
        index: dict[str, list[SignalObj | ImageObj | ObjectGroup]] = {}
        for group in self._groups:
            index.setdefault(get_title_reference(group), []).append(group)
        for uuid, obj in self._objects.items():
            index.setdefault(uuid[:UUID_DISPLAY_LENGTH], []).append(obj)
        return index

    def reorder_groups(self, group_ids: list[str]) -> None:
        """Reorder groups.

        Args:
            group_ids: list of group uuids
        """
        self._groups = [self.get_group(group_id) for group_id in group_ids]
        self.reset_short_ids()

    def reorder_objects(self, obj_ids: dict[str, list[str]]) -> None:
        """Reorder objects in groups.

        Args:
            obj_ids: dict of group uuids and list of object uuids
        """
        for group_id, obj_uuids in obj_ids.items():
            group = self.get_group(group_id)
            group.clear()
            for obj_uuid in obj_uuids:
                group.append(self._objects[obj_uuid])
        self.reset_short_ids()
