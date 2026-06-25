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
    if isinstance(obj, ObjectGroup):
        return f"{obj.prefix}{get_number(obj):03d}"
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


#: Regex matching short IDs as embedded in computation titles
#: (e.g. ``s001``, ``i012``, ``gs003``, ``gi007``).
SHORT_ID_REGEX = re.compile(r"\b(g?[si])(\d{3})\b")

#: Metadata key holding the per-object "deleted source references" registry.
#: This is a public key (no leading underscore) so it is copied with the object
#: (duplication, copy/paste of "other" metadata) and serialized to files.
DELETED_REF_KEY = "deleted_source_refs"

#: Regex matching deleted-source reference tokens embedded in titles
#: (e.g. ``sd001a`` for a deleted signal, ``id001`` for a deleted image,
#: ``gsd001`` for  deleted signal group, ``gid001`` for a deleted image group).
#: The trailing ``d`` and the 3+ digit count make this disjoint from
#: :data:`SHORT_ID_REGEX` (live short IDs always have exactly 3 digits and no
#: ``d`` separator).
DELETED_REF_REGEX = re.compile(r"\b(gs|gi|s|i)d(\d{3,})\b")


def find_short_ids_in_title(title: str) -> list[tuple[int, int, str]]:
    """Return a list of ``(start, end, short_id)`` tuples for every short ID
    occurrence found in ``title``.

    Args:
        title: title string to scan

    Returns:
        List of ``(start, end, short_id)`` tuples, sorted by ``start``.
    """
    return [(m.start(), m.end(), m.group(0)) for m in SHORT_ID_REGEX.finditer(title)]


def find_deleted_refs_in_title(title: str) -> list[tuple[int, int, str]]:
    """Return a list of ``(start, end, token)`` tuples for every deleted-source
    reference token found in ``title`` (e.g. ``sd001``, ``id002``, ``gsd001``).

    Args:
        title: title string to scan

    Returns:
        List of ``(start, end, token)`` tuples, sorted by ``start``.
    """
    return [(m.start(), m.end(), m.group(0)) for m in DELETED_REF_REGEX.finditer(title)]


def get_deleted_ref_prefix(obj_or_group: SignalObj | ImageObj | ObjectGroup) -> str:
    """Return the deleted-reference token prefix for an object or group.

    Args:
        obj_or_group: object or group that is about to be deleted

    Returns:
        Token prefix: ``"sd"`` (signal), ``"id"`` (image), ``"gsd"`` (signal
        group) or ``"gid"`` (image group).
    """
    if isinstance(obj_or_group, ObjectGroup):
        return f"{obj_or_group.prefix}d"  # "gs" -> "gsd", "gi" -> "gid"
    return f"{obj_or_group.PREFIX}d"  # "s" -> "sd", "i" -> "id"


class ObjectGroup:
    """Represents a DataLab object group

    Args:
        title: group title
        model: object model
        prefix: prefix for short ID ("gs" for signal groups, "gi" for image groups)
    """

    def __init__(self, title: str, model: ObjectModel, prefix: str) -> None:
        self.model = model
        self.prefix = prefix  # Instance-specific prefix
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
        # Sibling models (e.g. the image model is a sibling of the signal model).
        # Titles may embed *cross-panel* short IDs (e.g. a signal extracted from
        # an image keeps a reference like ``i001`` in its title). Such references
        # are kept in sync when *this* model is renumbered, so they keep pointing
        # to the same physical source after a reorder.
        self._sibling_models: list[ObjectModel] = []
        # Per-group "deleted source references" registries, keyed by group uuid.
        # Groups have no metadata dict (unlike SignalObj/ImageObj which store
        # their registry in ``metadata[DELETED_REF_KEY]``), so their registries
        # live here. Each registry maps a deleted-reference token (e.g.
        # ``"sd001"``) to the canonical title of the deleted source.
        self._group_deleted_refs: dict[str, dict[str, str]] = {}

    def add_sibling_model(self, model: ObjectModel) -> None:
        """Register a sibling model for cross-panel reference synchronization.

        Titles may embed cross-panel short IDs (e.g. a signal extracted from an
        image references that image as ``i001``). When *this* model is renumbered
        (e.g. on reorder), such references in the sibling's titles are updated so
        they keep pointing to the same physical source.

        Args:
            model: sibling object model (e.g. the image model for the signal one)
        """
        if model is not self and model not in self._sibling_models:
            self._sibling_models.append(model)

    def __iter_sibling_titled_items(
        self,
    ) -> Iterator[SignalObj | ImageObj | ObjectGroup]:
        """Iterate over every titled item (object or group) of sibling models.

        Yields:
            Sibling objects and groups whose titles may embed this model's short
            IDs.
        """
        for sibling in self._sibling_models:
            yield from sibling._objects.values()  # pylint: disable=protected-access
            yield from sibling._groups  # pylint: disable=protected-access

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
        """Check if an object with the given UUID exists in the model

        Args:
            uuid: UUID string to check

        Returns:
            True if an object with this UUID exists, False otherwise
        """
        return uuid in self._objects

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

    def find_by_short_id(
        self, short_id: str, include_siblings: bool = False
    ) -> SignalObj | ImageObj | ObjectGroup | None:
        """Return the object or group whose short ID matches ``short_id``,
        or ``None`` if no match is found.

        Args:
            short_id: short ID to look up (e.g. ``"s001"``, ``"i012"``,
             ``"gs003"`` or ``"gi007"``).
            include_siblings: if ``True``, also search the sibling models when
             no match is found in this model. Defaults to ``False`` (search
             this model only), which is the contract relied upon by callers
             that need to know which panel owns the match.

        Returns:
            The matching :class:`sigima.SignalObj`, :class:`sigima.ImageObj`
            or :class:`ObjectGroup` instance, or ``None``.
        """
        for group in self._groups:
            if get_short_id(group) == short_id:
                return group
        for obj in self._objects.values():
            if get_short_id(obj) == short_id:
                return obj
        if include_siblings:
            for sibling in self._sibling_models:
                source = sibling.find_by_short_id(short_id)
                if source is not None:
                    return source
        return None

    def __model_owning_group(self, group: ObjectGroup) -> ObjectModel:
        """Return the model (this one or a sibling) whose group list contains
        ``group``. Group deleted-reference registries live in their owning
        model's store, so cross-panel access must be routed to the right model.

        Args:
            group: group to locate

        Returns:
            The owning model (defaults to ``self`` if not found).
        """
        if any(group is grp for grp in self._groups):
            return self
        for sibling in self._sibling_models:
            # pylint: disable=protected-access
            if any(group is grp for grp in sibling._groups):
                return sibling
        return self

    def __get_registry(
        self, obj_or_group: SignalObj | ImageObj | ObjectGroup
    ) -> dict[str, str]:
        """Return the (read-only) deleted-source reference registry for an
        object or group, or an empty dict if none exists.

        Args:
            obj_or_group: object or group whose registry is requested

        Returns:
            Mapping ``token -> frozen canonical title`` (may be empty).
        """
        if isinstance(obj_or_group, ObjectGroup):
            owner = self.__model_owning_group(obj_or_group)
            # pylint: disable=protected-access
            return owner._group_deleted_refs.get(obj_or_group.uuid, {})
        return obj_or_group.metadata.get(DELETED_REF_KEY, {})

    def __ensure_registry(
        self, obj_or_group: SignalObj | ImageObj | ObjectGroup
    ) -> dict[str, str]:
        """Return the deleted-source reference registry for an object or group,
        creating it (in metadata or in the model's group registry store) if it
        does not exist yet.

        Args:
            obj_or_group: object or group whose registry is requested

        Returns:
            The mutable registry dict.
        """
        if isinstance(obj_or_group, ObjectGroup):
            owner = self.__model_owning_group(obj_or_group)
            # pylint: disable=protected-access
            return owner._group_deleted_refs.setdefault(obj_or_group.uuid, {})
        return obj_or_group.metadata.setdefault(DELETED_REF_KEY, {})

    def get_deleted_refs(
        self, obj_or_group: SignalObj | ImageObj | ObjectGroup
    ) -> dict[str, str]:
        """Return a copy of the deleted-source reference registry for an object
        or group (for display/tooltip purposes).

        Args:
            obj_or_group: object or group whose registry is requested

        Returns:
            Mapping ``token -> frozen canonical title`` (may be empty).
        """
        return dict(self.__get_registry(obj_or_group))

    def set_group_deleted_refs(self, group: ObjectGroup, refs: dict[str, str]) -> None:
        """Restore a group's deleted-source reference registry (e.g. on load).

        Group registries cannot live in metadata (groups have none), so they are
        persisted separately and re-injected here after deserialization.

        Args:
            group: group whose registry is restored
            refs: mapping ``token -> frozen canonical title`` (empty clears it)
        """
        if refs:
            self._group_deleted_refs[group.uuid] = dict(refs)
        else:
            self._group_deleted_refs.pop(group.uuid, None)

    def __allocate_deleted_token(
        self, dependent: SignalObj | ImageObj | ObjectGroup, prefix: str
    ) -> str:
        """Allocate the next free deleted-reference token for ``prefix`` in the
        registry of ``dependent`` (numbering starts at 1, per prefix).

        Args:
            dependent: object or group whose registry receives the token
            prefix: token prefix (``"sd"``, ``"id"``, ``"gsd"`` or ``"gid"``)

        Returns:
            A new token, e.g. ``"sd001"``.
        """
        registry = self.__get_registry(dependent)
        pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)$")
        max_n = 0
        for key in registry:
            match = pattern.match(key)
            if match:
                max_n = max(max_n, int(match.group(1)))
        return f"{prefix}{max_n + 1:03d}"

    def __freeze_deleted_object_refs(
        self, deleted: SignalObj | ImageObj | ObjectGroup
    ) -> None:
        """Freeze every reference to ``deleted`` found in the titles (and in the
        deleted-reference registries) of all other objects and groups.

        This must be called while titles are in canonical short-ID form, before
        ``deleted`` is actually removed from the model. Each dependent gets a
        stable per-object token (e.g. ``"sd001"``) substituted in place of the
        soon-to-be-invalid short ID, together with a registry entry mapping that
        token to the canonical title of ``deleted`` (so surviving live sources
        referenced by that title keep resolving dynamically).

        Args:
            deleted: object or group about to be removed from the model
        """
        short_id = get_short_id(deleted)
        frozen_title = deleted.title
        prefix = get_deleted_ref_prefix(deleted)
        token_re = re.compile(r"\b" + re.escape(short_id) + r"\b")
        dependents: list[SignalObj | ImageObj | ObjectGroup] = [
            obj for obj in self._objects.values() if obj is not deleted
        ]
        dependents += [group for group in self._groups if group is not deleted]
        # Cross-panel dependents: a title in the other panel may reference the
        # object being deleted (e.g. a signal extracted from a deleted image),
        # so freeze those references too.
        for sibling in self._sibling_models:
            # pylint: disable=protected-access
            dependents += [
                obj for obj in sibling._objects.values() if obj is not deleted
            ]
            dependents += [group for group in sibling._groups if group is not deleted]
        for dependent in dependents:
            registry = self.__get_registry(dependent)
            in_title = bool(token_re.search(dependent.title))
            in_registry = any(token_re.search(value) for value in registry.values())
            if not (in_title or in_registry):
                continue
            token = self.__allocate_deleted_token(dependent, prefix)
            self.__ensure_registry(dependent)[token] = frozen_title
            if in_title:
                dependent.title = token_re.sub(token, dependent.title)
            if in_registry:
                for key in registry:
                    registry[key] = token_re.sub(token, registry[key])

    def __render(
        self,
        title: str,
        registry: dict[str, str],
        seen: set[str],
    ) -> str:
        """Return ``title`` with embedded references resolved to source titles.

        Two kinds of references are resolved:

        - live short IDs (e.g. ``s001``): replaced by the (recursively rendered)
          title of the matching live object/group, using that source's own
          registry for nested deleted references;
        - deleted-source tokens (e.g. ``sd001``): replaced by the (recursively
          rendered) frozen title stored in ``registry``.

        Unresolved references (missing source, missing registry entry) or cycles
        are left untouched.

        Args:
            title: title string to render
            registry: deleted-reference registry of the object owning ``title``
            seen: set of references already being resolved (cycle guard)

        Returns:
            Rendered title string.
        """
        events: list[tuple[int, int, str, bool]] = []
        for start, end, sid in find_short_ids_in_title(title):
            events.append((start, end, sid, False))
        for start, end, token in find_deleted_refs_in_title(title):
            events.append((start, end, token, True))
        if not events:
            return title
        events.sort()
        parts: list[str] = []
        last = 0
        for start, end, ref, is_deleted in events:
            parts.append(title[last:start])
            if ref in seen:
                parts.append(title[start:end])
            elif is_deleted:
                frozen = registry.get(ref)
                if frozen is None:
                    parts.append(title[start:end])
                else:
                    rendered = self.__render(frozen, registry, seen | {ref})
                    # Keep the short reference if the source title is empty, so
                    # the reference stays visible (rather than rendering nothing):
                    parts.append(rendered if rendered.strip() else title[start:end])
            else:
                source = self.find_by_short_id(ref, include_siblings=True)
                if source is None:
                    parts.append(title[start:end])
                else:
                    rendered = self.__render(
                        source.title, self.__get_registry(source), seen | {ref}
                    )
                    # Keep the short reference if the source title is empty, so
                    # the reference stays visible (rather than rendering nothing):
                    parts.append(rendered if rendered.strip() else title[start:end])
            last = end
        parts.append(title[last:])
        return "".join(parts)

    def get_display_title(
        self, obj_or_group: SignalObj | ImageObj | ObjectGroup, use_titles: bool
    ) -> str:
        """Return the title to display for ``obj_or_group``.

        The stored title is always kept in canonical form: live source
        references use short IDs (e.g. ``"fft(s001)"``), and references to
        deleted sources use stable per-object tokens (e.g. ``"fft(sd001)"``).
        This method optionally renders it for display by replacing both kinds of
        references with the corresponding source titles, without altering the
        stored title.

        Args:
            obj_or_group: object or group whose display title is requested
            use_titles: if True, replace embedded references by source titles;
             if False, return the stored (short-ID/token) title unchanged

        Returns:
            Title string to display.
        """
        if not use_titles:
            return obj_or_group.title
        registry = self.__get_registry(obj_or_group)
        return self.__render(obj_or_group.title, registry, set())

    @staticmethod
    def __raise_ambiguous_title(
        title: str,
        matches: list[SignalObj | ImageObj | ObjectGroup],
        label: str,
    ) -> None:
        """Raise a ``ValueError`` describing an ambiguous title lookup.

        Args:
            title: requested title that matched more than one item
            matches: the matching objects or groups
            label: human-readable kind, e.g. ``"Object"`` or ``"Group"``

        Raises:
            ValueError: always, listing the short IDs to disambiguate
        """
        short_ids = ", ".join(get_short_id(match) for match in matches)
        raise ValueError(
            f"{label} title '{title}' is ambiguous: it matches "
            f"{len(matches)} items ({short_ids}). Use the short ID to "
            f"reference one unambiguously."
        )

    def __find_by_title(
        self,
        candidates: list[SignalObj | ImageObj | ObjectGroup],
        title: str,
        label: str,
    ) -> SignalObj | ImageObj | ObjectGroup:
        """Return the single object or group matching ``title``.

        The stored (canonical short-ID) title takes precedence: if exactly one
        candidate has this stored title, it is returned. Otherwise the lookup
        also accepts the title as shown in the GUI (source short IDs replaced by
        source titles). In both stages, more than one match is an error: titles
        are not unique identifiers (short IDs are), so an ambiguous lookup is
        rejected rather than silently returning an arbitrary match.

        Args:
            candidates: objects or groups to search
            title: stored title or title shown in the GUI
            label: human-readable kind, e.g. ``"Object"`` or ``"Group"``

        Returns:
            The single matching object or group.

        Raises:
            KeyError: if no candidate matches the title
            ValueError: if more than one candidate matches the title
        """
        stored = [c for c in candidates if c.title == title]
        if len(stored) == 1:
            return stored[0]
        if len(stored) > 1:
            self.__raise_ambiguous_title(title, stored, label)
        rendered = [c for c in candidates if self.get_display_title(c, True) == title]
        if len(rendered) == 1:
            return rendered[0]
        if len(rendered) > 1:
            self.__raise_ambiguous_title(title, rendered, label)
        raise KeyError(f"{label} with title '{title}' not found")

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

        As for :meth:`get_object_from_title`, the lookup first tries an exact
        match against the stored title, then against the title shown in the GUI
        (where embedded source short IDs are replaced by source titles). This
        lets macros reference groups by the title displayed in the GUI,
        regardless of the current display mode. Because titles are not unique
        identifiers, a title matching several groups is rejected.

        Args:
            title: group title (stored title or title shown in the GUI)

        Returns:
            Group

        Raises:
            KeyError: if no group with title found
            ValueError: if the title is ambiguous (matches several groups)
        """
        return self.__find_by_title(list(self._groups), title, "Group")

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
        """Add group to model

        Args:
            title: group title

        Returns:
            Created group object
        """
        group = ObjectGroup(title, self, self._group_prefix)
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
        # Freeze references to the group (and to its orphan objects) into stable
        # deleted-reference tokens, while titles are still in canonical form:
        self.__freeze_deleted_object_refs(group)
        orphans = [
            obj
            for obj in group
            if not any(obj in other for other in self._groups if other is not group)
        ]
        for obj in orphans:
            self.__freeze_deleted_object_refs(obj)
        self.replace_short_ids_by_uuids_in_titles()
        self._groups.remove(group)
        self._group_deleted_refs.pop(group.uuid, None)
        for obj in orphans:
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
        # Freeze references to this object into stable deleted-reference tokens,
        # while titles are still in canonical short-ID form:
        self.__freeze_deleted_object_refs(obj)
        # Keep references to *surviving* objects valid across the renumbering
        # (intra-panel and cross-panel), like remove_group does:
        self.replace_short_ids_by_uuids_in_titles()
        for group in self._groups:
            if obj in group:
                group.remove(obj)
        del self._objects[get_uuid(obj)]
        self.reset_short_ids()
        self.replace_uuids_by_short_ids_in_titles()

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

        The lookup first tries an exact match against the stored title. If that
        fails, it also accepts the title as shown in the GUI, where embedded
        source short IDs are replaced by source titles (e.g. ``"fft(My signal)"``
        matches an object whose stored title is ``"fft(s001)"`` when ``s001`` is
        titled ``"My signal"``). This lets macros and scripts reference objects
        by the title displayed in the GUI, regardless of the current result-title
        display mode. Because titles are not unique identifiers (short IDs are), a
        title matching several objects is rejected instead of returning one at
        random.

        Args:
            title: object title (stored title or title shown in the GUI)

        Returns:
            object with title

        Raises:
            KeyError: if no object with title found
            ValueError: if the title is ambiguous (matches several objects)
        """
        return self.__find_by_title(list(self._objects.values()), title, "Object")

    def __get_group_object_mapping_to_shortid(self) -> dict[str, str]:
        """Return dictionary mapping group/object uuids to their short ID"""
        mapping = {}
        for group in self._groups:
            mapping[get_uuid(group)] = get_short_id(group)
            for obj in group:
                mapping[get_uuid(obj)] = get_short_id(obj)
        return mapping

    def __iter_registries(
        self, other_objects: tuple[SignalObj | ImageObj] | None = None
    ) -> Iterator[dict[str, str]]:
        """Iterate over every non-empty deleted-reference registry in the model.

        Frozen titles stored as registry values may themselves contain live
        short IDs (when a deleted object derived from still-living sources). Those
        short IDs must be kept in sync through the same uuid swap as regular
        titles, so the swap methods process registries via this iterator.

        Args:
            other_objects: extra objects to consider (e.g. an object being added)

        Yields:
            Mutable registry dicts (``token -> frozen canonical title``).
        """
        objs = list(self._objects.values())
        if other_objects is not None:
            objs += list(other_objects)
        for obj in objs:
            registry = obj.metadata.get(DELETED_REF_KEY)
            if registry:
                yield registry
        for group in self._groups:
            registry = self._group_deleted_refs.get(group.uuid)
            if registry:
                yield registry

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
        # Cross-panel references: sibling titles may embed *this* model's short
        # IDs (e.g. a signal extracted from an image references it as "i001").
        # Freeze them to stable uuids too, so they survive this model's renumbering:
        for sib_item in self.__iter_sibling_titled_items():
            for obj_uuid, short_id in mapping.items():
                sib_item.title = sib_item.title.replace(short_id, obj_uuid)
        # Keep live short IDs embedded in frozen (deleted-reference) titles in sync:
        for registry in self.__iter_registries(other_objects):
            for token in registry:
                for obj_uuid, short_id in mapping.items():
                    registry[token] = registry[token].replace(short_id, obj_uuid)

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
        # Cross-panel references: restore *this* model's (new) short IDs in
        # sibling titles, so a reference like "i001" follows the physical source
        # after this model has been renumbered:
        for sib_item in self.__iter_sibling_titled_items():
            for obj_uuid, short_id in mapping.items():
                sib_item.title = sib_item.title.replace(obj_uuid, short_id)
        # Restore live short IDs embedded in frozen (deleted-reference) titles:
        for registry in self.__iter_registries():
            for token in registry:
                for obj_uuid, short_id in mapping.items():
                    registry[token] = registry[token].replace(obj_uuid, short_id)
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
