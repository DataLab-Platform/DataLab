# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Cross-panel references unit test.

Validates that *cross-panel* short IDs embedded in result titles stay in sync
when the referenced panel is renumbered. A typical case is a signal extracted
from an image: its title keeps a reference to the source image (e.g. ``i001``).
If the images are reordered so that this image becomes ``i002``, the signal
title must follow the physical source and now read ``i002`` (and no longer point
to whatever image happens to be ``i001`` after the reorder).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
from sigima.objects import create_image, create_signal

from datalab.objectmodel import ObjectModel, get_short_id, get_uuid


def _linked_models() -> tuple[ObjectModel, ObjectModel]:
    """Return signal and image object models linked as siblings."""
    smodel = ObjectModel(group_prefix="gs")
    imodel = ObjectModel(group_prefix="gi")
    smodel.add_sibling_model(imodel)
    imodel.add_sibling_model(smodel)
    return smodel, imodel


def test_cross_panel_reference_follows_image_reorder() -> None:
    """A signal title referencing an image follows it when images are reordered."""
    smodel, imodel = _linked_models()

    # Two images: img1 -> i001, img2 -> i002
    igroup = imodel.add_group("Images")
    img1 = create_image("First image", np.zeros((4, 4)))
    img2 = create_image("Second image", np.ones((4, 4)))
    imodel.add_object(img1, get_uuid(igroup))
    imodel.add_object(img2, get_uuid(igroup))
    assert get_short_id(img1) == "i001"
    assert get_short_id(img2) == "i002"

    # A signal extracted from the first image keeps a reference to it ("i001"):
    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0])
    smodel.add_object(sig, get_uuid(sgroup))
    assert sig.title == "average profile(i001)"

    # Reorder images so that img1 (the physical source) becomes i002:
    imodel.reorder_objects({get_uuid(igroup): [get_uuid(img2), get_uuid(img1)]})
    assert get_short_id(img1) == "i002"
    assert get_short_id(img2) == "i001"

    # The signal title must follow the physical source (img1 is now "i002"):
    assert sig.title == "average profile(i002)"


def test_cross_panel_reference_follows_image_group_reorder() -> None:
    """A signal title reference follows its image when image groups are reordered."""
    smodel, imodel = _linked_models()

    # Two image groups, one image each: img_a -> i001 (group A), img_b -> i002 (group B)
    group_a = imodel.add_group("Group A")
    group_b = imodel.add_group("Group B")
    img_a = create_image("Image A", np.zeros((4, 4)))
    img_b = create_image("Image B", np.ones((4, 4)))
    imodel.add_object(img_a, get_uuid(group_a))
    imodel.add_object(img_b, get_uuid(group_b))
    assert get_short_id(img_a) == "i001"

    # A signal referencing the first image ("i001"):
    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    # Reorder the image groups so that group B comes first: img_a becomes i002:
    imodel.reorder_groups([get_uuid(group_b), get_uuid(group_a)])
    assert get_short_id(img_a) == "i002"

    # The signal title must follow the physical source:
    assert sig.title == "average profile(i002)"


def test_intra_panel_reference_unaffected_by_sibling() -> None:
    """Linking siblings does not disturb regular intra-panel references."""
    smodel, imodel = _linked_models()

    sgroup = smodel.add_group("Signals")
    src = create_signal("Source", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(src, get_uuid(sgroup))  # s001
    res = create_signal("fft(s001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(res, get_uuid(sgroup))  # s002
    assert res.title == "fft(s001)"

    # Reorder signals so that the source becomes s002:
    smodel.reorder_objects({get_uuid(sgroup): [get_uuid(res), get_uuid(src)]})
    assert get_short_id(src) == "s002"
    assert res.title == "fft(s002)"

    # Image model has no objects referencing signals: it stays untouched.
    assert not list(imodel)


def test_cross_panel_reference_renders_long_name() -> None:
    """A cross-panel reference is rendered as the source image's long name."""
    smodel, imodel = _linked_models()

    igroup = imodel.add_group("Images")
    img = create_image("First image", np.zeros((4, 4)))
    imodel.add_object(img, get_uuid(igroup))  # i001

    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    # Raw (short-ID) form is preserved when long names are not requested:
    assert smodel.get_display_title(sig, False) == "average profile(i001)"

    # When long names are requested, the cross-panel reference resolves to the
    # referenced image's title:
    assert smodel.get_display_title(sig, True) == "average profile(First image)"


def test_cross_panel_deleted_image_freezes_reference() -> None:
    """Deleting a referenced image freezes the cross-panel reference (long name)."""
    smodel, imodel = _linked_models()

    igroup = imodel.add_group("Images")
    img = create_image("First image", np.zeros((4, 4)))
    imodel.add_object(img, get_uuid(igroup))  # i001

    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    # Delete the referenced image: the signal reference is frozen into a stable
    # deleted token, with the deleted image's title kept in the registry.
    imodel.remove_object(img)

    assert sig.title == "average profile(id001)"
    assert smodel.get_deleted_refs(sig) == {"id001": "First image"}

    # Short-ID display keeps the deleted token; long-name display shows the
    # frozen title of the deleted image:
    assert smodel.get_display_title(sig, False) == "average profile(id001)"
    assert smodel.get_display_title(sig, True) == "average profile(First image)"


def test_cross_panel_reference_follows_image_delete() -> None:
    """A surviving cross-panel reference follows renumbering on image deletion."""
    smodel, imodel = _linked_models()

    igroup = imodel.add_group("Images")
    img1 = create_image("First image", np.zeros((4, 4)))
    img2 = create_image("Second image", np.ones((4, 4)))
    imodel.add_object(img1, get_uuid(igroup))  # i001
    imodel.add_object(img2, get_uuid(igroup))  # i002

    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i002)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    # Delete the first image: the second image is renumbered i002 -> i001, and
    # the signal reference must follow the physical source:
    imodel.remove_object(img1)
    assert get_short_id(img2) == "i001"
    assert sig.title == "average profile(i001)"


def test_cross_panel_group_reference_renders_and_follows_rename() -> None:
    """A signal group referencing an image group renders its long name and
    follows the image group's rename (e.g. a projection of an image group)."""
    smodel, imodel = _linked_models()

    # An image group (e.g. "My images") containing one image -> gi001:
    igroup = imodel.add_group("My images")
    img = create_image("First image", np.zeros((4, 4)))
    imodel.add_object(img, get_uuid(igroup))
    assert get_short_id(igroup) == "gi001"

    # A signal group produced by a projection keeps a reference to the source
    # image group in its title (exactly as the processor builds it):
    sgroup = smodel.add_group(f"average profile({get_short_id(igroup)})")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))
    assert sgroup.title == "average profile(gi001)"

    # Raw form keeps the short ID; long-name form renders the image group title:
    assert smodel.get_display_title(sgroup, False) == "average profile(gi001)"
    assert smodel.get_display_title(sgroup, True) == "average profile(My images)"

    # Renaming the image group is reflected in the signal group's long name:
    igroup.title = "Renamed images"
    assert smodel.get_display_title(sgroup, True) == "average profile(Renamed images)"


def test_reference_to_empty_name_keeps_short_id() -> None:
    """When a referenced source has an empty title, its short ID is kept in the
    rendered (long-name) display instead of rendering an empty name."""
    smodel, imodel = _linked_models()

    # An image group with no name, containing one image -> gi001:
    igroup = imodel.add_group("")
    img = create_image("First image", np.zeros((4, 4)))
    imodel.add_object(img, get_uuid(igroup))
    assert get_short_id(igroup) == "gi001"

    sgroup = smodel.add_group("average profile(gi001)")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    # The empty image-group name must not blank out the reference: the short ID
    # stays visible in the long-name display too:
    assert smodel.get_display_title(sgroup, True) == "average profile(gi001)"

    # Once the image group gets a name, the long-name display uses it:
    igroup.title = "My images"
    assert smodel.get_display_title(sgroup, True) == "average profile(My images)"


def test_display_title_and_links_locates_resolved_references() -> None:
    """``get_display_title_and_links`` returns the long-name title together with
    the spans of the resolved live references, pointing to their short IDs."""
    smodel, imodel = _linked_models()

    igroup = imodel.add_group("Images")
    img = create_image("First image", np.zeros((4, 4)))
    imodel.add_object(img, get_uuid(igroup))  # i001

    sgroup = smodel.add_group("Signals")
    sig = create_signal("average profile(i001)", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    text, spans = smodel.get_display_title_and_links(sig)
    assert text == "average profile(First image)"
    # One resolved reference, covering the long name and pointing to "i001":
    assert len(spans) == 1
    start, end, target = spans[0]
    assert target == "i001"
    assert text[start:end] == "First image"


def test_display_title_and_links_empty_without_references() -> None:
    """A title without references yields no link spans."""
    smodel, _imodel = _linked_models()

    sgroup = smodel.add_group("Signals")
    sig = create_signal("plain title", x=[0.0, 1.0], y=[1.0, 2.0])
    smodel.add_object(sig, get_uuid(sgroup))

    text, spans = smodel.get_display_title_and_links(sig)
    assert text == "plain title"
    assert spans == []
