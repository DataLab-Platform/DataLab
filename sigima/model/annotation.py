# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotations for signals and images
----------------------------------

This module provides classes and functions to handle annotations for signals and images.

The `Annotation` class is the base class for all annotations, and it provides methods to
create, update, and delete annotations.

The following annotation types are supported:

.. list-table::
    :header-rows: 1
    :align: left

    * - Category
      - Class name
      - Description
    * - Signal or Image
      - :class:`sigima.model.annotation.Label`
      - A simple label annotation for a signal or image, defined by a text and a
        position.
    * - Signal
      - :class:`sigima.model.annotation.VCursor`
      - A vertical cursor annotation for a signal, defined by a X position.
    * - Signal
      - :class:`sigima.model.annotation.HCursor`
      - A horizontal cursor annotation for a signal, defined by a Y position.
    * - Signal
      - :class:`sigima.model.annotation.XCursor`
      - A cross cursor annotation for a signal, defined by a X and Y position.
    * - Signal
      - :class:`sigima.model.annotation.Range`
      - An horizontal range annotation for a signal, defined by a X interval.
    * - Signal or Image
      - :class:`sigima.model.annotation.Line`
      - A line annotation for a signal or image, defined by a start point and an end
        point.
    * - Signal or Image
      - :class:`sigima.model.annotation.Rectangle`
      - A rectangle annotation for a signal or image, defined by a top-left and
        bottom-right point.
    * - Image
      - :class:`sigima.model.annotation.Point`
      - A point annotation for an image, defined by a single point.
    * - Image
      - :class:`sigima.model.annotation.Circle`
      - A circle annotation for an image, defined by a center point and a radius.
    * - Image
      - :class:`sigima.model.annotation.Ellipse`
      - An ellipse annotation for an image, defined by a center point, a major axis and
        a minor axis.
    * - Image
      - :class:`sigima.model.annotation.Polygon`
      - A polygon annotation for an image, defined by a list of points.

The base class `Annotation` provides the common interface for all annotations, which
includes the following methods:

- `get_points()`: Returns the points defining the annotation.
- `set_points(points)`: Sets the points defining the annotation.

This module also provides a registry for annotations, allowing to register new
annotation types and retrieve them by name. The registry is used to create annotations
from JSON data and to manage the available annotation types. The conversion from
annotations to JSON and vice versa is handled by the `AnnotationRegistry` class.

.. note::

    Regarding the JSON serialization, please note that the `Annotation` class
    serializes the annotation type name, the points, and the text. Any other properties
    specific to the annotation type are not serialized by default, and should be
    calculated from the serialized information when needed (e.g. the `Circle` radius
    can be calculated from the bounding box points).

The `Annotation` subclasses constructors accept parameters that define the annotation
type and its properties. The parameters vary depending on the annotation type, but they
typically include the position and size of the annotation:

.. list-table::
    :header-rows: 1
    :align: left

    * - Annotation type
      - Parameters
    * - Label
      - `text`: The text of the label, `x`: X position, `y`: Y Position.
    * - VCursor
      - `x`: X position of the vertical cursor.
    * - HCursor
      - `y`: Y position of the horizontal cursor.
    * - XCursor
      - `x`: X position, `y`: Y position of the cross cursor.
    * - Range
      - `x1`: Start X position, `x2`: End X position of the range.
    * - Line
      - `text`: The text of the annotation, `x1`, `y1`: Start point coordinates, `x2`,
        `y2`: End point coordinates.
    * - Rectangle
      - `text`: The text of the annotation, `x1`, `y1`: Top-left corner coordinates,
        `x2`, `y2`: Bottom-right corner coordinates.
    * - Point
      - `text`: The text of the annotation, `x`, `y`: Coordinates of the point.
    * - Circle
      - `text`: The text of the annotation, `x`, `y`: Center coordinates, `radius`:
        Radius of the circle.
    * - Ellipse
      - `text`: The text of the annotation, `x`, `y`: Center coordinates, `major_axis`:
        Length of the major axis, `minor_axis`: Length of the minor axis.

The `Annotation.get_points()` method returns a list of points that depend on the type
of annotation:

.. list-table::
    :header-rows: 1
    :align: left

    * - Annotation type
      - Points returned by `get_points()`
    * - Label, XCursor, Point
      - A single point [(x, y)] representing the annotation position.
    * - VCursor
      - A single point [(x, 0)] representing the cursor position.
    * - HCursor
      - A single point [(0, y)] representing the cursor position.
    * - Range
      - Two points [(x1, 0), (x2, 0)] representing the start and end of the range.
    * - Line
      - Two points [(x1, y1), (x2, y2)] representing the start and end of the line.
    * - Rectangle
      - Two points [(x1, y1), (x2, y2)] representing the top-left and bottom-right
        corners of the rectangle.
    * - Circle, Ellipse
      - Two points [(x1, y1), (x2, y2)] representing the top-left and bottom-right
        corners of the bounding box of the circle or ellipse.
    * - Polygon
      - A list of points [(x1, y1), (x2, y2), ...] representing the vertices of
        the polygon.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import abc
from typing import Any, Callable, ClassVar, Self, Type

# --------------------------------------------------------------------------------------
# MARK: Registry / decorator


class AnnotationRegistry:
    """Registry for annotation types."""

    _registry: ClassVar[dict[str, Type[Annotation]]] = {}

    @classmethod
    def register(cls, name: str, annotation_cls: Type[Annotation]) -> None:
        """
        Register an annotation class.

        Args:
            name: Name of the annotation type.
            annotation_cls: Annotation class to register.
        """
        cls._registry[name] = annotation_cls

    @classmethod
    def get(cls, name: str) -> Type[Annotation]:
        """
        Get an annotation class by name.

        Args:
            name: Name of the annotation type.

        Returns:
            Annotation class.
        """
        return cls._registry[name]

    @classmethod
    def create_from_json(cls, json_dict: dict[str, Any]) -> Annotation:
        """
        Create an annotation from JSON dictionary

        Args:
            json_dict: JSON dictionary representation of the annotation

        Returns:
            Annotation instance
        """
        return cls.get(json_dict["type"]).from_json(json_dict)

    @classmethod
    def available_types(cls) -> list[str]:
        """
        Return a list of available annotation type names.

        Returns:
            List of annotation type names.
        """
        return list(cls._registry.keys())


def annotation_type(name: str) -> Callable[[Type[Annotation]], Type[Annotation]]:
    """
    Decorator for registering annotation classes.

    Args:
        name: Name of the annotation type.

    Returns:
        Decorator function.
    """

    def decorator(cls: Type[Annotation]) -> Type[Annotation]:
        AnnotationRegistry.register(name, cls)
        return cls

    return decorator


# --------------------------------------------------------------------------------------
# MARK: Base annotation class


class Annotation(abc.ABC):
    """
    Base class for all annotations.

    Args:
        points: List of (x, y) tuples defining the annotation position(s)
        text: Optional annotation text
    """

    STRUCTURE_VERSION: ClassVar[int] = 1
    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 0
    REQUIRE_MIN_N_POINTS: ClassVar[int] = 0

    def __init__(
        self, points: list[tuple[float, float]] | None = None, text: str = ""
    ) -> None:
        """
        Initialize an annotation.

        Args:
            points: List of (x, y) tuples defining the annotation position(s)
            text: Optional annotation text
        """
        if points is None:
            points = []
        self.check_points(points)
        self._points: list[tuple[float, float]] = points
        self._text: str = text

    @property
    def type_name(self) -> str:
        """
        Get the name of the annotation type.

        Returns:
            The name of the annotation type
        """
        return type(self).__name__

    @property
    def points(self) -> list[tuple[float, float]]:
        """
        Get the points defining the annotation.

        Returns:
            List of (x, y) tuples
        """
        return self._points.copy()

    @points.setter
    def points(self, pts: list[tuple[float, float]]) -> None:
        """
        Set the points defining the annotation, with validation.

        Args:
            pts: List of (x, y) tuples

        Raises:
            ValueError: If points are invalid for this annotation type
        """
        self.check_points(pts)
        self._points = pts

    def check_points(self, points: list[tuple[float, float]]) -> None:
        """
        Validate the points for this annotation type.

        Args:
            points: List of (x, y) tuples

        Raises:
            ValueError: If points are invalid for this annotation type
        """
        n_ex, n_min = self.REQUIRE_EXACTLY_N_POINTS, self.REQUIRE_MIN_N_POINTS
        if n_ex > 0 and len(points) != n_ex:
            raise ValueError(f"{self.type_name} annotation requires {n_ex} points.")
        if n_min > 0 and len(points) < n_min:
            raise ValueError(
                f"{self.type_name} annotation requires at least {n_min} points."
            )

    def get_points(self) -> list[tuple[float, float]]:
        """
        Return the points defining the annotation.

        Returns:
            List of (x, y) tuples
        """
        return self.points

    def set_points(self, points: list[tuple[float, float]]) -> None:
        """
        Set the points defining the annotation, with validation.

        Args:
            points: List of (x, y) tuples

        Raises:
            ValueError: If points are invalid for this annotation type
        """
        self.points = points  # calls setter, which validates

    def get_text(self) -> str:
        """
        Get the annotation text.

        Returns:
            The annotation text
        """
        return self._text

    def set_text(self, value: str) -> None:
        """
        Set the annotation text.

        Args:
            value: The new annotation text
        """
        self._text = value

    def to_json(self) -> dict[str, Any]:
        """
        Serialize annotation to JSON dictionary

        Returns:
            JSON dictionary representation of the annotation
        """
        return {
            "version": self.STRUCTURE_VERSION,
            "type": self.type_name,
            "points": self.get_points(),
            "text": self.get_text(),
        }

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """
        Deserialize annotation from JSON dictionary

        Args:
            json_dict: JSON dictionary representation of the annotation

        Returns:
            Annotation instance
        """
        assert json_dict["type"] == cls.__name__, (
            f"Expected type {cls.__name__}, got {json_dict['type']}"
        )
        if "points" not in json_dict or not isinstance(json_dict["points"], list):
            raise ValueError("Missing or invalid 'points' in JSON data")
        points = json_dict["points"]
        if not all(isinstance(coord, (int, float)) for coord in points[0]):
            raise ValueError("Invalid point coordinates in JSON data")
        version = json_dict.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported annotation version: {version}")
        instance = cls(text=json_dict.get("text", ""))
        instance.set_points(json_dict["points"])
        return instance

    def __repr__(self) -> str:
        """
        Return a string representation for debugging.

        Returns:
            String representation of the annotation.
        """
        cname = self.__class__.__name__
        return f"<{cname} points={self.points!r} text={self._text!r}>"


# --------------------------------------------------------------------------------------
# MARK: Annotation subclasses


@annotation_type("Label")
class Label(Annotation):
    """
    A simple label annotation for a signal or image.

    Args:
        x: X position of the label
        y: Y position of the label
        text: Optional text of the label
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 1

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "") -> None:
        super().__init__([(x, y)], text=text)

    @property
    def x(self) -> float:
        """X position of the label."""
        return self.points[0][0]

    @x.setter
    def x(self, value: float) -> None:
        self.set_points([(value, self.y)])

    @property
    def y(self) -> float:
        """Y position of the label."""
        return self.points[0][1]

    @y.setter
    def y(self, value: float) -> None:
        self.set_points([(self.x, value)])


@annotation_type("VCursor")
class VCursor(Annotation):
    """
    A vertical cursor annotation for a signal.

    Args:
        x: X position of the vertical cursor.
        text: Optional annotation text.
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 1

    def __init__(self, x: float = 0.0, text: str = "") -> None:
        super().__init__([(x, 0)], text=text)

    @property
    def x(self) -> float:
        """X position of the vertical cursor."""
        return self.points[0][0]

    @x.setter
    def x(self, value: float) -> None:
        self.set_points([(value, 0)])


@annotation_type("HCursor")
class HCursor(Annotation):
    """
    A horizontal cursor annotation for a signal.

    Args:
        y: Y position of the horizontal cursor
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 1

    def __init__(self, y: float = 0.0, text: str = "") -> None:
        super().__init__([(0, y)], text=text)

    @property
    def y(self) -> float:
        """Y position of the horizontal cursor."""
        return self.points[0][1]

    @y.setter
    def y(self, value: float) -> None:
        self.set_points([(0, value)])


@annotation_type("XCursor")
class XCursor(Annotation):
    """
    A cross cursor annotation for a signal.

    Args:
        x: X position of the cross cursor
        y: Y position of the cross cursor
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 1

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "") -> None:
        super().__init__([(x, y)], text=text)

    @property
    def x(self) -> float:
        """X position of the cross cursor."""
        return self.points[0][0]

    @x.setter
    def x(self, value: float) -> None:
        self.set_points([(value, self.y)])

    @property
    def y(self) -> float:
        """Y position of the cross cursor."""
        return self.points[0][1]

    @y.setter
    def y(self, value: float) -> None:
        self.set_points([(self.x, value)])


@annotation_type("Range")
class Range(Annotation):
    """
    A horizontal range annotation for a signal.

    Args:
        x1: Start X position
        x2: End X position
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 2

    def __init__(self, x1: float = 0.0, x2: float = 0.0, text: str = "") -> None:
        super().__init__([(x1, 0), (x2, 0)], text=text)

    @property
    def x1(self) -> float:
        """Start X position of the range."""
        return self.points[0][0]

    @x1.setter
    def x1(self, value: float) -> None:
        self.set_points([(value, 0), (self.x2, 0)])

    @property
    def x2(self) -> float:
        """End X position of the range."""
        return self.points[1][0]

    @x2.setter
    def x2(self, value: float) -> None:
        self.set_points([(self.x1, 0), (value, 0)])


@annotation_type("Line")
class Line(Annotation):
    """
    A line annotation for a signal or image.

    Args:
        x1: Start X coordinate
        y1: Start Y coordinate
        x2: End X coordinate
        y2: End Y coordinate
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 2

    def __init__(
        self,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 0.0,
        y2: float = 0.0,
        text: str = "",
    ) -> None:
        super().__init__([(x1, y1), (x2, y2)], text=text)

    @property
    def x1(self) -> float:
        """Start X coordinate of the line."""
        return self.points[0][0]

    @x1.setter
    def x1(self, value: float) -> None:
        self.set_points([(value, self.y1), (self.x2, self.y2)])

    @property
    def y1(self) -> float:
        """Start Y coordinate of the line."""
        return self.points[0][1]

    @y1.setter
    def y1(self, value: float) -> None:
        self.set_points([(self.x1, value), (self.x2, self.y2)])

    @property
    def x2(self) -> float:
        """End X coordinate of the line."""
        return self.points[1][0]

    @x2.setter
    def x2(self, value: float) -> None:
        self.set_points([(self.x1, self.y1), (value, self.y2)])

    @property
    def y2(self) -> float:
        """End Y coordinate of the line."""
        return self.points[1][1]

    @y2.setter
    def y2(self, value: float) -> None:
        self.set_points([(self.x1, self.y1), (self.x2, value)])


@annotation_type("Rectangle")
class Rectangle(Annotation):
    """
    A rectangle annotation for a signal or image.

    Args:
        x1: Top-left X coordinate
        y1: Top-left Y coordinate
        x2: Bottom-right X coordinate
        y2: Bottom-right Y coordinate
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 2

    def __init__(
        self,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 0.0,
        y2: float = 0.0,
        text: str = "",
    ) -> None:
        super().__init__([(x1, y1), (x2, y2)], text=text)

    @property
    def x1(self) -> float:
        """Top-left X coordinate of the rectangle."""
        return self.points[0][0]

    @x1.setter
    def x1(self, value: float) -> None:
        self.set_points([(value, self.y1), (self.x2, self.y2)])

    @property
    def y1(self) -> float:
        """Top-left Y coordinate of the rectangle."""
        return self.points[0][1]

    @y1.setter
    def y1(self, value: float) -> None:
        self.set_points([(self.x1, value), (self.x2, self.y2)])

    @property
    def x2(self) -> float:
        """Bottom-right X coordinate of the rectangle."""
        return self.points[1][0]

    @x2.setter
    def x2(self, value: float) -> None:
        self.set_points([(self.x1, self.y1), (value, self.y2)])

    @property
    def y2(self) -> float:
        """Bottom-right Y coordinate of the rectangle."""
        return self.points[1][1]

    @y2.setter
    def y2(self, value: float) -> None:
        self.set_points([(self.x1, self.y1), (self.x2, value)])


@annotation_type("Point")
class Point(Annotation):
    """
    A point annotation for an image.

    Args:
        x: X coordinate
        y: Y coordinate
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 1

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "") -> None:
        super().__init__([(x, y)], text=text)

    @property
    def x(self) -> float:
        """X coordinate of the point."""
        return self.points[0][0]

    @x.setter
    def x(self, value: float) -> None:
        self.set_points([(value, self.y)])

    @property
    def y(self) -> float:
        """Y coordinate of the point."""
        return self.points[0][1]

    @y.setter
    def y(self, value: float) -> None:
        self.set_points([(self.x, value)])


@annotation_type("Circle")
class Circle(Annotation):
    """
    A circle annotation for an image.

    Args:
        x: Center X coordinate
        y: Center Y coordinate
        radius: Radius of the circle
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 2

    def __init__(
        self, x: float = 0.0, y: float = 0.0, radius: float = 0.0, text: str = ""
    ) -> None:
        super().__init__(
            [(x - radius, y - radius), (x + radius, y + radius)], text=text
        )

    @property
    def x(self) -> float:
        """Center X coordinate of the circle."""
        return (self.points[0][0] + self.points[1][0]) / 2

    @x.setter
    def x(self, value: float) -> None:
        self.set_points(
            [
                (value - self.radius, self.y - self.radius),
                (value + self.radius, self.y + self.radius),
            ]
        )

    @property
    def y(self) -> float:
        """Center Y coordinate of the circle."""
        return (self.points[0][1] + self.points[1][1]) / 2

    @y.setter
    def y(self, value: float) -> None:
        self.set_points(
            [
                (self.x - self.radius, value - self.radius),
                (self.x + self.radius, value + self.radius),
            ]
        )

    @property
    def radius(self) -> float:
        """Radius of the circle."""
        return (self.points[1][0] - self.points[0][0]) / 2

    @radius.setter
    def radius(self, value: float) -> None:
        self.set_points(
            [(self.x - value, self.y - value), (self.x + value, self.y + value)]
        )


@annotation_type("Ellipse")
class Ellipse(Annotation):
    """
    An ellipse annotation for an image.

    Args:
        x: Center X coordinate
        y: Center Y coordinate
        major_semi_axis: Length of the major semi-axis
        minor_semi_axis: Length of the minor semi-axis
        text: Optional annotation text
    """

    REQUIRE_EXACTLY_N_POINTS: ClassVar[int] = 2

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        major_semi_axis: float = 0.0,
        minor_semi_axis: float = 0.0,
        text: str = "",
    ) -> None:
        super().__init__(
            [
                (x - major_semi_axis, y - minor_semi_axis),
                (x + major_semi_axis, y + minor_semi_axis),
            ],
            text=text,
        )

    @property
    def x(self) -> float:
        """Center X coordinate of the ellipse."""
        return (self.points[0][0] + self.points[1][0]) / 2

    @x.setter
    def x(self, value: float) -> None:
        self.set_points(
            [
                (value - self.major_semi_axis, self.y - self.minor_semi_axis),
                (value + self.major_semi_axis, self.y + self.minor_semi_axis),
            ]
        )

    @property
    def y(self) -> float:
        """Center Y coordinate of the ellipse."""
        return (self.points[0][1] + self.points[1][1]) / 2

    @y.setter
    def y(self, value: float) -> None:
        self.set_points(
            [
                (self.x - self.major_semi_axis, value - self.minor_semi_axis),
                (self.x + self.major_semi_axis, value + self.minor_semi_axis),
            ]
        )

    @property
    def major_semi_axis(self) -> float:
        """Length of the major semi-axis."""
        return abs(self.points[1][0] - self.points[0][0]) / 2

    @major_semi_axis.setter
    def major_semi_axis(self, value: float) -> None:
        self.set_points(
            [
                (self.x - value, self.y - self.minor_semi_axis),
                (self.x + value, self.y + self.minor_semi_axis),
            ]
        )

    @property
    def minor_semi_axis(self) -> float:
        """Length of the minor semi-axis."""
        return abs(self.points[1][1] - self.points[0][1]) / 2

    @minor_semi_axis.setter
    def minor_semi_axis(self, value: float) -> None:
        self.set_points(
            [
                (self.x - self.major_semi_axis, self.y - value),
                (self.x + self.major_semi_axis, self.y + value),
            ]
        )


@annotation_type("Polygon")
class Polygon(Annotation):
    """
    A polygon annotation for an image.

    Args:
        points: List of (x, y) tuples defining the vertices of the polygon
        text: Optional annotation text
    """

    REQUIRE_MIN_N_POINTS: ClassVar[int] = 3
