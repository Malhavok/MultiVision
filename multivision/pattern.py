"""Deterministic projector-space calibration-pattern metadata."""

from __future__ import annotations

import math
from collections.abc import (
    Mapping,
    Sequence,
    Set,
)
from numbers import Real
from typing import NamedTuple

from multivision.geometry import CoordinateBounds, Point2D
from multivision.types import (
    Resolution,
    is_valid_resolution,
)


APRILTAG_36H11 = 'DICT_APRILTAG_36h11'
APRILTAG_FAMILIES = frozenset(
    {
        'DICT_APRILTAG_16h5',
        'DICT_APRILTAG_25h9',
        'DICT_APRILTAG_36h10',
        APRILTAG_36H11,
    },
)
DEFAULT_MARKER_COUNT = 12
_DEFAULT_MARKER_SIZE_FRACTION = 0.1


class MarkerCorner(NamedTuple):
    """One ordered, projector-native corner of a calibration marker."""

    marker_id: int
    corner_index: int
    projector_position: Point2D


class CalibrationMarker(NamedTuple):
    """Renderer-facing metadata for one uniquely identified marker."""

    marker_id: int
    corners: tuple[Point2D, ...]

    @property
    def corner_metadata(self) -> tuple[MarkerCorner, ...]:
        return tuple(
            MarkerCorner(self.marker_id, corner_index, projector_position)
            for corner_index, projector_position in enumerate(self.corners)
        )

    @property
    def bounds(self) -> CoordinateBounds:
        x_positions = [corner.x for corner in self.corners]
        y_positions = [corner.y for corner in self.corners]
        return CoordinateBounds(
            min(x_positions),
            min(y_positions),
            max(x_positions),
            max(y_positions),
        )


class CalibrationPattern(NamedTuple):
    """Immutable pattern geometry in projector-native coordinates."""

    projector_resolution: Resolution
    usable_area: CoordinateBounds
    marker_family: str
    marker_size: float
    markers: tuple[CalibrationMarker, ...]

    @property
    def marker_corners(self) -> tuple[MarkerCorner, ...]:
        return tuple(
            corner
            for marker in self.markers
            for corner in marker.corner_metadata
        )

    def get_marker(self, marker_id: int) -> CalibrationMarker:
        """Return the marker with the requested dictionary ID."""
        for marker in self.markers:
            if marker.marker_id == marker_id:
                return marker
        raise KeyError(marker_id)


def build_calibration_pattern(
    projector_resolution: Resolution | Sequence[int],
    usable_area: CoordinateBounds | None = None,
    marker_count: int = DEFAULT_MARKER_COUNT,
    marker_size: float | None = None,
    marker_family: str = APRILTAG_36H11,
) -> CalibrationPattern:
    """Build a deterministic grid of uniquely identified AprilTag markers."""
    checked_resolution = _coerce_resolution(projector_resolution)
    checked_area = _coerce_usable_area(
        usable_area
        if usable_area is not None
        else CoordinateBounds(
            0.0,
            0.0,
            float(checked_resolution.width),
            float(checked_resolution.height),
        ),
        checked_resolution,
    )
    _validate_marker_count(marker_count)
    if not isinstance(marker_family, str) or marker_family not in APRILTAG_FAMILIES:
        raise ValueError(f'Unsupported AprilTag family: {marker_family!r}')

    checked_marker_size = marker_size
    if checked_marker_size is None:
        checked_marker_size = min(
            checked_area.right - checked_area.left,
            checked_area.bottom - checked_area.top,
        ) * _DEFAULT_MARKER_SIZE_FRACTION
    checked_marker_size = _coerce_marker_size(
        checked_marker_size,
        checked_area,
        marker_count,
    )

    columns, rows, omitted_slots = _layout_for_marker_count(marker_count)
    cell_width = (checked_area.right - checked_area.left) / columns
    cell_height = (checked_area.bottom - checked_area.top) / rows
    marker_positions = [
        (x_idx, y_idx)
        for y_idx in range(rows)
        for x_idx in range(columns)
        if (x_idx, y_idx) not in omitted_slots
    ]

    markers = tuple(
        CalibrationMarker(
            marker_id,
            _build_marker_corners(
                checked_area,
                x_idx,
                y_idx,
                cell_width,
                cell_height,
                checked_marker_size,
            ),
        )
        for marker_id, (x_idx, y_idx) in enumerate(marker_positions)
    )
    return CalibrationPattern(
        checked_resolution,
        checked_area,
        marker_family,
        float(checked_marker_size),
        markers,
    )


def _coerce_resolution(
    resolution: Resolution | Sequence[int],
) -> Resolution:
    if isinstance(resolution, Resolution):
        checked_resolution = resolution
    else:
        if isinstance(resolution, (Mapping, Set, str, bytes, bytearray)):
            raise ValueError('projector_resolution must contain width and height')
        try:
            width, height = resolution
        except (TypeError, ValueError):
            raise ValueError('projector_resolution must contain width and height') from None
        checked_resolution = Resolution(width, height)
    if not is_valid_resolution(checked_resolution):
        raise ValueError('projector_resolution must contain positive integer dimensions')
    try:
        width_as_float = float(checked_resolution.width)
        height_as_float = float(checked_resolution.height)
    except OverflowError:
        raise ValueError('projector_resolution dimensions are too large') from None
    if not math.isfinite(width_as_float) or not math.isfinite(height_as_float):
        raise ValueError('projector_resolution dimensions are too large')
    return checked_resolution


def _coerce_usable_area(
    usable_area: CoordinateBounds,
    projector_resolution: Resolution,
) -> CoordinateBounds:
    if not isinstance(usable_area, CoordinateBounds):
        raise ValueError('usable_area must be CoordinateBounds')
    try:
        checked_area = CoordinateBounds(
            _coerce_finite_float(usable_area.left, 'usable_area'),
            _coerce_finite_float(usable_area.top, 'usable_area'),
            _coerce_finite_float(usable_area.right, 'usable_area'),
            _coerce_finite_float(usable_area.bottom, 'usable_area'),
        )
    except ValueError:
        raise ValueError('usable_area must contain finite coordinates') from None
    if not checked_area.left < checked_area.right or not checked_area.top < checked_area.bottom:
        raise ValueError('usable_area must have positive dimensions')
    if (
        checked_area.left < 0
        or checked_area.top < 0
        or checked_area.right > projector_resolution.width
        or checked_area.bottom > projector_resolution.height
    ):
        raise ValueError('usable_area must be inside projector_resolution')
    return checked_area


def _validate_marker_count(marker_count: int) -> None:
    if (
        not isinstance(marker_count, int)
        or isinstance(marker_count, bool)
        or not 9 <= marker_count <= 12
    ):
        raise ValueError('marker_count must be between 9 and 12')


def _coerce_marker_size(
    marker_size: object,
    usable_area: CoordinateBounds,
    marker_count: int,
) -> float:
    try:
        checked_marker_size = _coerce_finite_float(marker_size, 'marker_size')
    except ValueError:
        raise ValueError('marker_size must be a finite, positive number') from None
    if checked_marker_size <= 0:
        raise ValueError('marker_size must be a finite, positive number')
    columns, rows, _ = _layout_for_marker_count(marker_count)
    cell_width = (usable_area.right - usable_area.left) / columns
    cell_height = (usable_area.bottom - usable_area.top) / rows
    if checked_marker_size >= min(cell_width, cell_height):
        raise ValueError('marker_size must fit inside each pattern cell')
    return checked_marker_size


def _layout_for_marker_count(marker_count: int) -> tuple[int, int, frozenset[tuple[int, int]]]:
    layouts = {
        9: (3, 3, frozenset()),
        10: (5, 2, frozenset()),
        11: (4, 3, frozenset({(1, 1)})),
        12: (4, 3, frozenset()),
    }
    return layouts[marker_count]


def _build_marker_corners(
    usable_area: CoordinateBounds,
    x_idx: int,
    y_idx: int,
    cell_width: float,
    cell_height: float,
    marker_size: float,
) -> tuple[Point2D, ...]:
    centre_x = usable_area.left + (x_idx + 0.5) * cell_width
    centre_y = usable_area.top + (y_idx + 0.5) * cell_height
    half_size = marker_size / 2
    return (
        Point2D(centre_x - half_size, centre_y - half_size),
        Point2D(centre_x + half_size, centre_y - half_size),
        Point2D(centre_x + half_size, centre_y + half_size),
        Point2D(centre_x - half_size, centre_y + half_size),
    )


def _coerce_finite_float(value: object, field_name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f'{field_name} must be a finite number')
    try:
        checked_value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f'{field_name} must be a finite number') from None
    if not math.isfinite(checked_value):
        raise ValueError(f'{field_name} must be a finite number')
    return checked_value


__all__ = [
    'APRILTAG_36H11',
    'APRILTAG_FAMILIES',
    'CalibrationMarker',
    'CalibrationPattern',
    'DEFAULT_MARKER_COUNT',
    'MarkerCorner',
    'build_calibration_pattern',
]
