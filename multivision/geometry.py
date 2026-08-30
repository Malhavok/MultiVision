"""Pure coordinate-space and homography operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from numbers import Real
from typing import (
    NamedTuple,
    TypeAlias,
)

from multivision.errors import (
    InvalidHomographyError,
    PointOutsideCalibratedRegionError,
    PointOutsideProjectorError,
    PointOutsidePreviewError,
)
from multivision.types import (
    Resolution,
    is_finite_real,
)


class Point2D(NamedTuple):
    x: float
    y: float


class CoordinateBounds(NamedTuple):
    """A half-open rectangle in one coordinate space."""

    left: float
    top: float
    right: float
    bottom: float

    def contains(self, point: PointLike) -> bool:
        """Return whether a finite point is inside this half-open rectangle."""
        return is_point_in_bounds(point, self)


class HomographyPair(NamedTuple):
    """Both directions of one camera/projector homography."""

    projector_to_camera: tuple[tuple[float, float, float], ...]
    camera_to_projector: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_projector_to_camera(cls, matrix: MatrixLike) -> 'HomographyPair':
        projector_to_camera = validate_homography(matrix)
        return cls(projector_to_camera, invert_homography(projector_to_camera))

    @classmethod
    def from_camera_to_projector(cls, matrix: MatrixLike) -> 'HomographyPair':
        camera_to_projector = validate_homography(matrix)
        return cls(invert_homography(camera_to_projector), camera_to_projector)


PointLike: TypeAlias = Point2D | Sequence[Real]
MatrixLike: TypeAlias = Sequence[Sequence[Real]]
RegionLike: TypeAlias = CoordinateBounds | Sequence[PointLike]
BoundsLike: TypeAlias = CoordinateBounds | Resolution | Sequence[int]
Polygon: TypeAlias = tuple[Point2D, ...]


class PreviewTransform(NamedTuple):
    """Aspect-preserving preview layout and its native-coordinate conversion."""

    preview_size: Resolution
    camera_resolution: Resolution
    scale: float
    content_bounds: CoordinateBounds

    def to_camera_native(self, preview_point: PointLike) -> Point2D:
        """Convert a preview-local point, rejecting letterbox padding."""
        point = coerce_point(preview_point)
        if not self.content_bounds.contains(point):
            raise PointOutsidePreviewError(
                f'Preview point {point!r} is outside the image content',
            )

        native_point = Point2D(
            (point.x - self.content_bounds.left) / self.scale,
            (point.y - self.content_bounds.top) / self.scale,
        )
        if not is_point_in_resolution(native_point, self.camera_resolution):
            raise PointOutsidePreviewError(
                f'Preview point {point!r} maps outside the camera image',
            )
        return native_point


def build_preview_transform(
    preview_size: Resolution | Sequence[int],
    camera_resolution: Resolution | Sequence[int],
) -> PreviewTransform:
    """Build an aspect-preserving, centred preview-to-native transform."""
    preview = _coerce_resolution(preview_size, 'preview_size')
    camera = _coerce_resolution(camera_resolution, 'camera_resolution')
    scale = min(preview.width / camera.width, preview.height / camera.height)
    content_width = camera.width * scale
    content_height = camera.height * scale
    left = max(0.0, (preview.width - content_width) / 2)
    top = max(0.0, (preview.height - content_height) / 2)
    return PreviewTransform(
        preview,
        camera,
        scale,
        CoordinateBounds(left, top, left + content_width, top + content_height),
    )


def preview_local_to_camera_native(
    preview_point: PointLike,
    preview_size: Resolution | Sequence[int],
    camera_resolution: Resolution | Sequence[int],
) -> Point2D:
    """Convert a preview-local point to native camera coordinates."""
    return build_preview_transform(preview_size, camera_resolution).to_camera_native(
        preview_point,
    )


def calculate_convex_hull(points: Sequence[Point2D]) -> tuple[Point2D, ...]:
    """Return the points on the convex hull in deterministic order."""
    unique_points = sorted(set(points))
    if len(unique_points) < 3:
        return tuple(unique_points)

    def cross(first: Point2D, second: Point2D, third: Point2D) -> float:
        return (
            (second.x - first.x) * (third.y - first.y)
            - (second.y - first.y) * (third.x - first.x)
        )

    lower: list[Point2D] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point2D] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def calculate_polygon_area(polygon: Sequence[Point2D]) -> float:
    """Return the absolute area of an ordered polygon."""
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[idx].x * polygon[(idx + 1) % len(polygon)].y
            - polygon[(idx + 1) % len(polygon)].x * polygon[idx].y
            for idx in range(len(polygon))
        )
        / 2
    )


def is_finite_point(point: object) -> bool:
    """Return whether a point contains exactly two finite numeric coordinates."""
    coordinates = _get_point_coordinates(point)
    if coordinates is None:
        return False
    x_value, y_value = coordinates
    return is_finite_real(x_value) and is_finite_real(y_value)


def is_point_in_bounds(point: PointLike, bounds: CoordinateBounds) -> bool:
    """Return whether a finite point is inside a half-open rectangle."""
    if not isinstance(bounds, CoordinateBounds):
        return False
    try:
        checked_point = coerce_point(point)
    except ValueError:
        return False
    if not _is_valid_bounds(bounds):
        return False
    return (
        bounds.left <= checked_point.x < bounds.right
        and bounds.top <= checked_point.y < bounds.bottom
    )


def is_point_in_resolution(
    point: PointLike,
    resolution: Resolution | Sequence[int],
) -> bool:
    """Return whether a point lies within native pixel-coordinate extents."""
    try:
        checked_resolution = _coerce_resolution(resolution, 'resolution')
    except ValueError:
        return False
    return is_point_in_bounds(
        point,
        CoordinateBounds(0, 0, checked_resolution.width, checked_resolution.height),
    )


def is_point_in_region(point: PointLike, region: RegionLike) -> bool:
    """Return whether a point is in rectangular or polygonal calibrated support."""
    if isinstance(region, CoordinateBounds):
        return is_point_in_bounds(point, region)
    if isinstance(region, (Mapping, Set)):
        return False
    try:
        checked_point = coerce_point(point)
    except ValueError:
        return False

    polygon = _normalise_polygon(region)
    if polygon is None:
        return False

    is_inside = False
    previous_point = polygon[-1]
    for current_point in polygon:
        if _is_point_on_segment(checked_point, previous_point, current_point):
            return True
        crosses_horizontal_ray = (current_point.y > checked_point.y) != (
            previous_point.y > checked_point.y
        )
        if crosses_horizontal_ray:
            crossing_x = (
                (previous_point.x - current_point.x)
                * (checked_point.y - current_point.y)
                / (previous_point.y - current_point.y)
                + current_point.x
            )
            if checked_point.x < crossing_x:
                is_inside = not is_inside
        previous_point = current_point
    return is_inside


def validate_point_in_region(point: PointLike, region: RegionLike) -> Point2D:
    """Validate and return a point supported by a calibrated region."""
    checked_point = coerce_point(point)
    if not is_point_in_region(checked_point, region):
        raise PointOutsideCalibratedRegionError(
            f'Point {checked_point!r} is outside the calibrated region',
        )
    return checked_point


def is_valid_homography(matrix: object) -> bool:
    """Return whether a matrix is finite, 3x3 and safely invertible."""
    try:
        _normalise_homography(matrix)  # type: ignore[arg-type]
    except (TypeError, ValueError, InvalidHomographyError):
        return False
    return True


def validate_homography(matrix: MatrixLike) -> tuple[tuple[float, float, float], ...]:
    """Validate and return a finite, normalised 3x3 homography."""
    return _normalise_homography(matrix)


def invert_homography(matrix: MatrixLike) -> tuple[tuple[float, float, float], ...]:
    """Invert a non-degenerate 3x3 homography without a hardware dependency."""
    normalised_matrix = _normalise_homography(matrix)
    determinant = _determinant(normalised_matrix)
    cofactors = (
        (
            normalised_matrix[1][1] * normalised_matrix[2][2]
            - normalised_matrix[1][2] * normalised_matrix[2][1],
            normalised_matrix[1][2] * normalised_matrix[2][0]
            - normalised_matrix[1][0] * normalised_matrix[2][2],
            normalised_matrix[1][0] * normalised_matrix[2][1]
            - normalised_matrix[1][1] * normalised_matrix[2][0],
        ),
        (
            normalised_matrix[0][2] * normalised_matrix[2][1]
            - normalised_matrix[0][1] * normalised_matrix[2][2],
            normalised_matrix[0][0] * normalised_matrix[2][2]
            - normalised_matrix[0][2] * normalised_matrix[2][0],
            normalised_matrix[0][1] * normalised_matrix[2][0]
            - normalised_matrix[0][0] * normalised_matrix[2][1],
        ),
        (
            normalised_matrix[0][1] * normalised_matrix[1][2]
            - normalised_matrix[0][2] * normalised_matrix[1][1],
            normalised_matrix[0][2] * normalised_matrix[1][0]
            - normalised_matrix[0][0] * normalised_matrix[1][2],
            normalised_matrix[0][0] * normalised_matrix[1][1]
            - normalised_matrix[0][1] * normalised_matrix[1][0],
        ),
    )
    return tuple(
        tuple(value / determinant for value in row)
        for row in zip(cofactors[0], cofactors[1], cofactors[2])
    )


def project_point(point: PointLike, matrix: MatrixLike) -> Point2D:
    """Project a point through a homography, rejecting invalid infinity results."""
    checked_point = coerce_point(point)
    normalised_matrix = _normalise_homography(matrix)
    projected_x = (
        normalised_matrix[0][0] * checked_point.x
        + normalised_matrix[0][1] * checked_point.y
        + normalised_matrix[0][2]
    )
    projected_y = (
        normalised_matrix[1][0] * checked_point.x
        + normalised_matrix[1][1] * checked_point.y
        + normalised_matrix[1][2]
    )
    denominator = (
        normalised_matrix[2][0] * checked_point.x
        + normalised_matrix[2][1] * checked_point.y
        + normalised_matrix[2][2]
    )
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise InvalidHomographyError('Homography projects the point to infinity')
    result = Point2D(projected_x / denominator, projected_y / denominator)
    if not is_finite_point(result):
        raise InvalidHomographyError('Homography produced a non-finite point')
    return result


def intersect_polygon_with_bounds(
    polygon: RegionLike,
    bounds: BoundsLike,
) -> Polygon | None:
    """Intersect a finite polygon with a finite rectangular bounds area.

    ``None`` is returned when the inputs are invalid or the intersection has
    no usable area.  Polygon boundaries use the closed geometric edges of the
    half-open coordinate rectangle, so a clipped edge may end at ``right`` or
    ``bottom`` while pixel-coordinate point checks remain half-open.
    """
    checked_polygon = _normalise_polygon(polygon)
    checked_bounds = _coerce_bounds(bounds)
    if checked_polygon is None or checked_bounds is None:
        return None

    clipped_polygon = checked_polygon
    for axis, boundary, keeps_greater in (
        ('x', checked_bounds.left, True),
        ('x', checked_bounds.right, False),
        ('y', checked_bounds.top, True),
        ('y', checked_bounds.bottom, False),
    ):
        clipped_polygon = _clip_polygon_against_boundary(
            clipped_polygon,
            axis,
            boundary,
            keeps_greater,
        )
        if clipped_polygon is None:
            return None

    return clipped_polygon


def project_polygon(
    polygon: RegionLike,
    homography: MatrixLike | HomographyPair,
) -> Polygon | None:
    """Project a polygon, rejecting invalid transforms and horizon crossings."""
    checked_polygon = _normalise_polygon(polygon)
    if checked_polygon is None:
        return None
    matrix = (
        homography.camera_to_projector
        if isinstance(homography, HomographyPair)
        else homography
    )
    try:
        normalised_matrix = _normalise_homography(matrix)
    except (OverflowError, TypeError, ValueError):
        return None

    denominators = tuple(
        normalised_matrix[2][0] * point.x
        + normalised_matrix[2][1] * point.y
        + normalised_matrix[2][2]
        for point in checked_polygon
    )
    if any(
        not math.isfinite(denominator)
        or abs(denominator) <= 1e-12
        for denominator in denominators
    ):
        return None
    denominator_signs = {denominator > 0 for denominator in denominators}
    if len(denominator_signs) != 1:
        return None

    try:
        projected_polygon = tuple(
            project_point(point, normalised_matrix)
            for point in checked_polygon
        )
    except InvalidHomographyError:
        return None
    return _normalise_polygon(projected_polygon)


def calculate_available_projector_area(
    camera_polygon: RegionLike,
    camera_resolution: Resolution | Sequence[int],
    homography: MatrixLike | HomographyPair,
    projector_resolution: Resolution | Sequence[int],
) -> Polygon | None:
    """Derive a usable projector polygon using a camera-to-projector transform."""
    camera_bounds = _coerce_bounds(camera_resolution)
    projector_bounds = _coerce_bounds(projector_resolution)
    if camera_bounds is None or projector_bounds is None:
        return None
    native_polygon = intersect_polygon_with_bounds(camera_polygon, camera_bounds)
    if native_polygon is None:
        return None
    projected_polygon = project_polygon(native_polygon, homography)
    if projected_polygon is None:
        return None
    return intersect_polygon_with_bounds(projected_polygon, projector_bounds)


def projector_to_camera(
    point: PointLike,
    homography: MatrixLike | HomographyPair,
) -> Point2D:
    """Transform a projector-native point into camera-native coordinates."""
    matrix = (
        homography.projector_to_camera
        if isinstance(homography, HomographyPair)
        else homography
    )
    return project_point(point, matrix)


def camera_to_projector(
    point: PointLike,
    homography: MatrixLike | HomographyPair,
) -> Point2D:
    """Transform a camera-native point into projector-native coordinates."""
    matrix = (
        homography.camera_to_projector
        if isinstance(homography, HomographyPair)
        else homography
    )
    return project_point(point, matrix)


def project_camera_to_projector(
    point: PointLike,
    homography: MatrixLike | HomographyPair,
    calibrated_region: RegionLike | None = None,
    projector_resolution: Resolution | Sequence[int] | None = None,
) -> Point2D:
    """Transform a supported camera point and enforce projector output bounds."""
    checked_point = coerce_point(point)
    if calibrated_region is not None:
        validate_point_in_region(checked_point, calibrated_region)
    projected_point = camera_to_projector(checked_point, homography)
    if projector_resolution is not None and not is_point_in_resolution(
        projected_point,
        projector_resolution,
    ):
        raise PointOutsideProjectorError(
            f'Projected point {projected_point!r} is outside projector bounds',
        )
    return projected_point


def coerce_point(point: PointLike) -> Point2D:
    coordinates = _get_point_coordinates(point)
    if coordinates is None:
        raise ValueError(f'Point must contain two finite numbers: {point!r}')
    x_value, y_value = coordinates
    if not is_finite_real(x_value) or not is_finite_real(y_value):
        raise ValueError(f'Point must contain two finite numbers: {point!r}')
    return Point2D(float(x_value), float(y_value))


def _get_point_coordinates(point: object) -> tuple[object, object] | None:
    if isinstance(point, (Mapping, Set, str, bytes, bytearray)):
        return None
    try:
        point_iterator = iter(point)  # type: ignore[arg-type]
        x_value = next(point_iterator)
        y_value = next(point_iterator)
    except (TypeError, StopIteration, ValueError):
        return None
    try:
        next(point_iterator)
    except StopIteration:
        return x_value, y_value
    return None


def _coerce_resolution(
    resolution: Resolution | Sequence[int],
    field_name: str,
) -> Resolution:
    if isinstance(resolution, Resolution):
        checked_resolution = resolution
    else:
        if isinstance(resolution, (Mapping, Set, str, bytes, bytearray)):
            raise ValueError(f'{field_name} must contain width and height')
        try:
            width, height = resolution
        except (TypeError, ValueError):
            raise ValueError(f'{field_name} must contain width and height') from None
        checked_resolution = Resolution(width, height)
    if (
        not isinstance(checked_resolution.width, int)
        or isinstance(checked_resolution.width, bool)
        or checked_resolution.width <= 0
        or not isinstance(checked_resolution.height, int)
        or isinstance(checked_resolution.height, bool)
        or checked_resolution.height <= 0
    ):
        raise ValueError(f'{field_name} must contain positive integer dimensions')
    return checked_resolution


def _coerce_bounds(bounds: BoundsLike) -> CoordinateBounds | None:
    if isinstance(bounds, CoordinateBounds):
        return bounds if _is_valid_bounds(bounds) else None
    try:
        resolution = _coerce_resolution(bounds, 'bounds')
    except (TypeError, ValueError):
        return None
    return CoordinateBounds(0, 0, resolution.width, resolution.height)


def _normalise_polygon(polygon: RegionLike) -> Polygon | None:
    if isinstance(polygon, CoordinateBounds):
        if not _is_valid_bounds(polygon):
            return None
        points = (
            Point2D(polygon.left, polygon.top),
            Point2D(polygon.right, polygon.top),
            Point2D(polygon.right, polygon.bottom),
            Point2D(polygon.left, polygon.bottom),
        )
    else:
        if isinstance(polygon, (Mapping, Set, str, bytes, bytearray)):
            return None
        try:
            points = tuple(coerce_point(point) for point in polygon)
        except (TypeError, ValueError, OverflowError):
            return None

    distinct_points: list[Point2D] = []
    for point in points:
        if len(distinct_points) == 0 or point != distinct_points[-1]:
            distinct_points.append(point)
    if len(distinct_points) > 1 and distinct_points[0] == distinct_points[-1]:
        distinct_points.pop()
    if len(distinct_points) < 3 or not _has_nonzero_polygon_area(distinct_points):
        return None
    return tuple(distinct_points)


def _clip_polygon_against_boundary(
    polygon: Polygon,
    axis: str,
    boundary: float,
    keeps_greater: bool,
) -> Polygon | None:
    clipped_points: list[Point2D] = []
    previous_point = polygon[-1]
    previous_value = previous_point.x if axis == 'x' else previous_point.y
    previous_inside = (
        previous_value >= boundary
        if keeps_greater
        else previous_value <= boundary
    )
    for current_point in polygon:
        current_value = current_point.x if axis == 'x' else current_point.y
        current_inside = (
            current_value >= boundary
            if keeps_greater
            else current_value <= boundary
        )
        if current_inside != previous_inside:
            value_delta = current_value - previous_value
            fraction = (boundary - previous_value) / value_delta
            intersection_x = previous_point.x + fraction * (
                current_point.x - previous_point.x
            )
            intersection_y = previous_point.y + fraction * (
                current_point.y - previous_point.y
            )
            if axis == 'x':
                intersection_x = boundary
            else:
                intersection_y = boundary
            clipped_points.append(Point2D(intersection_x, intersection_y))
        if current_inside:
            clipped_points.append(current_point)
        previous_point = current_point
        previous_value = current_value
        previous_inside = current_inside

    if len(clipped_points) == 0:
        return None
    return _normalise_polygon(clipped_points)


def _is_valid_bounds(bounds: CoordinateBounds) -> bool:
    return (
        is_finite_real(bounds.left)
        and is_finite_real(bounds.top)
        and is_finite_real(bounds.right)
        and is_finite_real(bounds.bottom)
        and bounds.left < bounds.right
        and bounds.top < bounds.bottom
    )


def _has_nonzero_polygon_area(polygon: list[Point2D]) -> bool:
    largest_coordinate = max(
        max(abs(point.x), abs(point.y))
        for point in polygon
    )
    if largest_coordinate == 0:
        return False

    scaled_polygon = [
        Point2D(point.x / largest_coordinate, point.y / largest_coordinate)
        for point in polygon
    ]
    twice_area = 0.0
    previous_point = scaled_polygon[-1]
    for current_point in scaled_polygon:
        twice_area += (
            previous_point.x * current_point.y
            - current_point.x * previous_point.y
        )
        previous_point = current_point
    return math.isfinite(twice_area) and abs(twice_area) > 1e-12


def _is_point_on_segment(
    point: Point2D,
    segment_start: Point2D,
    segment_end: Point2D,
) -> bool:
    cross_product = (
        (point.y - segment_start.y) * (segment_end.x - segment_start.x)
        - (point.x - segment_start.x) * (segment_end.y - segment_start.y)
    )
    if abs(cross_product) > 1e-9:
        return False
    return (
        min(segment_start.x, segment_end.x) <= point.x <= max(segment_start.x, segment_end.x)
        and min(segment_start.y, segment_end.y) <= point.y <= max(segment_start.y, segment_end.y)
    )


def _normalise_homography(matrix: MatrixLike) -> tuple[tuple[float, float, float], ...]:
    if isinstance(matrix, (Mapping, Set)):
        raise InvalidHomographyError('Homography must be a 3x3 matrix')
    try:
        raw_rows = tuple(matrix)
    except (TypeError, ValueError):
        raise InvalidHomographyError('Homography must be a 3x3 matrix') from None
    if any(isinstance(row, (Mapping, Set)) for row in raw_rows):
        raise InvalidHomographyError('Homography must be a 3x3 matrix')
    try:
        rows = tuple(tuple(row) for row in raw_rows)
    except (TypeError, ValueError):
        raise InvalidHomographyError('Homography must be a 3x3 matrix') from None
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise InvalidHomographyError('Homography must be a 3x3 matrix')
    if any(not is_finite_real(value) for row in rows for value in row):
        raise InvalidHomographyError('Homography must contain finite numbers')
    largest_value = max(abs(value) for row in rows for value in row)
    if largest_value == 0:
        raise InvalidHomographyError('Homography must not be all zero')
    normalised_matrix = tuple(
        tuple(float(value) / largest_value for value in row)
        for row in rows
    )
    if abs(_determinant(normalised_matrix)) <= 1e-12:
        raise InvalidHomographyError('Homography is degenerate')
    return normalised_matrix


def _determinant(matrix: tuple[tuple[float, float, float], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


__all__ = [
    'BoundsLike',
    'CoordinateBounds',
    'HomographyPair',
    'MatrixLike',
    'Polygon',
    'Point2D',
    'PointLike',
    'PreviewTransform',
    'RegionLike',
    'build_preview_transform',
    'calculate_available_projector_area',
    'calculate_convex_hull',
    'calculate_polygon_area',
    'camera_to_projector',
    'coerce_point',
    'intersect_polygon_with_bounds',
    'invert_homography',
    'is_finite_point',
    'is_point_in_bounds',
    'is_point_in_region',
    'is_point_in_resolution',
    'is_valid_homography',
    'preview_local_to_camera_native',
    'project_camera_to_projector',
    'project_point',
    'project_polygon',
    'projector_to_camera',
    'validate_homography',
    'validate_point_in_region',
]
