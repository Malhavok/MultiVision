"""Homography calibration and quality checks for one camera."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Real
from typing import (
    Any,
    NamedTuple,
)

from multivision.config import CalibrationThresholds
from multivision.errors import (
    CalibrationError,
    MultiVisionError,
)
from multivision.fiducials import CameraCorrespondences, FiducialCorrespondence
from multivision.geometry import (
    CoordinateBounds,
    HomographyPair,
    Point2D,
    calculate_convex_hull,
    calculate_polygon_area,
    is_finite_point,
    is_point_in_bounds,
    is_point_in_resolution,
    project_point,
)
from multivision.pattern import (
    APRILTAG_FAMILIES,
    CalibrationMarker,
    CalibrationPattern,
    SUPPORTED_MARKER_COUNTS,
)
from multivision.types import (
    Resolution,
    is_finite_real,
    is_valid_resolution,
)


class CalibrationMetrics(NamedTuple):
    """Measured quality of one RANSAC calibration."""

    unique_tag_count: int
    correspondence_corner_count: int
    ransac_inlier_count: int
    inlier_ratio: float
    mean_reprojection_error: float
    max_reprojection_error: float
    spatial_coverage: float
    capture_median_sigma_pixels: float | None = None
    capture_p95_sigma_pixels: float | None = None
    capture_max_sigma_pixels: float | None = None


class CalibrationResult(NamedTuple):
    """A validated camera/projector transform and its supported camera region."""

    homography: HomographyPair
    valid_region: tuple[Point2D, ...]
    metrics: CalibrationMetrics
    camera_id: str | None = None

    @property
    def projector_to_camera(self) -> tuple[tuple[float, float, float], ...]:
        return self.homography.projector_to_camera

    @property
    def camera_to_projector(self) -> tuple[tuple[float, float, float], ...]:
        return self.homography.camera_to_projector


def calibrate_homography(
    correspondences: CameraCorrespondences | Iterable[FiducialCorrespondence],
    pattern: CalibrationPattern,
    thresholds: CalibrationThresholds | None = None,
    ransac_reprojection_threshold: float = 3.0,
    cv2_module: Any | None = None,
    camera_resolution: Resolution | None = None,
) -> CalibrationResult:
    """Estimate and quality-check a projector-to-camera homography with RANSAC."""
    if not isinstance(pattern, CalibrationPattern):
        raise CalibrationError('pattern must be CalibrationPattern')
    _validate_pattern(pattern)
    checked_thresholds = thresholds if thresholds is not None else CalibrationThresholds()
    if not isinstance(checked_thresholds, CalibrationThresholds):
        raise CalibrationError('thresholds must be CalibrationThresholds')
    if camera_resolution is not None and not is_valid_resolution(camera_resolution):
        raise CalibrationError('camera_resolution must be a positive resolution')
    try:
        checked_ransac_reprojection_threshold = float(ransac_reprojection_threshold)
    except (OverflowError, TypeError, ValueError):
        raise CalibrationError(
            'ransac_reprojection_threshold must be positive and finite',
        ) from None
    if (
        not isinstance(ransac_reprojection_threshold, Real)
        or isinstance(ransac_reprojection_threshold, bool)
        or not math.isfinite(checked_ransac_reprojection_threshold)
        or checked_ransac_reprojection_threshold <= 0
    ):
        raise CalibrationError('ransac_reprojection_threshold must be positive and finite')

    checked_correspondences = _normalise_correspondences(correspondences)
    _validate_correspondences_against_pattern(checked_correspondences, pattern)
    if len(checked_correspondences) < 4:
        raise CalibrationError('At least four correspondence corners are required')
    unique_tag_count = len({item.marker_id for item in checked_correspondences})
    if unique_tag_count < checked_thresholds.min_unique_tags:
        raise CalibrationError(
            f'Calibration needs at least {checked_thresholds.min_unique_tags} unique tags',
        )

    projector_points = tuple(item.projector_position for item in checked_correspondences)
    camera_points = tuple(item.camera_position for item in checked_correspondences)
    _validate_projector_points(projector_points, pattern)
    if camera_resolution is not None and any(
        not is_point_in_resolution(point, camera_resolution)
        for point in camera_points
    ):
        raise CalibrationError(
            'Correspondences contain camera points outside the native resolution',
        )
    projector_hull = calculate_convex_hull(projector_points)
    if len(projector_hull) < 3 or len(calculate_convex_hull(camera_points)) < 3:
        raise CalibrationError('Correspondences must span two-dimensional regions')
    initial_spatial_coverage = _calculate_spatial_coverage(projector_hull, pattern)
    if initial_spatial_coverage < checked_thresholds.min_spatial_coverage:
        raise CalibrationError(
            f'Calibration spatial coverage is too small: {initial_spatial_coverage!r}',
        )

    cv2 = _load_cv2(cv2_module)
    try:
        import numpy
    except ImportError as ex:
        raise CalibrationError('NumPy is required for homography calibration') from ex

    source_array = numpy.asarray(
        [[point.x, point.y] for point in projector_points],
        dtype=numpy.float64,
    )
    destination_array = numpy.asarray(
        [[point.x, point.y] for point in camera_points],
        dtype=numpy.float64,
    )
    try:
        matrix, raw_mask = cv2.findHomography(
            source_array,
            destination_array,
            cv2.RANSAC,
            checked_ransac_reprojection_threshold,
        )
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise CalibrationError('Homography estimation failed') from ex

    try:
        homography = HomographyPair.from_projector_to_camera(matrix)
    except (TypeError, ValueError, MultiVisionError) as ex:
        raise CalibrationError('OpenCV returned an invalid homography') from ex

    inlier_mask = _normalise_inlier_mask(raw_mask, len(checked_correspondences))
    ransac_inlier_count = sum(inlier_mask)
    if ransac_inlier_count < 4:
        raise CalibrationError('RANSAC did not find enough inlier corners')
    inlier_ratio = ransac_inlier_count / len(checked_correspondences)
    if inlier_ratio < checked_thresholds.min_inlier_ratio:
        raise CalibrationError(f'Calibration inlier ratio is too small: {inlier_ratio!r}')

    inlier_projector_points = tuple(
        point
        for point, is_inlier in zip(projector_points, inlier_mask)
        if is_inlier
    )
    inlier_camera_points = tuple(
        point
        for point, is_inlier in zip(camera_points, inlier_mask)
        if is_inlier
    )
    projector_hull = calculate_convex_hull(inlier_projector_points)
    camera_hull = calculate_convex_hull(inlier_camera_points)
    if len(projector_hull) < 3 or len(camera_hull) < 3:
        raise CalibrationError('RANSAC inliers must span two-dimensional regions')

    spatial_coverage = _calculate_spatial_coverage(projector_hull, pattern)
    if spatial_coverage < checked_thresholds.min_spatial_coverage:
        raise CalibrationError(
            f'Calibration spatial coverage is too small: {spatial_coverage!r}',
        )

    reprojection_errors = _calculate_reprojection_errors(
        projector_points,
        camera_points,
        homography,
        inlier_mask,
    )
    if len(reprojection_errors) == 0:
        raise CalibrationError('Calibration produced no usable reprojection metrics')
    mean_reprojection_error = sum(reprojection_errors) / len(reprojection_errors)
    max_reprojection_error = max(reprojection_errors)
    if (
        mean_reprojection_error > checked_thresholds.max_mean_reprojection_error
        or max_reprojection_error > checked_thresholds.max_reprojection_error
    ):
        raise CalibrationError(
            'Calibration reprojection error exceeds configured thresholds',
        )

    valid_region = _expand_convex_hull(camera_hull, checked_thresholds.valid_region_margin)
    metrics = CalibrationMetrics(
        unique_tag_count,
        len(checked_correspondences),
        ransac_inlier_count,
        inlier_ratio,
        mean_reprojection_error,
        max_reprojection_error,
        spatial_coverage,
    )
    camera_id = (
        correspondences.camera_id
        if isinstance(correspondences, CameraCorrespondences)
        else None
    )
    return CalibrationResult(homography, valid_region, metrics, camera_id)


def _validate_pattern(pattern: CalibrationPattern) -> None:
    if (
        not is_valid_resolution(pattern.projector_resolution)
        or not is_finite_real(pattern.projector_resolution.width)
        or not is_finite_real(pattern.projector_resolution.height)
    ):
        raise CalibrationError('pattern has an invalid projector resolution')
    if not isinstance(pattern.usable_area, CoordinateBounds):
        raise CalibrationError('pattern has an invalid usable area')
    usable_area = pattern.usable_area
    if (
        not all(
            is_finite_real(value)
            for value in (
                usable_area.left,
                usable_area.top,
                usable_area.right,
                usable_area.bottom,
            )
        )
        or not usable_area.left < usable_area.right
        or not usable_area.top < usable_area.bottom
        or usable_area.left < 0
        or usable_area.top < 0
        or usable_area.right > pattern.projector_resolution.width
        or usable_area.bottom > pattern.projector_resolution.height
    ):
        raise CalibrationError('pattern has an invalid usable area')
    if (
        not isinstance(pattern.marker_family, str)
        or pattern.marker_family not in APRILTAG_FAMILIES
    ):
        raise CalibrationError('pattern has an unsupported marker family')
    if (
        not isinstance(pattern.marker_size, Real)
        or isinstance(pattern.marker_size, bool)
        or not is_finite_real(pattern.marker_size)
        or pattern.marker_size <= 0
    ):
        raise CalibrationError('pattern has an invalid marker size')
    if (
        not isinstance(pattern.markers, tuple)
        or len(pattern.markers) not in SUPPORTED_MARKER_COUNTS
    ):
        raise CalibrationError(
            'pattern must contain 9, 10, 11, 12 or 20 markers',
        )

    marker_ids: set[int] = set()
    for marker in pattern.markers:
        if not isinstance(marker, CalibrationMarker):
            raise CalibrationError('pattern contains an invalid marker')
        if (
            not isinstance(marker.marker_id, int)
            or isinstance(marker.marker_id, bool)
            or marker.marker_id < 0
            or marker.marker_id in marker_ids
            or not isinstance(marker.corners, tuple)
            or len(marker.corners) != 4
            or not all(isinstance(corner, Point2D) for corner in marker.corners)
            or any(
                not is_finite_point(corner)
                or not is_point_in_bounds(corner, usable_area)
                for corner in marker.corners
            )
            or len(set(marker.corners)) != 4
            or not math.isfinite(marker_area := calculate_polygon_area(marker.corners))
            or marker_area <= 0
        ):
            raise CalibrationError('pattern contains an invalid marker')
        marker_ids.add(marker.marker_id)


def validate_correspondences_against_pattern(
    correspondences: Sequence[FiducialCorrespondence],
    pattern: CalibrationPattern,
) -> None:
    if not isinstance(pattern, CalibrationPattern):
        raise CalibrationError('pattern must be CalibrationPattern')
    _validate_pattern(pattern)
    _validate_correspondences_against_pattern(
        _normalise_correspondences(correspondences),
        pattern,
    )


def _validate_correspondences_against_pattern(
    correspondences: Sequence[FiducialCorrespondence],
    pattern: CalibrationPattern,
) -> None:
    pattern_markers = {marker.marker_id: marker for marker in pattern.markers}
    corner_counts: dict[int, int] = {}
    for correspondence in correspondences:
        marker = pattern_markers.get(correspondence.marker_id)
        if marker is None:
            raise CalibrationError(
                f'Correspondence refers to unknown marker {correspondence.marker_id}',
            )
        expected_projector_position = marker.corners[correspondence.corner_index]
        if correspondence.projector_position != expected_projector_position:
            raise CalibrationError('Correspondence projector corner does not match pattern')
        corner_counts[correspondence.marker_id] = (
            corner_counts.get(correspondence.marker_id, 0) + 1
        )
    if any(corner_count != 4 for corner_count in corner_counts.values()):
        raise CalibrationError('Every detected marker must provide all four corners')


def _normalise_correspondences(
    correspondences: CameraCorrespondences | Iterable[FiducialCorrespondence],
) -> tuple[FiducialCorrespondence, ...]:
    raw_values = (
        correspondences.correspondences
        if isinstance(correspondences, CameraCorrespondences)
        else correspondences
    )
    try:
        values = tuple(raw_values)
    except Exception as ex:  # noqa: BLE001 (the correspondence boundary is caller input).
        raise CalibrationError('correspondences must be iterable') from ex

    normalised: list[FiducialCorrespondence] = []
    seen_corners: set[tuple[int, int]] = set()
    for correspondence in values:
        if not isinstance(correspondence, FiducialCorrespondence):
            raise CalibrationError('correspondences must contain FiducialCorrespondence values')
        if (
            not isinstance(correspondence.marker_id, int)
            or isinstance(correspondence.marker_id, bool)
            or correspondence.marker_id < 0
            or not isinstance(correspondence.corner_index, int)
            or isinstance(correspondence.corner_index, bool)
            or not 0 <= correspondence.corner_index < 4
            or not is_finite_point(correspondence.projector_position)
            or not is_finite_point(correspondence.camera_position)
        ):
            raise CalibrationError('Correspondences contain an invalid marker corner')
        corner_key = (correspondence.marker_id, correspondence.corner_index)
        if corner_key in seen_corners:
            raise CalibrationError('Correspondences contain a duplicate marker corner')
        seen_corners.add(corner_key)
        normalised.append(
            FiducialCorrespondence(
                correspondence.marker_id,
                correspondence.corner_index,
                Point2D(*correspondence.projector_position),
                Point2D(*correspondence.camera_position),
            ),
        )
    return tuple(normalised)


def _calculate_spatial_coverage(
    projector_hull: Sequence[Point2D],
    pattern: CalibrationPattern,
) -> float:
    usable_area = pattern.usable_area
    usable_area_size = (usable_area.right - usable_area.left) * (
        usable_area.bottom - usable_area.top
    )
    if not math.isfinite(usable_area_size) or usable_area_size <= 0:
        raise CalibrationError('Calibration usable area has an invalid size')
    spatial_coverage = calculate_polygon_area(projector_hull) / usable_area_size
    if not math.isfinite(spatial_coverage):
        raise CalibrationError('Calibration spatial coverage is not finite')
    return spatial_coverage


def _validate_projector_points(
    projector_points: Sequence[Point2D],
    pattern: CalibrationPattern,
) -> None:
    if any(not pattern.usable_area.contains(point) for point in projector_points):
        raise CalibrationError('Correspondences contain projector points outside the usable area')
    if len(set(projector_points)) < 4:
        raise CalibrationError('Correspondences contain too few distinct projector points')


def _load_cv2(cv2_module: Any | None) -> Any:
    if cv2_module is not None:
        if not hasattr(cv2_module, 'findHomography') or not hasattr(cv2_module, 'RANSAC'):
            raise CalibrationError('cv2_module must provide findHomography and RANSAC')
        return cv2_module
    try:
        import cv2
    except ImportError as ex:
        raise CalibrationError('OpenCV is not installed') from ex
    return cv2


def _normalise_inlier_mask(raw_mask: object, correspondence_count: int) -> tuple[bool, ...]:
    if raw_mask is None:
        raise CalibrationError('OpenCV returned no RANSAC inlier mask')
    try:
        mask_values = tuple(raw_mask)
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise CalibrationError('OpenCV returned an invalid RANSAC inlier mask') from ex
    if len(mask_values) != correspondence_count:
        raise CalibrationError('OpenCV returned an incomplete RANSAC inlier mask')
    normalised: list[bool] = []
    for value in mask_values:
        try:
            flattened_value = tuple(value)
        except TypeError:
            flattened_value = (value,)
        if len(flattened_value) != 1:
            raise CalibrationError('OpenCV returned an invalid RANSAC inlier mask')
        mask_value = flattened_value[0]
        if (
            not isinstance(mask_value, Real)
            or isinstance(mask_value, bool)
            or not math.isfinite(mask_value)
            or mask_value not in (0, 1)
        ):
            raise CalibrationError('OpenCV returned an invalid RANSAC inlier mask')
        normalised.append(bool(mask_value == 1))
    return tuple(normalised)


def _calculate_reprojection_errors(
    projector_points: Sequence[Point2D],
    camera_points: Sequence[Point2D],
    homography: HomographyPair,
    inlier_mask: Sequence[bool],
) -> tuple[float, ...]:
    errors: list[float] = []
    for projector_point, camera_point, is_inlier in zip(
        projector_points,
        camera_points,
        inlier_mask,
    ):
        if not is_inlier:
            continue
        try:
            projected_point = project_point(projector_point, homography.projector_to_camera)
        except MultiVisionError as ex:
            raise CalibrationError('Homography has an invalid reprojection') from ex
        error = math.hypot(
            projected_point.x - camera_point.x,
            projected_point.y - camera_point.y,
        )
        if not math.isfinite(error):
            raise CalibrationError('Calibration produced a non-finite reprojection error')
        errors.append(error)
    return tuple(errors)


def _expand_convex_hull(
    hull: Sequence[Point2D],
    margin: float,
) -> tuple[Point2D, ...]:
    if margin == 0:
        return tuple(hull)
    centre_x = sum(point.x for point in hull) / len(hull)
    centre_y = sum(point.y for point in hull) / len(hull)
    expanded_hull = tuple(
        Point2D(
            centre_x + (point.x - centre_x) * (1 + margin),
            centre_y + (point.y - centre_y) * (1 + margin),
        )
        for point in hull
    )
    if any(not is_finite_point(point) for point in expanded_hull):
        raise CalibrationError('Calibration valid region is not finite')
    return expanded_hull


__all__ = [
    'CalibrationError',
    'CalibrationMetrics',
    'CalibrationResult',
    'calibrate_homography',
    'validate_correspondences_against_pattern',
]
