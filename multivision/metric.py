"""Pure metric units and surface-to-projector geometry."""

from __future__ import annotations

import enum
import math
import threading
import time
from collections.abc import Iterable, Mapping, Sequence, Set
from numbers import Real
from typing import (
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
)

from multivision.config import (
    MetricCalibrationThresholds,
    ProjectorOutputDescriptor,
)
from multivision.errors import (
    CalibrationError,
    InvalidHomographyError,
    PointOutsideProjectorError,
)
from multivision.fiducials import (
    MetricTargetCorrespondence,
    MetricTargetCorrespondences,
)
from multivision.geometry import (
    CoordinateBounds,
    HomographyPair,
    MatrixLike,
    Point2D,
    PointLike,
    calculate_convex_hull,
    calculate_polygon_area,
    coerce_point,
    invert_homography,
    is_point_in_bounds,
    project_point,
    project_polygon,
    validate_homography,
)
from multivision.metric_target import (
    METRIC_TARGET,
    MetricTarget,
    validate_metric_target,
)
from multivision.types import (
    Resolution,
    is_finite_real,
    is_valid_resolution,
)


MetricUnit: TypeAlias = Literal['mm', 'cm', 'in']
MetricBounds: TypeAlias = CoordinateBounds | Resolution | Sequence[int]

UNIT_TO_MM: dict[MetricUnit, float] = {
    'mm': 1.0,
    'cm': 10.0,
    'in': 25.4,
}
MAX_METRIC_RULER_TICKS = 200
RULER_TICK_SPACING_SOURCE_UNITS = 5.0
METRIC_RULER_RASTER_MARGIN_PIXELS = 1


class MetricCalibrationStatus(str, enum.Enum):
    """Trust state for the one session-local metric calibration."""

    UNCALIBRATED = 'UNCALIBRATED'
    CALIBRATED = 'CALIBRATED'
    STALE = 'STALE'


class MetricValidationRecord(NamedTuple):
    """An optional, independently measured session validation observation."""

    requested_length_mm: float
    observed_length_mm: float
    absolute_error_mm: float
    timestamp: float


class MetricCalibrationRecord(NamedTuple):
    """The single shared metric calibration record for one service session."""

    state: MetricCalibrationStatus
    projector_output_descriptor: ProjectorOutputDescriptor
    homography: 'MetricHomographyPair | None' = None
    observation_camera_slot: str | None = None
    observation_camera_id: str | None = None
    observation_camera_calibration_version: int | None = None
    observation_camera_calibration_timestamp: float | None = None
    target_format: str | None = None
    target_version: int | None = None
    marker_family: str | None = None
    metrics: 'MetricCalibrationMetrics | None' = None
    timestamp: float | None = None
    validation_records: tuple[MetricValidationRecord, ...] = ()
    latest_physical_validation_error_mm: float | None = None

    @property
    def projector_resolution(self) -> Resolution:
        return self.projector_output_descriptor.projector_resolution

    @property
    def output_identity(self) -> str:
        return self.projector_output_descriptor.output_identity

    @property
    def projector_to_surface(self) -> tuple[tuple[float, float, float], ...] | None:
        return None if self.homography is None else self.homography.projector_to_surface

    @property
    def surface_to_projector(self) -> tuple[tuple[float, float, float], ...] | None:
        return None if self.homography is None else self.homography.surface_to_projector

    @property
    def fit_error_mm(self) -> float | None:
        return None if self.metrics is None else self.metrics.mean_fit_error_mm

    @property
    def format_version(self) -> int | None:
        return self.target_version

    @property
    def format_name(self) -> str | None:
        return self.target_format

    @property
    def calibration_timestamp(self) -> float | None:
        return self.timestamp


class MetricCalibrationRegistry:
    """Own exactly one non-persistent metric record for a service session."""

    def __init__(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> None:
        if projector_output_descriptor is not None and not isinstance(
            projector_output_descriptor,
            ProjectorOutputDescriptor,
        ):
            raise ValueError(
                'projector_output_descriptor must be ProjectorOutputDescriptor',
            )
        self._projector_output_descriptor = projector_output_descriptor
        self._record: MetricCalibrationRecord | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> MetricCalibrationStatus:
        with self._lock:
            return (
                MetricCalibrationStatus.UNCALIBRATED
                if self._record is None
                else self._record.state
            )

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor | None:
        with self._lock:
            return self._projector_output_descriptor

    @property
    def record(self) -> MetricCalibrationRecord | None:
        with self._lock:
            return self._record

    def get_status(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> MetricCalibrationStatus:
        """Return the current shared metric trust state."""
        if projector_output_descriptor is not None:
            self.is_usable(projector_output_descriptor)
        return self.state

    def get_record(self) -> MetricCalibrationRecord | None:
        """Return the one shared record, without creating per-camera state."""
        with self._lock:
            return self._record

    def register(
        self,
        result: MetricCalibrationResult,
        projector_output_descriptor: ProjectorOutputDescriptor,
        observation_camera_slot: str | None = None,
        observation_camera_calibration: object | None = None,
        timestamp: float | None = None,
    ) -> MetricCalibrationRecord:
        if not isinstance(result, MetricCalibrationResult):
            raise ValueError('result must be MetricCalibrationResult')
        checked_homography = _validate_metric_calibration_result(result)
        _validate_metric_descriptor(projector_output_descriptor)
        if result.projector_resolution != projector_output_descriptor.projector_resolution:
            raise ValueError(
                'Metric result resolution does not match projector output descriptor',
            )
        if not isinstance(observation_camera_slot, (str, type(None))):
            raise ValueError('observation_camera_slot must be a string or None')
        resolved_timestamp = time.time() if timestamp is None else timestamp
        if not is_finite_real(resolved_timestamp):
            raise ValueError('timestamp must be finite')
        calibration_version = getattr(
            observation_camera_calibration,
            'version',
            getattr(observation_camera_calibration, 'calibration_version', None),
        )
        calibration_timestamp = getattr(
            observation_camera_calibration,
            'timestamp',
            None,
        )
        if calibration_version is not None and (
            not isinstance(calibration_version, int)
            or isinstance(calibration_version, bool)
            or calibration_version <= 0
        ):
            raise ValueError('observation camera calibration version is invalid')
        if calibration_timestamp is not None and not is_finite_real(calibration_timestamp):
            raise ValueError('observation camera calibration timestamp is invalid')
        record = MetricCalibrationRecord(
            MetricCalibrationStatus.CALIBRATED,
            projector_output_descriptor,
            checked_homography,
            observation_camera_slot,
            result.observation_camera_id,
            calibration_version,
            None if calibration_timestamp is None else float(calibration_timestamp),
            result.target_format,
            result.target_version,
            result.marker_family,
            result.metrics,
            float(resolved_timestamp),
        )
        with self._lock:
            if (
                self._projector_output_descriptor is not None
                and projector_output_descriptor != self._projector_output_descriptor
            ):
                raise ValueError(
                    'Metric calibration descriptor does not match the active projector output',
                )
            self._projector_output_descriptor = projector_output_descriptor
            self._record = record
        return record

    def update_projector_descriptor(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
    ) -> MetricCalibrationStatus:
        _validate_metric_descriptor(projector_output_descriptor)
        with self._lock:
            if (
                self._record is not None
                and self._record.projector_output_descriptor != projector_output_descriptor
            ):
                self._record = self._record._replace(
                    state=MetricCalibrationStatus.STALE,
                )
            self._projector_output_descriptor = projector_output_descriptor
            return self.state

    def add_validation_record(
        self,
        requested_length: object,
        observed_length: object,
        requested_unit: object = 'mm',
        observed_unit: object = 'mm',
        timestamp: float | None = None,
    ) -> MetricValidationRecord:
        """Append one independently measured physical validation observation."""
        with self._lock:
            if not self.is_usable(self._projector_output_descriptor):
                raise ValueError(
                    'A current metric calibration is required for validation',
                )
            if self._record is None:
                raise ValueError('A metric calibration is required for validation')
            requested_length_mm = normalise_length_to_mm(
                requested_length,
                requested_unit,
            )
            observed_length_mm = normalise_length_to_mm(
                observed_length,
                observed_unit,
            )
            if requested_length_mm <= 0 or observed_length_mm <= 0:
                raise ValueError('Validation lengths must be positive')
            resolved_timestamp = time.time() if timestamp is None else timestamp
            if not is_finite_real(resolved_timestamp):
                raise ValueError('timestamp must be finite')
            validation = MetricValidationRecord(
                requested_length_mm,
                observed_length_mm,
                abs(observed_length_mm - requested_length_mm),
                float(resolved_timestamp),
            )
            self._record = self._record._replace(
                validation_records=self._record.validation_records + (validation,),
                latest_physical_validation_error_mm=validation.absolute_error_mm,
            )
            return validation

    def clear(self) -> None:
        with self._lock:
            self._record = None

    def is_usable(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> bool:
        with self._lock:
            descriptor = (
                projector_output_descriptor
                if projector_output_descriptor is not None
                else self._projector_output_descriptor
            )
            if self._record is None or self._record.state is not MetricCalibrationStatus.CALIBRATED:
                return False
            if descriptor is None or self._record.projector_output_descriptor != descriptor:
                if descriptor is not None:
                    self._record = self._record._replace(
                        state=MetricCalibrationStatus.STALE,
                    )
                return False
            try:
                _validate_metric_calibration_result(
                    MetricCalibrationResult(
                        self._record.homography,
                        self._record.metrics,
                        self._record.projector_resolution,
                        self._record.target_format,
                        self._record.target_version,
                        self._record.marker_family,
                    ),
                )
                _validate_metric_validation_records(self._record)
            except (TypeError, ValueError, CalibrationError):
                self._record = self._record._replace(
                    state=MetricCalibrationStatus.STALE,
                )
                return False
            return True


def _validate_metric_validation_records(record: MetricCalibrationRecord) -> None:
    if not isinstance(record.validation_records, tuple):
        raise ValueError('Metric validation records are invalid')
    for validation in record.validation_records:
        if not isinstance(validation, MetricValidationRecord):
            raise ValueError('Metric validation records are invalid')
        if not all(
            is_finite_real(value)
            for value in (
                validation.requested_length_mm,
                validation.observed_length_mm,
                validation.absolute_error_mm,
                validation.timestamp,
            )
        ) or validation.requested_length_mm <= 0 or validation.observed_length_mm <= 0:
            raise ValueError('Metric validation records are invalid')
        if not math.isclose(
            validation.absolute_error_mm,
            abs(validation.observed_length_mm - validation.requested_length_mm),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError('Metric validation records are invalid')

    expected_latest_error = (
        None
        if len(record.validation_records) == 0
        else record.validation_records[-1].absolute_error_mm
    )
    if record.latest_physical_validation_error_mm != expected_latest_error:
        raise ValueError('Metric validation latest error is invalid')
    if record.latest_physical_validation_error_mm is not None and not is_finite_real(
        record.latest_physical_validation_error_mm,
    ):
        raise ValueError('Metric validation latest error is invalid')


def _validate_metric_calibration_result(
    result: MetricCalibrationResult,
) -> MetricHomographyPair:
    if not isinstance(result.homography, MetricHomographyPair):
        raise ValueError('Metric result must contain a homography pair')
    try:
        projector_to_surface = validate_homography(
            result.homography.projector_to_surface,
        )
        surface_to_projector = validate_homography(
            result.homography.surface_to_projector,
        )
        expected_surface_to_projector = validate_homography(
            invert_homography(projector_to_surface),
        )
    except (InvalidHomographyError, TypeError, ValueError) as ex:
        raise ValueError('Metric result contains invalid homography matrices') from ex
    if not all(
        math.isclose(
            expected_surface_to_projector[row_idx][column_idx],
            surface_to_projector[row_idx][column_idx],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        for row_idx in range(3)
        for column_idx in range(3)
    ):
        raise ValueError('Metric result homography matrices are not inverses')

    if not is_valid_resolution(result.projector_resolution):
        raise ValueError('Metric result projector resolution is invalid')
    if result.target_format != METRIC_TARGET.format_name:
        raise ValueError('Metric result target format is invalid')
    if result.target_version != METRIC_TARGET.format_version:
        raise ValueError('Metric result target version is invalid')
    if result.marker_family != METRIC_TARGET.marker_family:
        raise ValueError('Metric result marker family is invalid')
    if result.observation_camera_id is not None and not isinstance(
        result.observation_camera_id,
        str,
    ):
        raise ValueError('Metric result observation camera ID is invalid')

    metrics = result.metrics
    if not isinstance(metrics, MetricCalibrationMetrics):
        raise ValueError('Metric result metrics are invalid')
    if (
        not isinstance(metrics.unique_target_fiducial_count, int)
        or isinstance(metrics.unique_target_fiducial_count, bool)
        or metrics.unique_target_fiducial_count <= 0
        or not isinstance(metrics.correspondence_corner_count, int)
        or isinstance(metrics.correspondence_corner_count, bool)
        or metrics.correspondence_corner_count < 4
        or not isinstance(metrics.ransac_inlier_count, int)
        or isinstance(metrics.ransac_inlier_count, bool)
        or metrics.ransac_inlier_count < 4
        or metrics.ransac_inlier_count > metrics.correspondence_corner_count
        or metrics.unique_target_fiducial_count > METRIC_TARGET.marker_count
        or metrics.correspondence_corner_count
        != metrics.unique_target_fiducial_count * 4
        or not all(
            is_finite_real(value)
            for value in (
                metrics.inlier_ratio,
                metrics.mean_fit_error_mm,
                metrics.max_fit_error_mm,
                metrics.target_page_spatial_coverage,
            )
        )
        or not 0 <= metrics.inlier_ratio <= 1
        or not math.isclose(
            metrics.inlier_ratio,
            metrics.ransac_inlier_count / metrics.correspondence_corner_count,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or metrics.mean_fit_error_mm < 0
        or metrics.max_fit_error_mm < metrics.mean_fit_error_mm
        or not 0 < metrics.target_page_spatial_coverage <= 1
    ):
        raise ValueError('Metric result metrics are invalid')
    return MetricHomographyPair(projector_to_surface, surface_to_projector)


def _validate_metric_descriptor(descriptor: object) -> None:
    if not isinstance(descriptor, ProjectorOutputDescriptor):
        raise ValueError('projector_output_descriptor must be ProjectorOutputDescriptor')


class MetricHomographyPair(NamedTuple):
    """Both directions of one projector-native/surface-mm transform."""

    projector_to_surface: tuple[tuple[float, float, float], ...]
    surface_to_projector: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_projector_to_surface(
        cls: type['MetricHomographyPair'],
        matrix: MatrixLike,
    ) -> 'MetricHomographyPair':
        projector_to_surface = validate_homography(matrix)
        return cls(projector_to_surface, invert_homography(projector_to_surface))

    @classmethod
    def from_surface_to_projector(
        cls: type['MetricHomographyPair'],
        matrix: MatrixLike,
    ) -> 'MetricHomographyPair':
        surface_to_projector = validate_homography(matrix)
        return cls(invert_homography(surface_to_projector), surface_to_projector)


class ProjectedSurfaceRuler(NamedTuple):
    """A validated physical line and its projector-native endpoints."""

    surface_start: Point2D
    surface_end: Point2D
    projector_start: Point2D
    projector_end: Point2D
    length_mm: float


class MetricRulerTick(NamedTuple):
    """One deterministic physical tick and its projected line."""

    distance_mm: float
    is_major: bool
    surface_start: Point2D
    surface_end: Point2D
    projector_start: Point2D
    projector_end: Point2D
    projector_position: Point2D


class MetricRulerMarker(NamedTuple):
    """A start or end marker with its complete physical and projected extent."""

    surface_position: Point2D
    projector_position: Point2D
    surface_extent: tuple[Point2D, ...]
    projector_extent: tuple[Point2D, ...]


class MetricRulerOverlay(NamedTuple):
    """The complete, validated session-local physical ruler overlay."""

    surface_start: Point2D
    surface_end: Point2D
    projector_start: Point2D
    projector_end: Point2D
    length_mm: float
    output_unit: MetricUnit
    length_in_output_unit: float
    label: str
    ticks: tuple[MetricRulerTick, ...]
    markers: tuple[MetricRulerMarker, ...]
    label_position: Point2D
    label_bounds: CoordinateBounds

    @property
    def unit(self) -> MetricUnit:
        return self.output_unit

    @property
    def tick_positions(self) -> tuple[Point2D, ...]:
        return tuple(tick.projector_position for tick in self.ticks)

    @property
    def marker_extents(self) -> tuple[tuple[Point2D, ...], ...]:
        return tuple(marker.projector_extent for marker in self.markers)

    def to_data(self) -> dict[str, Any]:
        """Return JSON-safe ruler data for service boundaries."""
        return {
            'surface_start': _metric_point_to_data(self.surface_start),
            'surface_end': _metric_point_to_data(self.surface_end),
            'projector_start': _metric_point_to_data(self.projector_start),
            'projector_end': _metric_point_to_data(self.projector_end),
            'length_mm': self.length_mm,
            'unit': self.output_unit,
            'length': self.length_in_output_unit,
            'label': self.label,
            'ticks': [
                {
                    'distance_mm': tick.distance_mm,
                    'is_major': tick.is_major,
                    'projector_position': _metric_point_to_data(tick.projector_position),
                    'projector_start': _metric_point_to_data(tick.projector_start),
                    'projector_end': _metric_point_to_data(tick.projector_end),
                }
                for tick in self.ticks
            ],
            'markers': [
                {
                    'projector_position': _metric_point_to_data(marker.projector_position),
                    'projector_extent': [
                        _metric_point_to_data(point) for point in marker.projector_extent
                    ],
                }
                for marker in self.markers
            ],
            'label_position': _metric_point_to_data(self.label_position),
            'label_bounds': list(self.label_bounds),
        }


def _metric_point_to_data(point: Point2D) -> list[float]:
    return [point.x, point.y]


class MetricCalibrationMetrics(NamedTuple):
    """Quality measurements for one projector-to-surface calibration."""

    unique_target_fiducial_count: int
    correspondence_corner_count: int
    ransac_inlier_count: int
    inlier_ratio: float
    mean_fit_error_mm: float
    max_fit_error_mm: float
    target_page_spatial_coverage: float

    @property
    def unique_target_count(self) -> int:
        return self.unique_target_fiducial_count

    @property
    def spatial_coverage(self) -> float:
        return self.target_page_spatial_coverage

    @property
    def fit_error_mm(self) -> float:
        return self.mean_fit_error_mm


class MetricCalibrationResult(NamedTuple):
    """A complete, session-local metric transform and its diagnostics."""

    homography: MetricHomographyPair
    metrics: MetricCalibrationMetrics
    projector_resolution: Resolution
    target_format: str
    target_version: int
    marker_family: str
    observation_camera_id: str | None = None

    @property
    def projector_to_surface(self) -> tuple[tuple[float, float, float], ...]:
        return self.homography.projector_to_surface

    @property
    def surface_to_projector(self) -> tuple[tuple[float, float, float], ...]:
        return self.homography.surface_to_projector

    @property
    def format_name(self) -> str:
        return self.target_format

    @property
    def format_version(self) -> int:
        return self.target_version

    @property
    def unique_target_count(self) -> int:
        return self.metrics.unique_target_fiducial_count

    @property
    def correspondence_corner_count(self) -> int:
        return self.metrics.correspondence_corner_count

    @property
    def ransac_inlier_count(self) -> int:
        return self.metrics.ransac_inlier_count

    @property
    def inlier_ratio(self) -> float:
        return self.metrics.inlier_ratio

    @property
    def mean_fit_error_mm(self) -> float:
        return self.metrics.mean_fit_error_mm

    @property
    def max_fit_error_mm(self) -> float:
        return self.metrics.max_fit_error_mm

    @property
    def target_page_spatial_coverage(self) -> float:
        return self.metrics.target_page_spatial_coverage

    @property
    def spatial_coverage(self) -> float:
        return self.metrics.target_page_spatial_coverage

    @property
    def fit_error_mm(self) -> float:
        return self.metrics.mean_fit_error_mm


def calibrate_metric_homography(
    correspondences: MetricTargetCorrespondences | Iterable[MetricTargetCorrespondence],
    camera_to_projector: object,
    projector_resolution: MetricBounds | None = None,
    thresholds: MetricCalibrationThresholds | None = None,
    target: MetricTarget = METRIC_TARGET,
    cv2_module: Any | None = None,
    projector_bounds: MetricBounds | None = None,
) -> MetricCalibrationResult:
    """Estimate projector-native to surface-mm geometry from target corners."""
    try:
        checked_target = validate_metric_target(target)
    except (TypeError, ValueError) as ex:
        raise CalibrationError('Metric target metadata is invalid') from ex
    checked_thresholds = (
        thresholds if thresholds is not None else MetricCalibrationThresholds()
    )
    if not isinstance(checked_thresholds, MetricCalibrationThresholds):
        raise CalibrationError('thresholds must be MetricCalibrationThresholds')
    if projector_bounds is not None:
        if projector_resolution is not None:
            raise CalibrationError(
                'Specify either projector_resolution or projector_bounds, not both',
            )
        projector_resolution = projector_bounds
    if projector_resolution is None:
        raise CalibrationError('projector_resolution is required')
    try:
        checked_projector_bounds, checked_resolution = _normalise_metric_projector_bounds(
            projector_resolution,
        )
    except (TypeError, ValueError) as ex:
        raise CalibrationError('projector_resolution is invalid') from ex
    checked_correspondences, camera_id = _normalise_metric_correspondences(
        correspondences,
        checked_target,
    )
    if len(checked_correspondences) < 4:
        raise CalibrationError('At least four metric correspondence corners are required')

    camera_matrix = _get_camera_to_projector_matrix(camera_to_projector)
    projector_points: list[Point2D] = []
    surface_points: list[Point2D] = []
    for correspondence in checked_correspondences:
        try:
            projector_point = project_point(
                correspondence.camera_position,
                camera_matrix,
            )
        except (InvalidHomographyError, ValueError) as ex:
            raise CalibrationError(
                'Camera-to-projector transform cannot map metric correspondences',
            ) from ex
        if not is_point_in_bounds(projector_point, checked_projector_bounds):
            raise CalibrationError(
                f'Mapped metric correspondence is outside projector bounds: '
                f'{projector_point!r}',
            )
        projector_points.append(projector_point)
        surface_points.append(correspondence.surface_position)

    unique_target_count = len({item.marker_id for item in checked_correspondences})
    if unique_target_count < checked_thresholds.min_unique_target_fiducials:
        raise CalibrationError(
            'Metric calibration does not contain enough unique target fiducials',
        )
    if len(set(projector_points)) < 4 or len(set(surface_points)) < 4:
        raise CalibrationError('Metric correspondences contain too few distinct points')
    if (
        len(calculate_convex_hull(projector_points)) < 3
        or len(calculate_convex_hull(surface_points)) < 3
    ):
        raise CalibrationError('Metric correspondences must span two-dimensional regions')

    cv2 = _load_metric_cv2(cv2_module)
    try:
        import numpy
    except ImportError as ex:
        raise CalibrationError('NumPy is required for metric calibration') from ex
    source_array = numpy.asarray(
        [[point.x, point.y] for point in projector_points],
        dtype=numpy.float64,
    )
    destination_array = numpy.asarray(
        [[point.x, point.y] for point in surface_points],
        dtype=numpy.float64,
    )
    try:
        matrix, raw_mask = cv2.findHomography(
            source_array,
            destination_array,
            cv2.RANSAC,
            checked_thresholds.ransac_reprojection_threshold_mm,
        )
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise CalibrationError('Metric homography estimation failed') from ex

    try:
        homography = MetricHomographyPair.from_projector_to_surface(matrix)
    except (OverflowError, TypeError, ValueError, InvalidHomographyError) as ex:
        raise CalibrationError('OpenCV returned an invalid metric homography') from ex
    inlier_mask = _normalise_metric_inlier_mask(raw_mask, len(checked_correspondences))
    ransac_inlier_count = sum(inlier_mask)
    if ransac_inlier_count < 4:
        raise CalibrationError('Metric RANSAC did not find enough inlier corners')
    inlier_ratio = ransac_inlier_count / len(checked_correspondences)
    if inlier_ratio < checked_thresholds.min_inlier_ratio:
        raise CalibrationError('Metric calibration inlier ratio is too small')

    inlier_surface_points = tuple(
        point for point, is_inlier in zip(surface_points, inlier_mask) if is_inlier
    )
    if len(calculate_convex_hull(inlier_surface_points)) < 3:
        raise CalibrationError('Metric inliers must span a two-dimensional region')
    spatial_coverage = _calculate_metric_spatial_coverage(
        inlier_surface_points,
        checked_target,
    )
    if spatial_coverage < checked_thresholds.min_spatial_coverage:
        raise CalibrationError('Metric target-page spatial coverage is too small')

    fit_errors = _calculate_metric_fit_errors(
        projector_points,
        surface_points,
        homography,
        inlier_mask,
    )
    if len(fit_errors) == 0:
        raise CalibrationError('Metric calibration produced no fit residuals')
    mean_fit_error_mm = sum(fit_errors) / len(fit_errors)
    max_fit_error_mm = max(fit_errors)
    if (
        mean_fit_error_mm > checked_thresholds.max_mean_fit_error_mm
        or max_fit_error_mm > checked_thresholds.max_fit_error_mm
    ):
        raise CalibrationError('Metric fit error exceeds configured thresholds')

    return MetricCalibrationResult(
        homography,
        MetricCalibrationMetrics(
            unique_target_count,
            len(checked_correspondences),
            ransac_inlier_count,
            inlier_ratio,
            mean_fit_error_mm,
            max_fit_error_mm,
            spatial_coverage,
        ),
        checked_resolution,
        checked_target.format_name,
        checked_target.format_version,
        checked_target.marker_family,
        camera_id,
    )


def validate_metric_correspondences(
    correspondences: MetricTargetCorrespondences | Iterable[MetricTargetCorrespondence],
    target: MetricTarget = METRIC_TARGET,
) -> MetricTargetCorrespondences:
    """Validate one complete set of target-aware metric correspondences."""
    checked_correspondences, camera_id = _normalise_metric_correspondences(
        correspondences,
        validate_metric_target(target),
    )
    return MetricTargetCorrespondences(checked_correspondences, camera_id)


def normalise_unit(unit: object) -> MetricUnit:
    """Validate an external metric unit and return its canonical spelling."""
    if not isinstance(unit, str) or unit not in UNIT_TO_MM:
        raise ValueError(f'Unit must be one of mm, cm or in: {unit!r}')
    return unit  # type: ignore[return-value]


def normalise_length_to_mm(length: object, unit: object = 'mm') -> float:
    """Convert one finite metric length to millimetres."""
    checked_unit = normalise_unit(unit)
    if not is_finite_real(length):
        raise ValueError(f'Length must be finite: {length!r}')
    length_mm = float(length) * UNIT_TO_MM[checked_unit]
    if not math.isfinite(length_mm):
        raise ValueError(f'Length must remain finite in millimetres: {length!r}')
    return length_mm


def validate_positive_length(length: object, unit: object = 'mm') -> float:
    """Convert a length to millimetres, rejecting zero and negative values."""
    length_mm = normalise_length_to_mm(length, unit)
    if length_mm <= 0:
        raise ValueError(f'Length must be positive: {length!r}')
    return length_mm


def validate_finite_point(point: PointLike) -> Point2D:
    """Validate and return a finite two-dimensional point."""
    return coerce_point(point)


def calculate_surface_distance_mm(
    first_point: PointLike,
    second_point: PointLike,
) -> float:
    """Calculate the finite Euclidean distance between surface-mm points."""
    first = validate_finite_point(first_point)
    second = validate_finite_point(second_point)
    x_difference = second.x - first.x
    y_difference = second.y - first.y
    distance_mm = math.hypot(x_difference, y_difference)
    if not math.isfinite(distance_mm):
        raise ValueError('Surface distance must be finite')
    return distance_mm


def calculate_ruler_tick_layout(
    measurement_length: object,
    maximum_tick_count: object,
) -> tuple[float, int]:
    """Calculate bounded, source-space ruler ticks without changing their spacing."""
    if not is_finite_real(measurement_length) or float(measurement_length) <= 0:
        raise ValueError('Ruler measurement length must be positive and finite')
    if (
        not isinstance(maximum_tick_count, int)
        or isinstance(maximum_tick_count, bool)
        or maximum_tick_count <= 0
    ):
        raise ValueError('Ruler tick budget must be a positive integer')
    length = float(measurement_length)
    nominal_tick_count = max(
        0,
        math.ceil(length / RULER_TICK_SPACING_SOURCE_UNITS) - 1,
    )
    if nominal_tick_count == 0:
        return RULER_TICK_SPACING_SOURCE_UNITS, 0
    # Retaining multiples of the base spacing keeps a bounded ruler honest
    # instead of resampling ticks to arbitrary fractions.
    spacing_multiplier = max(
        1,
        math.ceil(nominal_tick_count / maximum_tick_count),
    )
    tick_spacing = RULER_TICK_SPACING_SOURCE_UNITS * spacing_multiplier
    if not math.isfinite(tick_spacing):
        raise ValueError('Ruler tick spacing must remain finite')
    tick_count = min(
        maximum_tick_count,
        max(0, math.ceil(length / tick_spacing) - 1),
    )
    return tick_spacing, tick_count


def surface_to_projector(
    surface_point: PointLike,
    surface_to_projector_matrix: MatrixLike | MetricHomographyPair,
) -> Point2D:
    """Project one finite surface-mm point into projector-native coordinates."""
    checked_point = validate_finite_point(surface_point)
    matrix = _surface_to_projector_matrix(surface_to_projector_matrix)
    return project_point(checked_point, matrix)


def projector_to_surface(
    projector_point: PointLike,
    projector_to_surface_matrix: MatrixLike | MetricHomographyPair,
) -> Point2D:
    """Project one finite projector-native point into surface-mm coordinates."""
    checked_point = validate_finite_point(projector_point)
    matrix = _projector_to_surface_matrix(projector_to_surface_matrix)
    return project_point(checked_point, matrix)


def calculate_projector_surface_bounds(
    projector_to_surface_matrix: (
        MatrixLike | MetricHomographyPair | MetricCalibrationRecord
    ),
    projector_bounds: MetricBounds,
) -> CoordinateBounds:
    """Return finite surface bounds covering the complete projector output."""
    checked_projector_bounds = _coerce_metric_bounds(projector_bounds)
    projector_corners = (
        Point2D(checked_projector_bounds.left, checked_projector_bounds.top),
        Point2D(checked_projector_bounds.right, checked_projector_bounds.top),
        Point2D(checked_projector_bounds.right, checked_projector_bounds.bottom),
        Point2D(checked_projector_bounds.left, checked_projector_bounds.bottom),
    )
    projector_to_surface_matrix = _projector_to_surface_matrix(
        getattr(
            projector_to_surface_matrix,
            'projector_to_surface',
            projector_to_surface_matrix,
        ),
    )
    surface_points = tuple(
        project_point(point, projector_to_surface_matrix)
        for point in projector_corners
    )
    surface_bounds = CoordinateBounds(
        min(point.x for point in surface_points),
        min(point.y for point in surface_points),
        max(point.x for point in surface_points),
        max(point.y for point in surface_points),
    )
    if (
        surface_bounds.left >= surface_bounds.right
        or surface_bounds.top >= surface_bounds.bottom
    ):
        raise InvalidHomographyError(
            'Projector output has a degenerate surface footprint',
        )
    surface_to_projector_matrix = invert_homography(projector_to_surface_matrix)
    if project_polygon(
        (
            Point2D(surface_bounds.left, surface_bounds.top),
            Point2D(surface_bounds.right, surface_bounds.top),
            Point2D(surface_bounds.right, surface_bounds.bottom),
            Point2D(surface_bounds.left, surface_bounds.bottom),
        ),
        surface_to_projector_matrix,
    ) is None:
        raise InvalidHomographyError(
            'Projector output surface footprint crosses the homography horizon',
        )
    return surface_bounds


def project_surface_points(
    surface_points: Iterable[PointLike],
    surface_to_projector_matrix: MatrixLike | MetricHomographyPair,
    projector_bounds: MetricBounds | None = None,
) -> tuple[Point2D, ...]:
    """Project finite surface points, optionally enforcing projector bounds."""
    try:
        point_iterator = iter(surface_points)
    except TypeError:
        raise ValueError('surface_points must be iterable') from None

    projected_points = tuple(
        surface_to_projector(point, surface_to_projector_matrix)
        for point in point_iterator
    )
    if projector_bounds is not None:
        checked_bounds = _coerce_metric_bounds(projector_bounds)
        if any(not is_point_in_bounds(point, checked_bounds) for point in projected_points):
            raise PointOutsideProjectorError(
                'Projected surface point is outside projector bounds',
            )
    return projected_points


def project_surface_ruler(
    surface_start: PointLike,
    surface_end: PointLike,
    surface_to_projector_matrix: MatrixLike | MetricHomographyPair,
    projector_bounds: MetricBounds | None = None,
) -> ProjectedSurfaceRuler:
    """Project a positive surface-mm line without crossing its homography horizon."""
    checked_start = validate_finite_point(surface_start)
    checked_end = validate_finite_point(surface_end)
    length_mm = calculate_surface_distance_mm(checked_start, checked_end)
    if length_mm <= 0:
        raise ValueError('Ruler endpoints must be distinct')
    matrix = _surface_to_projector_matrix(surface_to_projector_matrix)
    _validate_line_denominators(checked_start, checked_end, matrix)
    projected_start = project_point(checked_start, matrix)
    projected_end = project_point(checked_end, matrix)

    if projector_bounds is not None:
        checked_bounds = _coerce_metric_bounds(projector_bounds)
        if not is_point_in_bounds(projected_start, checked_bounds):
            raise PointOutsideProjectorError(
                f'Projected ruler start is outside projector bounds: {projected_start!r}',
            )
        if not is_point_in_bounds(projected_end, checked_bounds):
            raise PointOutsideProjectorError(
                f'Projected ruler end is outside projector bounds: {projected_end!r}',
            )
        _validate_raster_point(projected_start, checked_bounds)
        _validate_raster_point(projected_end, checked_bounds)

    return ProjectedSurfaceRuler(
        checked_start,
        checked_end,
        projected_start,
        projected_end,
        length_mm,
    )


def build_metric_ruler(
    surface_start: PointLike,
    surface_end: PointLike,
    unit: object,
    surface_to_projector_matrix: MatrixLike | MetricHomographyPair,
    projector_bounds: MetricBounds,
) -> MetricRulerOverlay:
    """Build a complete ruler without changing its requested physical line."""
    checked_unit = normalise_unit(unit)
    checked_bounds = _coerce_metric_bounds(projector_bounds)
    projected_line = project_surface_ruler(
        surface_start,
        surface_end,
        surface_to_projector_matrix,
        checked_bounds,
    )
    _validate_raster_point(projected_line.projector_start, checked_bounds, 1)
    _validate_raster_point(projected_line.projector_end, checked_bounds, 1)
    matrix = _surface_to_projector_matrix(surface_to_projector_matrix)
    direction_x = (
        projected_line.surface_end.x - projected_line.surface_start.x
    ) / projected_line.length_mm
    direction_y = (
        projected_line.surface_end.y - projected_line.surface_start.y
    ) / projected_line.length_mm
    normal_x = -direction_y
    normal_y = direction_x

    # The physical primitives are made before projection so perspective changes
    # their projector lengths – rather than being hidden by a pixel approximation.
    ticks = _build_metric_ruler_ticks(
        projected_line,
        direction_x,
        direction_y,
        normal_x,
        normal_y,
        matrix,
        checked_bounds,
    )
    markers = tuple(
        _build_metric_ruler_marker(
            endpoint,
            direction_x,
            direction_y,
            normal_x,
            normal_y,
            matrix,
            checked_bounds,
        )
        for endpoint in (
            projected_line.surface_start,
            projected_line.surface_end,
        )
    )
    length_in_output_unit = projected_line.length_mm / UNIT_TO_MM[checked_unit]
    label = f'{length_in_output_unit:.1f} {checked_unit}'
    label_position, label_bounds = _clamp_metric_ruler_label(
        projected_line,
        label,
        checked_bounds,
    )
    return MetricRulerOverlay(
        projected_line.surface_start,
        projected_line.surface_end,
        projected_line.projector_start,
        projected_line.projector_end,
        projected_line.length_mm,
        checked_unit,
        length_in_output_unit,
        label,
        ticks,
        markers,
        label_position,
        label_bounds,
    )


def _build_metric_ruler_ticks(
    projected_line: ProjectedSurfaceRuler,
    direction_x: float,
    direction_y: float,
    normal_x: float,
    normal_y: float,
    matrix: MatrixLike,
    projector_bounds: CoordinateBounds,
) -> tuple[MetricRulerTick, ...]:
    ticks: list[MetricRulerTick] = []
    tick_spacing_mm, tick_count = calculate_ruler_tick_layout(
        projected_line.length_mm,
        MAX_METRIC_RULER_TICKS,
    )
    for tick_index in range(1, tick_count + 1):
        distance_mm = tick_index * tick_spacing_mm
        is_major = distance_mm % (2.0 * RULER_TICK_SPACING_SOURCE_UNITS) == 0
        half_tick_length_mm = 5.0 if is_major else 3.0
        centre = Point2D(
            projected_line.surface_start.x + direction_x * distance_mm,
            projected_line.surface_start.y + direction_y * distance_mm,
        )
        surface_tick_start = Point2D(
            centre.x - normal_x * half_tick_length_mm,
            centre.y - normal_y * half_tick_length_mm,
        )
        surface_tick_end = Point2D(
            centre.x + normal_x * half_tick_length_mm,
            centre.y + normal_y * half_tick_length_mm,
        )
        projected_tick = project_surface_ruler(
            surface_tick_start,
            surface_tick_end,
            matrix,
            projector_bounds,
        )
        _validate_raster_point(projected_tick.projector_start, projector_bounds, 1)
        _validate_raster_point(projected_tick.projector_end, projector_bounds, 1)
        projector_position = surface_to_projector(centre, matrix)
        _validate_raster_point(projector_position, projector_bounds, 1)
        ticks.append(
            MetricRulerTick(
                distance_mm,
                is_major,
                surface_tick_start,
                surface_tick_end,
                projected_tick.projector_start,
                projected_tick.projector_end,
                projector_position,
            ),
        )
    return tuple(ticks)


def _build_metric_ruler_marker(
    surface_position: Point2D,
    direction_x: float,
    direction_y: float,
    normal_x: float,
    normal_y: float,
    matrix: MatrixLike,
    projector_bounds: CoordinateBounds,
) -> MetricRulerMarker:
    half_marker_size_mm = 4.0
    surface_extent = tuple(
        Point2D(
            surface_position.x + direction_x * direction_sign * half_marker_size_mm
            + normal_x * normal_sign * half_marker_size_mm,
            surface_position.y + direction_y * direction_sign * half_marker_size_mm
            + normal_y * normal_sign * half_marker_size_mm,
        )
        for direction_sign, normal_sign in (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    )
    projector_extent = _project_surface_extent(
        surface_extent,
        matrix,
        projector_bounds,
    )
    projector_position = surface_to_projector(surface_position, matrix)
    _validate_raster_point(projector_position, projector_bounds, 1)
    return MetricRulerMarker(
        surface_position,
        projector_position,
        surface_extent,
        projector_extent,
    )


def _project_surface_extent(
    surface_extent: Sequence[Point2D],
    matrix: MatrixLike,
    projector_bounds: CoordinateBounds,
) -> tuple[Point2D, ...]:
    if len(surface_extent) < 3:
        raise ValueError('Ruler marker extent must contain at least three points')
    checked_matrix = validate_homography(matrix)
    denominators = tuple(
        checked_matrix[2][0] * point.x
        + checked_matrix[2][1] * point.y
        + checked_matrix[2][2]
        for point in surface_extent
    )
    if any(
        not math.isfinite(denominator) or abs(denominator) <= 1e-12
        for denominator in denominators
    ) or len({denominator > 0 for denominator in denominators}) != 1:
        raise InvalidHomographyError('Ruler marker extent crosses the homography horizon')
    projected_extent = tuple(
        surface_to_projector(point, checked_matrix) for point in surface_extent
    )
    for point in projected_extent:
        if not is_point_in_bounds(point, projector_bounds):
            raise PointOutsideProjectorError(
                f'Projected ruler marker is outside projector bounds: {point!r}',
            )
        _validate_raster_point(point, projector_bounds, 1)
    return projected_extent


def _validate_raster_point(
    point: Point2D,
    projector_bounds: CoordinateBounds,
    raster_margin_pixels: int = 0,
) -> None:
    if (
        not isinstance(raster_margin_pixels, int)
        or isinstance(raster_margin_pixels, bool)
        or raster_margin_pixels < 0
    ):
        raise ValueError('raster_margin_pixels must be a non-negative integer')
    # Leave room for the renderer's one-pixel stroke, rather than allowing a
    # mathematically in-bounds primitive to be clipped at the output edge.
    rounded_point = Point2D(float(round(point.x)), float(round(point.y)))
    if not (
        projector_bounds.left + raster_margin_pixels
        <= rounded_point.x
        < projector_bounds.right - raster_margin_pixels
        and projector_bounds.top + raster_margin_pixels
        <= rounded_point.y
        < projector_bounds.bottom - raster_margin_pixels
    ):
        raise PointOutsideProjectorError(
            f'Projected ruler primitive is not raster-safe: {point!r}',
        )


def _clamp_metric_ruler_label(
    projected_line: ProjectedSurfaceRuler,
    label: str,
    projector_bounds: CoordinateBounds,
) -> tuple[Point2D, CoordinateBounds]:
    label_width = min(
        projector_bounds.right - projector_bounds.left,
        max(1.0, len(label) * 8.0 + 8.0),
    )
    label_height = min(
        projector_bounds.bottom - projector_bounds.top,
        18.0,
    )
    midpoint = Point2D(
        (projected_line.projector_start.x + projected_line.projector_end.x) / 2,
        (projected_line.projector_start.y + projected_line.projector_end.y) / 2,
    )
    left = min(
        max(midpoint.x - label_width / 2, projector_bounds.left),
        projector_bounds.right - label_width,
    )
    top = min(
        max(midpoint.y - label_height / 2, projector_bounds.top),
        projector_bounds.bottom - label_height,
    )
    bounds = CoordinateBounds(left, top, left + label_width, top + label_height)
    return Point2D(left + label_width / 2, top + label_height / 2), bounds


def _normalise_metric_projector_bounds(
    projector_resolution: MetricBounds,
) -> tuple[CoordinateBounds, Resolution]:
    if isinstance(projector_resolution, Resolution):
        checked_bounds = _coerce_metric_bounds(projector_resolution)
        return checked_bounds, projector_resolution
    if isinstance(projector_resolution, CoordinateBounds):
        checked_bounds = _coerce_metric_bounds(projector_resolution)
        if (
            checked_bounds.left != 0
            or checked_bounds.top != 0
            or not float(checked_bounds.right).is_integer()
            or not float(checked_bounds.bottom).is_integer()
        ):
            raise CalibrationError(
                'projector_bounds must be an origin-based integer projector rectangle',
            )
        return checked_bounds, Resolution(
            int(checked_bounds.right),
            int(checked_bounds.bottom),
        )
    try:
        width, height = projector_resolution
    except (TypeError, ValueError):
        raise CalibrationError('projector_resolution must contain width and height') from None
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
    ):
        raise CalibrationError('projector_resolution must contain positive integer dimensions')
    checked_resolution = Resolution(width, height)
    return _coerce_metric_bounds(checked_resolution), checked_resolution


def _normalise_metric_correspondences(
    correspondences: MetricTargetCorrespondences | Iterable[MetricTargetCorrespondence],
    target: MetricTarget,
) -> tuple[tuple[MetricTargetCorrespondence, ...], str | None]:
    camera_id = (
        correspondences.camera_id
        if isinstance(correspondences, MetricTargetCorrespondences)
        else None
    )
    raw_values = (
        correspondences.correspondences
        if isinstance(correspondences, MetricTargetCorrespondences)
        else correspondences
    )
    try:
        values = tuple(raw_values)
    except Exception as ex:  # noqa: BLE001 (correspondences are caller input).
        raise CalibrationError('Metric correspondences must be iterable') from ex

    target_markers = {marker.marker_id: marker for marker in target.markers}
    seen_corners: set[tuple[int, int]] = set()
    marker_corner_counts: dict[int, int] = {}
    normalised: list[MetricTargetCorrespondence] = []
    for correspondence in values:
        if not isinstance(correspondence, MetricTargetCorrespondence):
            raise CalibrationError(
                'Metric correspondences must contain MetricTargetCorrespondence values',
            )
        target_marker = target_markers.get(correspondence.marker_id)
        if target_marker is None:
            raise CalibrationError(
                f'Metric correspondence refers to unknown target marker '
                f'{correspondence.marker_id}',
            )
        if (
            not isinstance(correspondence.corner_index, int)
            or isinstance(correspondence.corner_index, bool)
            or not 0 <= correspondence.corner_index < 4
        ):
            raise CalibrationError('Metric correspondence has an invalid corner index')
        corner_key = (correspondence.marker_id, correspondence.corner_index)
        if corner_key in seen_corners:
            raise CalibrationError('Metric correspondences contain a duplicate corner')
        try:
            surface_position = coerce_point(correspondence.surface_position)
            camera_position = coerce_point(correspondence.camera_position)
        except ValueError as ex:
            raise CalibrationError('Metric correspondence contains an invalid point') from ex
        if surface_position != target_marker.corners[correspondence.corner_index]:
            raise CalibrationError(
                'Metric correspondence does not match the target definition',
            )
        seen_corners.add(corner_key)
        marker_corner_counts[correspondence.marker_id] = (
            marker_corner_counts.get(correspondence.marker_id, 0) + 1
        )
        normalised.append(
            MetricTargetCorrespondence(
                correspondence.marker_id,
                correspondence.corner_index,
                surface_position,
                camera_position,
            ),
        )
    if any(corner_count != 4 for corner_count in marker_corner_counts.values()):
        raise CalibrationError('Every metric target marker must provide all four corners')
    normalised.sort(key=lambda item: (item.marker_id, item.corner_index))
    return tuple(normalised), camera_id


def _get_camera_to_projector_matrix(
    camera_to_projector: object,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(camera_to_projector, HomographyPair):
        matrix = camera_to_projector.camera_to_projector
    else:
        matrix = getattr(camera_to_projector, 'camera_to_projector', camera_to_projector)
    try:
        return validate_homography(matrix)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError, InvalidHomographyError) as ex:
        raise CalibrationError(
            'The camera-to-projector transform is invalid or unavailable',
        ) from ex


def _load_metric_cv2(cv2_module: Any | None) -> Any:
    if cv2_module is not None:
        if not hasattr(cv2_module, 'findHomography') or not hasattr(cv2_module, 'RANSAC'):
            raise CalibrationError(
                'cv2_module must provide findHomography and RANSAC',
            )
        return cv2_module
    try:
        import cv2
    except ImportError as ex:
        raise CalibrationError('OpenCV is not installed') from ex
    return cv2


def _normalise_metric_inlier_mask(
    raw_mask: object,
    correspondence_count: int,
) -> tuple[bool, ...]:
    if raw_mask is None:
        raise CalibrationError('OpenCV returned no metric RANSAC inlier mask')
    try:
        mask_values = tuple(raw_mask)  # type: ignore[arg-type]
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise CalibrationError('OpenCV returned an invalid metric inlier mask') from ex
    if len(mask_values) != correspondence_count:
        raise CalibrationError('OpenCV returned an incomplete metric inlier mask')
    normalised: list[bool] = []
    for value in mask_values:
        try:
            flattened_value = tuple(value)
        except TypeError:
            flattened_value = (value,)
        if len(flattened_value) != 1:
            raise CalibrationError('OpenCV returned an invalid metric inlier mask')
        mask_value = flattened_value[0]
        if (
            isinstance(mask_value, bool)
            or not isinstance(mask_value, Real)
            or not math.isfinite(float(mask_value))
            or mask_value not in (0, 1)
        ):
            raise CalibrationError('OpenCV returned an invalid metric inlier mask')
        normalised.append(bool(mask_value == 1))
    return tuple(normalised)


def _calculate_metric_fit_errors(
    projector_points: Sequence[Point2D],
    surface_points: Sequence[Point2D],
    homography: MetricHomographyPair,
    inlier_mask: Sequence[bool],
) -> tuple[float, ...]:
    errors: list[float] = []
    for projector_point, surface_point, is_inlier in zip(
        projector_points,
        surface_points,
        inlier_mask,
    ):
        if not is_inlier:
            continue
        try:
            fitted_surface_point = project_point(
                projector_point,
                homography.projector_to_surface,
            )
        except (InvalidHomographyError, ValueError) as ex:
            raise CalibrationError('Metric homography has an invalid fit') from ex
        error = math.dist(fitted_surface_point, surface_point)
        if not math.isfinite(error):
            raise CalibrationError('Metric fit produced a non-finite residual')
        errors.append(error)
    return tuple(errors)


def _calculate_metric_spatial_coverage(
    surface_points: Sequence[Point2D],
    target: MetricTarget,
) -> float:
    page_area = target.page_width_mm * target.page_height_mm
    if not math.isfinite(page_area) or page_area <= 0:
        raise CalibrationError('Metric target page has an invalid area')
    hull_area = calculate_polygon_area(calculate_convex_hull(surface_points))
    coverage = hull_area / page_area
    if not math.isfinite(coverage):
        raise CalibrationError('Metric target-page coverage is not finite')
    return coverage


def _surface_to_projector_matrix(
    transform: MatrixLike | MetricHomographyPair,
) -> MatrixLike:
    if isinstance(transform, MetricHomographyPair):
        return transform.surface_to_projector
    return validate_homography(transform)


def _projector_to_surface_matrix(
    transform: MatrixLike | MetricHomographyPair,
) -> MatrixLike:
    if isinstance(transform, MetricHomographyPair):
        return transform.projector_to_surface
    return validate_homography(transform)


def _validate_line_denominators(
    start: Point2D,
    end: Point2D,
    matrix: MatrixLike,
) -> None:
    checked_matrix = validate_homography(matrix)
    denominators = tuple(
        checked_matrix[2][0] * point.x
        + checked_matrix[2][1] * point.y
        + checked_matrix[2][2]
        for point in (start, end)
    )
    if any(not math.isfinite(denominator) for denominator in denominators):
        raise InvalidHomographyError('Ruler homography has a non-finite horizon')
    if any(abs(denominator) <= 1e-12 for denominator in denominators):
        raise InvalidHomographyError('Ruler endpoint lies on the homography horizon')
    if denominators[0] * denominators[1] < 0:
        raise InvalidHomographyError('Ruler line crosses the homography horizon')


def _coerce_metric_bounds(bounds: MetricBounds) -> CoordinateBounds:
    if isinstance(bounds, CoordinateBounds):
        checked_bounds = bounds
    else:
        if isinstance(bounds, (Mapping, Set, str, bytes, bytearray)):
            raise ValueError('projector_bounds must contain width and height')
        try:
            width, height = bounds
        except (TypeError, ValueError):
            raise ValueError('projector_bounds must contain width and height') from None
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError('projector_bounds must contain positive integer dimensions')
        checked_bounds = CoordinateBounds(0, 0, width, height)

    if (
        not is_finite_real(checked_bounds.left)
        or not is_finite_real(checked_bounds.top)
        or not is_finite_real(checked_bounds.right)
        or not is_finite_real(checked_bounds.bottom)
        or checked_bounds.left >= checked_bounds.right
        or checked_bounds.top >= checked_bounds.bottom
    ):
        raise ValueError('projector_bounds must be a finite positive rectangle')
    return checked_bounds


__all__ = [
    'MetricBounds',
    'MetricCalibrationRecord',
    'MetricCalibrationRegistry',
    'MetricCalibrationStatus',
    'MetricValidationRecord',
    'MetricCalibrationMetrics',
    'MetricCalibrationResult',
    'MetricHomographyPair',
    'MetricRulerMarker',
    'MetricRulerOverlay',
    'MetricRulerTick',
    'MetricUnit',
    'ProjectedSurfaceRuler',
    'MAX_METRIC_RULER_TICKS',
    'RULER_TICK_SPACING_SOURCE_UNITS',
    'METRIC_RULER_RASTER_MARGIN_PIXELS',
    'calculate_projector_surface_bounds',
    'calculate_ruler_tick_layout',
    'build_metric_ruler',
    'calibrate_metric_homography',
    'UNIT_TO_MM',
    'calculate_surface_distance_mm',
    'normalise_length_to_mm',
    'normalise_unit',
    'project_surface_points',
    'project_surface_ruler',
    'projector_to_surface',
    'surface_to_projector',
    'validate_finite_point',
    'validate_metric_correspondences',
    'validate_positive_length',
]
