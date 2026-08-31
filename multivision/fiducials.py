"""AprilTag detection and per-camera calibration correspondences."""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    NamedTuple,
    Protocol,
)

from multivision.errors import (
    FiducialDetectionError,
    InvalidHomographyError,
)
from multivision.geometry import (
    HomographyPair,
    MatrixLike,
    Point2D,
    TagGeometry,
    build_tag_geometry,
    calculate_convex_hull,
    calculate_polygon_area,
    project_point,
    project_tag_geometry,
    validate_homography,
    validate_planar_corners,
)
from multivision.metric_target import (
    METRIC_TARGET,
    MetricTarget,
    MetricTargetMarker,
    validate_metric_target,
)
from multivision.pattern import (
    APRILTAG_36H11,
    CalibrationPattern,
    validate_tag_dictionary,
)


class DetectedMarker(NamedTuple):
    """One marker returned by a fiducial detector in camera-native space."""

    marker_id: int
    corners: tuple[Point2D, ...]


class PlanarTagObservation(NamedTuple):
    """One immutable camera observation with optional projector geometry."""

    marker_id: int
    camera: TagGeometry
    projector: TagGeometry | None = None


class FiducialDetector(Protocol):
    """Detector boundary used by both OpenCV and deterministic test fakes."""

    def detect(self, frame: Any) -> Iterable[DetectedMarker]:
        ...


TagDetectorFactory = Callable[[str], FiducialDetector]


class FiducialCorrespondence(NamedTuple):
    """One marker corner correspondence for one camera."""

    marker_id: int
    corner_index: int
    projector_position: Point2D
    camera_position: Point2D


class CameraCorrespondences(NamedTuple):
    """All usable marker corners assembled for one logical camera."""

    correspondences: tuple[FiducialCorrespondence, ...]
    camera_id: str | None = None

    @property
    def unique_marker_ids(self) -> tuple[int, ...]:
        return tuple(
            marker_id
            for marker_id in dict.fromkeys(
                correspondence.marker_id
                for correspondence in self.correspondences
            )
        )

    @property
    def projector_points(self) -> tuple[Point2D, ...]:
        return tuple(
            correspondence.projector_position
            for correspondence in self.correspondences
        )

    @property
    def camera_points(self) -> tuple[Point2D, ...]:
        return tuple(
            correspondence.camera_position
            for correspondence in self.correspondences
        )


class MetricTargetCorrespondence(NamedTuple):
    """One target corner paired with its camera-native detection."""

    marker_id: int
    corner_index: int
    surface_position: Point2D
    camera_position: Point2D


class MetricTargetCorrespondences(NamedTuple):
    """Strict, target-aware corner correspondences for one camera frame."""

    correspondences: tuple[MetricTargetCorrespondence, ...]
    camera_id: str | None = None

    @property
    def unique_marker_ids(self) -> tuple[int, ...]:
        return tuple(
            marker_id
            for marker_id in dict.fromkeys(
                correspondence.marker_id
                for correspondence in self.correspondences
            )
        )

    @property
    def surface_points(self) -> tuple[Point2D, ...]:
        return tuple(
            correspondence.surface_position
            for correspondence in self.correspondences
        )

    @property
    def camera_points(self) -> tuple[Point2D, ...]:
        return tuple(
            correspondence.camera_position
            for correspondence in self.correspondences
        )


class _SynchronizedTagDetector:
    """Serialise calls into one detector instance."""

    def __init__(self, detector: FiducialDetector) -> None:
        self._detector = detector
        self._detect_lock = threading.Lock()

    def detect(self, frame: Any) -> Iterable[DetectedMarker]:
        with self._detect_lock:
            return self._detector.detect(frame)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._detector, name)
        if name != 'detect_strict' or not callable(attribute):
            return attribute

        def detect_strict(frame: Any) -> Iterable[DetectedMarker]:
            with self._detect_lock:
                return attribute(frame)

        return detect_strict


class CachedTagDetectorFactory:
    """Create and cache independent OpenCV detectors by dictionary name."""

    def __init__(
        self,
        detector_factory: Callable[[str], FiducialDetector] | None = None,
    ) -> None:
        if detector_factory is not None and not callable(detector_factory):
            raise TypeError('detector_factory must be callable')
        self._detector_factory = (
            detector_factory
            if detector_factory is not None
            else OpenCVArucoDetector
        )
        self._detectors: dict[str, FiducialDetector] = {}
        self._lock = threading.RLock()

    def __call__(self, dictionary_name: str) -> FiducialDetector:
        """Return the detector for one dictionary, creating it at most once."""
        checked_dictionary_name = validate_tag_dictionary(dictionary_name)
        with self._lock:
            detector = self._detectors.get(checked_dictionary_name)
            if detector is not None:
                return detector
            detector = _SynchronizedTagDetector(
                self._detector_factory(checked_dictionary_name),
            )
            self._detectors[checked_dictionary_name] = detector
            return detector


class OpenCVArucoDetector:
    """Detect AprilTags through OpenCV's ``cv2.aruco`` API."""

    def __init__(
        self,
        dictionary_name: str = APRILTAG_36H11,
        detector_parameters: Any | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        dictionary_name = validate_tag_dictionary(dictionary_name)

        if cv2_module is None:
            try:
                import cv2
            except ImportError as ex:
                raise FiducialDetectionError('OpenCV is not installed') from ex
            cv2_module = cv2

        aruco = getattr(cv2_module, 'aruco', None)
        if aruco is None:
            raise FiducialDetectionError('OpenCV was built without cv2.aruco')
        dictionary_constant = getattr(aruco, dictionary_name, None)
        if dictionary_constant is None:
            raise FiducialDetectionError(
                f'OpenCV does not provide {dictionary_name}',
            )

        try:
            dictionary = aruco.getPredefinedDictionary(dictionary_constant)
            parameters = (
                detector_parameters
                if detector_parameters is not None
                else _build_detector_parameters(aruco)
            )
            detector_class = getattr(aruco, 'ArucoDetector', None)
            self._detector = (
                detector_class(dictionary, parameters)
                if detector_class is not None
                else None
            )
        except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
            raise FiducialDetectionError('Could not initialise the ArUco detector') from ex

        self._aruco = aruco
        self._dictionary = dictionary
        self._parameters = parameters
        self.dictionary_name = dictionary_name

    def detect(self, frame: Any) -> tuple[DetectedMarker, ...]:
        """Return only well-formed four-corner detections from one frame."""
        raw_ids, raw_corners = self._detect_raw(frame)
        if raw_ids is None and raw_corners is None:
            return ()
        if raw_ids is None or raw_corners is None:
            if not _has_partial_detector_evidence(raw_ids, raw_corners):
                return ()
            raise FiducialDetectionError(
                'Detector returned only one of marker IDs and corners',
            )

        detections: list[DetectedMarker] = []
        try:
            marker_ids = list(raw_ids)
            marker_corners = list(raw_corners)
        except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
            raise FiducialDetectionError(
                'Detector returned non-iterable marker evidence',
            ) from ex
        if len(marker_ids) != len(marker_corners):
            raise FiducialDetectionError(
                'Detector returned mismatched marker IDs and corners',
            )
        for marker_id, corners in zip(marker_ids, marker_corners):
            normalised_marker = _normalise_marker(marker_id, corners)
            if normalised_marker is not None:
                detections.append(DetectedMarker(*normalised_marker))
        return tuple(sorted(detections, key=lambda marker: marker.marker_id))

    def detect_strict(self, frame: Any) -> tuple[DetectedMarker, ...]:
        """Return raw detector evidence, failing on malformed marker data."""
        raw_ids, raw_corners = self._detect_raw(frame)
        if raw_ids is None and raw_corners is None:
            return ()
        if raw_ids is None or raw_corners is None:
            if not _has_partial_detector_evidence(raw_ids, raw_corners):
                return ()
            raise FiducialDetectionError(
                'Detector returned only one of marker IDs and corners',
            )
        try:
            marker_ids = list(raw_ids)
            marker_corners = list(raw_corners)
        except Exception as ex:  # noqa: BLE001 (the detector is an external boundary).
            raise FiducialDetectionError(
                'Detector returned non-iterable marker evidence',
            ) from ex
        if len(marker_ids) != len(marker_corners):
            raise FiducialDetectionError(
                'Detector returned mismatched marker IDs and corners',
            )
        return _normalise_markers_strict(
            (marker_id, corners) for marker_id, corners in zip(
                marker_ids,
                marker_corners,
            )
        )

    def _detect_raw(self, frame: Any) -> tuple[object | None, object | None]:
        try:
            if self._detector is not None:
                raw_corners, raw_ids, _ = self._detector.detectMarkers(frame)
            else:
                raw_corners, raw_ids, _ = self._aruco.detectMarkers(
                    frame,
                    self._dictionary,
                    parameters=self._parameters,
                )
        except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
            raise FiducialDetectionError('Could not detect fiducials in the frame') from ex
        return raw_ids, raw_corners


def _has_partial_detector_evidence(
    raw_ids: object | None,
    raw_corners: object | None,
) -> bool:
    available_evidence = raw_corners if raw_ids is None else raw_ids
    try:
        return len(tuple(available_evidence)) > 0  # type: ignore[arg-type]
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise FiducialDetectionError(
            'Detector returned non-iterable marker evidence',
        ) from ex


def build_planar_tag_observation(
    detected_marker: object,
    homography: MatrixLike | HomographyPair | None = None,
) -> PlanarTagObservation:
    """Build one validated planar observation without filtering its marker ID."""
    normalised_marker = _normalise_marker_from_object(detected_marker)
    if normalised_marker is None:
        raise ValueError('detected_marker contains malformed evidence')
    marker_id, corners = normalised_marker
    camera_geometry = build_tag_geometry(corners)
    projector_geometry = (
        None
        if homography is None
        else project_tag_geometry(camera_geometry, homography)
    )
    return PlanarTagObservation(marker_id, camera_geometry, projector_geometry)


def build_planar_tag_observations(
    detected_markers: Iterable[DetectedMarker],
    homography: MatrixLike | HomographyPair | None = None,
) -> tuple[PlanarTagObservation, ...]:
    """Build all valid observations, retaining unknown and duplicate marker IDs."""
    if isinstance(detected_markers, (Mapping, str, bytes, bytearray)):
        raise ValueError('detected_markers must be an ordered marker collection')
    try:
        marker_iterator = iter(detected_markers)
    except TypeError:
        raise ValueError('detected_markers must be iterable') from None

    observations: list[PlanarTagObservation] = []
    try:
        for detected_marker in marker_iterator:
            try:
                observations.append(
                    build_planar_tag_observation(detected_marker, homography),
                )
            except InvalidHomographyError:
                raise
            except ValueError:
                continue
    except (FiducialDetectionError, InvalidHomographyError):
        raise
    except Exception as ex:  # noqa: BLE001 (detector evidence is an external boundary).
        raise FiducialDetectionError(
            'Detector failed while returning tag evidence',
        ) from ex
    return tuple(sorted(observations, key=_tag_observation_sort_key))


def detect_tag_observations(
    frame: Any,
    detector: FiducialDetector,
    homography: MatrixLike | HomographyPair | None = None,
) -> tuple[PlanarTagObservation, ...]:
    """Detect and build ordered planar observations through an injected detector."""
    try:
        detected_markers = detector.detect(frame)
    except FiducialDetectionError:
        raise
    except Exception as ex:  # noqa: BLE001 (the detector is an external boundary).
        raise FiducialDetectionError('Could not detect tags in the frame') from ex
    try:
        return build_planar_tag_observations(detected_markers, homography)
    except (FiducialDetectionError, InvalidHomographyError):
        raise
    except ValueError as ex:
        raise FiducialDetectionError(
            'Detector returned malformed tag evidence',
        ) from ex


def detect_fiducials(
    frame: Any,
    detector: FiducialDetector,
) -> tuple[DetectedMarker, ...]:
    """Run an injected detector and discard malformed marker results."""
    return _normalise_markers(detector.detect(frame))


def detect_metric_fiducials(
    frame: Any,
    detector: FiducialDetector,
) -> tuple[DetectedMarker, ...]:
    """Run a detector without discarding malformed target evidence."""
    strict_detector = getattr(detector, 'detect_strict', None)
    try:
        if callable(strict_detector):
            return _normalise_markers_strict(strict_detector(frame))
        raw_markers = detector.detect(frame)
        return _normalise_markers_strict(raw_markers)
    except FiducialDetectionError:
        raise
    except Exception as ex:  # noqa: BLE001 (the detector is an external boundary).
        raise FiducialDetectionError(
            'Could not perform strict fiducial detection',
        ) from ex


def assemble_metric_correspondences(
    detected_markers: Iterable[DetectedMarker],
    target: MetricTarget = METRIC_TARGET,
    camera_id: str | None = None,
    minimum_marker_count: int = 4,
    minimum_spatial_coverage: float = 0.5,
) -> MetricTargetCorrespondences:
    """Associate a verified target's ordered corners with camera coordinates."""
    checked_target = validate_metric_target(target)
    _validate_metric_assembly_limits(
        minimum_marker_count,
        minimum_spatial_coverage,
    )
    try:
        marker_iterator = iter(detected_markers)
    except TypeError:
        raise FiducialDetectionError('detected_markers must be iterable') from None

    target_markers = {marker.marker_id: marker for marker in checked_target.markers}
    markers_by_id: dict[int, DetectedMarker] = {}
    for detected_marker in _normalise_markers_strict(marker_iterator):
        if detected_marker.marker_id not in target_markers:
            raise FiducialDetectionError(
                f'Detected unknown metric target marker {detected_marker.marker_id}',
            )
        if detected_marker.marker_id in markers_by_id:
            raise FiducialDetectionError(
                f'Detected duplicate metric target marker {detected_marker.marker_id}',
            )
        markers_by_id[detected_marker.marker_id] = detected_marker

    if len(markers_by_id) < minimum_marker_count:
        raise FiducialDetectionError(
            f'Metric target needs at least {minimum_marker_count} markers',
        )
    provisional_homography = _estimate_target_centre_homography(
        tuple(
            (target_markers[marker_id], markers_by_id[marker_id])
            for marker_id in sorted(markers_by_id)
        ),
    )

    correspondences: list[MetricTargetCorrespondence] = []
    for marker_id in sorted(markers_by_id):
        target_marker = target_markers[marker_id]
        detected_marker = markers_by_id[marker_id]
        ordered_corners = _order_detected_corners(
            target_marker,
            detected_marker.corners,
            provisional_homography,
        )
        correspondences.extend(
            MetricTargetCorrespondence(
                marker_id,
                corner_index,
                surface_position,
                camera_position,
            )
            for corner_index, (surface_position, camera_position) in enumerate(
                zip(target_marker.corners, ordered_corners),
            )
        )

    coverage = _calculate_target_coverage(
        tuple(correspondence.surface_position for correspondence in correspondences),
        checked_target,
    )
    if coverage < minimum_spatial_coverage:
        raise FiducialDetectionError(
            f'Metric target spatial coverage is too small: {coverage!r}',
        )
    return MetricTargetCorrespondences(tuple(correspondences), camera_id)


def detect_and_assemble_metric_correspondences(
    frame: Any,
    detector: FiducialDetector,
    target: MetricTarget = METRIC_TARGET,
    camera_id: str | None = None,
    minimum_marker_count: int = 4,
    minimum_spatial_coverage: float = 0.5,
) -> MetricTargetCorrespondences:
    """Strictly detect and assemble one rotated metric target frame."""
    checked_target = validate_metric_target(target)
    _validate_detector_family(detector, checked_target)
    return assemble_metric_correspondences(
        detect_metric_fiducials(frame, detector),
        checked_target,
        camera_id,
        minimum_marker_count,
        minimum_spatial_coverage,
    )


def assemble_correspondences(
    detected_markers: Iterable[DetectedMarker],
    pattern: CalibrationPattern,
    camera_id: str | None = None,
) -> CameraCorrespondences:
    """Match known marker IDs and retain all four valid corners of each one."""
    if not isinstance(pattern, CalibrationPattern):
        raise ValueError('pattern must be CalibrationPattern')

    try:
        marker_iterator = iter(detected_markers)
    except TypeError:
        raise ValueError('detected_markers must be iterable') from None

    projector_markers = {
        marker.marker_id: marker
        for marker in pattern.markers
    }
    camera_corners_by_marker_id: dict[int, tuple[Point2D, ...]] = {}
    for detected_marker in _normalise_markers(marker_iterator):
        if detected_marker.marker_id not in projector_markers:
            continue
        if detected_marker.marker_id in camera_corners_by_marker_id:
            continue
        camera_corners_by_marker_id[detected_marker.marker_id] = detected_marker.corners

    correspondences: list[FiducialCorrespondence] = []
    for marker_id in sorted(camera_corners_by_marker_id):
        projector_corners = projector_markers[marker_id].corners
        for corner_index, (projector_corner, camera_corner) in enumerate(
            zip(projector_corners, camera_corners_by_marker_id[marker_id]),
        ):
            correspondences.append(
                FiducialCorrespondence(
                    marker_id,
                    corner_index,
                    projector_corner,
                    camera_corner,
                ),
            )

    return CameraCorrespondences(tuple(correspondences), camera_id)


def assemble_camera_correspondences(
    detected_markers_by_camera: Mapping[str, Iterable[DetectedMarker]],
    pattern: CalibrationPattern,
) -> dict[str, CameraCorrespondences]:
    """Assemble each camera independently without sharing detections or state."""
    if not isinstance(detected_markers_by_camera, Mapping):
        raise ValueError('detected_markers_by_camera must be a mapping')
    return {
        camera_id: assemble_correspondences(
            detected_markers,
            pattern,
            camera_id=camera_id,
        )
        for camera_id, detected_markers in sorted(detected_markers_by_camera.items())
    }


def detect_and_assemble_correspondences(
    frames_by_camera: Mapping[str, Any],
    detector: FiducialDetector,
    pattern: CalibrationPattern,
) -> dict[str, CameraCorrespondences]:
    """Detect and assemble independent correspondences for each camera frame."""
    if not isinstance(frames_by_camera, Mapping):
        raise ValueError('frames_by_camera must be a mapping')
    detected_markers_by_camera = {
        camera_id: detect_fiducials(frame, detector)
        for camera_id, frame in sorted(frames_by_camera.items())
    }
    return assemble_camera_correspondences(detected_markers_by_camera, pattern)


def _tag_observation_sort_key(
    observation: PlanarTagObservation,
) -> tuple[int, float, float, tuple[Point2D, ...]]:
    return (
        observation.marker_id,
        observation.camera.centre.y,
        observation.camera.centre.x,
        observation.camera.corners,
    )


def _normalise_markers_strict(markers: object) -> tuple[DetectedMarker, ...]:
    if markers is None:
        raise FiducialDetectionError('Detector returned no marker collection')
    try:
        marker_iterator = iter(markers)  # type: ignore[arg-type]
    except Exception as ex:  # noqa: BLE001 (the detector is an external boundary).
        raise FiducialDetectionError(
            'Detector returned a non-iterable marker collection',
        ) from ex

    normalised_markers: list[DetectedMarker] = []
    try:
        for detection_index, marker in enumerate(marker_iterator):
            normalised_marker = _normalise_marker_from_object(marker)
            if normalised_marker is None:
                raise FiducialDetectionError(
                    f'Detector returned malformed marker evidence at index {detection_index}',
                )
            normalised_markers.append(DetectedMarker(*normalised_marker))
    except FiducialDetectionError:
        raise
    except Exception as ex:  # noqa: BLE001 (the detector is an external boundary).
        raise FiducialDetectionError(
            'Detector failed while returning marker evidence',
        ) from ex
    return tuple(normalised_markers)


def _validate_detector_family(
    detector: FiducialDetector,
    target: MetricTarget,
) -> None:
    detector_family = getattr(detector, 'dictionary_name', None)
    if detector_family is not None and detector_family != target.marker_family:
        raise FiducialDetectionError(
            f'Detector family {detector_family!r} does not match the metric target',
        )


def _validate_metric_assembly_limits(
    minimum_marker_count: int,
    minimum_spatial_coverage: float,
) -> None:
    if (
        not isinstance(minimum_marker_count, int)
        or isinstance(minimum_marker_count, bool)
        or minimum_marker_count < 4
    ):
        raise ValueError('minimum_marker_count must be at least four')
    if (
        not isinstance(minimum_spatial_coverage, Real)
        or isinstance(minimum_spatial_coverage, bool)
        or not math.isfinite(float(minimum_spatial_coverage))
        or not 0 <= minimum_spatial_coverage <= 1
    ):
        raise ValueError('minimum_spatial_coverage must be between zero and one')


def _estimate_target_centre_homography(
    marker_pairs: Sequence[tuple[MetricTargetMarker, DetectedMarker]],
) -> tuple[tuple[float, float, float], ...]:
    centre_pairs = tuple(
        (
            _marker_centre(target_marker.corners),
            _marker_centre(detected_marker.corners),
        )
        for target_marker, detected_marker in marker_pairs
    )
    best_matrix: tuple[tuple[float, float, float], ...] | None = None
    best_score = math.inf
    best_indices: tuple[int, ...] | None = None
    for indices in itertools.combinations(range(len(centre_pairs)), 4):
        candidate_pairs = tuple(centre_pairs[idx] for idx in indices)
        if calculate_polygon_area(
            calculate_convex_hull(tuple(pair[0] for pair in candidate_pairs)),
        ) <= 1e-9:
            continue
        try:
            candidate_matrix = _solve_four_point_homography(candidate_pairs)
            errors = tuple(
                math.dist(project_point(source, candidate_matrix), destination)
                for source, destination in centre_pairs
            )
        except (InvalidHomographyError, ValueError, OverflowError):
            continue
        if any(not math.isfinite(error) for error in errors):
            continue
        score = sum(error * error for error in errors)
        candidate_key = (score, indices)
        best_key = (
            best_score,
            best_indices if best_indices is not None else tuple(),
        )
        if candidate_key < best_key:
            best_matrix = candidate_matrix
            best_score = score
            best_indices = indices

    if best_matrix is None:
        raise FiducialDetectionError(
            'Metric target markers do not provide sufficient two-dimensional spread',
        )
    centre_errors = tuple(
        math.dist(project_point(source, best_matrix), destination)
        for source, destination in centre_pairs
    )
    if max(centre_errors, default=math.inf) > _centre_error_limit(centre_pairs):
        raise FiducialDetectionError(
            'Metric target marker centres cannot be fitted reliably',
        )
    return best_matrix


def _order_detected_corners(
    target_marker: MetricTargetMarker,
    detected_corners: tuple[Point2D, ...],
    provisional_homography: tuple[tuple[float, float, float], ...],
) -> tuple[Point2D, ...]:
    predicted_corners = tuple(
        project_point(corner, provisional_homography)
        for corner in target_marker.corners
    )
    candidates = []
    for shift in range(4):
        ordered_corners = detected_corners[shift:] + detected_corners[:shift]
        error = sum(
            math.dist(predicted_corner, detected_corner)
            for predicted_corner, detected_corner in zip(
                predicted_corners,
                ordered_corners,
            )
        )
        candidates.append((error, shift, ordered_corners))
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    best_error, _, best_corners = candidates[0]
    second_error = candidates[1][0]
    observed_diagonal = math.dist(detected_corners[0], detected_corners[2])
    if (
        second_error - best_error <= max(1e-6, observed_diagonal * 0.02)
        or best_error / 4 > max(3.0, observed_diagonal * 0.25)
    ):
        raise FiducialDetectionError(
            f'Metric target marker {target_marker.marker_id} has unreliable corner order',
        )
    return best_corners


def _solve_four_point_homography(
    point_pairs: Sequence[tuple[Point2D, Point2D]],
) -> tuple[tuple[float, float, float], ...]:
    equations: list[list[float]] = []
    for source, destination in point_pairs:
        equations.append([
            source.x,
            source.y,
            1.0,
            0.0,
            0.0,
            0.0,
            -source.x * destination.x,
            -source.y * destination.x,
            destination.x,
        ])
        equations.append([
            0.0,
            0.0,
            0.0,
            source.x,
            source.y,
            1.0,
            -source.x * destination.y,
            -source.y * destination.y,
            destination.y,
        ])
    values = _solve_linear_system(equations)
    matrix = (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
        (values[6], values[7], 1.0),
    )
    return validate_homography(matrix)


def _solve_linear_system(equations: list[list[float]]) -> tuple[float, ...]:
    matrix = [row[:8] + [row[8]] for row in equations]
    for column in range(8):
        pivot_index = max(
            range(column, 8),
            key=lambda row_index: abs(matrix[row_index][column]),
        )
        pivot = matrix[pivot_index][column]
        if not math.isfinite(pivot) or abs(pivot) <= 1e-12:
            raise ValueError('Point pairs are degenerate')
        matrix[column], matrix[pivot_index] = matrix[pivot_index], matrix[column]
        pivot_row = matrix[column]
        pivot_value = pivot_row[column]
        for row_index in range(8):
            if row_index == column:
                continue
            scale = matrix[row_index][column] / pivot_value
            if scale == 0:
                continue
            for value_index in range(column, 9):
                matrix[row_index][value_index] -= scale * pivot_row[value_index]
    values = tuple(matrix[row_index][8] / matrix[row_index][row_index] for row_index in range(8))
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Point pairs produced a non-finite homography')
    return values


def _centre_error_limit(
    centre_pairs: Sequence[tuple[Point2D, Point2D]],
) -> float:
    observed_spread = max(
        math.dist(first, second)
        for first, second in itertools.combinations(
            (pair[1] for pair in centre_pairs),
            2,
        )
    )
    return max(5.0, observed_spread * 0.1)


def _marker_centre(corners: Sequence[Point2D]) -> Point2D:
    return Point2D(
        sum(corner.x for corner in corners) / len(corners),
        sum(corner.y for corner in corners) / len(corners),
    )


def _calculate_target_coverage(
    points: Sequence[Point2D],
    target: MetricTarget,
) -> float:
    hull_area = calculate_polygon_area(calculate_convex_hull(points))
    page_area = target.page_width_mm * target.page_height_mm
    coverage = hull_area / page_area
    if not math.isfinite(coverage):
        raise FiducialDetectionError('Metric target coverage is not finite')
    return coverage


def _build_detector_parameters(aruco: Any) -> Any:
    parameters_class = getattr(aruco, 'DetectorParameters', None)
    if parameters_class is not None:
        parameters = parameters_class()
    else:
        parameters_factory = getattr(aruco, 'DetectorParameters_create', None)
        if parameters_factory is None:
            raise FiducialDetectionError('OpenCV does not provide detector parameters')
        parameters = parameters_factory()

    # Projected tags span much larger regions than the default adaptive window,
    # and tabletop texture otherwise leaves the correct quads rejected before decoding.
    detector_settings = {
        'adaptiveThreshWinSizeMax': 101,
        'adaptiveThreshWinSizeStep': 10,
        'aprilTagMinWhiteBlackDiff': 2,
        'cornerRefinementMethod': getattr(aruco, 'CORNER_REFINE_SUBPIX', 1),
        'perspectiveRemovePixelPerCell': 8,
    }
    for setting_name, setting_value in detector_settings.items():
        if hasattr(parameters, setting_name):
            setattr(parameters, setting_name, setting_value)
    return parameters


def _normalise_markers(markers: object) -> tuple[DetectedMarker, ...]:
    try:
        marker_iterator = iter(markers)  # type: ignore[arg-type]
    except TypeError:
        return ()

    normalised_markers: list[DetectedMarker] = []
    for marker in marker_iterator:
        normalised_marker = _normalise_marker_from_object(marker)
        if normalised_marker is not None:
            normalised_markers.append(DetectedMarker(*normalised_marker))
    return tuple(sorted(normalised_markers, key=lambda item: item.marker_id))


def _normalise_marker_from_object(
    marker: object,
) -> tuple[int, tuple[Point2D, ...]] | None:
    if isinstance(marker, DetectedMarker):
        return _normalise_marker(marker.marker_id, marker.corners)
    marker_id = getattr(marker, 'marker_id', None)
    corners = getattr(marker, 'corners', None)
    if marker_id is None or corners is None:
        try:
            marker_id, corners = marker  # type: ignore[misc]
        except (TypeError, ValueError):
            return None
    return _normalise_marker(marker_id, corners)


def _normalise_marker(
    marker_id: object,
    corners: object,
) -> tuple[int, tuple[Point2D, ...]] | None:
    checked_marker_id = _normalise_marker_id(marker_id)
    if checked_marker_id is None:
        return None
    try:
        corner_values = tuple(corners)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(corner_values) == 1:
        try:
            unwrapped_corner_values = tuple(corner_values[0])
        except (TypeError, ValueError):
            return None
        if len(unwrapped_corner_values) == 4:
            corner_values = unwrapped_corner_values
    if len(corner_values) != 4:
        return None

    normalised_corners: list[Point2D] = []
    for corner in corner_values:
        normalised_corner = _normalise_point(corner)
        if normalised_corner is None:
            return None
        normalised_corners.append(normalised_corner)
    try:
        checked_corners = validate_planar_corners(normalised_corners)
    except ValueError:
        return None
    return checked_marker_id, checked_corners


def _normalise_marker_id(marker_id: object) -> int | None:
    if isinstance(marker_id, Integral) and not isinstance(marker_id, bool):
        return int(marker_id)
    if isinstance(marker_id, (str, bytes, bytearray, Mapping)):
        return None
    try:
        marker_id_values = tuple(marker_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(marker_id_values) != 1:
        return None
    return _normalise_marker_id(marker_id_values[0])


def _normalise_point(point: object) -> Point2D | None:
    if isinstance(point, (str, bytes, bytearray, Mapping)):
        return None
    try:
        coordinates = tuple(point)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(coordinates) != 2:
        return None
    if any(
        not isinstance(coordinate, Real) or isinstance(coordinate, bool)
        for coordinate in coordinates
    ):
        return None
    try:
        x_pos, y_pos = (float(coordinate) for coordinate in coordinates)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(x_pos) or not math.isfinite(y_pos):
        return None
    return Point2D(x_pos, y_pos)


__all__ = [
    'CachedTagDetectorFactory',
    'CameraCorrespondences',
    'DetectedMarker',
    'FiducialCorrespondence',
    'FiducialDetector',
    'MetricTargetCorrespondence',
    'MetricTargetCorrespondences',
    'OpenCVArucoDetector',
    'PlanarTagObservation',
    'TagDetectorFactory',
    'assemble_camera_correspondences',
    'assemble_correspondences',
    'assemble_metric_correspondences',
    'build_planar_tag_observation',
    'build_planar_tag_observations',
    'detect_and_assemble_correspondences',
    'detect_and_assemble_metric_correspondences',
    'detect_fiducials',
    'detect_metric_fiducials',
    'detect_tag_observations',
]
