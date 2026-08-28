"""AprilTag detection and per-camera calibration correspondences."""

from __future__ import annotations

import math
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from numbers import Integral, Real
from typing import (
    Any,
    NamedTuple,
    Protocol,
)

from multivision.errors import FiducialDetectionError
from multivision.geometry import Point2D
from multivision.pattern import (
    APRILTAG_36H11,
    APRILTAG_FAMILIES,
    CalibrationPattern,
)


class DetectedMarker(NamedTuple):
    """One marker returned by a fiducial detector in camera-native space."""

    marker_id: int
    corners: tuple[Point2D, ...]


class FiducialDetector(Protocol):
    """Detector boundary used by both OpenCV and deterministic test fakes."""

    def detect(self, frame: Any) -> Iterable[DetectedMarker]:
        ...


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


class OpenCVArucoDetector:
    """Detect AprilTags through OpenCV's ``cv2.aruco`` API."""

    def __init__(
        self,
        dictionary_name: str = APRILTAG_36H11,
        detector_parameters: Any | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        if not isinstance(dictionary_name, str) or dictionary_name not in APRILTAG_FAMILIES:
            raise ValueError(f'Unsupported AprilTag family: {dictionary_name!r}')

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

        if raw_ids is None or raw_corners is None:
            return ()

        detections: list[DetectedMarker] = []
        try:
            marker_ids = list(raw_ids)
            marker_corners = list(raw_corners)
        except (TypeError, ValueError):
            return ()
        if len(marker_ids) != len(marker_corners):
            raise FiducialDetectionError(
                'Detector returned mismatched marker IDs and corners',
            )
        for marker_id, corners in zip(marker_ids, marker_corners):
            normalised_marker = _normalise_marker(marker_id, corners)
            if normalised_marker is not None:
                detections.append(DetectedMarker(*normalised_marker))
        return tuple(sorted(detections, key=lambda marker: marker.marker_id))


def detect_fiducials(
    frame: Any,
    detector: FiducialDetector,
) -> tuple[DetectedMarker, ...]:
    """Run an injected detector and discard malformed marker results."""
    return _normalise_markers(detector.detect(frame))


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


def _build_detector_parameters(aruco: Any) -> Any:
    parameters_class = getattr(aruco, 'DetectorParameters', None)
    if parameters_class is not None:
        return parameters_class()
    parameters_factory = getattr(aruco, 'DetectorParameters_create', None)
    if parameters_factory is not None:
        return parameters_factory()
    raise FiducialDetectionError('OpenCV does not provide detector parameters')


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
    polygon_area = _polygon_area(normalised_corners)
    if (
        len(set(normalised_corners)) != 4
        or not math.isfinite(polygon_area)
        or polygon_area == 0
        or not _is_convex_polygon(normalised_corners)
    ):
        return None
    return checked_marker_id, tuple(normalised_corners)


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


def _polygon_area(corners: Sequence[Point2D]) -> float:
    return abs(
        sum(
            corners[idx].x * corners[(idx + 1) % len(corners)].y
            - corners[(idx + 1) % len(corners)].x * corners[idx].y
            for idx in range(len(corners))
        )
        / 2
    )


def _is_convex_polygon(corners: Sequence[Point2D]) -> bool:
    cross_products = [
        (
            corners[(idx + 1) % len(corners)].x - corners[idx].x
        ) * (
            corners[(idx + 2) % len(corners)].y - corners[(idx + 1) % len(corners)].y
        )
        - (
            corners[(idx + 1) % len(corners)].y - corners[idx].y
        ) * (
            corners[(idx + 2) % len(corners)].x - corners[(idx + 1) % len(corners)].x
        )
        for idx in range(len(corners))
    ]
    return all(cross_product > 0 for cross_product in cross_products) or all(
        cross_product < 0 for cross_product in cross_products
    )


__all__ = [
    'CameraCorrespondences',
    'DetectedMarker',
    'FiducialCorrespondence',
    'FiducialDetector',
    'OpenCVArucoDetector',
    'assemble_camera_correspondences',
    'assemble_correspondences',
    'detect_and_assemble_correspondences',
    'detect_fiducials',
]
