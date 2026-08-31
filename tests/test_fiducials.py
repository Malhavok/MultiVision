import math
import unittest
from typing import Any

import cv2
import numpy as np

from multivision.errors import FiducialDetectionError
from multivision.fiducials import (
    CachedTagDetectorFactory,
    DetectedMarker,
    OpenCVArucoDetector,
    assemble_camera_correspondences,
    assemble_correspondences,
    assemble_metric_correspondences,
    build_planar_tag_observations,
    detect_and_assemble_correspondences,
    detect_metric_fiducials,
    detect_tag_observations,
)
from multivision.geometry import Point2D
from multivision.metric_target import METRIC_TARGET
from multivision.pattern import (
    DICT_5X5_1000,
    build_calibration_pattern,
)
from multivision.types import Resolution


class FiducialTest(unittest.TestCase):
    def test_planar_observations_retain_unknown_duplicates_and_sort_by_centre(self) -> None:
        detections = [
            DetectedMarker(8, _corners(50, 50)),
            DetectedMarker(3, _corners(80, 10)),
            DetectedMarker(3, _corners(10, 10)),
            DetectedMarker(3, _corners(40, 0)),
            DetectedMarker(999, _corners(20, 20)),
            DetectedMarker(4, _corners(0, 0)[:3]),
        ]

        observations = build_planar_tag_observations(detections)

        assert [observation.marker_id for observation in observations] == [
            3,
            3,
            3,
            8,
            999,
        ], f'{observations=}'
        assert [
            observation.camera.centre
            for observation in observations[:3]
        ] == [
            Point2D(45, 5),
            Point2D(15, 15),
            Point2D(85, 15),
        ], f'{observations=}'
        assert observations[1].camera.corners == tuple(_corners(10, 10))
        assert observations[0].projector is None

    def test_planar_observation_projection_rejects_horizon_crossing(self) -> None:
        observations = build_planar_tag_observations([DetectedMarker(1, _corners(0, 0))])

        with self.assertRaises(ValueError):
            build_planar_tag_observations(
                [DetectedMarker(1, _corners(0, 0))],
                ((1, 0, 0), (0, 1, 0), (0.2, 0, -1)),
            )
        assert observations[0].projector is None

    def test_detection_preserves_camera_geometry_with_injected_detector(self) -> None:
        class FakeDetector:
            def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
                return (DetectedMarker(23, _corners(10, 20)),)

        observations = detect_tag_observations(object(), FakeDetector())

        assert len(observations) == 1, f'{observations=}'
        assert observations[0].marker_id == 23, f'{observations=}'
        assert observations[0].projector is None

    def test_planar_observation_detection_rejects_structural_evidence_failure(
        self,
    ) -> None:
        class BrokenDetector:
            def detect(self, _frame: object) -> None:
                return None

        with self.assertRaises(FiducialDetectionError):
            detect_tag_observations(object(), BrokenDetector())

    def test_opencv_detector_identifies_apriltags_in_a_synthetic_frame(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        frame = np.full((360, 520), 255, dtype=np.uint8)
        for marker_id, x_pos, y_pos in [(0, 30, 40), (4, 280, 180), (8, 390, 50)]:
            marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
            frame[y_pos:y_pos + 100, x_pos:x_pos + 100] = marker_image

        detections = OpenCVArucoDetector().detect(frame)

        assert [marker.marker_id for marker in detections] == [0, 4, 8], f'{detections=}'
        assert all(len(marker.corners) == 4 for marker in detections), f'{detections=}'
        assert all(
            all(math.isfinite(coordinate) for coordinate in point)
            for marker in detections
            for point in marker.corners
        ), f'{detections=}'

    def test_opencv_detector_identifies_a_5x5_tag(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        frame = np.full((180, 180), 255, dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(dictionary, 23, 100)
        frame[30:130, 40:140] = marker_image

        detections = OpenCVArucoDetector(DICT_5X5_1000).detect(frame)

        assert [marker.marker_id for marker in detections] == [23], f'{detections=}'

    def test_tag_detector_factory_validates_and_caches_by_dictionary(self) -> None:
        class FakeDetector:
            def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
                return ()

        created_dictionary_names: list[str] = []

        def make_detector(dictionary_name: str) -> FakeDetector:
            created_dictionary_names.append(dictionary_name)
            return FakeDetector()

        factory = CachedTagDetectorFactory(make_detector)
        first_detector = factory(DICT_5X5_1000)
        second_detector = factory(DICT_5X5_1000)

        assert first_detector is second_detector, f'{first_detector=}, {second_detector=}'
        assert created_dictionary_names == [DICT_5X5_1000], f'{created_dictionary_names=}'
        with self.assertRaises(ValueError):
            factory('DICT_UNKNOWN')
        assert created_dictionary_names == [DICT_5X5_1000], f'{created_dictionary_names=}'

    def test_detector_uses_projection_safe_parameters(self) -> None:
        class FakeParameters:
            adaptiveThreshWinSizeMax = 23
            adaptiveThreshWinSizeStep = 10
            aprilTagMinWhiteBlackDiff = 5
            cornerRefinementMethod = 0
            perspectiveRemovePixelPerCell = 4

        class FakeAruco:
            DICT_APRILTAG_36h11 = object()
            DetectorParameters = FakeParameters

            def getPredefinedDictionary(self, dictionary_constant: object) -> object:
                return dictionary_constant

            def detectMarkers(
                self,
                frame: Any,
                dictionary: object,
                parameters: Any,
            ) -> tuple[list[Any], list[Any], list[Any]]:
                return [], [], []

        detector = OpenCVArucoDetector(
            cv2_module=type('FakeCV2', (), {'aruco': FakeAruco()})(),
        )
        parameters = detector._parameters

        assert parameters.adaptiveThreshWinSizeMax == 101, f'{parameters=}'
        assert parameters.adaptiveThreshWinSizeStep == 10, f'{parameters=}'
        assert parameters.aprilTagMinWhiteBlackDiff == 2, f'{parameters=}'
        assert parameters.cornerRefinementMethod == 1, f'{parameters=}'
        assert parameters.perspectiveRemovePixelPerCell == 8, f'{parameters=}'

    def test_assembly_rejects_non_iterable_detections_and_frames(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))

        with self.assertRaises(ValueError):
            assemble_correspondences(None, pattern)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            detect_and_assemble_correspondences([], object(), pattern)  # type: ignore[arg-type]

    def test_detector_rejects_mismatched_ids_and_corners(self) -> None:
        class FakeParameters:
            pass

        class FakeAruco:
            DICT_APRILTAG_36h11 = object()
            DetectorParameters = FakeParameters

            def getPredefinedDictionary(self, dictionary_constant: object) -> object:
                return dictionary_constant

            def detectMarkers(
                self,
                frame: Any,
                dictionary: object,
                parameters: Any,
            ) -> tuple[list[Any], list[Any], list[Any]]:
                return [], [[0]], []

        detector = OpenCVArucoDetector(
            cv2_module=type('FakeCV2', (), {'aruco': FakeAruco()})(),
        )

        with self.assertRaises(FiducialDetectionError):
            detector.detect(object())

    def test_detector_rejects_one_sided_marker_evidence(self) -> None:
        class FakeParameters:
            pass

        class FakeAruco:
            DICT_APRILTAG_36h11 = object()
            DetectorParameters = FakeParameters

            def getPredefinedDictionary(self, dictionary_constant: object) -> object:
                return dictionary_constant

            def detectMarkers(
                self,
                frame: Any,
                dictionary: object,
                parameters: Any,
            ) -> tuple[None, list[Any], list[Any]]:
                return None, [[0]], []

        detector = OpenCVArucoDetector(
            cv2_module=type('FakeCV2', (), {'aruco': FakeAruco()})(),
        )

        with self.assertRaises(FiducialDetectionError):
            detector.detect(object())

    def test_strict_detector_accepts_an_empty_frame(self) -> None:
        frame = np.full((100, 100), 255, dtype=np.uint8)

        assert OpenCVArucoDetector().detect_strict(frame) == ()

    def test_non_finite_polygon_area_is_rejected(self) -> None:
        class FakeDetector:
            def detect(self, frame: Any) -> tuple[tuple[int, tuple[tuple[float, float], ...]], ...]:
                return (
                    (
                        0,
                        (
                            (0.0, 0.0),
                            (1e308, 0.0),
                            (1e308, 1e308),
                            (0.0, 1e308),
                        ),
                    ),
                )

        detections = detect_and_assemble_correspondences(
            {'camera-a': object()},
            FakeDetector(),
            build_calibration_pattern(Resolution(1000, 800)),
        )

        assert detections['camera-a'].correspondences == (), f'{detections=}'

    def test_invalid_and_unknown_detections_are_excluded_but_all_valid_corners_remain(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        detections = [
            DetectedMarker(
                0,
                (
                    Point2D(10, 20),
                    Point2D(30, 20),
                    Point2D(30, 40),
                    Point2D(10, 40),
                ),
            ),
            DetectedMarker(999, ((1, 1), (2, 1), (2, 2), (1, 2))),
            DetectedMarker(1, ((1, 1), (2, 1), (2, 2))),
            DetectedMarker(
                2,
                (
                    Point2D(1, 1),
                    Point2D(math.nan, 1),
                    Point2D(2, 2),
                    Point2D(1, 2),
                ),
            ),
        ]

        result = assemble_correspondences(detections, pattern, camera_id='overhead')

        assert result.camera_id == 'overhead', f'{result=}'
        assert len(result.correspondences) == 4, f'{result=}'
        assert [correspondence.corner_index for correspondence in result.correspondences] == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            correspondence.marker_id == 0
            for correspondence in result.correspondences
        ), f'{result=}'
        assert [
            correspondence.projector_position
            for correspondence in result.correspondences
        ] == list(pattern.get_marker(0).corners)

    def test_correspondences_are_independent_for_each_camera(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        detections_by_camera = {
            'side-left': [DetectedMarker(1, _corners(100, 200))],
            'overhead': [
                DetectedMarker(1, _corners(30, 40)),
                DetectedMarker(0, _corners(10, 20)),
            ],
        }

        results = assemble_camera_correspondences(detections_by_camera, pattern)

        assert list(results) == ['overhead', 'side-left'], f'{results=}'
        assert results['overhead'].unique_marker_ids == (0, 1), f'{results=}'
        assert results['side-left'].unique_marker_ids == (1,), f'{results=}'
        assert results['overhead'].correspondences[0].camera_position == Point2D(10, 20)
        assert results['side-left'].correspondences[0].camera_position == Point2D(100, 200)

    def test_concave_marker_corners_are_rejected(self) -> None:
        class FakeDetector:
            def detect(self, frame: Any) -> tuple[tuple[int, tuple[tuple[float, float], ...]], ...]:
                return (
                    (
                        0,
                        ((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 2.0)),
                    ),
                )

        detections = detect_and_assemble_correspondences(
            {'camera-a': object()},
            FakeDetector(),
            build_calibration_pattern(Resolution(1000, 800)),
        )

        assert detections['camera-a'].correspondences == (), f'{detections=}'

    def test_detector_boundary_can_be_replaced_with_a_fake(self) -> None:
        class FakeDetector:
            def detect(self, frame: Any) -> tuple[DetectedMarker, ...]:
                return (DetectedMarker(int(frame), _corners(1, 2)),)

        pattern = build_calibration_pattern(Resolution(1000, 800))
        results = detect_and_assemble_correspondences(
            {'camera-a': '0', 'camera-b': '1'},
            FakeDetector(),
            pattern,
        )

        assert len(results['camera-a'].correspondences) == 4, f'{results=}'
        assert len(results['camera-b'].correspondences) == 4, f'{results=}'
        assert results['camera-a'].unique_marker_ids == (0,), f'{results=}'
        assert results['camera-b'].unique_marker_ids == (1,), f'{results=}'

    def test_metric_target_assembly_resolves_rotated_cyclic_corner_order(self) -> None:
        angle_radians = math.radians(37)
        cosine = math.cos(angle_radians)
        sine = math.sin(angle_radians)

        def transform(point: Point2D) -> Point2D:
            return Point2D(
                cosine * point.x - sine * point.y + 500,
                sine * point.x + cosine * point.y + 100,
            )

        detections = []
        for marker_id in (0, 3, 16, 19):
            target_marker = METRIC_TARGET.markers[marker_id]
            transformed_corners = tuple(
                transform(point)
                for point in target_marker.corners
            )
            detections.append(
                DetectedMarker(
                    marker_id,
                    transformed_corners[2:] + transformed_corners[:2],
                ),
            )

        result = assemble_metric_correspondences(detections, camera_id='camera-0')

        assert result.camera_id == 'camera-0', f'{result=}'
        assert len(result.correspondences) == 16, f'{result=}'
        assert result.unique_marker_ids == (0, 3, 16, 19), f'{result=}'
        for correspondence in result.correspondences:
            expected_position = transform(correspondence.surface_position)
            assert math.isclose(
                correspondence.camera_position.x,
                expected_position.x,
                abs_tol=1e-8,
            ), f'{correspondence=}'
            assert math.isclose(
                correspondence.camera_position.y,
                expected_position.y,
                abs_tol=1e-8,
            ), f'{correspondence=}'

    def test_metric_target_assembly_rejects_unknown_duplicate_partial_and_weak_data(self) -> None:
        valid_detections: list[DetectedMarker] = []
        for marker_id in (0, 3, 16, 19):
            marker = METRIC_TARGET.markers[marker_id]
            valid_detections.append(
                DetectedMarker(
                    marker_id,
                    tuple(Point2D(point.x * 2, point.y * 2) for point in marker.corners),
                ),
            )
        invalid_detections = (
            valid_detections[:-1] + [DetectedMarker(999, _corners(1, 1))],
            valid_detections + [valid_detections[0]],
            valid_detections[:-1] + [DetectedMarker(19, _corners(1, 1)[:3])],
            [DetectedMarker(marker_id, _corners(marker_id, marker_id)) for marker_id in range(4)],
        )
        for detections in invalid_detections:
            with self.subTest(detections=detections):
                with self.assertRaises(FiducialDetectionError):
                    assemble_metric_correspondences(detections)

    def test_metric_detection_does_not_silently_drop_malformed_evidence(self) -> None:
        class FakeDetector:
            def detect(self, frame: Any) -> tuple[DetectedMarker, ...]:
                return (DetectedMarker(0, _corners(1, 1)[:3]),)

        with self.assertRaises(FiducialDetectionError):
            detect_metric_fiducials(object(), FakeDetector())

        class FailingDetector:
            def detect(self, frame: Any) -> tuple[DetectedMarker, ...]:
                raise RuntimeError('detector failed')

        with self.assertRaises(FiducialDetectionError):
            detect_metric_fiducials(object(), FailingDetector())


def _corners(x_pos: float, y_pos: float) -> tuple[Point2D, ...]:
    return (
        Point2D(x_pos, y_pos),
        Point2D(x_pos + 10, y_pos),
        Point2D(x_pos + 10, y_pos + 10),
        Point2D(x_pos, y_pos + 10),
    )


if __name__ == '__main__':
    unittest.main()
