import math
import unittest
from typing import Any

import cv2
import numpy as np

from multivision.errors import FiducialDetectionError
from multivision.fiducials import (
    DetectedMarker,
    OpenCVArucoDetector,
    assemble_camera_correspondences,
    assemble_correspondences,
    detect_and_assemble_correspondences,
)
from multivision.geometry import Point2D
from multivision.pattern import build_calibration_pattern
from multivision.types import Resolution


class FiducialTest(unittest.TestCase):
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


def _corners(x_pos: float, y_pos: float) -> tuple[Point2D, ...]:
    return (
        Point2D(x_pos, y_pos),
        Point2D(x_pos + 10, y_pos),
        Point2D(x_pos + 10, y_pos + 10),
        Point2D(x_pos, y_pos + 10),
    )


if __name__ == '__main__':
    unittest.main()
