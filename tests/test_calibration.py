import random
import unittest

from multivision.calibration import (
    CalibrationError,
    calibrate_homography,
    validate_correspondences_against_pattern,
)
from multivision.config import CalibrationThresholds
from multivision.fiducials import (
    CameraCorrespondences,
    FiducialCorrespondence,
)
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    is_point_in_region,
    project_point,
)
from multivision.pattern import (
    CalibrationPattern,
    build_calibration_pattern,
)
from multivision.types import Resolution


class CalibrationTest(unittest.TestCase):
    def test_perspective_recovery_stores_both_directions_and_metrics(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.1, 0.05, 40), (0.02, 1.15, 30), (0.0002, 0.0001, 1))
        correspondences = _correspondences(pattern, source_to_camera)

        result = calibrate_homography(correspondences, pattern)

        for source_point in (Point2D(150, 150), Point2D(500, 400), Point2D(850, 650)):
            camera_point = project_point(source_point, source_to_camera)
            recovered_point = project_point(camera_point, result.camera_to_projector)
            assert abs(recovered_point.x - source_point.x) < 0.01, f'{recovered_point=}'
            assert abs(recovered_point.y - source_point.y) < 0.01, f'{recovered_point=}'
        assert result.metrics.ransac_inlier_count == len(pattern.markers) * 4, (
            f'{result.metrics=}'
        )
        assert result.metrics.spatial_coverage > 0.5, f'{result.metrics=}'
        assert result.projector_to_camera != result.camera_to_projector, f'{result=}'
        assert is_point_in_region(Point2D(500, 400), result.valid_region), f'{result=}'

    def test_noisy_correspondences_are_accepted_with_quality_metrics(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.1, 0.05, 40), (0.02, 1.15, 30), (0.0002, 0.0001, 1))
        random_generator = random.Random(7)
        correspondences = _correspondences(
            pattern,
            source_to_camera,
            noise_radius=0.4,
            random_generator=random_generator,
        )

        result = calibrate_homography(correspondences, pattern)

        assert result.metrics.inlier_ratio > 0.9, f'{result.metrics=}'
        assert result.metrics.mean_reprojection_error < 1.0, f'{result.metrics=}'
        assert result.metrics.max_reprojection_error < 2.0, f'{result.metrics=}'

    def test_ransac_rejects_an_outlier_corner(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.1, 0.05, 40), (0.02, 1.15, 30), (0.0002, 0.0001, 1))
        correspondences = list(_correspondences(pattern, source_to_camera).correspondences)
        outlier = correspondences[17]
        correspondences[17] = outlier._replace(
            camera_position=Point2D(
                outlier.camera_position.x + 250,
                outlier.camera_position.y - 180,
            ),
        )

        result = calibrate_homography(CameraCorrespondences(tuple(correspondences)), pattern)

        assert (
            result.metrics.ransac_inlier_count < result.metrics.correspondence_corner_count
        ), f'{result.metrics=}'
        assert result.metrics.inlier_ratio > 0.9, f'{result.metrics=}'
        assert result.metrics.max_reprojection_error < 2.0, f'{result.metrics=}'

    def test_tightly_clustered_markers_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))
        correspondences = _clustered_correspondences(pattern, source_to_camera)
        thresholds = CalibrationThresholds(min_spatial_coverage=0.1)

        with self.assertRaises(CalibrationError):
            calibrate_homography(correspondences, pattern, thresholds)

    def test_insufficient_and_invalid_calibration_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))

        with self.assertRaises(CalibrationError):
            calibrate_homography(CameraCorrespondences(()), pattern)

        class InvalidOpenCV:
            RANSAC = 8

            def findHomography(
                self,
                source: object,
                destination: object,
                *args: object,
                **kwargs: object,
            ) -> tuple[None, None]:
                return None, None

        with self.assertRaises(CalibrationError):
            calibrate_homography(
                _correspondences(pattern, source_to_camera),
                pattern,
                cv2_module=InvalidOpenCV(),
            )

    def test_camera_points_outside_native_resolution_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))
        correspondences = list(
            _correspondences(pattern, source_to_camera).correspondences,
        )
        correspondences[0] = correspondences[0]._replace(
            camera_position=Point2D(1000, 100),
        )

        with self.assertRaises(CalibrationError):
            calibrate_homography(
                CameraCorrespondences(tuple(correspondences)),
                pattern,
                camera_resolution=Resolution(1000, 800),
            )

    def test_partial_marker_correspondences_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))
        correspondences = tuple(
            FiducialCorrespondence(
                marker.marker_id,
                0,
                marker.corners[0],
                project_point(marker.corners[0], source_to_camera),
            )
            for marker in pattern.markers[:4]
        )

        with self.assertRaises(CalibrationError):
            calibrate_homography(correspondences, pattern)

    def test_malformed_pattern_correspondences_are_domain_errors(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        correspondence = FiducialCorrespondence(
            marker_id=pattern.markers[0].marker_id,
            corner_index=99,
            projector_position=pattern.markers[0].corners[0],
            camera_position=Point2D(10, 10),
        )

        with self.assertRaises(CalibrationError):
            validate_correspondences_against_pattern([correspondence], pattern)

    def test_unknown_or_misassociated_pattern_corners_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))
        correspondences = list(_correspondences(pattern, source_to_camera).correspondences)

        correspondences[0] = correspondences[0]._replace(marker_id=999)
        with self.assertRaises(CalibrationError):
            calibrate_homography(correspondences, pattern)

        correspondences = list(_correspondences(pattern, source_to_camera).correspondences)
        correspondences[0] = correspondences[0]._replace(
            projector_position=pattern.markers[0].corners[1],
        )
        with self.assertRaises(CalibrationError):
            calibrate_homography(correspondences, pattern)

    def test_malformed_ransac_mask_and_pattern_are_refused(self) -> None:
        pattern = build_calibration_pattern(Resolution(1000, 800))
        source_to_camera = ((1.0, 0.0, 20), (0.0, 1.0, 30), (0.0, 0.0, 1))
        correspondences = _correspondences(pattern, source_to_camera)

        class MalformedMaskOpenCV:
            RANSAC = 8

            def findHomography(
                self,
                source: object,
                destination: object,
                *args: object,
                **kwargs: object,
            ) -> tuple[tuple[tuple[float, float, float], ...], list[list[int]]]:
                return source_to_camera, [[2] for _ in correspondences.correspondences]

        with self.assertRaises(CalibrationError):
            calibrate_homography(
                correspondences,
                pattern,
                cv2_module=MalformedMaskOpenCV(),
            )

        malformed_pattern = pattern._replace(
            usable_area=CoordinateBounds('invalid', 0, 1000, 800),  # type: ignore[arg-type]
        )
        with self.assertRaises(CalibrationError):
            calibrate_homography(correspondences, malformed_pattern)


def _correspondences(
    pattern: CalibrationPattern,
    source_to_camera: tuple[tuple[float, float, float], ...],
    noise_radius: float = 0.0,
    random_generator: random.Random | None = None,
) -> CameraCorrespondences:
    values: list[FiducialCorrespondence] = []
    for marker in pattern.markers:
        for corner_index, projector_position in enumerate(marker.corners):
            camera_position = project_point(projector_position, source_to_camera)
            if random_generator is not None:
                camera_position = Point2D(
                    camera_position.x + random_generator.uniform(-noise_radius, noise_radius),
                    camera_position.y + random_generator.uniform(-noise_radius, noise_radius),
                )
            values.append(
                FiducialCorrespondence(
                    marker.marker_id,
                    corner_index,
                    projector_position,
                    camera_position,
                ),
            )
    return CameraCorrespondences(tuple(values), camera_id='camera-a')


def _clustered_correspondences(
    pattern: CalibrationPattern,
    source_to_camera: tuple[tuple[float, float, float], ...],
) -> CameraCorrespondences:
    values: list[FiducialCorrespondence] = []
    for marker_id in range(4):
        centre_x = 490 + marker_id * 7
        centre_y = 390 + marker_id * 5
        corners = (
            Point2D(centre_x, centre_y),
            Point2D(centre_x + 4, centre_y),
            Point2D(centre_x + 4, centre_y + 4),
            Point2D(centre_x, centre_y + 4),
        )
        for corner_index, projector_position in enumerate(corners):
            values.append(
                FiducialCorrespondence(
                    marker_id,
                    corner_index,
                    projector_position,
                    project_point(projector_position, source_to_camera),
                ),
            )
    return CameraCorrespondences(tuple(values))


if __name__ == '__main__':
    unittest.main()
