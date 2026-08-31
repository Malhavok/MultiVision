import math
import random
import unittest

from multivision.application import _average_metric_correspondences
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
    Point2D,
    project_point,
)
from multivision.metric import (
    MetricCalibrationMetrics,
    MetricCalibrationRegistry,
    MetricCalibrationResult,
    MetricHomographyPair,
    build_metric_ruler,
    calculate_projector_surface_bounds,
    calibrate_metric_homography,
    calculate_surface_distance_mm,
    normalise_length_to_mm,
    project_surface_points,
    project_surface_ruler,
    projector_to_surface,
    surface_to_projector,
    validate_finite_point,
    validate_positive_length,
)
from multivision.metric_target import METRIC_TARGET
from multivision.types import Resolution


class MetricTest(unittest.TestCase):
    def test_units_are_normalised_to_millimetres(self) -> None:
        assert normalise_length_to_mm(1, 'mm') == 1.0
        assert normalise_length_to_mm(1, 'cm') == 10.0
        assert normalise_length_to_mm(1, 'in') == 25.4

        with self.assertRaises(ValueError):
            normalise_length_to_mm(1, 'pixel')

    def test_surface_distance_requires_finite_ruler_endpoints(self) -> None:
        assert calculate_surface_distance_mm((0, 0), (3, 4)) == 5.0
        assert calculate_surface_distance_mm((0, 0), (0, 0)) == 0.0

        for point in (
            (math.nan, 0),
            (0, math.inf),
            None,
            (),
            (0, ),
            (0, 0, 0),
            {'x': 0, 'y': 0},
        ):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    validate_finite_point(point)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            project_surface_ruler((0, 0), (0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_positive_lengths_reject_zero_negative_and_non_finite_values(self) -> None:
        for unit, multiplier in (('mm', 1.0), ('cm', 10.0), ('in', 25.4)):
            with self.subTest(unit=unit):
                assert normalise_length_to_mm(0, unit) == 0.0
                assert normalise_length_to_mm(-2, unit) == -2 * multiplier
                with self.assertRaises(ValueError):
                    validate_positive_length(0, unit)
                with self.assertRaises(ValueError):
                    validate_positive_length(-2, unit)
                for invalid_length in (math.nan, math.inf, -math.inf):
                    with self.subTest(invalid_length=invalid_length):
                        with self.assertRaises(ValueError):
                            normalise_length_to_mm(invalid_length, unit)

    def test_ruler_rejects_invalid_bounds_and_all_out_of_bounds_endpoints(self) -> None:
        identity_matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        invalid_bounds = (
            Resolution(0, 100),
            Resolution(100, 0),
            (-1, 100),
            (100, 100.0),
            (100, ),
            CoordinateBounds(0, 0, 0, 100),
            CoordinateBounds(0, 0, 100, math.inf),
            {'width': 100, 'height': 100},
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    project_surface_ruler(
                        (1, 1),
                        (2, 2),
                        identity_matrix,
                        bounds,
                    )  # type: ignore[arg-type]

        for start, end in (((-1, 1), (2, 2)), ((1, 1), (100, 2)), ((1, 100), (2, 2))):
            with self.subTest(start=start, end=end):
                with self.assertRaises(PointOutsideProjectorError):
                    project_surface_ruler(start, end, identity_matrix, Resolution(100, 100))

    def test_perspective_ruler_does_not_use_a_constant_pixel_scale(self) -> None:
        matrix = ((2, 0, 0), (0, 2, 0), (0.001, 0, 1))
        near_ruler = project_surface_ruler((0, 0), (10, 0), matrix)
        far_ruler = project_surface_ruler((100, 0), (110, 0), matrix)

        near_projector_length = near_ruler.projector_end.x - near_ruler.projector_start.x
        far_projector_length = far_ruler.projector_end.x - far_ruler.projector_start.x
        assert near_projector_length != far_projector_length, (
            f'{near_projector_length=}, {far_projector_length=}'
        )

    def test_surface_projector_round_trip_preserves_surface_points(self) -> None:
        surface_to_projector_matrix = (
            (1.2, 0.1, 40.0),
            (0.05, 0.9, 30.0),
            (0.0002, 0.0001, 1.0),
        )
        transform = MetricHomographyPair.from_surface_to_projector(
            surface_to_projector_matrix,
        )
        surface_point = Point2D(100.0, 80.0)

        projector_point = surface_to_projector(surface_point, transform)
        recovered_point = projector_to_surface(projector_point, transform)

        assert math.isclose(recovered_point.x, surface_point.x, abs_tol=1e-8), (
            f'{recovered_point=}'
        )
        assert math.isclose(recovered_point.y, surface_point.y, abs_tol=1e-8), (
            f'{recovered_point=}'
        )

    def test_projector_surface_bounds_cover_the_complete_output(self) -> None:
        surface_to_projector = (
            (1.8, 0.15, 20.0),
            (0.1, 1.4, 10.0),
            (0.0004, 0.0002, 1.0),
        )
        transform = MetricHomographyPair.from_surface_to_projector(
            surface_to_projector,
        )

        bounds = calculate_projector_surface_bounds(transform, Resolution(640, 480))

        assert bounds.left < bounds.right, f'{bounds=}'
        assert bounds.top < bounds.bottom, f'{bounds=}'
        for projector_point in (
            Point2D(0, 0),
            Point2D(640, 0),
            Point2D(640, 480),
            Point2D(0, 480),
        ):
            surface_point = projector_to_surface(projector_point, transform)
            assert (
                bounds.left <= surface_point.x <= bounds.right
                and bounds.top <= surface_point.y <= bounds.bottom
            ), f'{projector_point=}, {bounds=}'

    def test_projector_surface_bounds_reject_a_bbox_crossing_the_horizon(self) -> None:
        surface_to_projector = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        )
        transform = MetricHomographyPair.from_surface_to_projector(
            surface_to_projector,
        )

        with self.assertRaises(InvalidHomographyError):
            calculate_projector_surface_bounds(transform, Resolution(2, 2))

    def test_ruler_uses_projective_geometry_and_projector_bounds(self) -> None:
        matrix = ((1.5, 0, 0), (0, 1.5, 0), (0.005, 0, 1))
        ruler = project_surface_ruler(
            (0, 0),
            (100, 0),
            matrix,
            Resolution(101, 100),
        )

        assert ruler.length_mm == 100.0, f'{ruler=}'
        assert ruler.projector_start == Point2D(0.0, 0.0), f'{ruler=}'
        assert ruler.projector_end == Point2D(100.0, 0.0), f'{ruler=}'
        assert project_surface_points(
            [(0, 0), (10, 10)],
            matrix,
            Resolution(101, 100),
        ) == (Point2D(0.0, 0.0), Point2D(14.285714285714286, 14.285714285714286))

        with self.assertRaises(PointOutsideProjectorError):
            project_surface_ruler(
                (0, 0),
                (102, 0),
                matrix,
                Resolution(101, 100),
            )

    def test_ruler_rejects_horizon_non_finite_and_degenerate_geometry(self) -> None:
        with self.assertRaises(InvalidHomographyError):
            project_surface_ruler(
                (0, 0),
                (100, 0),
                ((1, 0, 0), (0, 1, 0), (0.02, 0, -1)),
            )
        with self.assertRaises(ValueError):
            project_surface_ruler(
                (0, 0),
                (100, 0),
                ((math.nan, 0, 0), (0, 1, 0), (0, 0, 1)),
            )
        with self.assertRaises(ValueError):
            project_surface_ruler(
                (0, 0),
                (100, 0),
                ((1, 2, 3), (2, 4, 6), (0, 0, 0)),
            )

    def test_metric_ruler_has_exact_label_ticks_and_projected_marker_extents(self) -> None:
        ruler = build_metric_ruler(
            (100.0, 100.0),
            (200.0, 100.0),
            'cm',
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            Resolution(400, 400),
        )

        assert ruler.length_mm == 100.0, f'{ruler=}'
        assert ruler.length_in_output_unit == 10.0, f'{ruler=}'
        assert ruler.label == '10.0 cm', f'{ruler=}'
        assert [tick.distance_mm for tick in ruler.ticks] == list(range(5, 100, 5)), (
            f'{ruler.ticks=}'
        )
        assert ruler.ticks[1].is_major
        assert ruler.markers[0].projector_extent == (
            Point2D(96.0, 96.0),
            Point2D(104.0, 96.0),
            Point2D(104.0, 104.0),
            Point2D(96.0, 104.0),
        ), f'{ruler.markers=}'
        assert ruler.label_bounds.contains(ruler.label_position), f'{ruler=}'

    def test_metric_ruler_uses_perspective_for_ticks_and_markers(self) -> None:
        matrix = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.001, 0.0, 1.0))
        ruler = build_metric_ruler(
            (100.0, 100.0),
            (200.0, 100.0),
            'mm',
            matrix,
            Resolution(400, 400),
        )

        assert ruler.projector_start != Point2D(100.0, 100.0), f'{ruler=}'
        assert ruler.projector_end != Point2D(200.0, 100.0), f'{ruler=}'
        assert ruler.ticks[0].projector_start != ruler.ticks[0].surface_start, (
            f'{ruler.ticks[0]=}'
        )
        expected_tick_position = surface_to_projector(
            Point2D(105.0, 100.0),
            matrix,
        )
        assert ruler.ticks[0].projector_position == expected_tick_position, (
            f'{ruler.ticks[0]=}, {expected_tick_position=}'
        )
        assert ruler.tick_positions[0] == expected_tick_position, (
            f'{ruler.tick_positions=}, {expected_tick_position=}'
        )
        assert ruler.markers[0].projector_extent != ruler.markers[0].surface_extent, (
            f'{ruler.markers[0]=}'
        )

    def test_metric_ruler_has_complete_raster_safe_primitives(self) -> None:
        matrix = ((1.4, 0.08, 20.0), (0.02, 1.2, 30.0), (0.0003, 0.0002, 1.0))
        bounds = Resolution(400, 400)
        ruler = build_metric_ruler(
            (100.0, 100.0),
            (180.0, 100.0),
            'mm',
            matrix,
            bounds,
        )

        primitive_points = [ruler.projector_start, ruler.projector_end]
        for tick in ruler.ticks:
            primitive_points.extend((tick.projector_start, tick.projector_end))
        for marker in ruler.markers:
            primitive_points.extend(marker.projector_extent)
        primitive_points.extend(ruler.tick_positions)
        assert all(
            0 <= round(point.x) < bounds.width
            and 0 <= round(point.y) < bounds.height
            for point in primitive_points
        ), f'{primitive_points=}'
        assert bounds.width >= ruler.label_bounds.right > ruler.label_bounds.left >= 0
        assert bounds.height >= ruler.label_bounds.bottom > ruler.label_bounds.top >= 0
        assert ruler.label_bounds.contains(ruler.label_position), f'{ruler=}'

    def test_metric_ruler_bounds_tick_generation_for_sparse_projected_lines(self) -> None:
        matrix = ((1e-8, 0, 10), (1e-7, 1, 20), (1e-8, 0, 1))
        ruler = build_metric_ruler(
            (10.0, 10.0),
            (1e20 + 10.0, 10.0),
            'mm',
            matrix,
            Resolution(100, 100),
        )

        assert ruler.length_mm == 1e20, f'{ruler=}'
        assert 0 < len(ruler.ticks) <= 200, f'{len(ruler.ticks)=}'
        assert all(tick.distance_mm % 5.0 == 0 for tick in ruler.ticks), (
            f'{ruler.ticks=}'
        )
        assert ruler.ticks[0].distance_mm < ruler.ticks[-1].distance_mm, (
            f'{ruler.ticks=}'
        )

    def test_metric_ruler_rejects_invalid_units_sizes_and_unsafe_geometry(self) -> None:
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for unit in ('pixel', None):
            with self.subTest(unit=unit):
                with self.assertRaises(ValueError):
                    build_metric_ruler(
                        (10.0, 50.0),
                        (20.0, 50.0),
                        unit,
                        identity,
                        Resolution(100, 100),
                    )
        with self.assertRaises(ValueError):
            build_metric_ruler(
                (10.0, 50.0),
                (10.0, 50.0),
                'mm',
                identity,
                Resolution(100, 100),
            )
        with self.assertRaises(PointOutsideProjectorError):
            build_metric_ruler(
                (3.9, 50.0),
                (80.0, 50.0),
                'mm',
                identity,
                Resolution(100, 100),
            )
        with self.assertRaises(PointOutsideProjectorError):
            build_metric_ruler(
                (4.0, 50.0),
                (80.0, 50.0),
                'mm',
                identity,
                Resolution(100, 100),
            )

        with self.assertRaises(InvalidHomographyError):
            build_metric_ruler(
                (-99.5, 50.0),
                (-90.0, 50.0),
                'mm',
                ((1, 0, 0), (0, 1, 0), (0.01, 0, 1)),
                CoordinateBounds(-30000, -100, 30000, 20000),
            )

    def test_metric_calibration_recovers_perspective_composition_and_inverse(self) -> None:
        surface_to_projector = (
            (2.1, 0.12, 70.0),
            (0.03, 1.9, 55.0),
            (0.00035, 0.00018, 1.0),
        )
        camera_to_projector = (
            (1.1, 0.04, 25.0),
            (0.02, 1.05, 18.0),
            (0.0001, 0.0002, 1.0),
        )
        correspondences = _build_metric_correspondences(
            surface_to_projector,
            camera_to_projector,
        )

        result = calibrate_metric_homography(
            MetricTargetCorrespondences(correspondences, 'camera-0'),
            camera_to_projector,
            Resolution(800, 700),
        )

        assert result.observation_camera_id == 'camera-0', f'{result=}'
        assert result.correspondence_corner_count == 80, f'{result=}'
        assert result.unique_target_count == 20, f'{result=}'
        assert result.ransac_inlier_count == 80, f'{result=}'
        expected_coverage = (202.0 - 8.0) * (285.0 - 40.0) / (210.0 * 297.0)
        assert math.isclose(
            result.target_page_spatial_coverage,
            expected_coverage,
        ), f'{result.target_page_spatial_coverage=}, {expected_coverage=}'
        assert result.fit_error_mm < 0.001, f'{result=}'
        for surface_point in (METRIC_TARGET.markers[0].corners[0], (100.0, 150.0)):
            projector_point = surface_to_projector_point(surface_point, surface_to_projector)
            recovered_surface_point = projector_to_surface(
                projector_point,
                result.homography,
            )
            assert math.isclose(
                recovered_surface_point.x,
                float(surface_point[0]),
                abs_tol=0.01,
            ), f'{recovered_surface_point=}'
            assert math.isclose(
                recovered_surface_point.y,
                float(surface_point[1]),
                abs_tol=0.01,
            ), f'{recovered_surface_point=}'

    def test_metric_calibration_uses_configured_mm_ransac_and_inlier_only_errors(self) -> None:
        surface_to_projector = (
            (2.1, 0.12, 70.0),
            (0.03, 1.9, 55.0),
            (0.00035, 0.00018, 1.0),
        )
        correspondences = list(
            _build_metric_correspondences(
                surface_to_projector,
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                noise_radius_mm=0.1,
                outlier_index=17,
            ),
        )
        thresholds = MetricCalibrationThresholds(
            ransac_reprojection_threshold_mm=0.25,
        )
        observed_thresholds: list[float] = []

        class RecordingOpenCV:
            RANSAC = 8

            def findHomography(
                self,
                source: object,
                destination: object,
                method: int,
                threshold: float,
            ) -> tuple[object, object]:
                observed_thresholds.append(threshold)
                assert len(source) == 80, f'{len(source)=}'
                assert len(destination) == 80, f'{len(destination)=}'
                import cv2

                return cv2.findHomography(source, destination, method, threshold)

        result = calibrate_metric_homography(
            correspondences,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            Resolution(800, 700),
            thresholds,
            cv2_module=RecordingOpenCV(),
        )

        assert observed_thresholds == [0.25], f'{observed_thresholds=}'
        assert result.ransac_inlier_count == 79, f'{result=}'
        assert result.inlier_ratio < 1.0, f'{result=}'
        assert result.max_fit_error_mm < 1.0, f'{result=}'

    def test_metric_calibration_rejects_degenerate_coverage_transforms_and_versions(self) -> None:
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        weak_correspondences = _build_metric_correspondences(
            identity,
            identity,
            marker_ids=(0, 1, 4, 5),
        )
        with self.assertRaises(CalibrationError):
            calibrate_metric_homography(
                weak_correspondences,
                identity,
                Resolution(400, 400),
            )

        degenerate_correspondences = tuple(
            correspondence._replace(camera_position=Point2D(10.0, 10.0))
            for correspondence in _build_metric_correspondences(identity, identity)
        )
        with self.assertRaises(CalibrationError):
            calibrate_metric_homography(
                degenerate_correspondences,
                identity,
                Resolution(400, 400),
            )

        with self.assertRaises(CalibrationError):
            calibrate_metric_homography(
                _build_metric_correspondences(
                    ((1, 0, 500), (0, 1, 500), (0, 0, 1)),
                    identity,
                ),
                identity,
                Resolution(400, 400),
            )

        with self.assertRaises(CalibrationError):
            calibrate_metric_homography(
                _build_metric_correspondences(identity, identity),
                ((1, 0, 0), (0, 1, 0), (0, 0, 0)),
                Resolution(400, 400),
            )

        malformed_target = METRIC_TARGET._replace(format_version=2)
        with self.assertRaises(CalibrationError):
            calibrate_metric_homography(
                _build_metric_correspondences(identity, identity),
                identity,
                Resolution(400, 400),
                target=malformed_target,
            )

    def test_three_metric_frames_are_averaged_only_within_stability_tolerance(self) -> None:
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        base_correspondences = _build_metric_correspondences(identity, identity)
        frames = tuple(
            MetricTargetCorrespondences(
                tuple(
                    correspondence._replace(
                        camera_position=Point2D(
                            correspondence.camera_position.x + offset_x,
                            correspondence.camera_position.y,
                        ),
                    )
                    for correspondence in base_correspondences
                ),
                'camera-0',
            )
            for offset_x in (0.0, 0.5, 1.0)
        )

        averaged = _average_metric_correspondences(frames, 1.1, 0.8, 'camera-0')

        assert averaged.camera_id == 'camera-0', f'{averaged=}'
        assert averaged.correspondences[0].camera_position == Point2D(8.5, 40.0), (
            f'{averaged.correspondences[0]=}'
        )
        moving_frames = tuple(
            MetricTargetCorrespondences(
                tuple(
                    correspondence._replace(
                        camera_position=Point2D(
                            correspondence.camera_position.x + offset_x,
                            correspondence.camera_position.y,
                        ),
                    )
                    for correspondence in base_correspondences
                ),
                'camera-0',
            )
            for offset_x in (0.0, 2.4, 4.8)
        )
        with self.assertRaises(CalibrationError):
            _average_metric_correspondences(
                moving_frames,
                1.1,
                0.8,
                'camera-0',
            )

        def without_markers(
            frame: MetricTargetCorrespondences,
            marker_ids: set[int],
        ) -> MetricTargetCorrespondences:
            return frame._replace(
                correspondences=tuple(
                    correspondence
                    for correspondence in frame.correspondences
                    if correspondence.marker_id not in marker_ids
                ),
            )

        base_frame = MetricTargetCorrespondences(base_correspondences, 'camera-0')
        stable_subset = _average_metric_correspondences(
            (
                without_markers(base_frame, {0}),
                base_frame,
                without_markers(base_frame, {1}),
            ),
            1.1,
            0.8,
            'camera-0',
        )
        assert len(stable_subset.unique_marker_ids) == 18, f'{stable_subset=}'
        assert len(stable_subset.correspondences) == 72, f'{stable_subset=}'

    def test_fit_error_does_not_become_physical_validation_error(self) -> None:
        projector_resolution = Resolution(800, 700)
        result = MetricCalibrationResult(
            MetricHomographyPair.from_projector_to_surface(
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ),
            MetricCalibrationMetrics(20, 80, 80, 1.0, 1.25, 2.0, 0.8),
            projector_resolution,
            METRIC_TARGET.format_name,
            METRIC_TARGET.format_version,
            METRIC_TARGET.marker_family,
        )
        descriptor = ProjectorOutputDescriptor(projector_resolution)
        record = MetricCalibrationRegistry(descriptor).register(result, descriptor)

        assert result.fit_error_mm > 0.0, f'{result=}'
        assert record.fit_error_mm == result.fit_error_mm, f'{record=}'
        assert record.latest_physical_validation_error_mm is None, f'{record=}'
        assert record.validation_records == (), f'{record=}'


def _build_metric_correspondences(
    surface_to_projector: tuple[tuple[float, float, float], ...],
    camera_to_projector: tuple[tuple[float, float, float], ...],
    marker_ids: tuple[int, ...] | None = None,
    noise_radius_mm: float = 0.0,
    outlier_index: int | None = None,
) -> tuple[MetricTargetCorrespondence, ...]:
    marker_ids = marker_ids or tuple(marker.marker_id for marker in METRIC_TARGET.markers)
    camera_to_projector_pair = HomographyPair.from_camera_to_projector(camera_to_projector)
    random_generator = random.Random(7)
    correspondences: list[MetricTargetCorrespondence] = []
    for marker_id in marker_ids:
        marker = METRIC_TARGET.markers[marker_id]
        for corner_index, surface_point in enumerate(marker.corners):
            projector_point = surface_to_projector_point(
                surface_point,
                surface_to_projector,
            )
            camera_point = project_point(
                projector_point,
                camera_to_projector_pair.projector_to_camera,
            )
            if noise_radius_mm > 0:
                camera_point = Point2D(
                    camera_point.x + random_generator.uniform(
                        -noise_radius_mm,
                        noise_radius_mm,
                    ),
                    camera_point.y + random_generator.uniform(
                        -noise_radius_mm,
                        noise_radius_mm,
                    ),
                )
            if outlier_index == len(correspondences):
                camera_point = Point2D(
                    camera_point.x + 300.0,
                    camera_point.y - 200.0,
                )
            correspondences.append(
                MetricTargetCorrespondence(
                    marker_id,
                    corner_index,
                    surface_point,
                    camera_point,
                ),
            )
    return tuple(correspondences)


def surface_to_projector_point(
    surface_point: Point2D | tuple[float, float],
    surface_to_projector: tuple[tuple[float, float, float], ...],
) -> Point2D:
    return project_point(surface_point, surface_to_projector)


if __name__ == '__main__':
    unittest.main()
