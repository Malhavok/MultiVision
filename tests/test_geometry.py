import math
import unittest

from multivision.geometry import (
    CoordinateBounds,
    HomographyPair,
    Point2D,
    build_preview_transform,
    calculate_available_projector_area,
    camera_to_projector,
    intersect_polygon_with_bounds,
    invert_homography,
    is_finite_point,
    is_point_in_bounds,
    is_point_in_resolution,
    is_point_in_region,
    is_valid_homography,
    projector_to_camera,
    project_camera_to_projector,
    project_point,
    project_polygon,
    preview_local_to_camera_native,
)
from multivision.types import Resolution


class GeometryTest(unittest.TestCase):
    def test_scaled_preview_maps_to_native_coordinates(self) -> None:
        native_resolution = Resolution(1920, 1080)
        preview_resolution = Resolution(960, 540)

        native_point = preview_local_to_camera_native(
            Point2D(480, 270),
            preview_resolution,
            native_resolution,
        )

        assert native_point == Point2D(960.0, 540.0), f'{native_point=}'

    def test_letterboxed_preview_maps_only_image_content(self) -> None:
        transform = build_preview_transform(
            Resolution(1000, 1000),
            Resolution(1920, 1080),
        )

        expected_bounds = CoordinateBounds(0, 218.75, 1000, 781.25)
        assert all(
            math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(transform.content_bounds, expected_bounds)
        ), f'{transform.content_bounds=}'
        native_point = transform.to_camera_native(Point2D(500, 500))
        assert math.isclose(native_point.x, 960.0, abs_tol=1e-9), f'{native_point=}'
        assert math.isclose(native_point.y, 540.0, abs_tol=1e-9), f'{native_point=}'
        with self.assertRaises(ValueError):
            transform.to_camera_native(Point2D(500, 100))

    def test_degenerate_homography_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            invert_homography(((1, 2, 3), (2, 4, 6), (0, 0, 0)))

        with self.assertRaises(ValueError):
            projector_to_camera(Point2D(1, 2), ((0, 0, 0), (0, 0, 0), (0, 0, 0)))

        unordered_matrix = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
        assert not is_valid_homography(unordered_matrix)

    def test_homography_round_trip(self) -> None:
        projector_to_camera_matrix = (
            (1.2, 0.1, 40.0),
            (0.05, 0.9, 30.0),
            (0.0002, 0.0001, 1.0),
        )
        transforms = HomographyPair.from_projector_to_camera(projector_to_camera_matrix)
        projector_point = Point2D(640, 360)

        camera_point = projector_to_camera(projector_point, transforms)
        round_trip_point = camera_to_projector(camera_point, transforms)

        assert math.isclose(
            round_trip_point.x,
            projector_point.x,
            abs_tol=1e-8,
        ), f'{round_trip_point=}'
        assert math.isclose(
            round_trip_point.y,
            projector_point.y,
            abs_tol=1e-8,
        ), f'{round_trip_point=}'

    def test_finite_and_bounds_checks_reject_invalid_points(self) -> None:
        bounds = CoordinateBounds(0, 0, 1280, 720)

        assert is_finite_point(Point2D(1, 2))
        assert is_finite_point((coordinate for coordinate in (1, 2)))
        assert project_point(
            (coordinate for coordinate in (1, 2)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ) == Point2D(1.0, 2.0)
        assert not is_finite_point(Point2D(math.nan, 2))
        assert not is_finite_point({1, 2})
        assert not is_finite_point(b'12')
        assert is_point_in_bounds(Point2D(1279.99, 719.99), bounds)
        assert not is_point_in_bounds(Point2D(1280, 719), bounds)
        assert not is_point_in_bounds(Point2D(-1, 20), bounds)
        assert not is_point_in_bounds({1279, 719}, bounds)
        assert not is_point_in_resolution(
            Point2D(10, 10),
            {1280, 720},  # type: ignore[arg-type]
        )

    def test_malformed_and_degenerate_regions_are_rejected(self) -> None:
        polygon = [(0, 0), (100, 0), (0, 100)]

        assert not is_point_in_region((10, 10), set(polygon))
        assert not is_point_in_region((10, 0), [(0, 0), (50, 0), (100, 0)])

    def test_polygon_intersection_and_projector_clipping_are_fail_closed(self) -> None:
        native_bounds = CoordinateBounds(0, 0, 100, 80)
        camera_polygon = [
            (-10, 10),
            (50, -10),
            (110, 10),
            (110, 70),
            (50, 90),
            (-10, 70),
        ]

        intersected_polygon = intersect_polygon_with_bounds(
            camera_polygon,
            native_bounds,
        )
        assert intersected_polygon is not None, f'{intersected_polygon=}'
        assert all(
            native_bounds.left <= point.x <= native_bounds.right
            for point in intersected_polygon
        )
        assert all(
            native_bounds.top <= point.y <= native_bounds.bottom
            for point in intersected_polygon
        )
        assert intersect_polygon_with_bounds(
            [(200, 200), (210, 200), (210, 210)],
            native_bounds,
        ) is None
        clipped_polygon = intersect_polygon_with_bounds(
            [(-10, -10), (110, -10), (110, 90), (-10, 90)],
            native_bounds,
        )
        assert clipped_polygon is not None, f'{clipped_polygon=}'
        assert set(clipped_polygon) == {
            Point2D(0, 0),
            Point2D(100, 0),
            Point2D(100, 80),
            Point2D(0, 80),
        }, f'{clipped_polygon=}'
        assert intersect_polygon_with_bounds(
            [(0, 0), (10, 0), (20, 0)],
            native_bounds,
        ) is None
        assert intersect_polygon_with_bounds(
            [(0, 0), (math.nan, 10), (10, 10)],
            native_bounds,
        ) is None

    def test_project_polygon_supports_perspective_and_identity(self) -> None:
        polygon = [(10, 10), (90, 10), (90, 70), (10, 70)]
        identity_polygon = project_polygon(
            polygon,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        assert identity_polygon == tuple(
            Point2D(*point)
            for point in polygon
        ), f'{identity_polygon=}'

        perspective_polygon = project_polygon(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            ((1, 0, 0), (0, 1, 0), (0.001, 0, 1)),
        )
        assert perspective_polygon is not None, f'{perspective_polygon=}'
        assert math.isclose(perspective_polygon[1].x, 100 / 1.1, abs_tol=1e-9)
        assert math.isclose(perspective_polygon[2].x, 100 / 1.1, abs_tol=1e-9)

    def test_project_polygon_rejects_invalid_and_horizon_crossing_transforms(self) -> None:
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert project_polygon(polygon, ((math.nan, 0, 0), (0, 1, 0), (0, 0, 1))) is None
        assert project_polygon(polygon, ((1, 0, 0), (0, 1, 0), (1, 0, -50))) is None
        assert project_polygon(
            [(0, 0), (1, 0), (2, 0)],
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ) is None

    def test_available_projector_area_intersects_projects_and_clips(self) -> None:
        available_area = calculate_available_projector_area(
            [(-10, 10), (50, -10), (110, 10), (110, 70), (50, 90), (-10, 70)],
            Resolution(100, 80),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            Resolution(100, 80),
        )
        assert available_area is not None, f'{available_area=}'
        assert all(
            0 <= point.x <= 100 and 0 <= point.y <= 80
            for point in available_area
        )

    def test_available_area_intersects_native_bounds_before_projector_clipping(self) -> None:
        camera_polygon = [(-20, -10), (120, -10), (120, 90), (-20, 90)]
        native_intersection = intersect_polygon_with_bounds(
            camera_polygon,
            Resolution(100, 80),
        )
        assert native_intersection is not None, f'{native_intersection=}'
        assert set(native_intersection) == {
            Point2D(0, 0),
            Point2D(100, 0),
            Point2D(100, 80),
            Point2D(0, 80),
        }, f'{native_intersection=}'

        available_area = calculate_available_projector_area(
            camera_polygon,
            Resolution(100, 80),
            ((1, 0, 20), (0, 1, 10), (0, 0, 1)),
            Resolution(70, 60),
        )
        assert available_area is not None, f'{available_area=}'
        assert set(available_area) == {
            Point2D(20, 10),
            Point2D(70, 10),
            Point2D(70, 60),
            Point2D(20, 60),
        }, f'{available_area=}'

    def test_perspective_projection_is_clipped_to_projector_bounds(self) -> None:
        available_area = calculate_available_projector_area(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            Resolution(100, 100),
            ((2, 0, 0), (0, 1, 0), (0.005, 0, 1)),
            Resolution(100, 100),
        )
        assert available_area is not None, f'{available_area=}'
        expected_points = (
            Point2D(0, 0),
            Point2D(100, 0),
            Point2D(100, 75),
            Point2D(0, 100),
        )
        assert len(available_area) == len(expected_points), f'{available_area=}'
        for expected_point in expected_points:
            assert any(
                math.isclose(actual_point.x, expected_point.x, abs_tol=1e-9)
                and math.isclose(actual_point.y, expected_point.y, abs_tol=1e-9)
                for actual_point in available_area
            ), f'{available_area=}, {expected_point=}'
        assert all(
            0 <= point.x <= 100 and 0 <= point.y <= 100
            for point in available_area
        ), f'{available_area=}'

    def test_available_area_fails_closed_for_invalid_or_empty_results(self) -> None:
        valid_polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        identity_matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        invalid_cases = [
            ([], identity_matrix),
            ([(0, 0), (50, 0), (100, 0)], identity_matrix),
            ([(0, 0), (math.nan, 50), (100, 100)], identity_matrix),
            (valid_polygon, ((math.nan, 0, 0), (0, 1, 0), (0, 0, 1))),
            (valid_polygon, ((1, 2, 3), (2, 4, 6), (0, 0, 0))),
            (valid_polygon, ((1, 0, 0), (0, 1, 0), (1, 0, -50))),
            (valid_polygon, ((1, 0, 200), (0, 1, 0), (0, 0, 1))),
        ]
        for camera_polygon, homography in invalid_cases:
            with self.subTest(camera_polygon=camera_polygon, homography=homography):
                available_area = calculate_available_projector_area(
                    camera_polygon,
                    Resolution(100, 100),
                    homography,
                    Resolution(100, 100),
                )
                assert available_area is None, f'{available_area=}'

    def test_outside_calibrated_region_and_projector_bounds_are_rejected(self) -> None:
        transforms = HomographyPair.from_projector_to_camera(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        calibrated_region = CoordinateBounds(100, 100, 900, 700)

        with self.assertRaises(ValueError):
            project_camera_to_projector(
                Point2D(50, 200),
                transforms,
                calibrated_region=calibrated_region,
                projector_resolution=Resolution(1000, 800),
            )
        with self.assertRaises(ValueError):
            project_camera_to_projector(
                Point2D(950, 200),
                transforms,
                calibrated_region=calibrated_region,
                projector_resolution=Resolution(900, 800),
            )
        assert is_point_in_region(Point2D(100, 100), calibrated_region)
        assert not is_point_in_region(Point2D(99.99, 100), calibrated_region)


if __name__ == '__main__':
    unittest.main()
