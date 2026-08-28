import math
import unittest

from multivision.geometry import (
    CoordinateBounds,
    HomographyPair,
    Point2D,
    build_preview_transform,
    camera_to_projector,
    invert_homography,
    is_finite_point,
    is_point_in_bounds,
    is_valid_homography,
    is_point_in_region,
    projector_to_camera,
    project_camera_to_projector,
    project_point,
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

    def test_malformed_and_degenerate_regions_are_rejected(self) -> None:
        polygon = [(0, 0), (100, 0), (0, 100)]

        assert not is_point_in_region((10, 10), set(polygon))
        assert not is_point_in_region((10, 0), [(0, 0), (50, 0), (100, 0)])

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
