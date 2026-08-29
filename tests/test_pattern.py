import math
import unittest

from multivision.geometry import CoordinateBounds, Point2D
from multivision.pattern import (
    APRILTAG_36H11,
    CalibrationPattern,
    build_calibration_pattern,
)
from multivision.types import Resolution


class CalibrationPatternTest(unittest.TestCase):
    def test_default_pattern_has_unique_markers_across_the_projector(self) -> None:
        pattern = build_calibration_pattern(Resolution(1920, 1080))

        assert isinstance(pattern, CalibrationPattern)
        assert pattern.marker_family == APRILTAG_36H11
        assert 9 <= len(pattern.markers) <= 12
        assert len({marker.marker_id for marker in pattern.markers}) == len(pattern.markers)
        assert all(
            pattern.usable_area.contains(corner)
            for marker in pattern.markers
            for corner in marker.corners
        ), f'{pattern=}'
        assert pattern.markers[0].marker_id == 0
        assert pattern.markers[-1].marker_id == len(pattern.markers) - 1

    def test_every_corner_has_identity_and_stable_projector_coordinates(self) -> None:
        pattern = build_calibration_pattern(
            Resolution(1000, 800),
            usable_area=CoordinateBounds(100, 50, 900, 750),
        )

        corner_metadata = pattern.marker_corners
        assert len(corner_metadata) == len(pattern.markers) * 4
        assert [corner.corner_index for corner in pattern.markers[0].corner_metadata] == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            corner.projector_position == pattern.markers[corner.marker_id].corners[
                corner.corner_index
            ]
            for corner in corner_metadata
        ), f'{corner_metadata=}'
        assert all(
            corner.projector_position == Point2D(
                float(corner.projector_position.x),
                float(corner.projector_position.y),
            )
            for corner in corner_metadata
        )

    def test_pattern_is_independent_of_camera_and_ui_dimensions(self) -> None:
        first_pattern = build_calibration_pattern(Resolution(1600, 900))
        second_pattern = build_calibration_pattern(Resolution(1600, 900))

        assert first_pattern == second_pattern
        assert not hasattr(first_pattern, 'camera_resolution')
        assert not hasattr(first_pattern, 'preview_size')

    def test_marker_count_is_tunable_only_within_contract(self) -> None:
        for marker_count in range(9, 13):
            pattern = build_calibration_pattern(
                Resolution(1200, 800),
                marker_count=marker_count,
            )
            assert len(pattern.markers) == marker_count, f'{marker_count=}'

        for marker_count in [8, 13]:
            with self.subTest(marker_count=marker_count):
                with self.assertRaises(ValueError):
                    build_calibration_pattern(
                        Resolution(1200, 800),
                        marker_count=marker_count,
                    )

    def test_all_layouts_use_raster_safe_usable_area_edges(self) -> None:
        usable_area = CoordinateBounds(37, 23, 973, 677)
        marker_size = 80
        for marker_count in range(9, 13):
            with self.subTest(marker_count=marker_count):
                pattern = build_calibration_pattern(
                    Resolution(1000, 700),
                    usable_area=usable_area,
                    marker_count=marker_count,
                    marker_size=marker_size,
                )
                first_marker = pattern.markers[0].bounds
                last_marker = pattern.markers[-1].bounds
                assert first_marker.left == usable_area.left, f'{pattern=}'
                assert first_marker.top == usable_area.top, f'{pattern=}'
                assert last_marker.right < usable_area.right, f'{pattern=}'
                assert last_marker.bottom < usable_area.bottom, f'{pattern=}'
                assert (
                    last_marker.right > usable_area.right - marker_size - 1
                ), f'{pattern=}'
                assert (
                    last_marker.bottom > usable_area.bottom - marker_size - 1
                ), f'{pattern=}'
                rendered_size = max(1, round(pattern.marker_size))
                assert round(first_marker.left) + rendered_size <= math.floor(
                    usable_area.right,
                )
                assert round(last_marker.left) + rendered_size <= math.floor(
                    usable_area.right,
                )
                assert round(first_marker.top) + rendered_size <= math.floor(
                    usable_area.bottom,
                )
                assert round(last_marker.top) + rendered_size <= math.floor(
                    usable_area.bottom,
                )
                assert build_calibration_pattern(
                    Resolution(1000, 700),
                    usable_area=usable_area,
                    marker_count=marker_count,
                    marker_size=marker_size,
                ) == pattern

    def test_fractional_edges_use_maximum_raster_safe_proximity(self) -> None:
        usable_area = CoordinateBounds(37.4, 23.4, 973.4, 677.4)
        marker_size = 80.6
        pattern = build_calibration_pattern(
            Resolution(1000, 700),
            usable_area=usable_area,
            marker_size=marker_size,
        )

        first_marker = pattern.markers[0].bounds
        last_marker = pattern.markers[-1].bounds
        assert first_marker.left == 37.5, f'{first_marker=}'
        assert first_marker.top == 23.5, f'{first_marker=}'
        assert last_marker.left == 892.5, f'{last_marker=}'
        assert last_marker.top == 596.5, f'{last_marker=}'
        assert last_marker.right == 973.1, f'{last_marker=}'
        assert last_marker.bottom == 677.1, f'{last_marker=}'
        assert (
            round(first_marker.left) + round(marker_size) <= usable_area.right
        )
        assert (
            round(first_marker.top) + round(marker_size) <= usable_area.bottom
        )
        assert (
            round(last_marker.left) + round(marker_size) <= usable_area.right
        )
        assert (
            round(last_marker.top) + round(marker_size) <= usable_area.bottom
        )
        assert all(
            usable_area.contains(corner)
            for marker in pattern.markers
            for corner in marker.corners
        ), f'{pattern=}'

    def test_rendered_marker_rectangles_stay_inside_surface_and_usable_area(self) -> None:
        projector_resolution = Resolution(1000, 700)
        usable_area = CoordinateBounds(37.4, 23.4, 973.4, 677.4)
        for marker_count in range(9, 13):
            with self.subTest(marker_count=marker_count):
                pattern = build_calibration_pattern(
                    projector_resolution,
                    usable_area=usable_area,
                    marker_count=marker_count,
                    marker_size=80.6,
                )
                for marker in pattern.markers:
                    rendered_left = round(marker.bounds.left)
                    rendered_top = round(marker.bounds.top)
                    rendered_width = max(
                        1,
                        round(marker.bounds.right - marker.bounds.left),
                    )
                    rendered_height = max(
                        1,
                        round(marker.bounds.bottom - marker.bounds.top),
                    )
                    rendered_bounds = CoordinateBounds(
                        rendered_left,
                        rendered_top,
                        rendered_left + rendered_width,
                        rendered_top + rendered_height,
                    )
                    assert rendered_bounds.left >= usable_area.left, f'{rendered_bounds=}'
                    assert rendered_bounds.top >= usable_area.top, f'{rendered_bounds=}'
                    assert rendered_bounds.right <= usable_area.right, f'{rendered_bounds=}'
                    assert rendered_bounds.bottom <= usable_area.bottom, f'{rendered_bounds=}'
                    assert rendered_bounds.left >= 0, f'{rendered_bounds=}'
                    assert rendered_bounds.top >= 0, f'{rendered_bounds=}'
                    assert rendered_bounds.right <= projector_resolution.width, f'{rendered_bounds=}'
                    assert rendered_bounds.bottom <= projector_resolution.height, f'{rendered_bounds=}'

    def test_marker_size_must_fit_every_supported_layout_cell(self) -> None:
        usable_area = CoordinateBounds(0, 0, 1000, 600)
        for marker_count in range(9, 13):
            with self.subTest(marker_count=marker_count):
                pattern = build_calibration_pattern(
                    Resolution(1000, 600),
                    usable_area=usable_area,
                    marker_count=marker_count,
                    marker_size=199,
                )
                assert len(pattern.markers) == marker_count, f'{pattern=}'
                with self.assertRaises(ValueError):
                    build_calibration_pattern(
                        Resolution(1000, 600),
                        usable_area=usable_area,
                        marker_count=marker_count,
                        marker_size=200,
                    )

    def test_malformed_dimensions_and_marker_sizes_fail_as_value_errors(self) -> None:
        invalid_resolutions = [
            None,
            (),
            (1920,),
            (1920, 1080, 1),
            {1: 1920, 2: 1080},
            {1920, 1080},
            b'12',
            bytearray(b'12'),
            (True, 1080),
            (0, 1080),
            (-1, 1080),
            (10**1000, 1080),
        ]
        for projector_resolution in invalid_resolutions:
            with self.subTest(projector_resolution=projector_resolution):
                with self.assertRaises(ValueError):
                    build_calibration_pattern(projector_resolution)  # type: ignore[arg-type]

        invalid_marker_sizes = [
            math.nan,
            math.inf,
            True,
            0,
            -1,
            10**1000,
        ]
        for marker_size in invalid_marker_sizes:
            with self.subTest(marker_size=marker_size):
                with self.assertRaises(ValueError):
                    build_calibration_pattern(
                        Resolution(1920, 1080),
                        marker_size=marker_size,
                    )

    def test_malformed_usable_area_and_family_fail_as_value_errors(self) -> None:
        invalid_areas = [
            CoordinateBounds(0, 0, 0, 100),
            CoordinateBounds(0, 0, math.nan, 100),
            CoordinateBounds(0, 0, math.inf, 100),
            CoordinateBounds(-1, 0, 100, 100),
            CoordinateBounds(0, 0, 100, 801),
            (0, 0, 100, 100),
        ]
        for usable_area in invalid_areas:
            with self.subTest(usable_area=usable_area):
                with self.assertRaises(ValueError):
                    build_calibration_pattern(
                        Resolution(1000, 800),
                        usable_area=usable_area,  # type: ignore[arg-type]
                    )

        for marker_family in [None, '', 'DICT_UNKNOWN']:
            with self.subTest(marker_family=marker_family):
                with self.assertRaises(ValueError):
                    build_calibration_pattern(
                        Resolution(1000, 800),
                        marker_family=marker_family,  # type: ignore[arg-type]
                    )

    def test_all_marker_metadata_is_complete_and_lookup_is_explicit(self) -> None:
        pattern = build_calibration_pattern(Resolution(1200, 800), marker_count=11)

        assert tuple(pattern.get_marker(marker_id) for marker_id in range(11)) == pattern.markers
        assert all(
            len(marker.corners) == 4
            and len(set(marker.corners)) == 4
            and marker.bounds == CoordinateBounds(
                marker.corners[0].x,
                marker.corners[0].y,
                marker.corners[2].x,
                marker.corners[2].y,
            )
            for marker in pattern.markers
        ), f'{pattern=}'
        with self.assertRaises(KeyError):
            pattern.get_marker(11)


if __name__ == '__main__':
    unittest.main()
