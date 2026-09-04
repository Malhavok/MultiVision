import base64
import math
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from multivision.errors import FiducialDetectionError
from multivision.fiducials import (
    DetectedMarker,
    assemble_metric_correspondences,
    detect_metric_fiducials,
)
from multivision.geometry import Point2D, project_point
from multivision.metric_target import (
    A4_PAGE_HEIGHT_MM,
    A4_PAGE_WIDTH_MM,
    METRIC_TARGET,
    METRIC_TARGET_FORMAT,
    METRIC_TARGET_FORMAT_VERSION,
    METRIC_TARGET_MARKER_COUNT,
    METRIC_TARGET_MARKER_FAMILY,
    METRIC_TARGET_MARKER_SIZE_MM,
    REFERENCE_SEGMENT_LABEL,
    REFERENCE_SEGMENT_LENGTH_MM,
    build_metric_target,
    generate_metric_target_svg,
    validate_metric_target,
    write_metric_target_svg,
)


class MetricTargetTest(unittest.TestCase):
    def test_target_is_deterministic_and_projector_independent(self) -> None:
        target = build_metric_target()

        assert target == METRIC_TARGET, f'{target=}'
        assert target.format_name == METRIC_TARGET_FORMAT, f'{target=}'
        assert target.format_version == METRIC_TARGET_FORMAT_VERSION, f'{target=}'
        assert target.marker_family == METRIC_TARGET_MARKER_FAMILY, f'{target=}'
        assert target.page_size_mm == (A4_PAGE_WIDTH_MM, A4_PAGE_HEIGHT_MM), f'{target=}'
        assert target.marker_count == METRIC_TARGET_MARKER_COUNT, f'{target=}'
        assert target.marker_size_mm == METRIC_TARGET_MARKER_SIZE_MM, f'{target=}'

    def test_marker_layout_has_ordered_unique_surface_corners(self) -> None:
        target = METRIC_TARGET
        marker_ids = [marker.marker_id for marker in target.markers]
        corners = [corner for marker in target.markers for corner in marker.corners]

        assert marker_ids == list(range(METRIC_TARGET_MARKER_COUNT)), f'{marker_ids=}'
        assert len(corners) == METRIC_TARGET_MARKER_COUNT * 4, f'{len(corners)=}'
        assert len(set(corners)) == len(corners), f'{corners=}'
        for marker in target.markers:
            assert len(marker.corners) == 4, f'{marker=}'
            assert all(
                math.isfinite(coordinate)
                for point in marker.corners
                for coordinate in point
            ), f'{marker=}'
            assert marker.corners[0].x < marker.corners[1].x, f'{marker=}'
            assert marker.corners[0].y < marker.corners[2].y, f'{marker=}'

    def test_layout_matches_declared_a4_geometry(self) -> None:
        target = METRIC_TARGET
        x_step_mm = (165.0 - 5.0) / 3
        y_step_mm = (249.0 - 34.0) / 4

        for marker_id, marker in enumerate(target.markers):
            y_idx, x_idx = divmod(marker_id, 4)
            x_start_mm = 5.0 + x_idx * x_step_mm
            y_start_mm = 34.0 + y_idx * y_step_mm
            expected_corners = (
                Point2D(x_start_mm, y_start_mm),
                Point2D(x_start_mm + 40.0, y_start_mm),
                Point2D(x_start_mm + 40.0, y_start_mm + 40.0),
                Point2D(x_start_mm, y_start_mm + 40.0),
            )
            assert marker.corners == expected_corners, f'{marker=}, {expected_corners=}'
            assert all(target.page_bounds.contains(point) for point in marker.corners), (
                f'{marker=}'
            )

    def test_target_assembly_resolves_rotated_projective_corner_order(self) -> None:
        target_to_camera = (
            (1.6, 0.08, 420.0),
            (0.04, 1.4, 110.0),
            (0.00035, 0.0002, 1.0),
        )
        detections = []
        for marker_id in (0, 3, 16, 19):
            marker = METRIC_TARGET.markers[marker_id]
            camera_corners = tuple(
                project_point(point, target_to_camera)
                for point in marker.corners
            )
            shift = marker_id % 4
            detections.append(
                DetectedMarker(
                    marker_id,
                    camera_corners[shift:] + camera_corners[:shift],
                ),
            )

        result = assemble_metric_correspondences(detections, camera_id='camera-0')

        assert result.camera_id == 'camera-0', f'{result=}'
        assert result.unique_marker_ids == (0, 3, 16, 19), f'{result=}'
        for correspondence in result.correspondences:
            expected_camera_point = project_point(
                correspondence.surface_position,
                target_to_camera,
            )
            assert correspondence.camera_position == expected_camera_point, (
                f'{correspondence=}, {expected_camera_point=}'
            )

    def test_target_assembly_rejects_unknown_duplicate_partial_and_weak_evidence(self) -> None:
        valid_detections = [
            DetectedMarker(marker_id, METRIC_TARGET.markers[marker_id].corners)
            for marker_id in (0, 3, 16, 19)
        ]
        invalid_detection_sets = (
            valid_detections[:-1] + [DetectedMarker(999, valid_detections[-1].corners)],
            valid_detections + [valid_detections[0]],
            valid_detections[:-1] + [
                DetectedMarker(19, valid_detections[-1].corners[:3]),
            ],
            [
                DetectedMarker(marker_id, METRIC_TARGET.markers[marker_id].corners)
                for marker_id in range(4)
            ],
        )

        for detections in invalid_detection_sets:
            with self.subTest(detections=detections):
                with self.assertRaises(FiducialDetectionError):
                    assemble_metric_correspondences(detections)

    def test_strict_detector_rejects_non_finite_and_non_convex_evidence(self) -> None:
        invalid_corners = (
            (
                Point2D(math.nan, 0.0),
                Point2D(1.0, 0.0),
                Point2D(1.0, 1.0),
                Point2D(0.0, 1.0),
            ),
            (
                Point2D(0.0, 0.0),
                Point2D(1.0, 1.0),
                Point2D(0.0, 1.0),
                Point2D(1.0, 0.0),
            ),
        )

        class FakeDetector:
            def __init__(self, corners: tuple[Point2D, ...]) -> None:
                self.corners = corners

            def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
                return (DetectedMarker(0, self.corners),)

        for corners in invalid_corners:
            with self.subTest(corners=corners):
                with self.assertRaises(FiducialDetectionError):
                    detect_metric_fiducials(object(), FakeDetector(corners))

    def test_orientation_and_reference_metadata_are_in_page(self) -> None:
        target = METRIC_TARGET

        for point in target.orientation_cue.corners:
            assert target.page_bounds.contains(point), f'{point=}'
        assert target.page_bounds.contains(
            target.orientation_cue.text_position,
        ), f'{target.orientation_cue.text_position=}'
        assert target.page_bounds.contains(
            target.version_text_position,
        ), f'{target.version_text_position=}'
        assert target.reference_segment.length_mm == REFERENCE_SEGMENT_LENGTH_MM, (
            f'{target.reference_segment.length_mm=}'
        )
        assert target.reference_segment.label == REFERENCE_SEGMENT_LABEL, (
            f'{target.reference_segment.label=}'
        )
        assert target.orientation_cue.corners[-1].x < target.orientation_cue.corners[0].x, (
            f'{target.orientation_cue.corners=}'
        )
        assert max(point.y for point in target.orientation_cue.corners) < min(
            point.y
            for marker in target.markers
            for point in marker.corners
        )

    def test_validation_rejects_a_different_target_definition(self) -> None:
        invalid_target = METRIC_TARGET._replace(
            format_version=METRIC_TARGET_FORMAT_VERSION + 1,
        )

        with self.assertRaises(ValueError):
            validate_metric_target(invalid_target)

    def test_svg_is_byte_stable_and_contains_physical_target_metadata(self) -> None:
        first_svg = generate_metric_target_svg()
        second_svg = generate_metric_target_svg()

        assert first_svg == second_svg, 'Repeated target generation changed the SVG'
        root = ElementTree.fromstring(first_svg)
        assert root.attrib['width'] == '210mm', f'{root.attrib=}'
        assert root.attrib['height'] == '297mm', f'{root.attrib=}'
        assert root.attrib['viewBox'] == '0 0 210 297', f'{root.attrib=}'
        assert root.attrib['data-target-version'] == str(METRIC_TARGET_FORMAT_VERSION)
        assert METRIC_TARGET_FORMAT in first_svg
        assert METRIC_TARGET_MARKER_FAMILY in first_svg
        assert '100% / Actual-size' in first_svg
        assert 'Fit to page' in first_svg
        assert 'printer scaling' in first_svg
        assert 'browser scaling' in first_svg
        assert METRIC_TARGET.reference_segment.label in first_svg

        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        marker_images = root.findall('.//svg:image', namespaces)
        assert len(marker_images) == METRIC_TARGET_MARKER_COUNT, f'{marker_images=}'
        for marker, image in zip(METRIC_TARGET.markers, marker_images):
            assert image.attrib['width'] == '40', f'{image.attrib=}'
            assert image.attrib['height'] == '40', f'{image.attrib=}'
            assert image.attrib['data-display-width-mm'] == '40', f'{image.attrib=}'
            assert image.attrib['data-display-height-mm'] == '40', f'{image.attrib=}'
            assert float(image.attrib['x']) == marker.corners[0].x, f'{image.attrib=}'
            assert float(image.attrib['y']) == marker.corners[0].y, f'{image.attrib=}'
            encoded_image = image.attrib['href'].split(',', maxsplit=1)[1]
            assert base64.b64decode(encoded_image).startswith(b'\x89PNG'), (
                f'{image.attrib=}'
            )

    def test_svg_generation_uses_only_a_software_marker_encoder(self) -> None:
        class FakeMarkerImage:
            def __init__(self, marker_id: int, size_pixels: int) -> None:
                self.marker_id = marker_id
                self.shape = (size_pixels, size_pixels)

        class FakeEncodedImage:
            def __init__(self, marker_id: int) -> None:
                self.marker_id = marker_id

            def tobytes(self) -> bytes:
                return b'\\x89PNG\\r\\n' + bytes((self.marker_id,))

        class FakeAruco:
            DICT_APRILTAG_36h11 = object()

            def getPredefinedDictionary(self, dictionary_constant: object) -> object:
                return dictionary_constant

            def generateImageMarker(
                self,
                _dictionary: object,
                marker_id: int,
                size_pixels: int,
                borderBits: int,
            ) -> FakeMarkerImage:
                assert borderBits == 1, f'{borderBits=}'
                return FakeMarkerImage(marker_id, size_pixels)

        class SoftwareOnlyOpenCV:
            aruco = FakeAruco()

            def imencode(
                self,
                _extension: str,
                marker_image: FakeMarkerImage,
            ) -> tuple[bool, FakeEncodedImage]:
                return True, FakeEncodedImage(marker_image.marker_id)

        software_only_cv2 = SoftwareOnlyOpenCV()
        first_svg = generate_metric_target_svg(cv2_module=software_only_cv2)
        second_svg = generate_metric_target_svg(cv2_module=software_only_cv2)

        assert first_svg == second_svg, 'Software-only SVG generation is not deterministic'
        assert not hasattr(software_only_cv2, 'VideoCapture'), 'SVG generation needs no camera'
        root = ElementTree.fromstring(first_svg)
        marker_images = root.findall(
            './/{http://www.w3.org/2000/svg}image',
        )
        assert len(marker_images) == METRIC_TARGET_MARKER_COUNT, f'{marker_images=}'
        for marker_image in marker_images:
            encoded_image = marker_image.attrib['href'].split(',', maxsplit=1)[1]
            assert base64.b64decode(encoded_image).startswith(b'\\x89PNG'), (
                f'{marker_image.attrib=}'
            )

    def test_svg_can_be_written_to_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / 'metric-target.svg'
            write_metric_target_svg(output_path)

            assert output_path.read_text(encoding='utf-8') == generate_metric_target_svg(), (
                f'{output_path=}'
            )

    def test_svg_rejects_a_different_target_definition(self) -> None:
        invalid_target = METRIC_TARGET._replace(
            format_version=METRIC_TARGET_FORMAT_VERSION + 1,
        )

        with self.assertRaises(ValueError):
            generate_metric_target_svg(invalid_target)


if __name__ == '__main__':
    unittest.main()
