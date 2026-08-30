"""Immutable, projector-independent metadata for the printable metric target."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pathlib
from typing import (
    Any,
    NamedTuple,
)
from xml.sax.saxutils import escape

from multivision.geometry import (
    CoordinateBounds,
    Point2D,
)
from multivision.pattern import APRILTAG_36H11
from multivision.types import is_finite_real


A4_PAGE_WIDTH_MM = 210.0
A4_PAGE_HEIGHT_MM = 297.0
METRIC_TARGET_FORMAT = 'multivision-metric-target'
METRIC_TARGET_FORMAT_VERSION = 1
METRIC_TARGET_MARKER_FAMILY = APRILTAG_36H11
METRIC_TARGET_MARKER_COUNT = 20
METRIC_TARGET_MARKER_COLUMNS = 4
METRIC_TARGET_MARKER_ROWS = 5
METRIC_TARGET_MARKER_SIZE_MM = 22.0
METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS = 440
REFERENCE_SEGMENT_LENGTH_MM = 100.0
REFERENCE_SEGMENT_LABEL = 'Expected reference: 100.0 mm'

# These positions are surface coordinates, not display or camera coordinates.
_MARKER_X_START_MM = 8.0
_MARKER_X_END_MM = 180.0
_MARKER_Y_START_MM = 40.0
_MARKER_Y_END_MM = 263.0
_ORIENTATION_CUE = (
    Point2D(20.0, 7.0),
    Point2D(29.0, 19.0),
    Point2D(24.0, 19.0),
    Point2D(24.0, 29.0),
    Point2D(16.0, 29.0),
    Point2D(16.0, 19.0),
    Point2D(11.0, 19.0),
)
_ORIENTATION_CUE_TEXT_POSITION = Point2D(8.0, 35.0)
_VERSION_TEXT = f'{METRIC_TARGET_FORMAT} v{METRIC_TARGET_FORMAT_VERSION}'
_VERSION_TEXT_POSITION = Point2D(145.0, 31.0)


class MetricTargetFrame(NamedTuple):
    """The fixed physical coordinate frame printed on the target."""

    page_width_mm: float
    page_height_mm: float
    origin: str
    x_axis_direction: str
    y_axis_direction: str

    @property
    def page_bounds(self) -> CoordinateBounds:
        return CoordinateBounds(
            0.0,
            0.0,
            self.page_width_mm,
            self.page_height_mm,
        )


class MetricTargetMarker(NamedTuple):
    """One AprilTag and its clockwise surface-mm corners."""

    marker_id: int
    corners: tuple[Point2D, ...]


class MetricOrientationCue(NamedTuple):
    """An asymmetric, human-readable cue for the target's top edge."""

    label: str
    corners: tuple[Point2D, ...]
    text_position: Point2D


class MetricReferenceSegment(NamedTuple):
    """A labelled physical reference segment in surface millimetres."""

    start: Point2D
    end: Point2D
    label: str

    @property
    def length_mm(self) -> float:
        return math.dist(self.start, self.end)


class MetricTarget(NamedTuple):
    """Complete immutable definition of the fixed A4 metric target."""

    format_name: str
    format_version: int
    marker_family: str
    target_frame: MetricTargetFrame
    marker_columns: int
    marker_rows: int
    marker_size_mm: float
    markers: tuple[MetricTargetMarker, ...]
    orientation_cue: MetricOrientationCue
    version_text: str
    version_text_position: Point2D
    reference_segment: MetricReferenceSegment

    @property
    def page_width_mm(self) -> float:
        return self.target_frame.page_width_mm

    @property
    def page_height_mm(self) -> float:
        return self.target_frame.page_height_mm

    @property
    def page_size_mm(self) -> tuple[float, float]:
        return self.page_width_mm, self.page_height_mm

    @property
    def page_bounds(self) -> CoordinateBounds:
        return self.target_frame.page_bounds

    @property
    def marker_count(self) -> int:
        return len(self.markers)

    @property
    def marker_ids(self) -> tuple[int, ...]:
        return tuple(marker.marker_id for marker in self.markers)

    @property
    def marker_corners(self) -> tuple[Point2D, ...]:
        return tuple(
            corner
            for marker in self.markers
            for corner in marker.corners
        )


_EXPECTED_TARGET_FRAME = MetricTargetFrame(
    A4_PAGE_WIDTH_MM,
    A4_PAGE_HEIGHT_MM,
    'top-left',
    'right',
    'down',
)
_EXPECTED_ORIENTATION_CUE = MetricOrientationCue(
    'TOP',
    _ORIENTATION_CUE,
    _ORIENTATION_CUE_TEXT_POSITION,
)
_EXPECTED_REFERENCE_SEGMENT = MetricReferenceSegment(
    Point2D(55.0, 20.0),
    Point2D(155.0, 20.0),
    REFERENCE_SEGMENT_LABEL,
)


def build_metric_target() -> MetricTarget:
    """Build the deterministic A4 target definition."""
    return validate_metric_target(
        MetricTarget(
            METRIC_TARGET_FORMAT,
            METRIC_TARGET_FORMAT_VERSION,
            METRIC_TARGET_MARKER_FAMILY,
            _EXPECTED_TARGET_FRAME,
            METRIC_TARGET_MARKER_COLUMNS,
            METRIC_TARGET_MARKER_ROWS,
            METRIC_TARGET_MARKER_SIZE_MM,
            _EXPECTED_MARKERS,
            _EXPECTED_ORIENTATION_CUE,
            _VERSION_TEXT,
            _VERSION_TEXT_POSITION,
            _EXPECTED_REFERENCE_SEGMENT,
        ),
    )


def validate_metric_target(target: MetricTarget) -> MetricTarget:
    """Validate target identity, ordering and finite page geometry."""
    if not isinstance(target, MetricTarget):
        raise ValueError('target must be MetricTarget')
    if target.format_name != METRIC_TARGET_FORMAT:
        raise ValueError('Unexpected metric target format')
    if target.format_version != METRIC_TARGET_FORMAT_VERSION:
        raise ValueError('Unexpected metric target version')
    if target.marker_family != METRIC_TARGET_MARKER_FAMILY:
        raise ValueError('Unexpected metric target marker family')
    if target.target_frame != _EXPECTED_TARGET_FRAME:
        raise ValueError('Metric target frame is not the deterministic A4 frame')
    if not isinstance(target.markers, tuple):
        raise ValueError('Metric target markers must be immutable')
    if (
        target.marker_columns != METRIC_TARGET_MARKER_COLUMNS
        or target.marker_rows != METRIC_TARGET_MARKER_ROWS
        or target.marker_size_mm != METRIC_TARGET_MARKER_SIZE_MM
        or len(target.markers) != METRIC_TARGET_MARKER_COUNT
    ):
        raise ValueError('Metric target layout metadata is invalid')

    page_bounds = target.page_bounds
    expected_markers = _EXPECTED_MARKERS
    for marker, expected_marker in zip(target.markers, expected_markers):
        if not isinstance(marker, MetricTargetMarker):
            raise ValueError('Metric target markers must use MetricTargetMarker')
        if not isinstance(marker.marker_id, int) or isinstance(marker.marker_id, bool):
            raise ValueError('Metric target marker IDs must be integers')
        if marker.marker_id != expected_marker.marker_id:
            raise ValueError('Metric target marker IDs must be ordered from zero')
        if marker.corners != expected_marker.corners:
            raise ValueError('Metric target marker corners are not deterministic')
        if len(marker.corners) != 4 or len(set(marker.corners)) != 4:
            raise ValueError('Every metric target marker needs four unique corners')
        _validate_ordered_polygon(marker.corners, page_bounds, 'marker')
    orientation_cue = target.orientation_cue
    if not isinstance(orientation_cue, MetricOrientationCue):
        raise ValueError('Metric target orientation cue is malformed')
    if orientation_cue != _EXPECTED_ORIENTATION_CUE:
        raise ValueError('Metric target orientation cue is not deterministic')
    _validate_ordered_polygon(orientation_cue.corners, page_bounds, 'orientation cue')
    _validate_point_in_page(
        orientation_cue.text_position,
        page_bounds,
        'orientation cue text position',
    )
    marker_top = min(point.y for marker in target.markers for point in marker.corners)
    if (
        max(point.y for point in orientation_cue.corners) >= marker_top
        or orientation_cue.text_position.y >= marker_top
    ):
        raise ValueError('Metric target orientation cue overlaps marker ink')

    _validate_point_in_page(
        target.version_text_position,
        page_bounds,
        'version text position',
    )
    if target.version_text_position.y >= marker_top:
        raise ValueError('Metric target version text overlaps marker ink')
    if target.version_text != _VERSION_TEXT:
        raise ValueError('Metric target version text is not deterministic')

    reference_segment = target.reference_segment
    if not isinstance(reference_segment, MetricReferenceSegment):
        raise ValueError('Metric target reference segment is malformed')
    if reference_segment != _EXPECTED_REFERENCE_SEGMENT:
        raise ValueError('Metric target reference segment is not deterministic')
    _validate_point_in_page(reference_segment.start, page_bounds, 'reference segment')
    _validate_point_in_page(reference_segment.end, page_bounds, 'reference segment')
    if reference_segment.length_mm != REFERENCE_SEGMENT_LENGTH_MM:
        raise ValueError('Metric target reference segment must be exactly 100 mm')
    return target


def build_expected_markers() -> tuple[MetricTargetMarker, ...]:
    """Return the fixed marker layout without constructing surrounding metadata."""
    x_starts = _linear_positions(
        _MARKER_X_START_MM,
        _MARKER_X_END_MM,
        METRIC_TARGET_MARKER_COLUMNS,
    )
    y_starts = _linear_positions(
        _MARKER_Y_START_MM,
        _MARKER_Y_END_MM,
        METRIC_TARGET_MARKER_ROWS,
    )
    return tuple(
        MetricTargetMarker(
            marker_id,
            _build_marker_corners(x_starts[x_idx], y_starts[y_idx]),
        )
        for marker_id, (y_idx, x_idx) in enumerate(
            (y_idx, x_idx)
            for y_idx in range(METRIC_TARGET_MARKER_ROWS)
            for x_idx in range(METRIC_TARGET_MARKER_COLUMNS)
        )
    )


def _linear_positions(start: float, end: float, count: int) -> tuple[float, ...]:
    step = (end - start) / (count - 1)
    return tuple(start + idx * step for idx in range(count))


def _build_marker_corners(x_start: float, y_start: float) -> tuple[Point2D, ...]:
    x_end = x_start + METRIC_TARGET_MARKER_SIZE_MM
    y_end = y_start + METRIC_TARGET_MARKER_SIZE_MM
    return (
        Point2D(x_start, y_start),
        Point2D(x_end, y_start),
        Point2D(x_end, y_end),
        Point2D(x_start, y_end),
    )


_EXPECTED_MARKERS = build_expected_markers()


def _validate_ordered_polygon(
    points: tuple[Point2D, ...],
    page_bounds: CoordinateBounds,
    geometry_name: str,
) -> None:
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError(f'{geometry_name} must contain unique points')
    for point in points:
        _validate_point_in_page(point, page_bounds, geometry_name)
    signed_area = sum(
        points[idx].x * points[(idx + 1) % len(points)].y
        - points[(idx + 1) % len(points)].x * points[idx].y
        for idx in range(len(points))
    ) / 2
    if not math.isfinite(signed_area) or signed_area <= 0:
        raise ValueError(f'{geometry_name} corners must be ordered clockwise')


def _validate_point_in_page(
    point: Point2D,
    page_bounds: CoordinateBounds,
    geometry_name: str,
) -> None:
    if not isinstance(point, Point2D):
        raise ValueError(f'{geometry_name} must be finite and inside the page')
    if any(not is_finite_real(coordinate) for coordinate in point):
        raise ValueError(f'{geometry_name} must be finite and inside the page')
    if not (
        page_bounds.left <= point.x <= page_bounds.right
        and page_bounds.top <= point.y <= page_bounds.bottom
    ):
        raise ValueError(f'{geometry_name} must be finite and inside the page')


METRIC_TARGET = build_metric_target()


def generate_metric_target_svg(
    target: MetricTarget = METRIC_TARGET,
    cv2_module: Any | None = None,
) -> str:
    """Generate a deterministic, physically dimensioned SVG target."""
    checked_target = validate_metric_target(target)
    target_definition_json = _serialise_target_definition(checked_target)
    target_definition_hash = hashlib.sha256(
        target_definition_json.encode('utf-8'),
    ).hexdigest()
    marker_images = tuple(
        _encode_marker_png(
            checked_target.marker_family,
            marker.marker_id,
            cv2_module,
        )
        for marker in checked_target.markers
    )
    formatted_page_width_mm = _format_mm(checked_target.page_width_mm)
    formatted_page_height_mm = _format_mm(checked_target.page_height_mm)
    formatted_marker_size_mm = _format_mm(checked_target.marker_size_mm)

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{formatted_page_width_mm}mm" '
            f'height="{formatted_page_height_mm}mm" '
            f'viewBox="0 0 {formatted_page_width_mm} {formatted_page_height_mm}" '
            f'data-target-format="{_escape_svg_attribute(checked_target.format_name)}" '
            f'data-target-version="{checked_target.format_version}" '
            f'data-marker-family="{_escape_svg_attribute(checked_target.marker_family)}" '
            f'data-target-definition-sha256="{target_definition_hash}">'
        ),
        (
            '<metadata id="multivision-metric-target-metadata" '
            f'data-target-format="{_escape_svg_attribute(checked_target.format_name)}" '
            f'data-target-version="{checked_target.format_version}" '
            f'data-marker-family="{_escape_svg_attribute(checked_target.marker_family)}" '
            f'data-target-definition-sha256="{target_definition_hash}">'
            f'{escape(target_definition_json)}'
            '</metadata>'
        ),
        f'<title>{escape(checked_target.version_text)}</title>',
        (
            '<desc>Portrait A4 metric calibration target. Print at 100% / '
            'Actual-size with no scaling.</desc>'
        ),
        '<g id="instructions" fill="#000000" font-family="sans-serif">',
        '<text x="32" y="7" font-size="3.2">PRINT AT 100% / Actual-size.</text>',
        (
            '<text x="32" y="11" font-size="2.7">WARNING: Do not use Fit to page, '
            'printer scaling or browser scaling.</text>'
        ),
        '</g>',
        '<g id="orientation" fill="#000000" stroke="#000000">',
        (
            '<polygon id="orientation-cue" '
            f'points="{_format_points(checked_target.orientation_cue.corners)}" '
            'stroke-width="0.4"/>'
        ),
        (
            f'<text x="{_format_mm(checked_target.orientation_cue.text_position.x)}" '
            f'y="{_format_mm(checked_target.orientation_cue.text_position.y)}" '
            'font-family="sans-serif" font-size="3.2" stroke="none">'
            f'{escape(checked_target.orientation_cue.label)}</text>'
        ),
        '</g>',
        '<g id="reference" fill="#000000" stroke="#000000">',
        (
            f'<line x1="{_format_mm(checked_target.reference_segment.start.x)}" '
            f'y1="{_format_mm(checked_target.reference_segment.start.y)}" '
            f'x2="{_format_mm(checked_target.reference_segment.end.x)}" '
            f'y2="{_format_mm(checked_target.reference_segment.end.y)}" '
            'stroke-width="0.5"/>'
        ),
        (
            f'<text x="{_format_mm(checked_target.reference_segment.start.x)}" '
            f'y="{_format_mm(checked_target.reference_segment.start.y - 3.0)}" '
            'font-family="sans-serif" font-size="3.0" stroke="none">'
            f'{escape(checked_target.reference_segment.label)}</text>'
        ),
        '</g>',
        '<g id="target-version" fill="#000000" font-family="sans-serif">',
        (
            f'<text x="{_format_mm(checked_target.version_text_position.x)}" '
            f'y="{_format_mm(checked_target.version_text_position.y)}" '
            'font-size="2.8">'
            f'{escape(checked_target.version_text)}</text>'
        ),
        '</g>',
        (
            '<g id="markers" '
            f'data-marker-image-size-pixels="{METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS}" '
            f'data-marker-size-mm="{formatted_marker_size_mm}">'
        ),
    ]
    for marker, marker_image in zip(checked_target.markers, marker_images):
        encoded_marker = base64.b64encode(marker_image).decode('ascii')
        svg_lines.append(
            (
                f'<image id="marker-{marker.marker_id}" '
                f'data-marker-id="{marker.marker_id}" '
                f'data-source-width-px="{METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS}" '
                f'data-source-height-px="{METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS}" '
                f'data-display-width-mm="{formatted_marker_size_mm}" '
                f'data-display-height-mm="{formatted_marker_size_mm}" '
                f'x="{_format_mm(marker.corners[0].x)}" '
                f'y="{_format_mm(marker.corners[0].y)}" '
                f'width="{formatted_marker_size_mm}" '
                f'height="{formatted_marker_size_mm}" '
                'preserveAspectRatio="none" image-rendering="pixelated" '
                f'href="data:image/png;base64,{encoded_marker}"/>'
            )
        )
    svg_lines.extend(('</g>', '</svg>', ''))
    return '\n'.join(svg_lines)


def write_metric_target_svg(
    output_path: pathlib.Path,
    target: MetricTarget = METRIC_TARGET,
    cv2_module: Any | None = None,
) -> None:
    """Write the deterministic SVG bytes to a local path."""
    if not isinstance(output_path, pathlib.Path):
        raise ValueError('output_path must be a pathlib.Path')
    svg = generate_metric_target_svg(target, cv2_module=cv2_module)
    output_path.write_bytes(svg.encode('utf-8'))


def _serialise_target_definition(target: MetricTarget) -> str:
    definition = {
        'format_name': target.format_name,
        'format_version': target.format_version,
        'marker_family': target.marker_family,
        'target_frame': {
            'page_width_mm': target.target_frame.page_width_mm,
            'page_height_mm': target.target_frame.page_height_mm,
            'origin': target.target_frame.origin,
            'x_axis_direction': target.target_frame.x_axis_direction,
            'y_axis_direction': target.target_frame.y_axis_direction,
        },
        'marker_columns': target.marker_columns,
        'marker_rows': target.marker_rows,
        'marker_size_mm': target.marker_size_mm,
        'markers': [
            {
                'marker_id': marker.marker_id,
                'corners': [_serialise_point(corner) for corner in marker.corners],
            }
            for marker in target.markers
        ],
        'orientation_cue': {
            'label': target.orientation_cue.label,
            'corners': [
                _serialise_point(corner)
                for corner in target.orientation_cue.corners
            ],
            'text_position': _serialise_point(
                target.orientation_cue.text_position,
            ),
        },
        'version_text': target.version_text,
        'version_text_position': _serialise_point(target.version_text_position),
        'reference_segment': {
            'start': _serialise_point(target.reference_segment.start),
            'end': _serialise_point(target.reference_segment.end),
            'label': target.reference_segment.label,
        },
    }
    return json.dumps(
        definition,
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    )


def _serialise_point(point: Point2D) -> tuple[float, float]:
    return point.x, point.y


def _encode_marker_png(
    marker_family: str,
    marker_id: int,
    cv2_module: Any | None,
) -> bytes:
    if cv2_module is None:
        try:
            import cv2
        except ImportError as ex:
            raise RuntimeError('OpenCV is required to generate metric targets') from ex
        cv2_module = cv2

    aruco_module = getattr(cv2_module, 'aruco', None)
    if aruco_module is None:
        raise RuntimeError('OpenCV was built without aruco support')
    dictionary_id = getattr(aruco_module, marker_family, None)
    if dictionary_id is None:
        raise ValueError(f'Unsupported AprilTag family: {marker_family!r}')

    try:
        dictionary = aruco_module.getPredefinedDictionary(dictionary_id)
        marker_image = aruco_module.generateImageMarker(
            dictionary,
            marker_id,
            METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS,
            borderBits=1,
        )
        image_shape = getattr(marker_image, 'shape', None)
        if image_shape is not None and tuple(image_shape[:2]) != (
            METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS,
            METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS,
        ):
            raise RuntimeError('OpenCV generated a marker with unexpected dimensions')
        encoded_successfully, encoded_image = cv2_module.imencode(
            '.png',
            marker_image,
        )
        if not encoded_successfully:
            raise RuntimeError('OpenCV could not encode a metric marker')
        to_bytes = getattr(encoded_image, 'tobytes', None)
        encoded_bytes = to_bytes() if callable(to_bytes) else bytes(encoded_image)
    except RuntimeError:
        raise
    except Exception as ex:  # noqa: BLE001 (OpenCV is an external boundary).
        raise RuntimeError('Could not generate an OpenCV metric marker') from ex
    if len(encoded_bytes) == 0:
        raise RuntimeError('OpenCV generated an empty metric marker')
    return encoded_bytes


def _format_mm(value: float) -> str:
    formatted_value = format(value, '.17g')
    if '.' in formatted_value:
        formatted_value = formatted_value.rstrip('0').rstrip('.')
    return formatted_value


def _format_points(points: tuple[Point2D, ...]) -> str:
    return ' '.join(
        f'{_format_mm(point.x)},{_format_mm(point.y)}'
        for point in points
    )


def _escape_svg_attribute(value: object) -> str:
    return escape(str(value), {'"': '&quot;'})


__all__ = [
    'A4_PAGE_HEIGHT_MM',
    'A4_PAGE_WIDTH_MM',
    'METRIC_TARGET',
    'METRIC_TARGET_FORMAT',
    'METRIC_TARGET_FORMAT_VERSION',
    'METRIC_TARGET_MARKER_COLUMNS',
    'METRIC_TARGET_MARKER_COUNT',
    'METRIC_TARGET_MARKER_FAMILY',
    'METRIC_TARGET_MARKER_IMAGE_SIZE_PIXELS',
    'METRIC_TARGET_MARKER_ROWS',
    'METRIC_TARGET_MARKER_SIZE_MM',
    'MetricOrientationCue',
    'MetricReferenceSegment',
    'MetricTarget',
    'MetricTargetFrame',
    'MetricTargetMarker',
    'REFERENCE_SEGMENT_LABEL',
    'REFERENCE_SEGMENT_LENGTH_MM',
    'build_expected_markers',
    'build_metric_target',
    'generate_metric_target_svg',
    'validate_metric_target',
    'write_metric_target_svg',
]
