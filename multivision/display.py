"""Main-thread Pygame rendering for camera previews and projector output."""

from __future__ import annotations

import math
import threading
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    NamedTuple,
    Protocol,
)

from multivision.config import ProjectorOutputDescriptor
from multivision.errors import MultiVisionError
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    PointLike,
    PreviewTransform,
    build_preview_transform,
)
from multivision.overlays import (
    MAX_OVERLAY_LABEL_SCALE,
    MIN_OVERLAY_LABEL_SCALE,
    OverlayStyle,
    ProjectorLabel,
    ProjectorMaterialisation,
    ProjectorPolygon,
    ProjectorSegment,
    apply_overlay_intensity_to_colour,
    is_circle_intersecting_protected_regions,
    is_polygon_intersecting_protected_regions,
    materialise_presentation,
    normalise_overlay_intensity,
)
from multivision.pattern import CalibrationPattern
from multivision.session import SessionCamera
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    Frame,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
    is_valid_resolution,
)


BLACK = (0, 0, 0)
WHITE = (235, 235, 235)
PROJECTOR_AREA_LABEL_FONT_SIZE = 48
GREY = (145, 145, 145)
DARK_GREY = (35, 35, 35)
GREEN = (85, 205, 115)
ORANGE = (235, 175, 75)
RED = (220, 75, 75)
METRIC_RULER_COLOUR = WHITE
METRIC_RULER_LINE_WIDTH = 2
METRIC_RULER_MINOR_TICK_WIDTH = 1
METRIC_RULER_MAJOR_TICK_WIDTH = 2


class MarkerImageRenderer(Protocol):
    def __call__(
        self,
        marker_family: str,
        marker_id: int,
        pixel_size: int,
        pygame_module: Any,
    ) -> Any:
        ...


class FrameSurfaceConverter(Protocol):
    def __call__(self, frame: Frame, pygame_module: Any) -> Any:
        ...


class ProjectorOutput(Protocol):
    def present(self, surface: Any) -> None:
        ...

    def shutdown(self) -> None:
        ...


class ProjectorPointOverlayLike(Protocol):
    projector_point: Point2D
    radius: int
    colour: tuple[int, int, int]


ProjectorOutputFactory = Callable[[Any, 'DisplayConfiguration'], ProjectorOutput | None]


class CalibrationMetricsLike(Protocol):
    unique_tag_count: int
    correspondence_corner_count: int
    ransac_inlier_count: int
    inlier_ratio: float
    mean_reprojection_error: float
    max_reprojection_error: float
    spatial_coverage: float


class ProjectorAreaLike(Protocol):
    slot_id: str
    display_name: str
    area_enabled: bool
    available_area: Sequence[Point2D] | None
    area_colour: tuple[int, int, int]


class MetricRulerTickLike(Protocol):
    is_major: bool
    projector_start: Point2D
    projector_end: Point2D


class MetricRulerMarkerLike(Protocol):
    projector_extent: Sequence[Point2D]


class MetricRulerLike(Protocol):
    projector_start: Point2D
    projector_end: Point2D
    ticks: Sequence[MetricRulerTickLike]
    markers: Sequence[MetricRulerMarkerLike]
    label: str
    label_position: Point2D
    label_bounds: CoordinateBounds


class ProjectorOverlayLike(Protocol):
    kind: str
    visible: bool
    materialised_primitives: ProjectorMaterialisation
    insertion_sequence: int


class DisplayServiceLike(Protocol):
    @property
    def overlay(self) -> ProjectorPointOverlayLike | None:
        ...

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor:
        ...

    @property
    def metric_state(self) -> object:
        ...

    def get_metric_status(self) -> object:
        ...

    @property
    def metric_ruler(self) -> MetricRulerLike | None:
        ...

    @property
    def metric_capture_active(self) -> bool:
        ...

    def mark_metric_capture_presented(self) -> None:
        ...

    def get_camera_statuses(self) -> list[CameraStatus]:
        ...

    def get_camera_areas(self) -> list[ProjectorAreaLike]:
        ...

    def list_overlays(self) -> list[ProjectorOverlayLike]:
        ...

    def get_render_snapshot(self) -> object | None:
        ...

    def get_overlay_intensity(self) -> float:
        ...

    def snapshot(self, logical_name: str) -> Frame:
        ...

    def get_calibration_metrics(self, logical_name: str) -> CalibrationMetricsLike | None:
        ...

    def point_from_preview(
        self,
        logical_name: str,
        preview_point: Point2D,
        preview_transform: PreviewTransform,
    ) -> ProjectorPointOverlayLike:
        ...

    @property
    def calibration_pattern_visible(self) -> bool:
        ...

    def mark_calibration_pattern_presented(self) -> None:
        ...


class CameraPreviewLayout(NamedTuple):
    """The window rectangle and local transform assigned to one camera."""

    logical_name: str
    panel_bounds: CoordinateBounds
    preview_bounds: CoordinateBounds
    preview_transform: PreviewTransform | None
    slot_id: str | None = None


class _DisplayCameraSnapshot(NamedTuple):
    slot_id: str
    logical_name: str
    status: CameraStatus
    camera_state: SessionCameraState
    lifecycle_generation: int
    frame: Frame | None
    calibration_metrics: CalibrationMetricsLike | None


class _DisplayCamera(NamedTuple):
    slot_id: str
    logical_name: str
    status: CameraStatus
    session_camera: SessionCamera | None
    render_snapshot: _DisplayCameraSnapshot | None = None


@dataclass(frozen=True)
class DisplayConfiguration:
    """Window settings plus the service-selected projector output descriptor."""

    window_resolution: Resolution = Resolution(1280, 720)
    projector_resolution: Resolution = Resolution(1920, 1080)
    frames_per_second: int = 30
    caption: str = 'MultiVision'
    projector_output_descriptor: ProjectorOutputDescriptor | None = None
    preview_mode: str = 'active'
    preview_low_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        _validate_resolution(self.window_resolution, 'window_resolution')
        if self.projector_output_descriptor is not None:
            if not isinstance(
                self.projector_output_descriptor,
                ProjectorOutputDescriptor,
            ):
                raise ValueError(
                    'projector_output_descriptor must be ProjectorOutputDescriptor',
                )
            # The service descriptor is authoritative – callers need not duplicate
            # its resolution in the display-only configuration.
            object.__setattr__(
                self,
                'projector_resolution',
                self.projector_output_descriptor.projector_resolution,
            )
        else:
            _validate_resolution(self.projector_resolution, 'projector_resolution')
            object.__setattr__(
                self,
                'projector_output_descriptor',
                ProjectorOutputDescriptor(self.projector_resolution),
            )
        if (
            not isinstance(self.frames_per_second, int)
            or isinstance(self.frames_per_second, bool)
            or self.frames_per_second <= 0
        ):
            raise ValueError('frames_per_second must be a positive integer')
        if not isinstance(self.caption, str) or len(self.caption) == 0:
            raise ValueError('caption must be a non-empty string')
        if not isinstance(self.preview_mode, str) or self.preview_mode not in {
            'active',
            'low_rate',
            'off',
        }:
            raise ValueError('preview_mode must be active, low_rate or off')
        if (
            isinstance(self.preview_low_rate_hz, bool)
            or not isinstance(self.preview_low_rate_hz, (int, float))
            or not math.isfinite(self.preview_low_rate_hz)
            or not 1.0 <= self.preview_low_rate_hz <= 15.0
        ):
            raise ValueError('preview_low_rate_hz must be between 1.0 and 15.0')
        object.__setattr__(self, 'preview_low_rate_hz', float(self.preview_low_rate_hz))


class Sdl2ProjectorOutput:
    """Present the projector surface in a borderless window on its display."""

    def __init__(
        self,
        projector_resolution: Resolution,
        display_index: int = 1,
        window_resolution: Resolution | None = None,
        top_inset_pixels: int = 0,
        fullscreen: bool = False,
    ) -> None:
        from pygame._sdl2 import video

        if (
            not isinstance(display_index, int)
            or isinstance(display_index, bool)
            or display_index < 0
        ):
            raise ValueError('display_index must be a non-negative integer')
        if not is_valid_resolution(projector_resolution):
            raise ValueError('projector_resolution must be a positive Resolution')
        if window_resolution is None:
            window_resolution = projector_resolution
        if not is_valid_resolution(window_resolution):
            raise ValueError('window_resolution must be a positive Resolution')
        if (
            not isinstance(top_inset_pixels, int)
            or isinstance(top_inset_pixels, bool)
            or not 0 <= top_inset_pixels < window_resolution.height
        ):
            raise ValueError(
                'top_inset_pixels must be a non-negative value smaller than window height',
            )
        if not isinstance(fullscreen, bool):
            raise ValueError('fullscreen must be a bool')
        centred_position = video.WINDOWPOS_CENTERED + display_index
        window = video.Window(
            'MultiVision Projector',
            size=tuple(window_resolution),
            position=(centred_position, centred_position),
            borderless=True,
        )
        try:
            renderer = video.Renderer(window)
            window.show()
            if fullscreen:
                window.restore()
                window.set_fullscreen(desktop=True)
                window.show()
                window.focus()
        except BaseException:  # noqa: BLE001 (Release a window if projector setup fails).
            try:
                window.destroy()
            except Exception:  # noqa: BLE001 (Preserve the projector setup failure).
                pass
            raise
        self._window = window
        self._window_resolution = window_resolution
        self._top_inset_pixels = top_inset_pixels
        self._renderer = renderer
        self._texture: Any | None = None

    def present(self, surface: Any) -> None:
        from pygame import Rect
        from pygame._sdl2 import video

        if self._texture is None:
            self._texture = video.Texture.from_surface(self._renderer, surface)
        else:
            self._texture.update(surface)
        self._renderer.clear()
        self._renderer.blit(
            self._texture,
            Rect(
                0,
                self._top_inset_pixels,
                self._window_resolution.width,
                self._window_resolution.height - self._top_inset_pixels,
            ),
        )
        self._renderer.present()

    def shutdown(self) -> None:
        window = self._window
        self._texture = None
        self._renderer = None
        self._window = None
        if window is not None:
            window.destroy()


class ProjectorRenderer:
    """Render projector-native content without changing calibration data."""

    def __init__(
        self,
        pygame_module: Any,
        marker_image_renderer: MarkerImageRenderer | None = None,
    ) -> None:
        self._pygame = pygame_module
        self._marker_image_renderer = (
            marker_image_renderer
            if marker_image_renderer is not None
            else _render_apriltag_image
        )

    def clear(self, surface: Any) -> None:
        self._require_main_thread()
        surface.fill(BLACK)

    def render_calibration_pattern(
        self,
        surface: Any,
        pattern: CalibrationPattern,
    ) -> None:
        """Draw the known pattern in projector-native pixels only."""
        self._require_main_thread()
        if not isinstance(pattern, CalibrationPattern):
            raise TypeError('pattern must be CalibrationPattern')
        surface.fill(WHITE)
        for marker in pattern.markers:
            marker_width = max(1, round(marker.bounds.right - marker.bounds.left))
            marker_height = max(1, round(marker.bounds.bottom - marker.bounds.top))
            if marker_width != marker_height:
                raise ValueError('Calibration markers must be square')
            marker_surface = self._marker_image_renderer(
                pattern.marker_family,
                marker.marker_id,
                marker_width,
                self._pygame,
            )
            marker_surface_size = _get_surface_size(marker_surface, marker_width)
            surface.blit(
                marker_surface,
                (
                    round(marker.bounds.left)
                    - (marker_surface_size[0] - marker_width) // 2,
                    round(marker.bounds.top)
                    - (marker_surface_size[1] - marker_width) // 2,
                ),
            )

    def render_areas(
        self,
        surface: Any,
        areas: Sequence[ProjectorAreaLike],
        font: Any | None = None,
        intensity: float = 1.0,
        protected_regions: Sequence[Sequence[Point2D]] = (),
    ) -> None:
        """Draw the current enabled diagnostic areas in slot order."""
        self._require_main_thread()
        checked_intensity = normalise_overlay_intensity(intensity)
        if not isinstance(areas, Sequence) or isinstance(areas, (str, bytes)):
            raise TypeError('areas must be a sequence of projector areas')
        if font is None:
            font = self._pygame.font.Font(None, PROJECTOR_AREA_LABEL_FONT_SIZE)
        for area in sorted(areas, key=lambda value: _slot_sort_key(value.slot_id)):
            if not area.area_enabled or area.available_area is None:
                continue
            area_points = tuple(
                (round(point.x), round(point.y))
                for point in area.available_area
            )
            if len(area_points) < 3:
                continue
            area_polygon = tuple(
                Point2D(float(x_pos), float(y_pos))
                for x_pos, y_pos in area_points
            )
            if is_polygon_intersecting_protected_regions(
                area_polygon,
                protected_regions,
            ):
                continue
            area_colour = apply_overlay_intensity_to_colour(
                area.area_colour,
                checked_intensity,
            )
            self._pygame.draw.polygon(
                surface,
                area_colour,
                area_points,
                2,
            )
            label_surface = font.render(area.display_name, True, area_colour)
            surface_size = _get_surface_size(surface, 0)
            label_size = _get_surface_size(label_surface, 0)
            first_point = area_points[0]
            surface.blit(
                label_surface,
                (
                    min(
                        max(first_point[0], 0),
                        max(0, surface_size[0] - label_size[0]),
                    ),
                    min(
                        max(first_point[1], 0),
                        max(0, surface_size[1] - label_size[1]),
                    ),
                ),
            )

    def render_metric_ruler(
        self,
        surface: Any,
        ruler: MetricRulerLike | None,
        font: Any | None = None,
        intensity: float = 1.0,
        protected_regions: Sequence[Sequence[Point2D]] = (),
    ) -> None:
        """Draw service-produced projector-native ruler primitives unchanged."""
        self._require_main_thread()
        checked_intensity = normalise_overlay_intensity(intensity)
        if ruler is None:
            return
        if not isinstance(ruler.label, str):
            raise TypeError('metric ruler labels must be strings')
        if font is None:
            font = self._pygame.font.Font(None, 16)
        surface_size = _get_surface_size(surface, 0)
        marker_points = tuple(
            _round_projector_points(marker.projector_extent)
            for marker in ruler.markers
        )
        projector_start = _round_projector_point(ruler.projector_start)
        projector_end = _round_projector_point(ruler.projector_end)
        tick_points = tuple(
            (
                _round_projector_point(tick.projector_start),
                _round_projector_point(tick.projector_end),
                METRIC_RULER_MAJOR_TICK_WIDTH
                if tick.is_major
                else METRIC_RULER_MINOR_TICK_WIDTH,
            )
            for tick in ruler.ticks
        )
        for points in marker_points:
            _validate_raster_projector_points(
                points,
                surface_size,
                METRIC_RULER_LINE_WIDTH,
            )
        _validate_raster_projector_points(
            (projector_start, projector_end),
            surface_size,
            METRIC_RULER_LINE_WIDTH,
        )
        for tick_start, tick_end, line_width in tick_points:
            _validate_raster_projector_points(
                (tick_start, tick_end),
                surface_size,
                line_width,
            )

        if _metric_ruler_intersects_protected_regions(
            marker_points,
            (projector_start, projector_end),
            tuple((tick_start, tick_end) for tick_start, tick_end, _line_width in tick_points),
            ruler.label_position,
            protected_regions,
        ):
            return

        ruler_colour = apply_overlay_intensity_to_colour(
            METRIC_RULER_COLOUR,
            checked_intensity,
        )
        for points in marker_points:
            self._pygame.draw.polygon(
                surface,
                ruler_colour,
                points,
                METRIC_RULER_LINE_WIDTH,
            )
        self._pygame.draw.line(
            surface,
            ruler_colour,
            projector_start,
            projector_end,
            METRIC_RULER_LINE_WIDTH,
        )
        for tick_start, tick_end, line_width in tick_points:
            self._pygame.draw.line(
                surface,
                ruler_colour,
                tick_start,
                tick_end,
                line_width,
            )

        label_surface = font.render(ruler.label, True, ruler_colour)
        label_size = _get_surface_size(label_surface, 0)
        label_left = round(ruler.label_position.x - label_size[0] / 2)
        label_top = round(ruler.label_position.y - label_size[1] / 2)
        surface.blit(
            label_surface,
            (
                min(max(label_left, 0), max(0, surface_size[0] - label_size[0])),
                min(max(label_top, 0), max(0, surface_size[1] - label_size[1])),
            ),
        )

    def render_overlay(
        self,
        surface: Any,
        overlay: ProjectorPointOverlayLike | None,
        intensity: float = 1.0,
        protected_regions: Sequence[Sequence[Point2D]] = (),
    ) -> None:
        self._require_main_thread()
        checked_intensity = normalise_overlay_intensity(intensity)
        if overlay is None:
            return
        projector_point = getattr(overlay, 'projector_point', None)
        radius = getattr(overlay, 'radius', None)
        colour = getattr(overlay, 'colour', None)
        if (
            not _is_finite_display_point(projector_point)
            or not isinstance(radius, int)
            or isinstance(radius, bool)
            or radius <= 0
            or not _is_display_colour(colour)
        ):
            raise TypeError('overlay must contain a finite projector point, radius and colour')
        if is_circle_intersecting_protected_regions(
            projector_point,
            radius,
            protected_regions,
        ):
            return
        self._pygame.draw.circle(
            surface,
            apply_overlay_intensity_to_colour(colour, checked_intensity),
            (round(projector_point.x), round(projector_point.y)),
            radius,
        )

    def render_generic_overlays(
        self,
        surface: Any,
        overlays: Sequence[ProjectorOverlayLike],
        font: Any | None = None,
        layer: str | None = None,
        intensity: float = 1.0,
        protected_regions: Sequence[Sequence[Point2D]] = (),
        presentation_safe: bool = False,
    ) -> None:
        """Draw immutable, already-materialised projector overlay primitives."""
        self._require_main_thread()
        checked_intensity = normalise_overlay_intensity(intensity)
        if not isinstance(overlays, Sequence) or isinstance(overlays, (str, bytes)):
            raise TypeError('overlays must be a sequence of projector overlays')
        if layer not in {None, 'grid', 'shape', 'line', 'label'}:
            raise ValueError(f'Unknown projector overlay layer: {layer!r}')
        if len(overlays) == 0:
            return
        selected_layer = {
            'grid': 0,
            'shape': 1,
            'line': 2,
        }.get(layer)
        surface_size = _get_surface_size(surface, 0)
        if surface_size[0] <= 0 or surface_size[1] <= 0:
            raise ValueError('projector surface must have a positive size')
        ordered_overlays = sorted(
            overlays,
            key=lambda overlay: (
                _projector_overlay_layer_key(overlay.kind),
                overlay.insertion_sequence,
            ),
        )
        draw_primitives: list[ProjectorPolygon | ProjectorSegment] = []
        draw_labels: list[ProjectorLabel] = []
        for overlay in ordered_overlays:
            if not isinstance(overlay.visible, bool):
                raise TypeError('projector overlay visibility must be a bool')
            if not overlay.visible:
                continue
            overlay_layer = _projector_overlay_layer_key(overlay.kind)
            if selected_layer is not None and overlay_layer != selected_layer:
                continue
            materialisation = overlay.materialised_primitives
            if not isinstance(materialisation, ProjectorMaterialisation):
                raise TypeError(
                    'projector overlays must contain ProjectorMaterialisation values',
                )
            if not presentation_safe:
                materialisation = materialise_presentation(
                    materialisation,
                    checked_intensity,
                    protected_regions,
                )
            if not all(
                isinstance(primitives, tuple)
                for primitives in (
                    materialisation.segments,
                    materialisation.polygons,
                    materialisation.labels,
                )
            ):
                raise TypeError('projector materialisations must be immutable tuples')
            for polygon in materialisation.polygons:
                if not isinstance(polygon, ProjectorPolygon):
                    raise TypeError('projector polygons must be ProjectorPolygon values')
                if not isinstance(polygon.points, tuple):
                    raise TypeError('projector polygon points must be immutable tuples')
                if not all(
                    _is_finite_display_point(point)
                    for point in polygon.points
                ):
                    raise ValueError('projector polygon points must be finite')
                _validate_overlay_style(polygon.style)
            for segment in materialisation.segments:
                if not isinstance(segment, ProjectorSegment):
                    raise TypeError('projector segments must be ProjectorSegment values')
                if (
                    not _is_finite_display_point(segment.start)
                    or not _is_finite_display_point(segment.end)
                ):
                    raise ValueError('projector segment points must be finite')
                _validate_overlay_style(segment.style)
            for label in materialisation.labels:
                if not isinstance(label, ProjectorLabel):
                    raise TypeError('projector labels must be ProjectorLabel values')
                if not isinstance(label.text, str):
                    raise TypeError('projector labels must contain string text')
                if not _is_finite_display_point(label.position):
                    raise ValueError('projector label positions must be finite')
                if (
                    not _is_finite_display_number(label.angle_deg)
                    or not _is_finite_display_number(label.scale)
                    or not (
                        MIN_OVERLAY_LABEL_SCALE
                        <= label.scale
                        <= MAX_OVERLAY_LABEL_SCALE
                    )
                ):
                    raise ValueError('projector label rotation and scale are invalid')
                _validate_overlay_style(label.style)
            if layer == 'label':
                draw_labels.extend(materialisation.labels)
                continue
            draw_primitives.extend(materialisation.polygons)
            draw_primitives.extend(materialisation.segments)
            if layer is None:
                draw_labels.extend(materialisation.labels)

        for primitive in draw_primitives:
            if isinstance(primitive, ProjectorPolygon):
                points = _round_generic_projector_points(primitive.points)
                self._pygame.draw.polygon(surface, primitive.style.colour, points)
                continue
            start = _round_generic_projector_point(primitive.start)
            end = _round_generic_projector_point(primitive.end)
            self._pygame.draw.line(
                surface,
                primitive.style.colour,
                start,
                end,
                primitive.style.line_width_px,
            )
        if len(draw_labels) == 0:
            return
        if font is None:
            font = self._pygame.font.Font(None, 16)
        for label in draw_labels:
            position = _round_generic_projector_point(label.position)
            label_surfaces = [
                font.render(line.rstrip('\r') or ' ', True, label.style.colour)
                for line in label.text.split('\n')
            ]
            if label.angle_deg != 0.0 or label.scale != 1.0:
                rotozoom = getattr(self._pygame.transform, 'rotozoom', None)
                if not callable(rotozoom):
                    raise RuntimeError('Pygame transform.rotozoom is unavailable')
                label_surfaces = [
                    rotozoom(label_surface, label.angle_deg, label.scale)
                    for label_surface in label_surfaces
                ]
            line_sizes = [
                _get_surface_size(label_surface, 0)
                for label_surface in label_surfaces
            ]
            line_height = max(size[1] for size in line_sizes)
            block_height = line_height * len(label_surfaces)
            block_top = min(
                max(position[1] - block_height // 2, 0),
                max(0, surface_size[1] - block_height),
            )
            for line_index, (label_surface, label_size) in enumerate(
                zip(label_surfaces, line_sizes),
            ):
                label_left = min(
                    max(position[0] - label_size[0] // 2, 0),
                    max(0, surface_size[0] - label_size[0]),
                )
                label_top = block_top + line_index * line_height
                surface.blit(label_surface, (label_left, label_top))

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Pygame display operations must run on the main thread')


class PygameDisplayRuntime:
    """Own the main-thread Pygame surfaces while reading camera frames by reference."""

    def __init__(
        self,
        service: DisplayServiceLike,
        configuration: DisplayConfiguration | None = None,
        calibration_pattern: CalibrationPattern | None = None,
        pygame_module: Any | None = None,
        frame_surface_converter: FrameSurfaceConverter | None = None,
        marker_image_renderer: MarkerImageRenderer | None = None,
        projector_output: ProjectorOutput | None = None,
        projector_output_factory: ProjectorOutputFactory | None = None,
        preview_clock: Callable[[], float] | None = None,
    ) -> None:
        if configuration is None:
            configuration = DisplayConfiguration()
        if not isinstance(configuration, DisplayConfiguration):
            raise TypeError('configuration must be DisplayConfiguration')
        self.service = service
        service_descriptor = _get_service_projector_descriptor(service)
        if service_descriptor is not None:
            configuration = replace(
                configuration,
                projector_resolution=service_descriptor.projector_resolution,
                projector_output_descriptor=service_descriptor,
            )
        if calibration_pattern is not None and not isinstance(
            calibration_pattern,
            CalibrationPattern,
        ):
            raise TypeError('calibration_pattern must be CalibrationPattern')
        if (
            calibration_pattern is not None
            and calibration_pattern.projector_resolution
            != configuration.projector_resolution
        ):
            raise ValueError(
                'calibration_pattern and configuration must use the same projector resolution',
            )

        self.configuration = configuration
        self._projector_output_descriptor = (
            service_descriptor
            if service_descriptor is not None
            else configuration.projector_output_descriptor
        )
        self.calibration_pattern = calibration_pattern
        self._last_point_error: str | None = None
        self._last_metric_error: str | None = None
        self._last_projector_error: str | None = None
        self._pygame = pygame_module
        self._frame_surface_converter = (
            frame_surface_converter
            if frame_surface_converter is not None
            else _frame_to_surface
        )
        self._marker_image_renderer = marker_image_renderer
        self._projector_renderer: ProjectorRenderer | None = None
        self._window_surface: Any | None = None
        self._projector_surface: Any | None = None
        self._projector_output = projector_output
        self._projector_output_is_injected = projector_output is not None
        self._projector_output_factory = (
            projector_output_factory
            if projector_output_factory is not None
            else _make_projector_output
        )
        self._font: Any | None = None
        self._projector_area_font: Any | None = None
        self._area_colours: dict[str, tuple[int, int, int]] = {}
        self._is_running = False
        self._is_initialised = False
        self._preview_layouts: dict[str, CameraPreviewLayout] = {}
        self._uncalibrated_click_cameras: set[str] = set()
        self._camera_lifecycle_generations: dict[str, int] = {}
        if preview_clock is not None and not callable(preview_clock):
            raise ValueError('preview_clock must be callable')
        self._preview_clock = preview_clock or time.monotonic
        self._preview_last_update_seconds: dict[str, float] = {}

    @property
    def window_surface(self) -> Any | None:
        return self._window_surface

    @property
    def projector_surface(self) -> Any | None:
        return self._projector_surface

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor:
        return self._projector_output_descriptor

    @property
    def preview_layouts(self) -> dict[str, CameraPreviewLayout]:
        return dict(self._preview_layouts)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_point_error(self) -> str | None:
        return self._last_point_error

    def initialise(self) -> None:
        """Create Pygame resources on the process main thread."""
        self._require_main_thread()
        if self._is_initialised:
            return
        pygame_module = self._get_pygame()
        projector_output = self._projector_output
        try:
            pygame_module.init()
            window_surface = pygame_module.display.set_mode(
                tuple(self.configuration.window_resolution),
            )
            pygame_module.display.set_caption(self.configuration.caption)
            projector_surface = pygame_module.Surface(
                tuple(self.configuration.projector_resolution),
            )
            projector_renderer = ProjectorRenderer(
                pygame_module,
                self._marker_image_renderer,
            )
            if projector_output is None:
                projector_output = self._projector_output_factory(
                    pygame_module,
                    self.configuration,
                )
            font = pygame_module.font.Font(None, 16)
            projector_area_font = pygame_module.font.Font(
                None,
                PROJECTOR_AREA_LABEL_FONT_SIZE,
            )
        except BaseException:  # noqa: BLE001 (Clean up Pygame before propagating interruption).
            try:
                if 'projector_output' in locals() and projector_output is not None:
                    projector_output.shutdown()
            finally:
                self._projector_output = None
                pygame_module.quit()
            raise

        self._window_surface = window_surface
        self._projector_surface = projector_surface
        self._projector_renderer = projector_renderer
        self._projector_output = projector_output
        self._font = font
        self._projector_area_font = projector_area_font
        self._is_initialised = True

    def run(self, max_frames: int | None = None) -> None:
        """Run the Pygame event loop until quit, optionally for a test-sized run."""
        self._require_main_thread()
        if max_frames is not None and (
            not isinstance(max_frames, int)
            or isinstance(max_frames, bool)
            or max_frames <= 0
        ):
            raise ValueError('max_frames must be a positive integer or None')
        self.initialise()
        pygame_module = self._get_pygame()
        clock = pygame_module.time.Clock()
        self._is_running = True
        rendered_frames = 0
        try:
            while self._is_running:
                self.process_events()
                if not self._is_running:
                    break
                self.render_once()
                self._present_projector_surface()
                pygame_module.display.flip()
                rendered_frames += 1
                if max_frames is not None and rendered_frames >= max_frames:
                    break
                clock.tick(self.configuration.frames_per_second)
        finally:
            self._is_running = False

    def process_events(self) -> None:
        """Handle window events without allowing worker threads into Pygame."""
        self._require_main_thread()
        self.initialise()
        pygame_module = self._get_pygame()
        # Layouts are rebuilt with the same complete snapshot as the frame.
        quit_event = getattr(pygame_module, 'QUIT', None)
        key_down_event = getattr(pygame_module, 'KEYDOWN', None)
        escape_key = getattr(pygame_module, 'K_ESCAPE', None)
        mouse_button_event = getattr(pygame_module, 'MOUSEBUTTONDOWN', None)
        for event in pygame_module.event.get():
            event_type = getattr(event, 'type', None)
            if quit_event is not None and event_type == quit_event:
                self.stop()
                continue
            if (
                key_down_event is not None
                and event_type == key_down_event
                and escape_key is not None
                and getattr(event, 'key', None) == escape_key
            ):
                self.stop()
                continue
            if (
                mouse_button_event is not None
                and event_type == mouse_button_event
                and getattr(event, 'button', 1) == 1
            ):
                self._handle_preview_click(getattr(event, 'pos', None))

    def render_once(self) -> None:
        """Render one frame from current statuses and latest camera snapshots."""
        started_seconds = time.perf_counter()
        started_cpu_seconds = time.thread_time()
        try:
            self._render_once()
        finally:
            self._record_service_timing(
                'presentation_render',
                time.perf_counter() - started_seconds,
                max(0.0, time.thread_time() - started_cpu_seconds),
                2.0 / self.configuration.frames_per_second,
            )

    def _render_once(self) -> None:
        """Render one frame from current statuses and latest camera snapshots."""
        self._require_main_thread()
        self.initialise()
        assert self._window_surface is not None
        assert self._projector_surface is not None
        assert self._projector_renderer is not None
        self._last_metric_error = None
        self._last_projector_error = None
        self._window_surface.fill(DARK_GREY)
        has_snapshot_provider = callable(
            getattr(self.service, 'get_render_snapshot', None),
        )
        render_snapshot = self._get_render_snapshot()
        snapshot_unavailable = False
        if render_snapshot is not None:
            try:
                self._synchronise_projector_descriptor(
                    getattr(render_snapshot, 'projector_output_descriptor', None),
                    getattr(render_snapshot, 'calibration_pattern', None),
                    True,
                )
                display_cameras = self._get_snapshot_display_cameras(render_snapshot)
                pattern_visible = getattr(
                    render_snapshot,
                    'calibration_pattern_visible',
                    False,
                )
                metric_capture_active = getattr(
                    render_snapshot,
                    'metric_capture_active',
                    False,
                )
                metric_state = getattr(
                    render_snapshot,
                    'metric_state',
                    'uncalibrated',
                )
                metric_ruler = getattr(render_snapshot, 'metric_ruler', None)
                projector_areas = list(getattr(render_snapshot, 'projector_areas', ()))
                generic_overlays = list(getattr(render_snapshot, 'overlays', ()))
                point_overlay = getattr(render_snapshot, 'point_overlay', None)
                snapshot_intensity = normalise_overlay_intensity(
                    getattr(render_snapshot, 'global_overlay_intensity', 1.0),
                )
                protected_regions = tuple(
                    getattr(render_snapshot, 'protected_projector_regions', ()),
                )
                generic_intensity = 1.0
            except Exception as ex:  # noqa: BLE001 (Malformed snapshots are retryable input).
                self._last_projector_error = f'Render snapshot unavailable: {ex}'
                snapshot_unavailable = True
                display_cameras = []
                pattern_visible = False
                metric_capture_active = False
                metric_state = 'uncalibrated'
                metric_ruler = None
                projector_areas = []
                generic_overlays = []
                point_overlay = None
                protected_regions = ()
                snapshot_intensity = 1.0
                generic_intensity = 1.0
        elif has_snapshot_provider:
            # A failed complete snapshot must fail closed.  In particular, do not
            # reconstruct one from individually changing service properties.
            self._synchronise_projector_descriptor()
            display_cameras = []
            pattern_visible = False
            metric_capture_active = False
            metric_state = 'uncalibrated'
            metric_ruler = None
            projector_areas = []
            generic_overlays = []
            point_overlay = None
            protected_regions = ()
            snapshot_intensity = 1.0
            generic_intensity = 1.0
        else:
            self._synchronise_projector_descriptor()
            display_cameras = self._get_display_cameras()
            pattern_visible = self.service.calibration_pattern_visible
            if not isinstance(pattern_visible, bool):
                raise TypeError('service returned an invalid calibration pattern state')
            metric_capture_active = self._get_metric_capture_active()
            metric_state, metric_ruler = self._get_metric_snapshot()
            snapshot_intensity = self._get_overlay_intensity()
            projector_areas = (
                [] if pattern_visible or metric_capture_active else self._get_projector_areas()
            )
            generic_overlays = (
                []
                if pattern_visible or metric_capture_active
                else self._get_generic_overlays()
            )
            point_overlay = self.service.overlay
            protected_regions = ()
            generic_intensity = snapshot_intensity
        self._preview_layouts = self._build_preview_layouts(display_cameras)
        if not isinstance(pattern_visible, bool) or not isinstance(metric_capture_active, bool):
            raise TypeError('render snapshot contains invalid presentation flags')
        if not _is_metric_state(metric_state):
            raise TypeError('render snapshot contains invalid metric state')
        render_snapshot_unavailable = snapshot_unavailable or (
            render_snapshot is None
            and callable(getattr(self.service, 'get_render_snapshot', None))
        )
        if not pattern_visible and not metric_capture_active:
            self._area_colours = {
                area.slot_id: area.area_colour
                for area in projector_areas
            }
        area_colours = self._area_colours
        for camera in display_cameras:
            camera_snapshot = getattr(camera, 'render_snapshot', None)
            self._render_camera_card(
                camera.status,
                self._preview_layouts[camera.slot_id],
                camera.session_camera,
                area_colours.get(camera.slot_id),
                metric_state,
                None if camera_snapshot is None else camera_snapshot.frame,
                camera_snapshot is not None,
                None if camera_snapshot is None else camera_snapshot.calibration_metrics,
                None if camera_snapshot is None else camera_snapshot.camera_state,
            )

        try:
            if metric_capture_active:
                self._projector_renderer.clear(self._projector_surface)
                mark_metric_capture_presented = getattr(
                    self.service,
                    'mark_metric_capture_presented',
                    None,
                )
                if callable(mark_metric_capture_presented):
                    mark_metric_capture_presented()
            elif self.calibration_pattern is None or not pattern_visible:
                self._projector_renderer.clear(self._projector_surface)
            else:
                self._projector_renderer.render_calibration_pattern(
                    self._projector_surface,
                    self.calibration_pattern,
                )
                mark_pattern_presented = getattr(
                    self.service,
                    'mark_calibration_pattern_presented',
                    None,
                )
                if callable(mark_pattern_presented):
                    mark_pattern_presented()
            if (
                not pattern_visible
                and not metric_capture_active
                and not render_snapshot_unavailable
            ):
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'grid',
                    generic_intensity,
                    protected_regions,
                    True,
                )
                self._projector_renderer.render_areas(
                    self._projector_surface,
                    projector_areas,
                    self._projector_area_font,
                    snapshot_intensity,
                    protected_regions,
                )
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'shape',
                    generic_intensity,
                    protected_regions,
                    True,
                )
                if _is_metric_calibrated(metric_state):
                    try:
                        self._projector_renderer.render_metric_ruler(
                            self._projector_surface,
                            metric_ruler,
                            self._font,
                            snapshot_intensity,
                            protected_regions,
                        )
                    except Exception as ex:  # noqa: BLE001 (Bad metric snapshot).
                        self._last_metric_error = f'Metric ruler unavailable: {ex}'
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'line',
                    generic_intensity,
                    protected_regions,
                    True,
                )
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'label',
                    generic_intensity,
                    protected_regions,
                    True,
                )
                self._projector_renderer.render_overlay(
                    self._projector_surface,
                    point_overlay,
                    snapshot_intensity,
                    protected_regions,
                )
        except Exception as ex:  # noqa: BLE001 (Projector failures must not stop previews).
            self._last_projector_error = f'Projector unavailable: {ex}'
            try:
                self._projector_renderer.clear(self._projector_surface)
            except Exception:  # noqa: BLE001 (The original projector failure is authoritative).
                pass
        if self._last_projector_error is not None:
            self._draw_text(self._last_projector_error, 8, 8, RED)
        if self._last_metric_error is not None:
            self._draw_text(
                self._last_metric_error,
                8,
                self.configuration.window_resolution.height - 38,
                RED,
            )
        if self._last_point_error is not None:
            self._draw_text(
                self._last_point_error,
                8,
                self.configuration.window_resolution.height - 20,
                RED,
            )

    def stop(self) -> None:
        self._is_running = False

    def shutdown(self) -> None:
        """Release only Pygame resources; camera ownership remains with CameraRuntime."""
        self._require_main_thread()
        if not self._is_initialised:
            return
        pygame_module = self._get_pygame()
        self._is_running = False
        projector_output = self._projector_output
        output_error: Exception | None = None
        try:
            if projector_output is not None:
                projector_output.shutdown()
        except Exception as ex:  # noqa: BLE001 (Pygame cleanup must still complete).
            output_error = ex
        finally:
            self._projector_output = None
            try:
                pygame_module.quit()
            finally:
                self._window_surface = None
                self._projector_surface = None
                self._projector_renderer = None
                self._font = None
                self._projector_area_font = None
                self._area_colours = {}
                self._preview_last_update_seconds.clear()
                self._last_projector_error = None
                self._last_metric_error = None
                self._is_initialised = False
        if output_error is not None:
            raise output_error

    def _present_projector_surface(self) -> None:
        if self._projector_output is None or self._projector_surface is None:
            return
        started_seconds = time.perf_counter()
        started_cpu_seconds = time.thread_time()
        try:
            self._projector_output.present(self._projector_surface)
        except Exception as ex:  # noqa: BLE001 (A projector seam must not stop the UI loop).
            self._last_projector_error = f'Projector unavailable: {ex}'
            if self._window_surface is not None and self._font is not None:
                self._draw_text(self._last_projector_error, 8, 8, RED)
        finally:
            self._record_service_timing(
                'projector_presentation',
                time.perf_counter() - started_seconds,
                max(0.0, time.thread_time() - started_cpu_seconds),
                2.0 / self.configuration.frames_per_second,
            )

    def _record_service_timing(
        self,
        component: str,
        elapsed_seconds: float,
        cpu_seconds: float,
        stall_threshold_seconds: float | None = None,
    ) -> None:
        record_timing = getattr(self.service, 'record_benchmark_timing', None)
        if callable(record_timing):
            record_timing(
                component,
                elapsed_seconds,
                cpu_seconds,
                stall_threshold_seconds is not None
                and elapsed_seconds > stall_threshold_seconds,
            )

    def _render_camera_card(
        self,
        status: CameraStatus,
        layout: CameraPreviewLayout,
        session_camera: SessionCamera | None = None,
        area_colour: tuple[int, int, int] | None = None,
        metric_state: object = 'uncalibrated',
        frame: Frame | None = None,
        snapshot_input: bool = False,
        calibration_metrics: CalibrationMetricsLike | None = None,
        camera_state: SessionCameraState | None = None,
    ) -> None:
        assert self._window_surface is not None
        self._draw_rectangle(self._window_surface, layout.panel_bounds, DARK_GREY)
        connection_colour = _status_colour(status.runtime_status)
        calibration_status = self._get_calibration_status(status)
        calibration_colour = _calibration_colour(calibration_status)
        slot_id = layout.slot_id or status.logical_name
        effective_camera_state = camera_state or _get_camera_state(status, session_camera)
        self._draw_text(
            f'{layout.logical_name}  connection: {status.runtime_status.value}',
            layout.panel_bounds.left + 8,
            layout.panel_bounds.top + 6,
            connection_colour,
        )
        self._draw_text(
            f'calibration: {calibration_status.value}',
            layout.panel_bounds.left + 8,
            layout.panel_bounds.top + 24,
            calibration_colour,
        )
        self._draw_text(
            f'metrics-calibrated: {_metric_status_value(metric_state)}',
            layout.panel_bounds.left + 250,
            layout.panel_bounds.top + 24,
            _metric_calibration_colour(metric_state),
        )
        self._draw_text(
            f'slot: {slot_id}  state: {effective_camera_state.value}',
            layout.panel_bounds.left + 8,
            layout.panel_bounds.top + 42,
            WHITE,
        )
        resolution_text = _format_resolution(status.native_resolution)
        self._draw_text(
            f'native: {resolution_text}',
            layout.panel_bounds.left + 8,
            layout.panel_bounds.top + 60,
            WHITE,
        )
        metrics = (
            calibration_metrics
            if snapshot_input
            else self._get_calibration_metrics(status)
        )
        if metrics is not None:
            self._draw_text(
                _format_metrics(metrics),
                layout.panel_bounds.left + 8,
                layout.panel_bounds.top + 78,
                WHITE,
            )

        if layout.preview_transform is None:
            self._draw_text(
                status.error_message or 'No active preview',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                GREY,
            )
            self._draw_uncalibrated_click_frame(slot_id, layout)
            return
        if effective_camera_state in {
            SessionCameraState.CLOSED,
            SessionCameraState.UNAVAILABLE,
        } or status.runtime_status in {
            RuntimeStatus.UNAVAILABLE,
            RuntimeStatus.STOPPED,
        }:
            self._draw_text(
                status.error_message or 'Camera is not available',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                GREY,
            )
            self._draw_uncalibrated_click_frame(slot_id, layout)
            return
        if self.configuration.preview_mode == 'off':
            self._draw_text(
                'Preview disabled',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                GREY,
            )
            self._draw_preview_border(layout.preview_bounds, area_colour)
            self._draw_uncalibrated_click_frame(slot_id, layout)
            return
        if not self._should_update_preview(slot_id):
            self._draw_text(
                'Preview paused',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                GREY,
            )
            self._draw_preview_border(layout.preview_bounds, area_colour)
            self._draw_uncalibrated_click_frame(slot_id, layout)
            return
        try:
            preview_frame = frame if snapshot_input else self.service.snapshot(slot_id)
            if preview_frame is None:
                raise RuntimeError('no retained frame')
            started_seconds = time.perf_counter()
            started_cpu_seconds = time.thread_time()
            try:
                frame_surface = self._frame_surface_converter(
                    preview_frame,
                    self._get_pygame(),
                )
            finally:
                self._record_service_timing(
                    'preview_conversion',
                    time.perf_counter() - started_seconds,
                    max(0.0, time.thread_time() - started_cpu_seconds),
                )
            self._render_frame_surface(frame_surface, layout)
            self._mark_preview_updated(slot_id)
        except Exception as ex:  # noqa: BLE001 (A bad frame must not stop the UI loop).
            self._draw_text(
                f'Preview unavailable: {ex}',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                RED,
            )
        self._draw_preview_border(layout.preview_bounds, area_colour)
        self._draw_uncalibrated_click_frame(slot_id, layout)

    def _should_update_preview(self, slot_id: str) -> bool:
        if self.configuration.preview_mode == 'active':
            return True
        if self.configuration.preview_mode == 'off':
            return False
        try:
            now_seconds = float(self._preview_clock())
        except (TypeError, ValueError, OverflowError):
            return True
        previous_seconds = self._preview_last_update_seconds.get(slot_id)
        if previous_seconds is None:
            return True
        return now_seconds - previous_seconds >= 1.0 / self.configuration.preview_low_rate_hz

    def _mark_preview_updated(self, slot_id: str) -> None:
        try:
            now_seconds = float(self._preview_clock())
        except (TypeError, ValueError, OverflowError):
            return
        if math.isfinite(now_seconds):
            self._preview_last_update_seconds[slot_id] = now_seconds

    def _draw_preview_border(
        self,
        bounds: CoordinateBounds,
        colour: tuple[int, int, int] | None,
    ) -> None:
        if colour is None:
            return
        assert self._window_surface is not None
        self._get_pygame().draw.rect(
            self._window_surface,
            colour,
            (
                round(bounds.left),
                round(bounds.top),
                max(1, round(bounds.right - bounds.left)),
                max(1, round(bounds.bottom - bounds.top)),
            ),
            3,
        )

    def _handle_preview_click(self, window_position: object) -> None:
        if not isinstance(window_position, (tuple, list)) or len(window_position) != 2:
            return
        try:
            window_point = Point2D(float(window_position[0]), float(window_position[1]))
        except (OverflowError, TypeError, ValueError) as ex:
            self._last_point_error = f'INVALID_POINT: {ex}'
            return
        for slot_id, layout in self._preview_layouts.items():
            if not layout.preview_bounds.contains(window_point):
                continue
            if layout.preview_transform is None:
                return
            preview_point = Point2D(
                window_point.x - layout.preview_bounds.left,
                window_point.y - layout.preview_bounds.top,
            )
            try:
                self.service.point_from_preview(
                    layout.slot_id or slot_id,
                    preview_point,
                    layout.preview_transform,
                )
            except (MultiVisionError, OverflowError, TypeError, ValueError) as ex:
                error_code = getattr(ex, 'code', type(ex).__name__)
                self._last_point_error = f'{error_code}: {ex}'
                if error_code == 'CALIBRATION_UNCALIBRATED':
                    self._uncalibrated_click_cameras.add(layout.slot_id or slot_id)
            else:
                self._last_point_error = None
            return

    def _render_frame_surface(
        self,
        frame_surface: Any,
        layout: CameraPreviewLayout,
    ) -> None:
        assert self._window_surface is not None
        assert layout.preview_transform is not None
        preview_bounds = layout.preview_bounds
        transform = layout.preview_transform
        self._draw_rectangle(self._window_surface, preview_bounds, BLACK)
        content_bounds = transform.content_bounds
        scaled_size = (
            max(1, round(content_bounds.right - content_bounds.left)),
            max(1, round(content_bounds.bottom - content_bounds.top)),
        )
        scaled_surface = self._get_pygame().transform.smoothscale(
            frame_surface,
            scaled_size,
        )
        self._window_surface.blit(
            scaled_surface,
            (
                round(preview_bounds.left + content_bounds.left),
                round(preview_bounds.top + content_bounds.top),
            ),
        )

    def _draw_text(
        self,
        text: str,
        x_pos: float,
        y_pos: float,
        colour: tuple[int, int, int],
    ) -> None:
        assert self._window_surface is not None
        assert self._font is not None
        text_surface = self._font.render(text, True, colour)
        self._window_surface.blit(text_surface, (round(x_pos), round(y_pos)))

    def _draw_rectangle(
        self,
        surface: Any,
        bounds: CoordinateBounds,
        colour: tuple[int, int, int],
    ) -> None:
        self._get_pygame().draw.rect(
            surface,
            colour,
            (
                round(bounds.left),
                round(bounds.top),
                max(1, round(bounds.right - bounds.left)),
                max(1, round(bounds.bottom - bounds.top)),
            ),
        )

    def _get_snapshot_display_cameras(
        self,
        render_snapshot: object,
    ) -> list[_DisplayCamera]:
        camera_snapshots = getattr(render_snapshot, 'camera_snapshots', ())
        if not isinstance(camera_snapshots, Sequence) or isinstance(
            camera_snapshots,
            (str, bytes),
        ):
            raise TypeError('render snapshot contains invalid camera inputs')
        display_cameras: list[_DisplayCamera] = []
        for camera_snapshot in camera_snapshots:
            slot_id = getattr(camera_snapshot, 'slot_id', None)
            logical_name = getattr(camera_snapshot, 'logical_name', None)
            status = getattr(camera_snapshot, 'status', None)
            camera_state = getattr(camera_snapshot, 'camera_state', None)
            lifecycle_generation = getattr(
                camera_snapshot,
                'lifecycle_generation',
                None,
            )
            frame = getattr(camera_snapshot, 'frame', None)
            calibration_metrics = getattr(
                camera_snapshot,
                'calibration_metrics',
                None,
            )
            if (
                not isinstance(slot_id, str)
                or len(slot_id) == 0
                or not isinstance(logical_name, str)
                or len(logical_name) == 0
                or not isinstance(status, CameraStatus)
                or not isinstance(camera_state, SessionCameraState)
                or not isinstance(lifecycle_generation, int)
                or isinstance(lifecycle_generation, bool)
                or lifecycle_generation < 0
                or (frame is not None and not isinstance(frame, Frame))
            ):
                raise TypeError('render snapshot contains invalid camera input')
            display_cameras.append(
                _DisplayCamera(
                    slot_id,
                    logical_name,
                    status,
                    None,
                    _DisplayCameraSnapshot(
                        slot_id,
                        logical_name,
                        status,
                        camera_state,
                        lifecycle_generation,
                        frame,
                        calibration_metrics,
                    ),
                ),
            )
        if len({camera.slot_id for camera in display_cameras}) != len(display_cameras):
            raise ValueError('render snapshot camera slots must be unique')
        self._clear_lifecycle_invalidated_clicks(display_cameras)
        self._prune_uncalibrated_click_cameras(display_cameras)
        return display_cameras

    def _build_preview_layouts(
        self,
        display_cameras: Sequence[_DisplayCamera] | None = None,
    ) -> dict[str, CameraPreviewLayout]:
        if display_cameras is None:
            display_cameras = self._get_display_cameras()
        statuses = [camera.status for camera in display_cameras]
        session_cameras = [
            camera.session_camera
            for camera in display_cameras
            if camera.session_camera is not None
        ]
        if any(camera.render_snapshot is not None for camera in display_cameras):
            # Snapshot camera names are display-only aliases; slot IDs remain the
            # stable keys used by the service and by preview clicks.
            slot_statuses = [
                camera.status._replace(logical_name=camera.slot_id)
                for camera in display_cameras
            ]
            slot_layouts = build_camera_preview_layouts(
                slot_statuses,
                self.configuration.window_resolution,
            )
            return {
                camera.slot_id: slot_layouts[camera.slot_id]._replace(
                    logical_name=camera.logical_name,
                    slot_id=camera.slot_id,
                )
                for camera in display_cameras
            }
        return build_camera_preview_layouts(
            statuses,
            self.configuration.window_resolution,
            session_cameras=session_cameras if len(session_cameras) > 0 else None,
        )

    def _get_render_snapshot(self) -> object | None:
        get_render_snapshot = getattr(self.service, 'get_render_snapshot', None)
        if not callable(get_render_snapshot):
            return None
        try:
            snapshot = get_render_snapshot()
        except Exception as ex:  # noqa: BLE001 (Legacy display fallback remains usable.)
            self._last_projector_error = f'Render snapshot unavailable: {ex}'
            return None
        if not hasattr(snapshot, 'overlays'):
            self._last_projector_error = 'Render snapshot unavailable'
            return None
        snapshot_descriptor = getattr(snapshot, 'projector_output_descriptor', None)
        if snapshot_descriptor is not None and not isinstance(
            snapshot_descriptor,
            ProjectorOutputDescriptor,
        ):
            self._last_projector_error = 'Render snapshot unavailable'
            return None
        return snapshot

    def _get_overlay_intensity(self, render_snapshot: object | None = None) -> float:
        if render_snapshot is not None:
            value = getattr(render_snapshot, 'global_overlay_intensity', 1.0)
        else:
            getter = getattr(self.service, 'get_overlay_intensity', None)
            value = getter() if callable(getter) else getattr(
                self.service,
                'overlay_intensity',
                1.0,
            )
        try:
            return normalise_overlay_intensity(value)
        except ValueError as ex:
            self._last_projector_error = f'Overlay intensity unavailable: {ex}'
            return 1.0

    def _get_projector_areas(self) -> list[ProjectorAreaLike]:
        get_camera_areas = getattr(self.service, 'get_camera_areas', None)
        if not callable(get_camera_areas):
            return []
        areas = get_camera_areas()
        if not isinstance(areas, list):
            raise TypeError('service returned an invalid projector area list')
        return areas

    def _get_generic_overlays(self) -> list[ProjectorOverlayLike]:
        list_overlays = getattr(self.service, 'list_overlays', None)
        if not callable(list_overlays):
            return []
        overlays = list_overlays()
        if not isinstance(overlays, list):
            raise TypeError('service returned an invalid projector overlay list')
        return overlays

    def _get_metric_capture_active(self) -> bool:
        try:
            metric_capture_active = getattr(self.service, 'metric_capture_active', False)
        except Exception as ex:  # noqa: BLE001 (A display snapshot must not stop the UI loop).
            self._last_metric_error = f'Metric capture state unavailable: {ex}'
            return True
        if not isinstance(metric_capture_active, bool):
            self._last_metric_error = 'Metric capture state unavailable'
            return True
        return metric_capture_active

    def _get_metric_snapshot(
        self,
    ) -> tuple[object, MetricRulerLike | None]:
        try:
            get_metric_status = getattr(self.service, 'get_metric_status', None)
        except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
            self._last_metric_error = f'Metric status unavailable: {ex}'
            return 'uncalibrated', None
        if callable(get_metric_status):
            try:
                raw_state = get_metric_status()
            except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
                self._last_metric_error = f'Metric status unavailable: {ex}'
                return 'uncalibrated', None
        else:
            missing_state = object()
            try:
                raw_state = getattr(self.service, 'metric_state', missing_state)
            except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
                self._last_metric_error = f'Metric status unavailable: {ex}'
                return 'uncalibrated', None
            if raw_state is missing_state:
                self._last_metric_error = 'Metric status unavailable'
                return 'uncalibrated', None
        if not _is_metric_state(raw_state):
            self._last_metric_error = 'Metric status unavailable'
            return 'uncalibrated', None
        if not _is_metric_calibrated(raw_state):
            return raw_state, None
        try:
            ruler = getattr(self.service, 'metric_ruler', None)
        except Exception as ex:  # noqa: BLE001 (A bad snapshot must not stop the UI loop).
            self._last_metric_error = f'Metric ruler unavailable: {ex}'
            return raw_state, None
        if ruler is None:
            return raw_state, None
        if not _is_metric_ruler_snapshot(ruler):
            self._last_metric_error = 'Metric ruler unavailable'
            return raw_state, None
        return raw_state, ruler

    def _synchronise_projector_descriptor(
        self,
        snapshot_descriptor: ProjectorOutputDescriptor | None = None,
        snapshot_calibration_pattern: CalibrationPattern | None = None,
        use_snapshot_pattern: bool = False,
    ) -> None:
        descriptor = snapshot_descriptor
        if descriptor is None:
            descriptor = _get_service_projector_descriptor(self.service)
        if descriptor is None or descriptor == self._projector_output_descriptor:
            if use_snapshot_pattern:
                self.calibration_pattern = (
                    snapshot_calibration_pattern
                    if isinstance(snapshot_calibration_pattern, CalibrationPattern)
                    else None
                )
            return
        next_configuration = replace(
            self.configuration,
            projector_resolution=descriptor.projector_resolution,
            projector_output_descriptor=descriptor,
        )
        replacement_output = self._projector_output
        replacement_surface: Any | None = None
        if self._is_initialised:
            if not self._projector_output_is_injected:
                replacement_output = self._projector_output_factory(
                    self._get_pygame(),
                    next_configuration,
                )
            try:
                replacement_surface = self._get_pygame().Surface(
                    tuple(descriptor.projector_resolution),
                )
            except BaseException:
                if (
                    replacement_output is not self._projector_output
                    and replacement_output is not None
                ):
                    replacement_output.shutdown()
                raise
            if (
                self._projector_output is not None
                and self._projector_output is not replacement_output
            ):
                self._projector_output.shutdown()
        self._projector_output_descriptor = descriptor
        self._preview_last_update_seconds.clear()
        self.configuration = next_configuration
        self._projector_output = replacement_output
        if use_snapshot_pattern:
            self.calibration_pattern = (
                snapshot_calibration_pattern
                if isinstance(snapshot_calibration_pattern, CalibrationPattern)
                else None
            )
        else:
            try:
                calibration_pattern = getattr(self.service, 'calibration_pattern', None)
            except Exception:  # noqa: BLE001 (A bad service pattern must not stop the UI loop).
                calibration_pattern = None
            self.calibration_pattern = (
                calibration_pattern
                if isinstance(calibration_pattern, CalibrationPattern)
                else None
            )
        if replacement_surface is not None:
            self._projector_surface = replacement_surface

    def _get_display_cameras(self) -> list[_DisplayCamera]:
        statuses = self.service.get_camera_statuses()
        if not isinstance(statuses, list):
            raise TypeError('service returned an invalid camera status list')
        session_camera_getter = getattr(self.service, 'get_session_cameras', None)
        if not callable(session_camera_getter):
            display_cameras = list(_build_display_cameras(statuses))
        else:
            session_cameras = session_camera_getter()
            if not isinstance(session_cameras, list):
                raise TypeError('service returned an invalid session camera list')
            display_cameras = list(
                _build_display_cameras(
                    statuses,
                    session_cameras if len(session_cameras) > 0 else None,
                ),
            )
        self._clear_lifecycle_invalidated_clicks(display_cameras)
        self._prune_uncalibrated_click_cameras(display_cameras)
        return display_cameras

    def _clear_lifecycle_invalidated_clicks(
        self,
        display_cameras: Sequence[_DisplayCamera],
    ) -> None:
        for camera in display_cameras:
            session_camera = camera.session_camera
            lifecycle_generation = (
                getattr(camera.render_snapshot, 'lifecycle_generation', None)
                if camera.render_snapshot is not None
                else (
                    None
                    if session_camera is None
                    else session_camera.lifecycle_generation
                )
            )
            if not isinstance(lifecycle_generation, int):
                continue
            previous_generation = self._camera_lifecycle_generations.get(camera.slot_id)
            if (
                previous_generation is not None
                and previous_generation != lifecycle_generation
            ):
                self._uncalibrated_click_cameras.discard(camera.slot_id)
                self._preview_last_update_seconds.pop(camera.slot_id, None)
            self._camera_lifecycle_generations[camera.slot_id] = lifecycle_generation

    def _prune_uncalibrated_click_cameras(
        self,
        display_cameras: Sequence[_DisplayCamera],
    ) -> None:
        eligible_cameras = {
            camera.slot_id
            for camera in display_cameras
            if (
                camera.status.runtime_status is RuntimeStatus.AVAILABLE
                and _get_camera_state(camera.status, camera.session_camera)
                is SessionCameraState.OPEN
                and camera.status.native_resolution is not None
                and camera.status.calibration_status is CalibrationStatus.UNCALIBRATED
            )
        }
        self._uncalibrated_click_cameras.intersection_update(eligible_cameras)

    def _draw_uncalibrated_click_frame(
        self,
        slot_id: str,
        layout: CameraPreviewLayout,
    ) -> None:
        if slot_id not in self._uncalibrated_click_cameras:
            return
        assert self._window_surface is not None
        bounds = layout.preview_bounds
        self._get_pygame().draw.rect(
            self._window_surface,
            RED,
            (
                round(bounds.left),
                round(bounds.top),
                max(1, round(bounds.right - bounds.left)),
                max(1, round(bounds.bottom - bounds.top)),
            ),
            3,
        )

    def _get_calibration_status(self, status: CameraStatus) -> CalibrationStatus:
        return status.calibration_status

    def _get_calibration_metrics(
        self,
        status: CameraStatus,
    ) -> CalibrationMetricsLike | None:
        return self.service.get_calibration_metrics(status.logical_name)

    def _get_pygame(self) -> Any:
        if self._pygame is None:
            try:
                import pygame
            except ImportError as ex:
                raise RuntimeError('Pygame is not installed') from ex
            self._pygame = pygame
        return self._pygame

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Pygame display operations must run on the main thread')


def build_camera_preview_layouts(
    statuses: Sequence[CameraStatus],
    window_resolution: Resolution,
    *,
    session_cameras: Sequence[SessionCamera] | None = None,
) -> dict[str, CameraPreviewLayout]:
    """Build a deterministic grid without changing camera-native geometry."""
    _validate_resolution(window_resolution, 'window_resolution')
    status_values = tuple(statuses)
    if len(status_values) == 0:
        return {}
    if any(not isinstance(status, CameraStatus) for status in status_values):
        raise TypeError('statuses must contain CameraStatus values')
    for status in status_values:
        if not isinstance(status.logical_name, str) or len(status.logical_name) == 0:
            raise ValueError('camera logical names must be non-empty strings')
        if not isinstance(status.runtime_status, RuntimeStatus):
            raise TypeError('camera runtime statuses must be RuntimeStatus values')
        if not isinstance(status.calibration_status, CalibrationStatus):
            raise TypeError(
                'camera calibration statuses must be CalibrationStatus values',
            )
        if status.device_id is not None and (
            not isinstance(status.device_id, str) or len(status.device_id) == 0
        ):
            raise ValueError('camera device IDs must be non-empty strings or None')
        if status.native_resolution is not None:
            _validate_resolution(status.native_resolution, 'native_resolution')
        if (
            not isinstance(status.frame_counter, int)
            or isinstance(status.frame_counter, bool)
            or status.frame_counter < 0
        ):
            raise ValueError('camera frame counters must be non-negative integers')
        if status.error_message is not None and not isinstance(status.error_message, str):
            raise TypeError('camera error messages must be strings or None')
    logical_names = [status.logical_name for status in status_values]
    if len(set(logical_names)) != len(logical_names):
        raise ValueError('camera logical names must be unique')
    camera_values = _build_display_cameras(status_values, session_cameras)
    column_count = max(1, math.ceil(math.sqrt(len(camera_values))))
    row_count = math.ceil(len(camera_values) / column_count)
    panel_width = window_resolution.width / column_count
    panel_height = window_resolution.height / row_count
    layouts: dict[str, CameraPreviewLayout] = {}
    for idx, camera in enumerate(camera_values):
        status = camera.status
        x_idx = idx % column_count
        y_idx = idx // column_count
        panel_bounds = CoordinateBounds(
            x_idx * panel_width,
            y_idx * panel_height,
            (x_idx + 1) * panel_width,
            (y_idx + 1) * panel_height,
        )
        preview_bounds = CoordinateBounds(
            panel_bounds.left + 8
            if panel_width > 16
            else panel_bounds.left,
            panel_bounds.top + 100
            if panel_height > 108
            else panel_bounds.top,
            panel_bounds.right - 8
            if panel_width > 16
            else panel_bounds.right,
            panel_bounds.bottom - 8
            if panel_height > 90
            else panel_bounds.bottom,
        )
        preview_size = Resolution(
            max(1, round(preview_bounds.right - preview_bounds.left)),
            max(1, round(preview_bounds.bottom - preview_bounds.top)),
        )
        preview_transform = (
            build_preview_transform(preview_size, status.native_resolution)
            if status.native_resolution is not None
            else None
        )
        layouts[camera.slot_id] = CameraPreviewLayout(
            camera.logical_name,
            panel_bounds,
            preview_bounds,
            preview_transform,
            camera.slot_id if camera.session_camera is not None else None,
        )
    return layouts


def _get_service_projector_descriptor(
    service: object,
) -> ProjectorOutputDescriptor | None:
    try:
        descriptor = getattr(service, 'projector_output_descriptor', None)
    except Exception:  # noqa: BLE001 (A display snapshot must not stop the UI loop).
        return None
    if not isinstance(descriptor, ProjectorOutputDescriptor):
        return None
    return descriptor


def _is_metric_ruler_snapshot(value: object) -> bool:
    try:
        projector_start = getattr(value, 'projector_start')
        projector_end = getattr(value, 'projector_end')
        label_position = getattr(value, 'label_position')
        label_bounds = getattr(value, 'label_bounds')
        if not _is_finite_display_point(projector_start):
            return False
        if not _is_finite_display_point(projector_end):
            return False
        if not isinstance(getattr(value, 'label'), str):
            return False
        if not isinstance(label_bounds, CoordinateBounds):
            return False
        if not _is_finite_display_point(label_position):
            return False
        if not all(_is_finite_display_number(bound) for bound in label_bounds):
            return False
        ticks = getattr(value, 'ticks')
        markers = getattr(value, 'markers')
        if not isinstance(ticks, Sequence) or not isinstance(markers, Sequence):
            return False
        if len(markers) != 2:
            return False
        if any(
            not isinstance(getattr(tick, 'is_major'), bool)
            or not _is_finite_display_point(getattr(tick, 'projector_start'))
            or not _is_finite_display_point(getattr(tick, 'projector_end'))
            for tick in ticks
        ):
            return False
        return all(
            isinstance(getattr(marker, 'projector_extent'), Sequence)
            and len(getattr(marker, 'projector_extent')) >= 3
            and all(
                _is_finite_display_point(point)
                for point in getattr(marker, 'projector_extent')
            )
            for marker in markers
        )
    except Exception:  # noqa: BLE001 (Malformed service snapshots are display input).
        return False


def _metric_ruler_intersects_protected_regions(
    marker_points: Sequence[Sequence[PointLike]],
    main_segment: Sequence[PointLike],
    tick_segments: Sequence[Sequence[PointLike]],
    label_position: PointLike,
    protected_regions: Sequence[Sequence[PointLike]],
) -> bool:
    if len(protected_regions) == 0:
        return False
    if any(
        is_polygon_intersecting_protected_regions(points, protected_regions)
        for points in marker_points
    ):
        return True
    style = OverlayStyle(colour='#ebebeb', line_width_px=METRIC_RULER_LINE_WIDTH)
    materialisation = ProjectorMaterialisation(
        segments=(
            tuple(
                ProjectorSegment(
                    Point2D(float(start[0]), float(start[1])),
                    Point2D(float(end[0]), float(end[1])),
                    style,
                )
                for start, end in (tuple(main_segment), *tuple(tick_segments))
            )
        ),
        labels=(
            ProjectorLabel(
                Point2D(float(label_position[0]), float(label_position[1])),
                '',
                style,
            ),
        ),
    )
    return materialise_presentation(materialisation, 1.0, protected_regions) != materialisation


def _projector_overlay_layer_key(kind: str) -> int:
    if not isinstance(kind, str):
        raise TypeError('projector overlay kinds must be strings')
    try:
        return {
            'grid': 0,
            'rect': 1,
            'circle': 1,
            'line': 2,
            'ruler': 2,
            'arrow': 2,
            'text': 1,
        }[kind]
    except KeyError as ex:
        raise ValueError(f'Unknown projector overlay kind: {kind!r}') from ex


def _validate_overlay_style(style: object) -> None:
    if not isinstance(style, OverlayStyle):
        raise TypeError('projector primitives must contain OverlayStyle values')
    if (
        not isinstance(style.colour, tuple)
        or len(style.colour) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in style.colour
        )
        or not isinstance(style.line_width_px, int)
        or isinstance(style.line_width_px, bool)
        or style.line_width_px <= 0
        or not _is_finite_display_number(style.intensity)
        or not 0.0 <= style.intensity <= 1.0
    ):
        raise ValueError('projector primitive styles are not raster-safe')


def _round_generic_projector_point(point: Point2D) -> tuple[int, int]:
    if not _is_finite_display_point(point):
        raise ValueError('projector primitive points must be finite Point2D values')
    return round(point.x), round(point.y)


def _round_generic_projector_points(
    points: Sequence[Point2D],
) -> tuple[tuple[int, int], ...]:
    if not isinstance(points, Sequence) or len(points) < 3:
        raise ValueError('projector polygons must contain at least three points')
    return tuple(_round_generic_projector_point(point) for point in points)


def _is_finite_display_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_finite_display_point(value: object) -> bool:
    return (
        isinstance(value, Point2D)
        and _is_finite_display_number(value.x)
        and _is_finite_display_number(value.y)
    )


def _is_display_colour(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(
            isinstance(channel, int)
            and not isinstance(channel, bool)
            and 0 <= channel <= 255
            for channel in value
        )
    )


def _make_projector_output(
    pygame_module: Any,
    configuration: DisplayConfiguration,
) -> ProjectorOutput | None:
    if not isinstance(pygame_module, types.ModuleType):
        return None
    display_count = pygame_module.display.get_num_displays()
    display_index = min(1, max(0, display_count - 1))
    desktop_sizes = pygame_module.display.get_desktop_sizes()
    desktop_width, desktop_height = desktop_sizes[display_index]
    return Sdl2ProjectorOutput(
        configuration.projector_resolution,
        display_index,
        Resolution(desktop_width, desktop_height),
        fullscreen=True,
    )


def _frame_to_surface(frame: Frame, pygame_module: Any) -> Any:
    if not isinstance(frame, Frame):
        raise TypeError('frame must be Frame')
    if isinstance(frame.data, pygame_module.Surface):
        return frame.data
    try:
        import cv2
    except ImportError as ex:
        raise RuntimeError('OpenCV is required to render camera frames') from ex
    frame_data = frame.data
    if not hasattr(frame_data, 'shape') or len(frame_data.shape) < 2:
        raise ValueError('Camera frame must be an image or Pygame surface')
    height, width = frame_data.shape[:2]
    if len(frame_data.shape) < 3 or frame_data.shape[2] < 3:
        raise ValueError('Camera frame must have colour channels')
    rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
    return pygame_module.image.frombuffer(
        rgb_frame.tobytes(),
        (int(width), int(height)),
        'RGB',
    )


def _render_apriltag_image(
    marker_family: str,
    marker_id: int,
    pixel_size: int,
    pygame_module: Any,
) -> Any:
    try:
        import cv2
    except ImportError as ex:
        raise RuntimeError('OpenCV is required to render calibration markers') from ex
    aruco_module = getattr(cv2, 'aruco', None)
    if aruco_module is None:
        raise RuntimeError('OpenCV was built without aruco support')
    dictionary_id = getattr(aruco_module, marker_family, None)
    if dictionary_id is None:
        raise ValueError(f'Unsupported AprilTag family: {marker_family!r}')
    dictionary = aruco_module.getPredefinedDictionary(dictionary_id)
    marker_image = aruco_module.generateImageMarker(
        dictionary,
        marker_id,
        pixel_size,
        borderBits=1,
    )
    rgb_marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)
    marker_surface = pygame_module.image.frombuffer(
        rgb_marker_image.tobytes(),
        (pixel_size, pixel_size),
        'RGB',
    )
    quiet_zone = max(4, round(pixel_size * 0.1))
    quiet_surface = pygame_module.Surface(
        (pixel_size + 2 * quiet_zone, pixel_size + 2 * quiet_zone),
    )
    quiet_surface.fill(WHITE)
    quiet_surface.blit(marker_surface, (quiet_zone, quiet_zone))
    return quiet_surface


def _round_projector_point(point: Point2D) -> tuple[int, int]:
    if not isinstance(point, Point2D):
        raise TypeError('projector primitive points must be Point2D values')
    return round(point.x), round(point.y)


def _round_projector_points(points: Sequence[Point2D]) -> tuple[tuple[int, int], ...]:
    if len(points) < 3:
        raise ValueError('projector marker primitives must contain at least three points')
    return tuple(_round_projector_point(point) for point in points)


def _validate_raster_projector_points(
    points: Sequence[tuple[int, int]],
    surface_size: tuple[int, int],
    line_width: int = 1,
) -> None:
    if surface_size[0] <= 0 or surface_size[1] <= 0:
        raise ValueError('projector surface must have a positive size')
    if line_width <= 0:
        raise ValueError('metric ruler line width must be positive')
    raster_margin = math.ceil(line_width / 2)
    if any(
        not raster_margin <= x_pos < surface_size[0] - raster_margin
        or not raster_margin <= y_pos < surface_size[1] - raster_margin
        for x_pos, y_pos in points
    ):
        raise ValueError('metric ruler primitive is outside the projector surface')


def _get_surface_size(surface: Any, fallback_size: int) -> tuple[int, int]:
    get_size = getattr(surface, 'get_size', None)
    if callable(get_size):
        size = get_size()
        if (
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(value, int) and value > 0 for value in size)
        ):
            return size
    size = getattr(surface, 'size', None)
    if (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(value, int) and value > 0 for value in size)
    ):
        return size
    return fallback_size, fallback_size


def _build_display_cameras(
    statuses: Sequence[CameraStatus],
    session_cameras: Sequence[SessionCamera] | None = None,
) -> tuple[_DisplayCamera, ...]:
    if session_cameras is None:
        return tuple(
            _DisplayCamera(
                status.logical_name,
                status.logical_name,
                status,
                None,
            )
            for status in sorted(statuses, key=_camera_status_sort_key)
        )

    session_camera_values = tuple(session_cameras)
    if any(
        not isinstance(camera, SessionCamera)
        for camera in session_camera_values
    ):
        raise TypeError('session_cameras must contain SessionCamera values')
    if len({camera.slot_id for camera in session_camera_values}) != len(
        session_camera_values,
    ):
        raise ValueError('session camera slots must be unique')
    statuses_by_slot = {
        status.logical_name: status
        for status in statuses
    }
    if len(statuses_by_slot) != len(session_camera_values):
        raise ValueError('session cameras and statuses must describe the same slots')
    display_cameras: list[_DisplayCamera] = []
    for camera in sorted(
        session_camera_values,
        key=lambda value: _slot_sort_key(value.slot_id),
    ):
        status = statuses_by_slot.get(camera.slot_id)
        if status is None:
            raise ValueError(
                f'No runtime status was returned for session camera {camera.slot_id!r}',
            )
        display_cameras.append(
            _DisplayCamera(
                camera.slot_id,
                camera.display_name,
                status,
                camera,
            ),
        )
    return tuple(display_cameras)


def _slot_sort_key(slot_id: str) -> tuple[int, int | str]:
    prefix, separator, index = slot_id.rpartition('-')
    if prefix == 'camera' and separator and index.isdigit():
        return (0, int(index))
    return (1, slot_id)


def _camera_status_sort_key(status: CameraStatus) -> tuple[int, int | str]:
    return _slot_sort_key(status.logical_name)


def _get_camera_state(
    status: CameraStatus,
    session_camera: SessionCamera | None,
) -> SessionCameraState:
    if session_camera is not None:
        return session_camera.state
    if status.runtime_status is RuntimeStatus.STOPPED:
        return SessionCameraState.CLOSED
    if status.runtime_status is RuntimeStatus.UNAVAILABLE:
        return SessionCameraState.UNAVAILABLE
    return SessionCameraState.OPEN


def _status_colour(status: RuntimeStatus) -> tuple[int, int, int]:
    if status is RuntimeStatus.AVAILABLE:
        return GREEN
    if status in {RuntimeStatus.ERROR, RuntimeStatus.UNAVAILABLE}:
        return RED
    return ORANGE


def _calibration_colour(status: CalibrationStatus) -> tuple[int, int, int]:
    if status is CalibrationStatus.CALIBRATED:
        return GREEN
    if status is CalibrationStatus.STALE:
        return RED
    return ORANGE


def _metric_status_value(status: object) -> str:
    value = getattr(status, 'value', status)
    if isinstance(value, str) and value.casefold() in {
        'calibrated',
        'stale',
        'uncalibrated',
    }:
        return value
    return 'uncalibrated'


def _is_metric_state(status: object) -> bool:
    return _metric_status_value(status).casefold() in {
        'calibrated',
        'stale',
        'uncalibrated',
    }


def _is_metric_calibrated(status: object) -> bool:
    return _metric_status_value(status).casefold() == 'calibrated'


def _metric_calibration_colour(status: object) -> tuple[int, int, int]:
    if _is_metric_calibrated(status):
        return GREEN
    if _metric_status_value(status).casefold() == 'stale':
        return RED
    return ORANGE


def _format_resolution(resolution: Resolution | None) -> str:
    if resolution is None:
        return 'unknown'
    return f'{resolution.width}x{resolution.height}'


def _format_metrics(metrics: CalibrationMetricsLike) -> str:
    return (
        f'tags: {metrics.unique_tag_count}  '
        f'corners: {metrics.correspondence_corner_count}  '
        f'inliers: {metrics.ransac_inlier_count} ({metrics.inlier_ratio:.2f})  '
        f'error: {metrics.mean_reprojection_error:.2f}/{metrics.max_reprojection_error:.2f}  '
        f'coverage: {metrics.spatial_coverage:.2f}'
    )


def _validate_resolution(resolution: object, field_name: str) -> None:
    if not is_valid_resolution(resolution):
        raise ValueError(f'{field_name} must be a positive Resolution')


__all__ = [
    'BLACK',
    'METRIC_RULER_COLOUR',
    'CameraPreviewLayout',
    'DisplayConfiguration',
    'FrameSurfaceConverter',
    'MarkerImageRenderer',
    'MetricRulerLike',
    'MetricRulerMarkerLike',
    'MetricRulerTickLike',
    'DisplayServiceLike',
    'ProjectorAreaLike',
    'ProjectorPointOverlayLike',
    'ProjectorOutput',
    'ProjectorOutputFactory',
    'PygameDisplayRuntime',
    'ProjectorRenderer',
    'Sdl2ProjectorOutput',
    'build_camera_preview_layouts',
]
