"""Main-thread Pygame rendering for camera previews and projector output."""

from __future__ import annotations

import math
import threading
import types
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    NamedTuple,
    Protocol,
)

from multivision.calibration import CalibrationMetrics
from multivision.config import ProjectorOutputDescriptor
from multivision.errors import MultiVisionError
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    PreviewTransform,
    build_preview_transform,
)
from multivision.metric import MetricCalibrationStatus
from multivision.overlays import (
    MAX_OVERLAY_LABEL_SCALE,
    MIN_OVERLAY_LABEL_SCALE,
    OverlayStyle,
    ProjectorLabel,
    ProjectorMaterialisation,
    ProjectorPolygon,
    ProjectorSegment,
)
from multivision.pattern import CalibrationPattern
from multivision.service import RedCircleOverlay
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


ProjectorOutputFactory = Callable[[Any, 'DisplayConfiguration'], ProjectorOutput | None]


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
    def overlay(self) -> RedCircleOverlay | None:
        ...

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor:
        ...

    @property
    def metric_state(self) -> MetricCalibrationStatus:
        ...

    def get_metric_status(self) -> MetricCalibrationStatus:
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

    def snapshot(self, logical_name: str) -> Frame:
        ...

    def get_calibration_metrics(self, logical_name: str) -> CalibrationMetrics | None:
        ...

    def point_from_preview(
        self,
        logical_name: str,
        preview_point: Point2D,
        preview_transform: PreviewTransform,
    ) -> RedCircleOverlay:
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


class _DisplayCamera(NamedTuple):
    slot_id: str
    logical_name: str
    status: CameraStatus
    session_camera: SessionCamera | None


@dataclass(frozen=True)
class DisplayConfiguration:
    """Window settings plus the service-selected projector output descriptor."""

    window_resolution: Resolution = Resolution(1280, 720)
    projector_resolution: Resolution = Resolution(1920, 1080)
    frames_per_second: int = 30
    caption: str = 'MultiVision'
    projector_output_descriptor: ProjectorOutputDescriptor | None = None

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
    ) -> None:
        """Draw the current enabled diagnostic areas in slot order."""
        self._require_main_thread()
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
            self._pygame.draw.polygon(
                surface,
                area.area_colour,
                area_points,
                2,
            )
            label_surface = font.render(area.display_name, True, area.area_colour)
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
    ) -> None:
        """Draw service-produced projector-native ruler primitives unchanged."""
        self._require_main_thread()
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

        for points in marker_points:
            self._pygame.draw.polygon(
                surface,
                METRIC_RULER_COLOUR,
                points,
                METRIC_RULER_LINE_WIDTH,
            )
        self._pygame.draw.line(
            surface,
            METRIC_RULER_COLOUR,
            projector_start,
            projector_end,
            METRIC_RULER_LINE_WIDTH,
        )
        for tick_start, tick_end, line_width in tick_points:
            self._pygame.draw.line(
                surface,
                METRIC_RULER_COLOUR,
                tick_start,
                tick_end,
                line_width,
            )

        label_surface = font.render(ruler.label, True, METRIC_RULER_COLOUR)
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
        overlay: RedCircleOverlay | None,
    ) -> None:
        self._require_main_thread()
        if overlay is None:
            return
        if not isinstance(overlay, RedCircleOverlay):
            raise TypeError('overlay must be RedCircleOverlay or None')
        self._pygame.draw.circle(
            surface,
            overlay.colour,
            (round(overlay.projector_point.x), round(overlay.projector_point.y)),
            overlay.radius,
        )

    def render_generic_overlays(
        self,
        surface: Any,
        overlays: Sequence[ProjectorOverlayLike],
        font: Any | None = None,
        layer: str | None = None,
    ) -> None:
        """Draw immutable, already-materialised projector overlay primitives."""
        self._require_main_thread()
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
        self._preview_layouts = self._build_preview_layouts()
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
        self._require_main_thread()
        self._synchronise_projector_descriptor()
        self.initialise()
        assert self._window_surface is not None
        assert self._projector_surface is not None
        assert self._projector_renderer is not None
        self._last_metric_error = None
        self._last_projector_error = None
        self._window_surface.fill(DARK_GREY)
        display_cameras = self._get_display_cameras()
        self._preview_layouts = self._build_preview_layouts(display_cameras)
        pattern_visible = self.service.calibration_pattern_visible
        if not isinstance(pattern_visible, bool):
            raise TypeError('service returned an invalid calibration pattern state')
        metric_capture_active = self._get_metric_capture_active()
        metric_state, metric_ruler = self._get_metric_snapshot()
        projector_areas = (
            []
            if pattern_visible or metric_capture_active
            else self._get_projector_areas()
        )
        generic_overlays = (
            []
            if pattern_visible or metric_capture_active
            else self._get_generic_overlays()
        )
        if not pattern_visible and not metric_capture_active:
            self._area_colours = {
                area.slot_id: area.area_colour
                for area in projector_areas
            }
        area_colours = self._area_colours
        for camera in display_cameras:
            self._render_camera_card(
                camera.status,
                self._preview_layouts[camera.slot_id],
                camera.session_camera,
                area_colours.get(camera.slot_id),
                metric_state,
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
            if not pattern_visible and not metric_capture_active:
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'grid',
                )
                self._projector_renderer.render_areas(
                    self._projector_surface,
                    projector_areas,
                    self._projector_area_font,
                )
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'shape',
                )
                if metric_state is MetricCalibrationStatus.CALIBRATED:
                    try:
                        self._projector_renderer.render_metric_ruler(
                            self._projector_surface,
                            metric_ruler,
                            self._font,
                        )
                    except Exception as ex:  # noqa: BLE001 (Bad metric snapshot).
                        self._last_metric_error = f'Metric ruler unavailable: {ex}'
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'line',
                )
                self._projector_renderer.render_generic_overlays(
                    self._projector_surface,
                    generic_overlays,
                    self._font,
                    'label',
                )
                self._projector_renderer.render_overlay(
                    self._projector_surface,
                    self.service.overlay,
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
                self._last_projector_error = None
                self._last_metric_error = None
                self._is_initialised = False
        if output_error is not None:
            raise output_error

    def _present_projector_surface(self) -> None:
        if self._projector_output is None or self._projector_surface is None:
            return
        try:
            self._projector_output.present(self._projector_surface)
        except Exception as ex:  # noqa: BLE001 (A projector seam must not stop the UI loop).
            self._last_projector_error = f'Projector unavailable: {ex}'
            if self._window_surface is not None and self._font is not None:
                self._draw_text(self._last_projector_error, 8, 8, RED)

    def _render_camera_card(
        self,
        status: CameraStatus,
        layout: CameraPreviewLayout,
        session_camera: SessionCamera | None = None,
        area_colour: tuple[int, int, int] | None = None,
        metric_state: MetricCalibrationStatus = MetricCalibrationStatus.UNCALIBRATED,
    ) -> None:
        assert self._window_surface is not None
        self._draw_rectangle(self._window_surface, layout.panel_bounds, DARK_GREY)
        connection_colour = _status_colour(status.runtime_status)
        calibration_status = self._get_calibration_status(status)
        calibration_colour = _calibration_colour(calibration_status)
        slot_id = layout.slot_id or status.logical_name
        camera_state = _get_camera_state(status, session_camera)
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
            f'metrics-calibrated: {metric_state.value}',
            layout.panel_bounds.left + 250,
            layout.panel_bounds.top + 24,
            _metric_calibration_colour(metric_state),
        )
        self._draw_text(
            f'slot: {slot_id}  state: {camera_state.value}',
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
        metrics = self._get_calibration_metrics(status)
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
        if camera_state in {
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
        try:
            frame = self.service.snapshot(slot_id)
            frame_surface = self._frame_surface_converter(frame, self._get_pygame())
            self._render_frame_surface(frame_surface, layout)
        except Exception as ex:  # noqa: BLE001 (A bad frame must not stop the UI loop).
            self._draw_text(
                f'Preview unavailable: {ex}',
                layout.preview_bounds.left + 8,
                layout.preview_bounds.top + 8,
                RED,
            )
        self._draw_preview_border(layout.preview_bounds, area_colour)
        self._draw_uncalibrated_click_frame(slot_id, layout)

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
        return build_camera_preview_layouts(
            statuses,
            self.configuration.window_resolution,
            session_cameras=session_cameras if len(session_cameras) > 0 else None,
        )

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
    ) -> tuple[MetricCalibrationStatus, MetricRulerLike | None]:
        try:
            get_metric_status = getattr(self.service, 'get_metric_status', None)
        except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
            self._last_metric_error = f'Metric status unavailable: {ex}'
            return MetricCalibrationStatus.UNCALIBRATED, None
        if callable(get_metric_status):
            try:
                raw_state = get_metric_status()
            except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
                self._last_metric_error = f'Metric status unavailable: {ex}'
                return MetricCalibrationStatus.UNCALIBRATED, None
        else:
            missing_state = object()
            try:
                raw_state = getattr(self.service, 'metric_state', missing_state)
            except Exception as ex:  # noqa: BLE001 (Status is a display boundary).
                self._last_metric_error = f'Metric status unavailable: {ex}'
                return MetricCalibrationStatus.UNCALIBRATED, None
            if raw_state is missing_state:
                self._last_metric_error = 'Metric status unavailable'
                return MetricCalibrationStatus.UNCALIBRATED, None
        if not isinstance(raw_state, MetricCalibrationStatus):
            self._last_metric_error = 'Metric status unavailable'
            return MetricCalibrationStatus.UNCALIBRATED, None
        if raw_state is not MetricCalibrationStatus.CALIBRATED:
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

    def _synchronise_projector_descriptor(self) -> None:
        descriptor = _get_service_projector_descriptor(self.service)
        if descriptor is None or descriptor == self._projector_output_descriptor:
            return
        self._projector_output_descriptor = descriptor
        self.configuration = replace(
            self.configuration,
            projector_resolution=descriptor.projector_resolution,
            projector_output_descriptor=descriptor,
        )
        try:
            calibration_pattern = getattr(self.service, 'calibration_pattern', None)
        except Exception:  # noqa: BLE001 (A bad service pattern must not stop the UI loop).
            calibration_pattern = None
        self.calibration_pattern = (
            calibration_pattern
            if isinstance(calibration_pattern, CalibrationPattern)
            else None
        )
        if not self._is_initialised:
            return
        self._projector_surface = self._get_pygame().Surface(
            tuple(descriptor.projector_resolution),
        )

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
            if session_camera is None:
                continue
            previous_generation = self._camera_lifecycle_generations.get(camera.slot_id)
            if (
                previous_generation is not None
                and previous_generation != session_camera.lifecycle_generation
            ):
                self._uncalibrated_click_cameras.discard(camera.slot_id)
            self._camera_lifecycle_generations[camera.slot_id] = (
                session_camera.lifecycle_generation
            )

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
    ) -> CalibrationMetrics | None:
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


def _metric_calibration_colour(status: MetricCalibrationStatus) -> tuple[int, int, int]:
    if status is MetricCalibrationStatus.CALIBRATED:
        return GREEN
    if status is MetricCalibrationStatus.STALE:
        return RED
    return ORANGE


def _format_resolution(resolution: Resolution | None) -> str:
    if resolution is None:
        return 'unknown'
    return f'{resolution.width}x{resolution.height}'


def _format_metrics(metrics: CalibrationMetrics) -> str:
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
    'ProjectorOutput',
    'ProjectorOutputFactory',
    'PygameDisplayRuntime',
    'ProjectorRenderer',
    'Sdl2ProjectorOutput',
    'build_camera_preview_layouts',
]
