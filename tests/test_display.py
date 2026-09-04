import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from multivision.application import CameraArea, CameraRenderSnapshot
from multivision.calibration import CalibrationMetrics
from multivision.config import ProjectorOutputDescriptor
from multivision.display import (
    BLACK,
    DARK_GREY,
    METRIC_RULER_COLOUR,
    DisplayConfiguration,
    PygameDisplayRuntime,
    ProjectorRenderer,
    Sdl2ProjectorOutput,
    build_camera_preview_layouts,
)
from multivision.geometry import CoordinateBounds, Point2D
from multivision.metric import MetricCalibrationStatus, build_metric_ruler
from multivision.overlays import (
    OverlayStyle,
    ProjectorLabel,
    apply_overlay_intensity_to_colour,
    ProjectorMaterialisation,
    ProjectorPolygon,
    ProjectorSegment,
)
from multivision.pattern import build_calibration_pattern
from multivision.service import RedCircleOverlay
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
)


class FakeSurface:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.fills: list[tuple[int, int, int]] = []
        self.blits: list[tuple[object, tuple[int, int]]] = []

    def fill(self, colour: tuple[int, int, int]) -> None:
        self.fills.append(colour)

    def blit(self, surface: object, position: tuple[int, int]) -> None:
        self.blits.append((surface, position))


class FakeProjectorOutput:
    def __init__(self) -> None:
        self.presented_surfaces: list[FakeSurface] = []
        self.shutdown_called = False

    def present(self, surface: FakeSurface) -> None:
        self.presented_surfaces.append(surface)

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeFont:
    def __init__(self, rendered_text: list[str]) -> None:
        self.rendered_text = rendered_text

    def render(
        self,
        text: str,
        _antialias: bool,
        _colour: tuple[int, int, int],
    ) -> FakeSurface:
        self.rendered_text.append(text)
        return FakeSurface((1, 1))


class FakePygame:
    Surface = FakeSurface
    QUIT = 1
    KEYDOWN = 2
    K_ESCAPE = 27

    def __init__(self) -> None:
        self.window_surface = FakeSurface((1280, 720))
        self.rendered_text: list[str] = []
        self.font_sizes: list[int] = []
        self.display = SimpleNamespace(
            set_mode=lambda _size: self.window_surface,
            set_caption=lambda _caption: None,
            flip=lambda: None,
        )
        def make_font(_name: object, size: int) -> FakeFont:
            self.font_sizes.append(size)
            return FakeFont(self.rendered_text)

        self.font = SimpleNamespace(Font=make_font)
        self.time = SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda _rate: None))
        self.event = SimpleNamespace(get=lambda: [])
        self.draw_calls: list[tuple[str, tuple[object, ...]]] = []
        self.draw = SimpleNamespace(
            rect=lambda *arguments: self.draw_calls.append(('rect', arguments)),
            polygon=lambda *arguments: self.draw_calls.append(('polygon', arguments)),
            line=lambda *arguments: self.draw_calls.append(('line', arguments)),
            circle=lambda *arguments: self.draw_calls.append(('circle', arguments)),
        )
        self.transform = SimpleNamespace(
            smoothscale=lambda _surface, size: FakeSurface(size),
            rotozoom=lambda _surface, _angle, scale: FakeSurface(
                (round(scale), round(scale)),
            ),
        )
        self.initialise_count = 0
        self.quit_count = 0
        self.Surface = FakeSurface

    def init(self) -> None:
        self.initialise_count += 1

    def quit(self) -> None:
        self.quit_count += 1


class FakeCameraRuntime:
    def __init__(self) -> None:
        self.statuses = [
            CameraStatus(
                'overhead',
                'overhead-device',
                RuntimeStatus.AVAILABLE,
                CalibrationStatus.CALIBRATED,
                Resolution(1920, 1080),
            ),
        ]
        self.snapshot_count = 0
        self.open_count = 0
        self.calibration_metrics: CalibrationMetrics | None = None
        self.calibration_pattern_visible = False
        self.overlay = None

    def get_camera_statuses(self) -> list[CameraStatus]:
        return self.statuses

    def get_calibration_metrics(self, _logical_name: str) -> CalibrationMetrics | None:
        return self.calibration_metrics

    def point_from_preview(
        self,
        _logical_name: str,
        _preview_point: object,
        _preview_transform: object,
    ) -> object:
        raise AssertionError('pointing is not used by this fake')

    def snapshot(self, logical_name: str) -> Frame:
        assert logical_name == 'overhead'
        self.snapshot_count += 1
        return Frame(f'frame-{self.snapshot_count}', self.snapshot_count, 0.0)


class AreaDisplayService(FakeCameraRuntime):
    def __init__(self, areas: list[CameraArea]) -> None:
        super().__init__()
        self.areas = areas

    def get_camera_areas(self) -> list[CameraArea]:
        return self.areas


class MetricDisplayService(AreaDisplayService):
    def __init__(self, ruler: object | None, areas: list[CameraArea]) -> None:
        super().__init__(areas)
        self.metric_ruler = ruler
        self.metric_state = MetricCalibrationStatus.CALIBRATED
        self.metric_capture_active = False
        self.metric_capture_presented_count = 0

    def mark_metric_capture_presented(self) -> None:
        self.metric_capture_presented_count += 1


class SessionDisplayService:
    def __init__(self, camera_count: int) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    f'device-{idx}',
                    f'Camera {idx}',
                    capture_index=idx,
                    native_resolution=Resolution(640, 480),
                )
                for idx in range(camera_count)
            ],
        )
        self.snapshot_requests: list[str] = []

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_camera_statuses(self) -> list[CameraStatus]:
        statuses: list[CameraStatus] = []
        for camera in self.registry.get_cameras():
            assert camera.device_info is not None
            runtime_status = {
                SessionCameraState.OPEN: RuntimeStatus.AVAILABLE,
                SessionCameraState.CLOSED: RuntimeStatus.STOPPED,
                SessionCameraState.UNAVAILABLE: RuntimeStatus.UNAVAILABLE,
            }[camera.state]
            frame_counter = (
                0
                if camera.frame_metadata is None
                else camera.frame_metadata.frame_counter
            )
            statuses.append(
                CameraStatus(
                    camera.slot_id,
                    None,
                    runtime_status,
                    camera.calibration_status,
                    camera.device_info.native_resolution,
                    frame_counter,
                ),
            )
        return statuses

    def snapshot(self, slot_id: str) -> Frame:
        self.snapshot_requests.append(slot_id)
        return Frame(slot_id, len(self.snapshot_requests), 0.0)

    def get_calibration_metrics(self, _slot_id: str) -> CalibrationMetrics | None:
        return None

    def point_from_preview(
        self,
        _slot_id: str,
        _preview_point: object,
        _preview_transform: object,
    ) -> object:
        raise AssertionError('pointing is not used by this fake')

    @property
    def calibration_pattern_visible(self) -> bool:
        return False

    @property
    def overlay(self) -> None:
        return None


class DisplayTest(unittest.TestCase):
    def test_render_clears_stale_window_content_between_snapshot_frames(self) -> None:
        pygame_module = FakePygame()
        service = FakeCameraRuntime()
        display_runtime = PygameDisplayRuntime(
            service,
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()
        service.statuses = []
        display_runtime.render_once()

        assert pygame_module.window_surface.fills == [DARK_GREY, DARK_GREY], (
            f'{pygame_module.window_surface.fills=}'
        )

    def test_snapshot_preview_layouts_follow_stable_slot_order(self) -> None:
        descriptor = ProjectorOutputDescriptor(Resolution(100, 80))
        statuses = (
            CameraStatus(
                'zulu',
                'device-0',
                RuntimeStatus.AVAILABLE,
                CalibrationStatus.CALIBRATED,
                Resolution(640, 480),
            ),
            CameraStatus(
                'alpha',
                'device-1',
                RuntimeStatus.AVAILABLE,
                CalibrationStatus.CALIBRATED,
                Resolution(640, 480),
            ),
        )
        snapshot = SimpleNamespace(
            overlays=(),
            projector_output_descriptor=descriptor,
            global_overlay_intensity=1.0,
            protected_projector_regions=(),
            camera_snapshots=(
                CameraRenderSnapshot(
                    'camera-0',
                    'zulu',
                    statuses[0],
                    SessionCameraState.OPEN,
                    0,
                    Frame(object(), 1, 0.0),
                ),
                CameraRenderSnapshot(
                    'camera-1',
                    'alpha',
                    statuses[1],
                    SessionCameraState.OPEN,
                    0,
                    Frame(object(), 1, 0.0),
                ),
            ),
            projector_areas=(),
            metric_state=MetricCalibrationStatus.UNCALIBRATED,
            metric_ruler=None,
            point_overlay=None,
            calibration_pattern_visible=False,
            metric_capture_active=False,
            calibration_pattern=None,
        )

        class SnapshotService:
            def get_render_snapshot(self) -> object:
                return snapshot

        display_runtime = PygameDisplayRuntime(
            SnapshotService(),  # type: ignore[arg-type]
            DisplayConfiguration(
                window_resolution=Resolution(1000, 700),
                projector_resolution=Resolution(100, 80),
            ),
            pygame_module=FakePygame(),
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
            projector_output=FakeProjectorOutput(),
        )
        display_runtime.render_once()

        assert display_runtime.preview_layouts['camera-0'].panel_bounds.left == 0
        assert display_runtime.preview_layouts['camera-1'].panel_bounds.left == 500

    def test_lifecycle_rebuild_resets_low_rate_preview_cadence(self) -> None:
        pygame_module = FakePygame()
        service = SessionDisplayService(1)
        clock_seconds = [0.0]
        converted_frames: list[Frame] = []
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(
                window_resolution=Resolution(500, 400),
                preview_mode='low_rate',
                preview_low_rate_hz=10.0,
            ),
            pygame_module=pygame_module,
            frame_surface_converter=lambda frame, _pygame: (
                converted_frames.append(frame) or FakeSurface((1, 1))
            ),
            preview_clock=lambda: clock_seconds[0],
        )

        display_runtime.render_once()
        clock_seconds[0] = 0.01
        service.registry.close('camera-0')
        service.registry.open('camera-0')
        display_runtime.render_once()

        assert len(converted_frames) == 2, f'{converted_frames=}'

    def test_snapshot_rendering_is_single_read_and_preview_mode_is_independent(self) -> None:
        class SnapshotService:
            projector_output_descriptor = ProjectorOutputDescriptor(Resolution(100, 80))
            calibration_pattern = None
            overlay = None

            def __init__(self) -> None:
                self.snapshot_calls = 0
                self.forbidden_calls: list[str] = []
                self.render_snapshot = SimpleNamespace(
                    overlays=(),
                    projector_output_descriptor=self.projector_output_descriptor,
                    global_overlay_intensity=1.0,
                    protected_projector_regions=(),
                    camera_snapshots=(
                        CameraRenderSnapshot(
                            'camera-0',
                            'overhead',
                            CameraStatus(
                                'camera-0',
                                'device-0',
                                RuntimeStatus.AVAILABLE,
                                CalibrationStatus.CALIBRATED,
                                Resolution(640, 480),
                            ),
                            SessionCameraState.OPEN,
                            0,
                            Frame(object(), 1, 0.0),
                        ),
                    ),
                    projector_areas=(),
                    metric_state=MetricCalibrationStatus.UNCALIBRATED,
                    metric_ruler=None,
                    point_overlay=None,
                    calibration_pattern_visible=False,
                    metric_capture_active=False,
                    calibration_pattern=None,
                )

            def get_render_snapshot(self) -> object:
                self.snapshot_calls += 1
                return self.render_snapshot

            def __getattr__(self, name: str) -> object:
                if name in {
                    'snapshot',
                    'get_camera_statuses',
                    'get_camera_areas',
                    'get_calibration_metrics',
                }:
                    self.forbidden_calls.append(name)
                    raise AssertionError(f'Unexpected display service call: {name}')
                raise AttributeError(name)

        for preview_mode, expected_conversion_count in (
            ('active', 3),
            ('low_rate', 2),
            ('off', 0),
        ):
            service = SnapshotService()
            clock_seconds = [0.0]
            converted_frames: list[Frame] = []

            def convert_frame(frame: Frame, _pygame: object) -> FakeSurface:
                converted_frames.append(frame)
                return FakeSurface((1, 1))

            display_runtime = PygameDisplayRuntime(
                service,  # type: ignore[arg-type]
                DisplayConfiguration(
                    window_resolution=Resolution(500, 400),
                    projector_resolution=Resolution(100, 80),
                    preview_mode=preview_mode,
                    preview_low_rate_hz=10.0,
                ),
                pygame_module=FakePygame(),
                frame_surface_converter=convert_frame,
                projector_output=FakeProjectorOutput(),
                preview_clock=lambda: clock_seconds[0],
            )
            display_runtime.render_once()
            clock_seconds[0] = 0.01
            display_runtime.render_once()
            clock_seconds[0] = 0.11
            display_runtime.render_once()

            assert len(converted_frames) == expected_conversion_count, (
                f'{preview_mode=}, {converted_frames=}'
            )
            assert service.snapshot_calls == 3, (
                f'{preview_mode=}, {service.snapshot_calls=}'
            )
            assert service.forbidden_calls == [], f'{service.forbidden_calls=}'

    def test_legacy_projector_layers_respect_protected_regions(self) -> None:
        pygame_module = FakePygame()
        renderer = ProjectorRenderer(pygame_module)
        surface = FakeSurface((200, 100))
        protected_regions = (
            (
                Point2D(40, 40),
                Point2D(160, 40),
                Point2D(160, 60),
                Point2D(40, 60),
            ),
        )
        area = CameraArea(
            'camera-0',
            'overhead',
            True,
            (Point2D(40, 40), Point2D(160, 40), Point2D(160, 60)),
            (70, 190, 255),
        )
        ruler = build_metric_ruler(
            (50.0, 50.0),
            (150.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )

        renderer.render_areas(
            surface,
            [area],
            protected_regions=protected_regions,
        )
        renderer.render_metric_ruler(
            surface,
            ruler,
            protected_regions=protected_regions,
        )
        renderer.render_overlay(
            surface,
            RedCircleOverlay(
                'overhead',
                'overhead-device',
                Point2D(1, 2),
                Point2D(100, 50),
            ),
            protected_regions=protected_regions,
        )

        assert pygame_module.draw_calls == [], f'{pygame_module.draw_calls=}'

    def test_global_intensity_applies_to_legacy_projector_layers(self) -> None:
        area = CameraArea(
            'camera-0',
            'overhead',
            True,
            (Point2D(10, 10), Point2D(40, 10), Point2D(40, 30)),
            (70, 190, 255),
        )
        area_pygame = FakePygame()
        ProjectorRenderer(area_pygame).render_areas(
            FakeSurface((100, 80)),
            [area],
            FakeFont(area_pygame.rendered_text),
            intensity=0.5,
        )
        assert area_pygame.draw_calls[0][1][1] == apply_overlay_intensity_to_colour(
            area.area_colour,
            0.5,
        ), f'{area_pygame.draw_calls=}'

        ruler_pygame = FakePygame()
        ruler = build_metric_ruler(
            (20.0, 20.0),
            (60.0, 20.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(100, 80),
        )
        ProjectorRenderer(ruler_pygame).render_metric_ruler(
            FakeSurface((100, 80)),
            ruler,
            FakeFont(ruler_pygame.rendered_text),
            intensity=0.5,
        )
        ruler_colour = apply_overlay_intensity_to_colour(
            METRIC_RULER_COLOUR,
            0.5,
        )
        assert len(ruler_pygame.draw_calls) > 0, f'{ruler_pygame.draw_calls=}'
        assert all(
            call[1][1] == ruler_colour
            for call in ruler_pygame.draw_calls
        ), f'{ruler_pygame.draw_calls=}'

        overlay_pygame = FakePygame()
        ProjectorRenderer(overlay_pygame).render_overlay(
            FakeSurface((100, 80)),
            RedCircleOverlay(
                'overhead',
                'overhead-device',
                Point2D(20, 20),
                Point2D(10, 10),
            ),
            intensity=0.5,
        )
        assert overlay_pygame.draw_calls[0][1][1] == apply_overlay_intensity_to_colour(
            (255, 0, 0),
            0.5,
        ), f'{overlay_pygame.draw_calls=}'

    def test_renderer_preserves_overlay_layers_and_visibility(self) -> None:
        pygame_module = FakePygame()
        renderer = ProjectorRenderer(pygame_module)
        surface = FakeSurface((100, 80))
        style = OverlayStyle(colour='#123456', line_width_px=2)
        shape_overlay = SimpleNamespace(
            kind='rect',
            visible=True,
            insertion_sequence=1,
            materialised_primitives=ProjectorMaterialisation(
                polygons=(
                    ProjectorPolygon(
                        (Point2D(10, 10), Point2D(20, 10), Point2D(20, 20)),
                        style,
                    ),
                ),
            ),
        )
        line_overlay = SimpleNamespace(
            kind='line',
            visible=True,
            insertion_sequence=0,
            materialised_primitives=ProjectorMaterialisation(
                segments=(ProjectorSegment(Point2D(1, 1), Point2D(5, 5), style),),
                labels=(ProjectorLabel(Point2D(3, 3), 'line', style),),
            ),
        )
        hidden_overlay = SimpleNamespace(
            kind='circle',
            visible=False,
            insertion_sequence=2,
            materialised_primitives=ProjectorMaterialisation(
                polygons=(
                    ProjectorPolygon(
                        (Point2D(30, 30), Point2D(40, 30), Point2D(40, 40)),
                        style,
                    ),
                ),
            ),
        )

        renderer.render_generic_overlays(
            surface,
            [line_overlay, hidden_overlay, shape_overlay],
            FakeFont(pygame_module.rendered_text),
        )

        assert [call[0] for call in pygame_module.draw_calls] == ['polygon', 'line']
        assert pygame_module.rendered_text == ['line'], (
            f'{pygame_module.rendered_text=}'
        )
        assert len(surface.blits) == 1, f'{surface.blits=}'

    def test_generic_labels_apply_rotation_and_scale_before_blitting(self) -> None:
        pygame_module = FakePygame()
        renderer = ProjectorRenderer(pygame_module)
        style = OverlayStyle(colour='#123456', line_width_px=1)
        overlay = SimpleNamespace(
            kind='text',
            visible=True,
            insertion_sequence=0,
            materialised_primitives=ProjectorMaterialisation(
                labels=(ProjectorLabel(Point2D(50, 40), 'one\ntwo', style, 30, 2),),
            ),
        )

        renderer.render_generic_overlays(
            FakeSurface((100, 80)),
            [overlay],
            FakeFont(pygame_module.rendered_text),
            'label',
        )

        assert pygame_module.rendered_text == ['one', 'two']

    def test_generic_overlays_are_suppressed_during_calibration_and_blank_capture(self) -> None:
        class GenericOverlayService(FakeCameraRuntime):
            def __init__(self) -> None:
                super().__init__()
                self.metric_capture_active = False
                self.generic_overlays: list[object] = []

            def list_overlays(self) -> list[object]:
                return self.generic_overlays

        pygame_module = FakePygame()
        service = GenericOverlayService()
        style = OverlayStyle(colour='#102030', line_width_px=2)
        service.generic_overlays = [
            SimpleNamespace(
                kind='line',
                visible=True,
                insertion_sequence=0,
                materialised_primitives=ProjectorMaterialisation(
                    segments=(
                        ProjectorSegment(Point2D(2, 3), Point2D(20, 30), style),
                    ),
                ),
            ),
        ]
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(
                window_resolution=Resolution(500, 400),
                projector_resolution=Resolution(100, 80),
            ),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
            projector_output=FakeProjectorOutput(),
        )

        display_runtime.render_once()
        normal_line_count = len(
            [
                call
                for call in pygame_module.draw_calls
                if call[0] == 'line'
            ],
        )
        service.calibration_pattern_visible = True
        display_runtime.render_once()
        service.calibration_pattern_visible = False
        service.metric_capture_active = True
        display_runtime.render_once()

        assert normal_line_count == 1, f'{pygame_module.draw_calls=}'
        assert len(
            [call for call in pygame_module.draw_calls if call[0] == 'line'],
        ) == normal_line_count, f'{pygame_module.draw_calls=}'

    def test_session_previews_follow_slot_order_and_rebuild_after_lifecycle_changes(self) -> None:
        pygame_module = FakePygame()
        service = SessionDisplayService(5)
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(window_resolution=Resolution(1000, 700)),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()

        assert list(display_runtime.preview_layouts) == [
            'camera-0',
            'camera-1',
            'camera-2',
            'camera-3',
            'camera-4',
        ]
        assert service.snapshot_requests == [
            'camera-0',
            'camera-1',
            'camera-2',
            'camera-3',
        ], f'{service.snapshot_requests=}'
        assert 'slot: camera-0  state: OPEN' in pygame_module.rendered_text

        service.registry.rename('camera-1', 'overhead')
        service.registry.close('camera-0')
        display_runtime.render_once()

        assert display_runtime.preview_layouts['camera-1'].logical_name == 'overhead'
        assert service.snapshot_requests[-3:] == [
            'camera-1',
            'camera-2',
            'camera-3',
        ], f'{service.snapshot_requests=}'
        assert 'overhead  connection: AVAILABLE' in pygame_module.rendered_text
        assert 'slot: camera-0  state: CLOSED' in pygame_module.rendered_text

        service.registry.open('camera-4')
        display_runtime.render_once()
        assert service.snapshot_requests[-4:] == [
            'camera-1',
            'camera-2',
            'camera-3',
            'camera-4',
        ], f'{service.snapshot_requests=}'

    def test_lifecycle_change_clears_uncalibrated_click_frame_before_reopen(self) -> None:
        display_runtime = PygameDisplayRuntime(
            SessionDisplayService(1),  # type: ignore[arg-type]
            DisplayConfiguration(window_resolution=Resolution(500, 400)),
            pygame_module=FakePygame(),
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )
        display_runtime.render_once()
        display_runtime._uncalibrated_click_cameras.add('camera-0')

        service = display_runtime.service
        service.registry.close('camera-0')
        service.registry.open('camera-0')
        display_runtime.render_once()

        assert 'camera-0' not in display_runtime._uncalibrated_click_cameras

    def test_layout_keeps_native_resolution_out_of_window_geometry(self) -> None:
        layouts = build_camera_preview_layouts(
            [
                CameraStatus(
                    'overhead',
                    'device',
                    RuntimeStatus.AVAILABLE,
                    CalibrationStatus.UNCALIBRATED,
                    Resolution(1920, 1080),
                ),
            ],
            Resolution(1000, 700),
        )

        layout = layouts['overhead']
        assert layout.preview_transform is not None
        assert layout.preview_transform.camera_resolution == Resolution(1920, 1080)
        assert layout.preview_transform.preview_size != Resolution(1920, 1080)

    def test_run_presents_the_projector_surface_and_shuts_it_down(self) -> None:
        pygame_module = FakePygame()
        projector_output = FakeProjectorOutput()
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
            projector_output=projector_output,
        )

        display_runtime.run(max_frames=1)
        display_runtime.shutdown()

        assert len(projector_output.presented_surfaces) == 1
        assert projector_output.presented_surfaces[0] is not None
        assert projector_output.shutdown_called

    def test_render_uses_latest_snapshot_without_opening_camera(self) -> None:
        pygame_module = FakePygame()
        camera_runtime = FakeCameraRuntime()
        converted_frames: list[str] = []

        def convert_frame(frame: Frame, _pygame_module: object) -> FakeSurface:
            converted_frames.append(frame.data)
            return FakeSurface((1920, 1080))

        calibration_metrics = CalibrationMetrics(9, 36, 34, 34 / 36, 1.2, 3.4, 0.8)
        camera_runtime.calibration_metrics = calibration_metrics
        display_runtime = PygameDisplayRuntime(
            camera_runtime,
            DisplayConfiguration(
                window_resolution=Resolution(1000, 700),
                projector_resolution=Resolution(1200, 800),
            ),
            pygame_module=pygame_module,
            frame_surface_converter=convert_frame,
        )
        display_runtime.render_once()
        display_runtime.render_once()

        assert converted_frames == ['frame-1', 'frame-2'], f'{converted_frames=}'
        assert camera_runtime.open_count == 0, f'{camera_runtime.open_count=}'
        assert 'overhead  connection: AVAILABLE' in pygame_module.rendered_text
        assert 'calibration: CALIBRATED' in pygame_module.rendered_text
        assert 'native: 1920x1080' in pygame_module.rendered_text
        assert any(
            text.startswith('tags: 9  corners: 36  inliers: 34')
            for text in pygame_module.rendered_text
        ), f'{pygame_module.rendered_text=}'

        display_runtime.shutdown()
        assert pygame_module.quit_count == 1, f'{pygame_module.quit_count=}'
        assert camera_runtime.open_count == 0, f'{camera_runtime.open_count=}'

    def test_layout_order_is_deterministic_and_duplicate_names_are_rejected(self) -> None:
        statuses = [
            CameraStatus(
                'side-left',
                'left-device',
                RuntimeStatus.UNAVAILABLE,
                CalibrationStatus.UNCALIBRATED,
                Resolution(1280, 720),
            ),
            CameraStatus(
                'overhead',
                'overhead-device',
                RuntimeStatus.UNAVAILABLE,
                CalibrationStatus.UNCALIBRATED,
                Resolution(1280, 720),
            ),
        ]
        forward_layouts = build_camera_preview_layouts(statuses, Resolution(1000, 700))
        reverse_layouts = build_camera_preview_layouts(
            list(reversed(statuses)),
            Resolution(1000, 700),
        )

        assert forward_layouts == reverse_layouts, (
            f'{forward_layouts=}, {reverse_layouts=}'
        )
        duplicate_statuses = statuses + [statuses[0]]
        with self.assertRaises(ValueError):
            build_camera_preview_layouts(duplicate_statuses, Resolution(1000, 700))
        with self.assertRaises(ValueError):
            build_camera_preview_layouts(
                statuses,
                Resolution(1000, 700),
                session_cameras=SessionCameraRegistry.from_capture_indexes(
                    [0],
                ).get_cameras(),
            )

    def test_tiny_window_keeps_preview_bounds_valid(self) -> None:
        layouts = build_camera_preview_layouts(
            [
                CameraStatus(
                    'overhead',
                    'device',
                    RuntimeStatus.UNAVAILABLE,
                    CalibrationStatus.UNCALIBRATED,
                    Resolution(1920, 1080),
                ),
            ],
            Resolution(1, 1),
        )

        preview_bounds = layouts['overhead'].preview_bounds
        assert preview_bounds.right > preview_bounds.left, f'{preview_bounds=}'
        assert preview_bounds.bottom > preview_bounds.top, f'{preview_bounds=}'

    def test_malformed_event_does_not_stop_event_processing(self) -> None:
        pygame_module = FakePygame()
        pygame_module.event = SimpleNamespace(get=lambda: [SimpleNamespace()])
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
        )

        display_runtime.process_events()

    def test_error_status_keeps_the_latest_usable_preview(self) -> None:
        pygame_module = FakePygame()
        camera_runtime = FakeCameraRuntime()
        camera_runtime.statuses[0] = camera_runtime.statuses[0]._replace(
            runtime_status=RuntimeStatus.ERROR,
            error_message='temporary read failure',
        )
        converted_frames: list[str] = []

        def convert_frame(frame: Frame, _pygame: object) -> FakeSurface:
            converted_frames.append(frame.data)
            return FakeSurface((1, 1))

        display_runtime = PygameDisplayRuntime(
            camera_runtime,
            pygame_module=pygame_module,
            frame_surface_converter=convert_frame,
        )

        display_runtime.render_once()

        assert converted_frames == ['frame-1'], f'{converted_frames=}'

    def test_frame_rendering_failure_is_visible_and_retryable(self) -> None:
        pygame_module = FakePygame()
        conversion_count = 0

        def convert_frame(_frame: Frame, _pygame: object) -> FakeSurface:
            nonlocal conversion_count
            conversion_count += 1
            if conversion_count == 1:
                raise RuntimeError('temporary display failure')
            return FakeSurface((1, 1))

        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
            frame_surface_converter=convert_frame,
        )

        display_runtime.render_once()
        display_runtime.render_once()
        assert conversion_count == 2, f'{conversion_count=}'
        assert any(
            text.startswith('Preview unavailable: temporary display failure')
            for text in pygame_module.rendered_text
        ), f'{pygame_module.rendered_text=}'

    def test_calibration_pattern_is_hidden_without_calibration_request(self) -> None:
        pygame_module = FakePygame()
        pattern = build_calibration_pattern(Resolution(1200, 800))
        marker_count = 0

        def render_marker(
            _family: str,
            _marker_id: int,
            _pixel_size: int,
            _pygame_module: object,
        ) -> FakeSurface:
            nonlocal marker_count
            marker_count += 1
            return FakeSurface((_pixel_size, _pixel_size))

        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            DisplayConfiguration(projector_resolution=Resolution(1200, 800)),
            calibration_pattern=pattern,
            pygame_module=pygame_module,
            marker_image_renderer=render_marker,
        )

        display_runtime.render_once()

        assert marker_count == 0, f'{marker_count=}'

    def test_projector_rendering_failure_is_visible_and_retryable(self) -> None:
        pygame_module = FakePygame()
        pattern = build_calibration_pattern(Resolution(1200, 800))
        marker_count = 0

        def render_marker(
            _family: str,
            _marker_id: int,
            _pixel_size: int,
            _pygame_module: object,
        ) -> FakeSurface:
            nonlocal marker_count
            marker_count += 1
            if marker_count == 1:
                raise RuntimeError('temporary marker failure')
            return FakeSurface((_pixel_size, _pixel_size))

        camera_runtime = FakeCameraRuntime()
        camera_runtime.calibration_pattern_visible = True
        display_runtime = PygameDisplayRuntime(
            camera_runtime,
            DisplayConfiguration(projector_resolution=Resolution(1200, 800)),
            calibration_pattern=pattern,
            pygame_module=pygame_module,
            marker_image_renderer=render_marker,
        )

        display_runtime.render_once()
        display_runtime.render_once()
        assert marker_count == len(pattern.markers) + 1, f'{marker_count=}'
        assert any(
            text.startswith('Projector unavailable: temporary marker failure')
            for text in pygame_module.rendered_text
        ), f'{pygame_module.rendered_text=}'

    def test_projector_window_is_destroyed_if_renderer_initialisation_fails(self) -> None:
        from pygame._sdl2 import video

        class FakeWindow:
            def __init__(self) -> None:
                self.destroyed = False

            def destroy(self) -> None:
                self.destroyed = True

        class FailingRenderer:
            def __init__(self, _window: FakeWindow) -> None:
                raise RuntimeError('temporary renderer failure')

        window = FakeWindow()
        with (
            patch.object(video, 'Window', return_value=window),
            patch.object(video, 'Renderer', FailingRenderer),
        ):
            with self.assertRaisesRegex(RuntimeError, 'temporary renderer failure'):
                Sdl2ProjectorOutput(Resolution(1200, 800))

        assert window.destroyed, f'{window.destroyed=}'

    def test_projector_window_targets_the_second_display(self) -> None:
        from pygame._sdl2 import video

        class FakeWindow:
            def __init__(self) -> None:
                self.fullscreen_calls: list[bool] = []
                self.show_count = 0
                self.restore_count = 0
                self.focus_count = 0

            def destroy(self) -> None:
                return None

            def show(self) -> None:
                self.show_count += 1

            def restore(self) -> None:
                self.restore_count += 1

            def set_fullscreen(self, *, desktop: bool) -> None:
                self.fullscreen_calls.append(desktop)

            def focus(self) -> None:
                self.focus_count += 1

        window_arguments: list[dict[str, object]] = []
        created_windows: list[FakeWindow] = []

        def make_window(*_args: object, **kwargs: object) -> FakeWindow:
            window_arguments.append(kwargs)
            window = FakeWindow()
            created_windows.append(window)
            return window

        class FakeRenderer:
            def __init__(self, _window: FakeWindow) -> None:
                return None

        with (
            patch.object(video, 'Window', side_effect=make_window),
            patch.object(video, 'Renderer', FakeRenderer),
        ):
            Sdl2ProjectorOutput(
                Resolution(1200, 800),
                window_resolution=Resolution(800, 600),
                fullscreen=True,
            )

        assert window_arguments[0]['size'] == (800, 600), f'{window_arguments=}'
        assert window_arguments[0]['position'] == (
            video.WINDOWPOS_CENTERED + 1,
            video.WINDOWPOS_CENTERED + 1,
        ), f'{window_arguments=}'
        assert created_windows[0].fullscreen_calls == [True], (
            f'{created_windows[0].fullscreen_calls=}'
        )
        assert created_windows[0].show_count == 2, f'{created_windows[0].show_count=}'
        assert created_windows[0].restore_count == 1, (
            f'{created_windows[0].restore_count=}'
        )
        assert created_windows[0].focus_count == 1, f'{created_windows[0].focus_count=}'

    def test_early_initialisation_failure_shuts_down_injected_projector(self) -> None:
        pygame_module = FakePygame()
        projector_output = FakeProjectorOutput()

        def fail_initialisation() -> None:
            raise RuntimeError('temporary Pygame initialisation failure')

        pygame_module.init = fail_initialisation  # type: ignore[method-assign]
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
            projector_output=projector_output,
        )

        with self.assertRaisesRegex(RuntimeError, 'temporary Pygame initialisation failure'):
            display_runtime.initialise()

        assert projector_output.shutdown_called, f'{projector_output=}'
        assert pygame_module.quit_count == 1, f'{pygame_module.quit_count=}'

    def test_initialisation_failure_cleans_up_for_retry(self) -> None:
        pygame_module = FakePygame()
        font_call_count = 0

        def make_font(_name: object, _size: int) -> FakeFont:
            nonlocal font_call_count
            font_call_count += 1
            if font_call_count == 1:
                raise RuntimeError('temporary font failure')
            return FakeFont(pygame_module.rendered_text)

        pygame_module.font = SimpleNamespace(Font=make_font)
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
        )

        with self.assertRaisesRegex(RuntimeError, 'temporary font failure'):
            display_runtime.initialise()
        assert not display_runtime.is_running, f'{display_runtime.is_running=}'
        assert display_runtime.window_surface is None
        assert display_runtime.projector_surface is None
        assert pygame_module.quit_count == 1, f'{pygame_module.quit_count=}'

        display_runtime.initialise()
        assert display_runtime.window_surface is pygame_module.window_surface
        assert pygame_module.initialise_count == 2, f'{pygame_module.initialise_count=}'

    def test_initialisation_cleans_up_when_interrupted(self) -> None:
        pygame_module = FakePygame()

        def interrupt_font(_name: object, _size: int) -> FakeFont:
            raise KeyboardInterrupt()

        pygame_module.font = SimpleNamespace(Font=interrupt_font)
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
        )

        with self.assertRaises(KeyboardInterrupt):
            display_runtime.initialise()

        assert not display_runtime.is_running, f'{display_runtime.is_running=}'
        assert display_runtime.window_surface is None
        assert display_runtime.projector_surface is None
        assert pygame_module.quit_count == 1, f'{pygame_module.quit_count=}'

    def test_projector_area_rendering_uses_slot_order_names_and_colours(self) -> None:
        pygame_module = FakePygame()
        font = FakeFont(pygame_module.rendered_text)
        surface = FakeSurface((1200, 800))
        areas = [
            CameraArea(
                'camera-1',
                'side-left',
                True,
                (Point2D(20, 30), Point2D(200, 30), Point2D(200, 150)),
                (255, 180, 70),
            ),
            CameraArea(
                'camera-0',
                'overhead',
                True,
                (Point2D(10, 20), Point2D(100, 20), Point2D(100, 120)),
                (70, 190, 255),
            ),
            CameraArea('camera-2', 'disabled', False, None, (180, 100, 255)),
        ]

        ProjectorRenderer(pygame_module).render_areas(surface, areas, font)

        assert [call[0] for call in pygame_module.draw_calls] == [
            'polygon',
            'polygon',
        ], f'{pygame_module.draw_calls=}'
        assert [call[1][1] for call in pygame_module.draw_calls] == [
            (70, 190, 255),
            (255, 180, 70),
        ], f'{pygame_module.draw_calls=}'
        assert pygame_module.rendered_text == ['overhead', 'side-left'], (
            f'{pygame_module.rendered_text=}'
        )
        assert [position for _surface, position in surface.blits] == [
            (10, 20),
            (20, 30),
        ], f'{surface.blits=}'

    def test_projector_area_default_font_is_three_times_the_ui_size(self) -> None:
        pygame_module = FakePygame()
        surface = FakeSurface((1200, 800))
        area = CameraArea(
            'camera-0',
            'overhead',
            True,
            (Point2D(10, 20), Point2D(200, 20), Point2D(200, 150)),
            (70, 190, 255),
        )

        ProjectorRenderer(pygame_module).render_areas(surface, [area])

        assert pygame_module.font_sizes == [48], f'{pygame_module.font_sizes=}'

    def test_metric_ruler_renderer_draws_projector_primitives_and_explicit_label(self) -> None:
        pygame_module = FakePygame()
        surface = FakeSurface((200, 100))
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )

        ProjectorRenderer(pygame_module).render_metric_ruler(surface, ruler)

        assert [call[0] for call in pygame_module.draw_calls] == [
            'polygon',
            'polygon',
            'line',
            *(['line'] * len(ruler.ticks)),
        ], f'{pygame_module.draw_calls=}'
        assert pygame_module.rendered_text == ['60.0 mm'], (
            f'{pygame_module.rendered_text=}'
        )
        assert not any(call[0] == 'circle' for call in pygame_module.draw_calls)
        for call in pygame_module.draw_calls:
            points = call[1][2] if call[0] == 'polygon' else call[1][2:4]
            assert all(
                0 <= x_pos < surface.size[0] and 0 <= y_pos < surface.size[1]
                for x_pos, y_pos in points
            ), f'{call=}'
        assert surface.blits[0][1] == (50, 50), f'{surface.blits=}'

    def test_metric_ruler_renderer_rejects_non_raster_safe_primitives(self) -> None:
        pygame_module = FakePygame()
        surface = FakeSurface((200, 100))
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )._replace(projector_start=Point2D(-1.0, 50.0))

        with self.assertRaises(ValueError):
            ProjectorRenderer(pygame_module).render_metric_ruler(surface, ruler)

        assert pygame_module.draw_calls == [], f'{pygame_module.draw_calls=}'
        assert surface.blits == [], f'{surface.blits=}'

    def test_metric_ruler_renderer_is_main_thread_only(self) -> None:
        pygame_module = FakePygame()
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'cm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        errors: list[BaseException] = []

        def render_from_worker() -> None:
            try:
                ProjectorRenderer(pygame_module).render_metric_ruler(
                    FakeSurface((200, 100)),
                    ruler,
                )
            except BaseException as ex:
                errors.append(ex)

        worker = threading.Thread(target=render_from_worker)
        worker.start()
        worker.join()

        assert len(errors) == 1, f'{errors=}'
        assert isinstance(errors[0], RuntimeError), f'{errors=}'

    def test_projector_ruler_is_ordered_between_areas_and_point_overlay(self) -> None:
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        service = MetricDisplayService(
            ruler,
            [
                CameraArea(
                    'camera-0',
                    'overhead',
                    True,
                    (Point2D(10, 20), Point2D(100, 20), Point2D(100, 80)),
                    (70, 190, 255),
                ),
            ],
        )
        service.overlay = RedCircleOverlay(
            'overhead',
            'overhead-device',
            Point2D(1, 2),
            Point2D(150, 50),
        )
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()

        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert projector_draw_calls[0][0] == 'polygon', f'{projector_draw_calls=}'
        assert projector_draw_calls[1][0:1] == ('polygon',), f'{projector_draw_calls=}'
        assert projector_draw_calls[3][0] == 'line', f'{projector_draw_calls=}'
        assert projector_draw_calls[-1][0] == 'circle', f'{projector_draw_calls=}'

    def test_projector_normal_layers_are_suppressed_during_metric_blank_capture(self) -> None:
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        service = MetricDisplayService(
            ruler,
            [
                CameraArea(
                    'camera-0',
                    'overhead',
                    True,
                    (Point2D(10, 20), Point2D(100, 20), Point2D(100, 80)),
                    (70, 190, 255),
                ),
            ],
        )
        service.overlay = RedCircleOverlay(
            'overhead',
            'overhead-device',
            Point2D(1, 2),
            Point2D(150, 50),
        )
        service.metric_capture_active = True
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()

        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert projector_draw_calls == [], f'{projector_draw_calls=}'
        assert display_runtime.projector_surface.fills == [BLACK]
        assert service.metric_capture_presented_count == 1

    def test_projector_ruler_is_suppressed_during_camera_calibration(self) -> None:
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'in',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        service = MetricDisplayService(ruler, [])
        service.calibration_pattern_visible = True
        pygame_module = FakePygame()
        pattern = build_calibration_pattern(Resolution(200, 100))
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            calibration_pattern=pattern,
            pygame_module=pygame_module,
            marker_image_renderer=lambda _family, _id, size, _pygame: FakeSurface(
                (size, size),
            ),
        )

        display_runtime.render_once()

        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert projector_draw_calls == [], f'{projector_draw_calls=}'
        assert '2.4 in' not in pygame_module.rendered_text, (
            f'{pygame_module.rendered_text=}'
        )

    def test_metric_state_is_fail_closed_when_missing_or_stale(self) -> None:
        ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        service = MetricDisplayService(ruler, [])
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
        )

        del service.metric_state
        display_runtime.render_once()
        assert '60.0 mm' not in pygame_module.rendered_text

        service.metric_state = MetricCalibrationStatus.STALE
        display_runtime.render_once()
        assert '60.0 mm' not in pygame_module.rendered_text

    def test_malformed_metric_ruler_does_not_clear_other_projector_layers(self) -> None:
        service = MetricDisplayService(
            build_metric_ruler(
                (20.0, 50.0),
                (80.0, 50.0),
                'mm',
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                Resolution(200, 100),
            )._replace(projector_start=Point2D(float('nan'), 50.0)),
            [],
        )
        service.overlay = RedCircleOverlay(
            'overhead',
            'overhead-device',
            Point2D(1, 2),
            Point2D(150, 50),
        )
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
        )

        display_runtime.render_once()

        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert [call[0] for call in projector_draw_calls] == ['circle']
        assert 'Metric ruler unavailable' in pygame_module.rendered_text

    def test_metric_render_failure_preserves_point_overlay(self) -> None:
        service = MetricDisplayService(
            build_metric_ruler(
                (20.0, 50.0),
                (80.0, 50.0),
                'mm',
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                Resolution(200, 100),
            ),
            [],
        )
        service.overlay = RedCircleOverlay(
            'overhead',
            'overhead-device',
            Point2D(1, 2),
            Point2D(150, 50),
        )
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
        )

        with patch(
            'multivision.display.ProjectorRenderer.render_metric_ruler',
            side_effect=RuntimeError('malformed ruler'),
        ):
            display_runtime.render_once()

        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert [call[0] for call in projector_draw_calls] == ['circle']
        assert 'Metric ruler unavailable: malformed ruler' in pygame_module.rendered_text
        assert 'Projector unavailable: malformed ruler' not in pygame_module.rendered_text

    def test_projector_descriptor_change_updates_surface_and_suppresses_stale_ruler(self) -> None:
        service = MetricDisplayService(
            build_metric_ruler(
                (20.0, 50.0),
                (80.0, 50.0),
                'mm',
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                Resolution(200, 100),
            ),
            [],
        )
        service.projector_output_descriptor = ProjectorOutputDescriptor(
            Resolution(200, 100),
            'projector-a',
        )
        service.metric_state = MetricCalibrationStatus.CALIBRATED
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
        )
        display_runtime.render_once()
        rendered_text_before_change = list(pygame_module.rendered_text)

        service.projector_output_descriptor = ProjectorOutputDescriptor(
            Resolution(300, 150),
            'projector-b',
        )
        service.metric_state = MetricCalibrationStatus.STALE
        display_runtime.render_once()

        assert display_runtime.projector_output_descriptor == service.projector_output_descriptor
        assert display_runtime.projector_surface.size == (300, 150)
        new_rendered_text = pygame_module.rendered_text[len(rendered_text_before_change):]
        assert '60.0 mm' not in new_rendered_text

    def test_projector_output_rebuilds_when_snapshot_output_changes(self) -> None:
        class SnapshotService:
            def __init__(self) -> None:
                self.projector_output_descriptor = ProjectorOutputDescriptor(
                    Resolution(100, 80),
                    'projector-a',
                )
                self.calibration_pattern_visible = False
                self.created_outputs: list[FakeProjectorOutput] = []

            def get_camera_statuses(self) -> list[CameraStatus]:
                return []

            def get_camera_areas(self) -> list[CameraArea]:
                return []

            def list_overlays(self) -> list[object]:
                return []

            def get_metric_status(self) -> MetricCalibrationStatus:
                return MetricCalibrationStatus.UNCALIBRATED

            @property
            def metric_capture_active(self) -> bool:
                return False

            @property
            def metric_ruler(self) -> None:
                return None

            @property
            def overlay(self) -> None:
                return None

            def get_calibration_metrics(self, _logical_name: str) -> None:
                return None

            def snapshot(self, _logical_name: str) -> Frame:
                return Frame(object(), 1, 0.0)

            def point_from_preview(
                self,
                _logical_name: str,
                _preview_point: Point2D,
                _preview_transform: object,
            ) -> None:
                return None

        service = SnapshotService()
        outputs: list[FakeProjectorOutput] = []

        def make_output(_pygame: object, _configuration: DisplayConfiguration) -> FakeProjectorOutput:
            output = FakeProjectorOutput()
            outputs.append(output)
            return output

        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(projector_resolution=Resolution(100, 80)),
            pygame_module=FakePygame(),
            projector_output_factory=make_output,
        )
        display_runtime.render_once()
        first_output = outputs[0]
        service.projector_output_descriptor = ProjectorOutputDescriptor(
            Resolution(120, 90),
            'projector-b',
        )
        display_runtime.render_once()

        assert len(outputs) == 2, f'{outputs=}'
        assert first_output.shutdown_called, f'{first_output.shutdown_called=}'
        assert display_runtime.projector_surface.size == (120, 90)

    def test_malformed_render_snapshot_fails_closed_and_remains_retryable(self) -> None:
        class SnapshotService:
            projector_output_descriptor = ProjectorOutputDescriptor(Resolution(100, 80))

            def __init__(self) -> None:
                self.snapshots = [
                    SimpleNamespace(
                        overlays=(),
                        projector_output_descriptor=self.projector_output_descriptor,
                        camera_snapshots=None,
                    ),
                    SimpleNamespace(
                        overlays=(),
                        projector_output_descriptor=self.projector_output_descriptor,
                        camera_snapshots=(),
                    ),
                ]

            def get_render_snapshot(self) -> object:
                return self.snapshots.pop(0)

        service = SnapshotService()
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            pygame_module=pygame_module,
            projector_output=FakeProjectorOutput(),
        )

        display_runtime.render_once()
        assert any(
            text.startswith('Render snapshot unavailable')
            for text in pygame_module.rendered_text
        ), f'{pygame_module.rendered_text=}'
        display_runtime.render_once()

        assert display_runtime.projector_surface.fills == [BLACK, BLACK]

    def test_projector_output_failure_does_not_stop_the_next_frame(self) -> None:
        class FlakyProjectorOutput(FakeProjectorOutput):
            def __init__(self) -> None:
                super().__init__()
                self.present_count = 0

            def present(self, surface: FakeSurface) -> None:
                self.present_count += 1
                if self.present_count == 1:
                    raise RuntimeError('temporary projector failure')
                super().present(surface)

        pygame_module = FakePygame()
        projector_output = FlakyProjectorOutput()
        display_runtime = PygameDisplayRuntime(
            FakeCameraRuntime(),
            pygame_module=pygame_module,
            projector_output=projector_output,
        )

        display_runtime.run(max_frames=2)

        assert projector_output.present_count == 2
        assert len(projector_output.presented_surfaces) == 1
        assert 'Projector unavailable: temporary projector failure' in pygame_module.rendered_text

    def test_projector_ruler_replacement_and_clear_are_deterministic(self) -> None:
        first_ruler = build_metric_ruler(
            (20.0, 50.0),
            (80.0, 50.0),
            'mm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        second_ruler = build_metric_ruler(
            (30.0, 40.0),
            (90.0, 40.0),
            'cm',
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            Resolution(200, 100),
        )
        service = MetricDisplayService(first_ruler, [])
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(200, 100)),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()
        first_frame_calls = list(pygame_module.draw_calls)
        display_runtime.render_once()
        repeated_frame_calls = pygame_module.draw_calls[len(first_frame_calls):]
        service.metric_ruler = second_ruler
        display_runtime.render_once()
        second_frame_calls = pygame_module.draw_calls[
            len(first_frame_calls) + len(repeated_frame_calls):
        ]
        service.metric_ruler = None
        display_runtime.render_once()
        clear_frame_calls = pygame_module.draw_calls[
            len(first_frame_calls)
            + len(repeated_frame_calls)
            + len(second_frame_calls):
        ]

        assert repeated_frame_calls == first_frame_calls, (
            f'{repeated_frame_calls=}, {first_frame_calls=}'
        )
        assert second_frame_calls != first_frame_calls, f'{second_frame_calls=}'
        clear_projector_calls = [
            call
            for call in clear_frame_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert clear_projector_calls == [], f'{clear_projector_calls=}'
        assert display_runtime.projector_surface.fills == [BLACK, BLACK, BLACK, BLACK]

    def test_camera_preview_border_uses_projector_area_colour(self) -> None:
        pygame_module = FakePygame()
        service = AreaDisplayService(
            [
                CameraArea(
                    'overhead',
                    'overhead',
                    True,
                    (Point2D(10, 20), Point2D(200, 20), Point2D(200, 150)),
                    (70, 190, 255),
                ),
            ],
        )
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(window_resolution=Resolution(1000, 700)),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()

        assert any(
            call[0] == 'rect'
            and call[1][1] == (70, 190, 255)
            and call[1][3] == 3
            for call in pygame_module.draw_calls
        ), f'{pygame_module.draw_calls=}'

    def test_projector_areas_are_separate_from_point_overlay_and_stale_areas_disappear(self) -> None:
        area = CameraArea(
            'camera-0',
            'overhead',
            True,
            (Point2D(10, 20), Point2D(200, 20), Point2D(200, 150)),
            (70, 190, 255),
        )
        service = AreaDisplayService([area])
        service.overlay = RedCircleOverlay(
            'overhead',
            'overhead-device',
            Point2D(1, 2),
            Point2D(300, 400),
        )
        pygame_module = FakePygame()
        display_runtime = PygameDisplayRuntime(
            service,
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: FakeSurface((1, 1)),
        )

        display_runtime.render_once()
        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert [call[0] for call in projector_draw_calls] == [
            'polygon',
            'circle',
        ], f'{projector_draw_calls=}'

        service.areas = []
        display_runtime.render_once()
        projector_draw_calls = [
            call
            for call in pygame_module.draw_calls
            if call[1][0] is display_runtime.projector_surface
        ]
        assert [call[0] for call in projector_draw_calls] == [
            'polygon',
            'circle',
            'circle',
        ], f'{projector_draw_calls=}'
        assert display_runtime.projector_surface.fills == [BLACK, BLACK]

    def test_projector_areas_are_suppressed_while_calibration_pattern_is_visible(self) -> None:
        area = CameraArea(
            'camera-0',
            'overhead',
            True,
            (Point2D(10, 20), Point2D(200, 20), Point2D(200, 150)),
            (70, 190, 255),
        )
        service = AreaDisplayService([area])
        service.calibration_pattern_visible = True
        pygame_module = FakePygame()
        pattern = build_calibration_pattern(Resolution(1200, 800))
        display_runtime = PygameDisplayRuntime(
            service,
            DisplayConfiguration(projector_resolution=Resolution(1200, 800)),
            calibration_pattern=pattern,
            pygame_module=pygame_module,
            marker_image_renderer=lambda _family, _id, size, _pygame: FakeSurface(
                (size, size),
            ),
        )

        display_runtime.render_once()

        assert not any(
            call[0] == 'polygon'
            for call in pygame_module.draw_calls
        ), f'{pygame_module.draw_calls=}'
        assert 'overhead' not in pygame_module.rendered_text
        assert 'side-left' not in pygame_module.rendered_text
        assert display_runtime.projector_surface.fills == [
            (235, 235, 235),
        ]

    def test_projector_pattern_rendering_is_independent_of_camera_runtime(self) -> None:
        pygame_module = FakePygame()
        marker_calls: list[tuple[str, int, int]] = []

        def render_marker(
            family: str,
            marker_id: int,
            pixel_size: int,
            _pygame_module: object,
        ) -> FakeSurface:
            marker_calls.append((family, marker_id, pixel_size))
            return FakeSurface((pixel_size, pixel_size))

        usable_area = CoordinateBounds(50, 50, 1150, 750)
        pattern = build_calibration_pattern(
            Resolution(1200, 800),
            usable_area=usable_area,
        )
        projector_surface = FakeSurface((1200, 800))
        ProjectorRenderer(pygame_module, render_marker).render_calibration_pattern(
            projector_surface,
            pattern,
        )

        assert len(marker_calls) == len(pattern.markers), f'{marker_calls=}'
        assert projector_surface.fills == [(235, 235, 235)], f'{projector_surface.fills=}'
        assert len(projector_surface.blits) == len(pattern.markers), f'{projector_surface.blits=}'
        assert all(
            isinstance(marker_surface, FakeSurface)
            and position[0] >= usable_area.left
            and position[1] >= usable_area.top
            and position[0] + marker_surface.size[0] <= usable_area.right
            and position[1] + marker_surface.size[1] <= usable_area.bottom
            for marker_surface, position in projector_surface.blits
        ), f'{projector_surface.blits=}'
        assert pattern == build_calibration_pattern(
            Resolution(1200, 800),
            usable_area=CoordinateBounds(50, 50, 1150, 750),
        )


if __name__ == '__main__':
    unittest.main()
