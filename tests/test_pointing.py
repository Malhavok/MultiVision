import threading
import unittest
from types import SimpleNamespace

from multivision.calibration import CalibrationMetrics
from multivision.display import RED, DisplayConfiguration, PygameDisplayRuntime
from multivision.errors import (
    CameraUnavailableError,
    InvalidCalibrationStateError,
    InvalidHomographyError,
    PointOutsideCalibratedRegionError,
    PointOutsidePreviewError,
    PointOutsideProjectorError,
)
from multivision.geometry import CoordinateBounds, Point2D, PreviewTransform
from multivision.persistence import PersistedCalibration
from multivision.service import PointOverlayService, RedCircleOverlay
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
)


class FakeCameraRuntime:
    def __init__(self, runtime_status: RuntimeStatus = RuntimeStatus.AVAILABLE) -> None:
        self.status = CameraStatus(
            'overhead',
            'device-a',
            runtime_status,
            CalibrationStatus.CALIBRATED,
            Resolution(800, 600),
        )

    def get_status(self, logical_name: str) -> CameraStatus:
        assert logical_name == 'overhead'
        return self.status

    def get_statuses(self) -> list[CameraStatus]:
        return [self.status]

    def snapshot(self, logical_name: str) -> Frame:
        assert logical_name == 'overhead'
        return Frame('frame', 1, 0.0)


class FakeCalibrationRegistry:
    def __init__(self, status: CalibrationStatus = CalibrationStatus.CALIBRATED) -> None:
        self.status = status
        self.calibrations = {
            'device-a': SimpleNamespace(
                camera_resolution=Resolution(800, 600),
                projector_resolution=Resolution(1000, 700),
            ),
        }
        self.camera_points: list[Point2D] = []
        self.projected_point = Point2D(300, 200)

    def get_status(
        self,
        camera_id: str,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
    ) -> CalibrationStatus:
        assert camera_id == 'device-a'
        record = self.calibrations.get(camera_id)
        if record is None:
            return CalibrationStatus.UNCALIBRATED
        if record.camera_resolution != camera_resolution:
            return CalibrationStatus.STALE
        if record.projector_resolution != projector_resolution:
            return CalibrationStatus.STALE
        return self.status

    def get_status_error_code(
        self,
        camera_id: str,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
    ) -> str:
        record = self.calibrations.get(camera_id)
        if record is None:
            return 'CALIBRATION_UNCALIBRATED'
        if record.camera_resolution != camera_resolution:
            return 'CAMERA_RESOLUTION_CHANGED'
        if record.projector_resolution != projector_resolution:
            return 'PROJECTOR_RESOLUTION_CHANGED'
        return {
            CalibrationStatus.UNCALIBRATED: 'CALIBRATION_UNCALIBRATED',
            CalibrationStatus.UNVERIFIED: 'CALIBRATION_UNVERIFIED',
            CalibrationStatus.STALE: 'CALIBRATION_STALE',
        }[self.status]

    def project_camera_points_to_projector(
        self,
        _camera_id: str,
        points: tuple[Point2D, ...],
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        _projector_output_descriptor: object | None = None,
    ) -> tuple[Point2D, ...]:
        assert camera_resolution == Resolution(800, 600)
        assert projector_resolution == Resolution(1000, 700)
        self.camera_points.extend(points)
        return tuple(self.projected_point for _point in points)


class FakeDisplayService:
    def __init__(
        self,
        camera_runtime: FakeCameraRuntime,
        point_service: PointOverlayService,
    ) -> None:
        self.camera_runtime = camera_runtime
        self.point_service = point_service

    @property
    def overlay(self) -> RedCircleOverlay | None:
        return self.point_service.overlay

    @property
    def calibration_pattern_visible(self) -> bool:
        return False

    def get_camera_statuses(self) -> list[CameraStatus]:
        return self.camera_runtime.get_statuses()

    def get_calibration_metrics(self, _logical_name: str) -> None:
        return None

    def snapshot(self, logical_name: str) -> Frame:
        return self.camera_runtime.snapshot(logical_name)

    def point_from_preview(
        self,
        logical_name: str,
        preview_point: Point2D,
        preview_transform: PreviewTransform,
    ) -> RedCircleOverlay:
        return self.point_service.point_from_preview(
            logical_name,
            preview_point,
            preview_transform,
        )


class BlockingCalibrationRegistry(FakeCalibrationRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.first_projection_started = threading.Event()
        self.release_first_projection = threading.Event()
        self.projection_count = 0

    def project_camera_points_to_projector(
        self,
        camera_id: str,
        points: tuple[Point2D, ...],
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        _projector_output_descriptor: object | None = None,
    ) -> tuple[Point2D, ...]:
        self.projection_count += 1
        projection_number = self.projection_count
        if projection_number == 1:
            self.first_projection_started.set()
            assert self.release_first_projection.wait(1), f'{self.release_first_projection=}'
        super().project_camera_points_to_projector(
            camera_id,
            points,
            camera_resolution,
            projector_resolution,
            _projector_output_descriptor,
        )
        return tuple(
            Point2D(300 + projection_number, 200 + projection_number)
            for _point in points
        )


class PointingTest(unittest.TestCase):
    def test_preview_and_native_callers_share_replacement_and_clear_path(self) -> None:
        camera_runtime = FakeCameraRuntime()
        calibration_registry = FakeCalibrationRegistry()
        service = PointOverlayService(
            camera_runtime,
            calibration_registry,
            Resolution(1000, 700),
        )
        preview_transform = PreviewTransform(
            Resolution(400, 300),
            Resolution(800, 600),
            0.5,
            CoordinateBounds(0.0, 0.0, 400.0, 300.0),
        )

        first_overlay = service.point_from_camera('overhead', (10, 20))
        second_overlay = service.point_from_preview(
            'overhead',
            (100, 100),
            preview_transform,
        )

        assert first_overlay.projector_point == Point2D(300, 200)
        assert second_overlay.camera_point == Point2D(200, 200)
        assert service.overlay is second_overlay
        assert calibration_registry.camera_points == [
            Point2D(10, 20),
            Point2D(200, 200),
        ]
        service.clear_overlay()
        assert service.overlay is None

    def test_multi_point_projection_is_side_effect_free_and_uses_point_authority(self) -> None:
        service = PointOverlayService(
            FakeCameraRuntime(),
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        existing_overlay = service.point_from_camera('overhead', (10, 20))

        projected_points = service.project_camera_points(
            'overhead',
            ((30, 40), (50, 60)),
        )

        assert projected_points == (Point2D(300, 200), Point2D(300, 200))
        assert service.overlay is existing_overlay

    def test_overlay_management_requires_session_identity_not_just_display_name(self) -> None:
        service = PointOverlayService(
            FakeCameraRuntime(),
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        service._overlay = RedCircleOverlay(
            'overhead',
            'camera-1',
            Point2D(10, 20),
            Point2D(300, 200),
        )

        service.rename_overlay_camera('camera-0', 'renamed')
        service.clear_overlay_for_camera('camera-0')

        assert service.overlay is not None
        assert service.overlay.logical_name == 'overhead'
        assert service.overlay.camera_id == 'camera-1'

    def test_session_preview_click_resolves_name_and_uses_only_that_slot_calibration(self) -> None:
        class SessionRuntime:
            def __init__(self) -> None:
                self.registry = SessionCameraRegistry.from_devices(
                    [
                        DeviceInfo(
                            'device-a',
                            'Camera A',
                            capture_index=0,
                            native_resolution=Resolution(800, 600),
                        ),
                        DeviceInfo(
                            'device-b',
                            'Camera B',
                            capture_index=1,
                            native_resolution=Resolution(800, 600),
                        ),
                    ],
                )
                self.statuses = {
                    camera.slot_id: CameraStatus(
                        camera.slot_id,
                        None,
                        RuntimeStatus.AVAILABLE,
                        CalibrationStatus.UNCALIBRATED,
                        Resolution(800, 600),
                    )
                    for camera in self.registry.get_cameras()
                }

            def get_session_cameras(self) -> list[object]:
                return self.registry.get_cameras()

            def get_status(self, slot_id: str) -> CameraStatus:
                return self.statuses[slot_id]

        runtime = SessionRuntime()
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                valid_region=CoordinateBounds(0, 0, 800, 600),
            ),
        )
        runtime.registry.set_calibration(
            'camera-1',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=((1, 0, 100), (0, 1, 100), (0, 0, 1)),
                valid_region=CoordinateBounds(0, 0, 800, 600),
            ),
        )
        service = PointOverlayService(
            runtime,  # type: ignore[arg-type]
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        preview_transform = PreviewTransform(
            Resolution(400, 300),
            Resolution(800, 600),
            0.5,
            CoordinateBounds(0.0, 0.0, 400.0, 300.0),
        )

        first_overlay = service.point_from_preview(
            'camera-1',
            (100, 100),
            preview_transform,
        )
        runtime.registry.rename('camera-1', 'side')
        second_overlay = service.point_from_preview(
            'side',
            (100, 100),
            preview_transform,
        )
        other_overlay = service.point_from_preview(
            'camera-0',
            (100, 100),
            preview_transform,
        )

        assert first_overlay.camera_id == 'camera-1', f'{first_overlay=}'
        assert first_overlay.projector_point == Point2D(300, 300), f'{first_overlay=}'
        assert second_overlay.camera_id == 'camera-1', f'{second_overlay=}'
        assert second_overlay.logical_name == 'side', f'{second_overlay=}'
        assert second_overlay.projector_point == first_overlay.projector_point, (
            f'{second_overlay=}, {first_overlay=}'
        )
        assert other_overlay.camera_id == 'camera-0', f'{other_overlay=}'
        assert other_overlay.projector_point == Point2D(200, 200), f'{other_overlay=}'

    def test_session_calibration_metadata_mismatch_fails_closed(self) -> None:
        class SessionRuntime:
            def __init__(self) -> None:
                self.registry = SessionCameraRegistry.from_devices(
                    [DeviceInfo('device-a', 'Camera A', capture_index=0)],
                )
                self.status = CameraStatus(
                    'camera-0',
                    None,
                    RuntimeStatus.AVAILABLE,
                    CalibrationStatus.CALIBRATED,
                    Resolution(800, 600),
                )

            def get_session_cameras(self) -> list[object]:
                return self.registry.get_cameras()

            def get_status(self, _slot_id: str) -> CameraStatus:
                return self.status

        runtime = SessionRuntime()
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            PersistedCalibration(
                'camera-0',
                Resolution(640, 480),
                Resolution(1000, 700),
                1,
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                CalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 0.5),
                1.0,
                (
                    Point2D(0, 0),
                    Point2D(640, 0),
                    Point2D(640, 480),
                    Point2D(0, 480),
                ),
            ),
        )
        service = PointOverlayService(
            runtime,  # type: ignore[arg-type]
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )

        with self.assertRaises(InvalidCalibrationStateError) as context:
            service.point_from_camera('camera-0', (100, 100))

        assert context.exception.code == 'CAMERA_RESOLUTION_CHANGED'
        assert service.overlay is None

    def test_preview_padding_is_rejected_without_replacing_overlay(self) -> None:
        calibration_registry = FakeCalibrationRegistry()
        service = PointOverlayService(
            FakeCameraRuntime(),
            calibration_registry,
            Resolution(1000, 700),
        )
        existing_overlay = service.point_from_camera('overhead', (10, 20))
        transform = PreviewTransform(
            Resolution(500, 500),
            Resolution(800, 600),
            0.625,
            CoordinateBounds(0.0, 62.5, 500.0, 437.5),
        )

        with self.assertRaises(PointOutsidePreviewError) as context:
            service.point_from_preview('overhead', (20, 20), transform)

        assert context.exception.code == 'POINT_OUTSIDE_PREVIEW'
        assert service.overlay is existing_overlay
        assert len(calibration_registry.camera_points) == 1

    def test_every_calibration_failure_is_explicit(self) -> None:
        for calibration_status, expected_code in [
            (CalibrationStatus.UNCALIBRATED, 'CALIBRATION_UNCALIBRATED'),
            (CalibrationStatus.UNVERIFIED, 'CALIBRATION_UNVERIFIED'),
            (CalibrationStatus.STALE, 'CALIBRATION_STALE'),
        ]:
            with self.subTest(calibration_status=calibration_status):
                service = PointOverlayService(
                    FakeCameraRuntime(),
                    FakeCalibrationRegistry(calibration_status),
                    Resolution(1000, 700),
                )
                with self.assertRaises(InvalidCalibrationStateError) as context:
                    service.point_from_camera('overhead', (10, 20))
                assert context.exception.code == expected_code
                assert service.overlay is None

        unavailable_service = PointOverlayService(
            FakeCameraRuntime(RuntimeStatus.UNAVAILABLE),
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        with self.assertRaises(CameraUnavailableError) as context:
            unavailable_service.point_from_camera('overhead', (10, 20))
        assert context.exception.code == 'CAMERA_UNAVAILABLE'

    def test_malformed_native_points_and_preview_transforms_are_rejected(self) -> None:
        service = PointOverlayService(
            FakeCameraRuntime(),
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        with self.assertRaises(ValueError):
            service.point_from_camera('overhead', (1, 2, 3))
        with self.assertRaises(ValueError):
            service.point_from_camera('overhead', (float('inf'), 2))
        with self.assertRaises(PointOutsideCalibratedRegionError) as context:
            service.point_from_camera('overhead', (-1, 20))
        assert context.exception.code == 'POINT_OUTSIDE_CALIBRATED_REGION'

        malformed_transform = PreviewTransform(
            Resolution(400, 300),
            Resolution(800, 600),
            0.0,
            CoordinateBounds(0.0, 0.0, 400.0, 300.0),
        )
        with self.assertRaises(ValueError):
            service.point_from_preview('overhead', (10, 20), malformed_transform)

    def test_invalid_status_identity_and_metadata_fail_closed(self) -> None:
        camera_runtime = FakeCameraRuntime()
        camera_runtime.status = CameraStatus(
            'side-left',
            'device-a',
            RuntimeStatus.AVAILABLE,
            CalibrationStatus.CALIBRATED,
            Resolution(800, 600),
        )
        service = PointOverlayService(
            camera_runtime,
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        with self.assertRaises(CameraUnavailableError):
            service.point_from_camera('overhead', (10, 20))

        camera_runtime.status = CameraStatus(
            'overhead',
            '',
            RuntimeStatus.AVAILABLE,
            CalibrationStatus.CALIBRATED,
            Resolution(800, 600),
        )
        with self.assertRaises(CameraUnavailableError):
            service.point_from_camera('overhead', (10, 20))

    def test_concurrent_points_are_serialised_and_latest_success_replaces(self) -> None:
        registry = BlockingCalibrationRegistry()
        service = PointOverlayService(
            FakeCameraRuntime(),
            registry,
            Resolution(1000, 700),
        )
        errors: list[BaseException] = []

        def point_first() -> None:
            try:
                service.point_from_camera('overhead', (10, 20))
            except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                errors.append(ex)

        def point_second() -> None:
            try:
                service.point_from_camera('overhead', (30, 40))
            except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                errors.append(ex)

        first_thread = threading.Thread(target=point_first)
        second_thread = threading.Thread(target=point_second)
        first_thread.start()
        assert registry.first_projection_started.wait(1), f'{registry.first_projection_started=}'
        second_thread.start()
        assert not registry.release_first_projection.is_set()
        registry.release_first_projection.set()
        first_thread.join(2)
        second_thread.join(2)

        assert not first_thread.is_alive(), f'{first_thread=}'
        assert not second_thread.is_alive(), f'{second_thread=}'
        assert errors == [], f'{errors=}'
        assert registry.projection_count == 2, f'{registry.projection_count=}'
        assert service.overlay is not None
        assert service.overlay.projector_point == Point2D(302, 202)

    def test_resolution_homography_region_and_projector_failures_are_explicit(self) -> None:
        camera_runtime = FakeCameraRuntime()
        registry = FakeCalibrationRegistry()
        registry.calibrations['device-a'].camera_resolution = Resolution(640, 480)
        service = PointOverlayService(camera_runtime, registry, Resolution(1000, 700))
        with self.assertRaises(InvalidCalibrationStateError) as context:
            service.point_from_camera('overhead', (10, 20))
        assert context.exception.code == 'CAMERA_RESOLUTION_CHANGED'

        registry = FakeCalibrationRegistry()
        registry.calibrations['device-a'].projector_resolution = Resolution(900, 700)
        service = PointOverlayService(camera_runtime, registry, Resolution(1000, 700))
        with self.assertRaises(InvalidCalibrationStateError) as context:
            service.point_from_camera('overhead', (10, 20))
        assert context.exception.code == 'PROJECTOR_RESOLUTION_CHANGED'

        registry = FakeCalibrationRegistry()
        registry.project_camera_points_to_projector = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PointOutsideCalibratedRegionError('outside region'),
            )
        )
        with self.assertRaises(PointOutsideCalibratedRegionError) as context:
            PointOverlayService(camera_runtime, registry, Resolution(1000, 700)).point_from_camera(
                'overhead',
                (10, 20),
            )
        assert context.exception.code == 'POINT_OUTSIDE_CALIBRATED_REGION'

        registry = FakeCalibrationRegistry()
        registry.projected_point = Point2D(float('nan'), 20)
        with self.assertRaises(InvalidHomographyError):
            PointOverlayService(camera_runtime, registry, Resolution(1000, 700)).point_from_camera(
                'overhead',
                (10, 20),
            )

        registry = FakeCalibrationRegistry()
        registry.projected_point = Point2D(1000, 20)
        with self.assertRaises(PointOutsideProjectorError):
            PointOverlayService(camera_runtime, registry, Resolution(1000, 700)).point_from_camera(
                'overhead',
                (10, 20),
            )


class FakePygame:
    MOUSEBUTTONDOWN = 3
    QUIT = 1
    KEYDOWN = 2
    K_ESCAPE = 27
    def __init__(self) -> None:
        self.events: list[object] = []
        self.circles: list[tuple[object, object, object, object]] = []
        self.rectangles: list[tuple[object, ...]] = []
        self.window_surface = SimpleNamespace(fill=lambda _colour: None, blit=lambda *_args: None)
        self.Surface = lambda _size: SimpleNamespace(
            fill=lambda _colour: None,
            blit=lambda *_args: None,
        )
        self.display = SimpleNamespace(
            set_mode=lambda _size: self.window_surface,
            set_caption=lambda _caption: None,
            flip=lambda: None,
        )
        self.event = SimpleNamespace(get=lambda: self.events)
        self.font = SimpleNamespace(
            Font=lambda _name, _size: SimpleNamespace(
                render=lambda *_args: self.window_surface,
            ),
        )
        self.time = SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda _rate: None))
        self.draw = SimpleNamespace(
            rect=lambda *args: self.rectangles.append(args),
            circle=lambda *args: self.circles.append(args),
        )
        self.transform = SimpleNamespace(smoothscale=lambda *_args: self.window_surface)

    def init(self) -> None:
        pass

    def quit(self) -> None:
        pass


class DisplayPointingTest(unittest.TestCase):
    def test_uncalibrated_preview_click_draws_only_a_persistent_camera_frame(self) -> None:
        pygame_module = FakePygame()
        camera_runtime = FakeCameraRuntime()
        camera_runtime.status = camera_runtime.status._replace(
            calibration_status=CalibrationStatus.UNCALIBRATED,
        )
        registry = FakeCalibrationRegistry(CalibrationStatus.UNCALIBRATED)
        service = PointOverlayService(camera_runtime, registry, Resolution(1000, 700))
        display_runtime = PygameDisplayRuntime(
            FakeDisplayService(camera_runtime, service),
            DisplayConfiguration(
                window_resolution=Resolution(500, 400),
                projector_resolution=Resolution(1000, 700),
            ),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: pygame_module.window_surface,
        )

        display_runtime.render_once()
        layout = display_runtime.preview_layouts['overhead']
        pygame_module.events.append(
            SimpleNamespace(
                type=pygame_module.MOUSEBUTTONDOWN,
                button=1,
                pos=(1, 1),
            ),
        )
        display_runtime.process_events()
        display_runtime.render_once()
        assert not any(
            len(arguments) == 4 and arguments[1] == RED
            for arguments in pygame_module.rectangles
        ), f'{pygame_module.rectangles=}'

        pygame_module.events.append(
            SimpleNamespace(
                type=pygame_module.MOUSEBUTTONDOWN,
                button=1,
                pos=(
                    round(layout.preview_bounds.left + 100),
                    round(layout.preview_bounds.top + 100),
                ),
            ),
        )
        display_runtime.process_events()
        display_runtime.render_once()
        red_frames = [
            arguments
            for arguments in pygame_module.rectangles
            if len(arguments) == 4 and arguments[1] == RED
        ]

        assert display_runtime.last_point_error is not None
        assert display_runtime.last_point_error.startswith('CALIBRATION_UNCALIBRATED:')
        assert service.overlay is None
        assert len(red_frames) == 1, f'{red_frames=}'
        assert red_frames[0][2] == (
            round(layout.preview_bounds.left),
            round(layout.preview_bounds.top),
            round(layout.preview_bounds.right - layout.preview_bounds.left),
            round(layout.preview_bounds.bottom - layout.preview_bounds.top),
        ), f'{red_frames=}'

        display_runtime.render_once()
        persistent_red_frames = [
            arguments
            for arguments in pygame_module.rectangles
            if len(arguments) == 4 and arguments[1] == RED
        ]
        assert len(persistent_red_frames) == 2, f'{persistent_red_frames=}'
        red_frame_count = len(pygame_module.rectangles)
        camera_runtime.status = camera_runtime.status._replace(
            calibration_status=CalibrationStatus.CALIBRATED,
        )
        registry.status = CalibrationStatus.CALIBRATED
        display_runtime.render_once()
        assert not any(
            len(arguments) == 4 and arguments[1] == RED
            for arguments in pygame_module.rectangles[red_frame_count:]
        ), f'{pygame_module.rectangles=}'

    def test_gui_click_failure_is_reported_without_stopping_event_loop(self) -> None:
        pygame_module = FakePygame()
        camera_runtime = FakeCameraRuntime(RuntimeStatus.UNAVAILABLE)
        service = PointOverlayService(
            camera_runtime,
            FakeCalibrationRegistry(),
            Resolution(1000, 700),
        )
        display_runtime = PygameDisplayRuntime(
            FakeDisplayService(camera_runtime, service),
            DisplayConfiguration(
                window_resolution=Resolution(500, 400),
                projector_resolution=Resolution(1000, 700),
            ),
            pygame_module=pygame_module,
        )

        display_runtime.render_once()
        layout = display_runtime.preview_layouts['overhead']
        pygame_module.events.append(
            SimpleNamespace(
                type=pygame_module.MOUSEBUTTONDOWN,
                button=1,
                pos=(
                    round(layout.preview_bounds.left + 100),
                    round(layout.preview_bounds.top + 100),
                ),
            ),
        )

        display_runtime.process_events()

        assert display_runtime.last_point_error is not None
        assert display_runtime.last_point_error.startswith('CAMERA_UNAVAILABLE:')
        assert service.overlay is None

    def test_gui_click_uses_shared_service_and_renders_overlay(self) -> None:
        pygame_module = FakePygame()
        camera_runtime = FakeCameraRuntime()
        registry = FakeCalibrationRegistry()
        service = PointOverlayService(camera_runtime, registry, Resolution(1000, 700))
        display_runtime = PygameDisplayRuntime(
            FakeDisplayService(camera_runtime, service),
            DisplayConfiguration(
                window_resolution=Resolution(500, 400),
                projector_resolution=Resolution(1000, 700),
            ),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: pygame_module.window_surface,
        )

        display_runtime.render_once()
        layout = display_runtime.preview_layouts['overhead']
        assert layout.preview_transform is not None
        click_x = round(layout.preview_bounds.left + 100)
        click_y = round(layout.preview_bounds.top + 100)
        pygame_module.events.append(
            SimpleNamespace(
                type=pygame_module.MOUSEBUTTONDOWN,
                button=1,
                pos=(click_x, click_y),
            ),
        )
        display_runtime.process_events()
        display_runtime.render_once()

        assert service.overlay is not None
        assert service.overlay.logical_name == 'overhead'
        assert len(pygame_module.circles) == 1
        assert pygame_module.circles[0][2] == (300, 200)


if __name__ == '__main__':
    unittest.main()
