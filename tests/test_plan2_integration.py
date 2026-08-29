import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.camera import CameraRuntime
from multivision.cli import MultiVisionClient, ServiceResponse, main as cli_main
from multivision.config import Configuration
from multivision.display import DisplayConfiguration, PygameDisplayRuntime
from multivision.errors import CameraUnavailableError
from multivision.geometry import CoordinateBounds, Point2D
from multivision.types import (
    CalibrationStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
)


CAMERA_RESOLUTION = Resolution(640, 480)


class IntegrationCapture:
    def __init__(self, frame_prefix: str) -> None:
        self.frame_prefix = frame_prefix
        self._lock = threading.Lock()
        self._is_released = False
        self._is_disconnected = False
        self.read_count = 0
        self.first_frame = threading.Event()

    @property
    def is_released(self) -> bool:
        with self._lock:
            return self._is_released

    def is_opened(self) -> bool:
        with self._lock:
            return not self._is_released and not self._is_disconnected

    def get_native_resolution(self) -> Resolution:
        return CAMERA_RESOLUTION

    def read(self) -> tuple[bool, object]:
        with self._lock:
            if self._is_released or self._is_disconnected:
                return False, None
            self.read_count += 1
            frame = f'{self.frame_prefix}-{self.read_count}'
            self.first_frame.set()
            return True, frame

    def release(self) -> None:
        with self._lock:
            self._is_released = True

    def disconnect(self) -> None:
        with self._lock:
            self._is_disconnected = True


class IntegrationDiscovery:
    def __init__(self, device_count: int) -> None:
        self.devices = [
            DeviceInfo(
                f'device-{capture_index}',
                f'Camera {capture_index}',
                capture_index=capture_index,
                native_resolution=CAMERA_RESOLUTION,
            )
            for capture_index in range(device_count)
        ]
        self.call_count = 0

    def discover_devices(self) -> list[DeviceInfo]:
        self.call_count += 1
        return list(self.devices)


class IntegrationCaptureFactory:
    def __init__(self) -> None:
        self.captures: dict[str, list[IntegrationCapture]] = {}
        self.opened_device_ids: list[str] = []

    def open_capture(self, device: DeviceInfo) -> IntegrationCapture:
        capture = IntegrationCapture(device.device_id)
        self.captures.setdefault(device.device_id, []).append(capture)
        self.opened_device_ids.append(device.device_id)
        return capture

    def get_capture(self, device_id: str, generation: int = 0) -> IntegrationCapture:
        return self.captures[device_id][generation]


class IntegrationSurface:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def fill(self, _colour: tuple[int, int, int]) -> None:
        return None

    def blit(self, _surface: object, _position: tuple[int, int]) -> None:
        return None


class IntegrationPygame:
    MOUSEBUTTONDOWN = 3
    QUIT = 1
    KEYDOWN = 2
    K_ESCAPE = 27
    Surface = IntegrationSurface

    def __init__(self) -> None:
        self.events: list[object] = []
        self.rectangles: list[tuple[object, ...]] = []
        self.circles: list[tuple[object, ...]] = []
        self.window_surface = IntegrationSurface((800, 600))
        self.display = SimpleNamespace(
            set_mode=lambda _size: self.window_surface,
            set_caption=lambda _caption: None,
            flip=lambda: None,
        )
        self.event = SimpleNamespace(get=self._get_events)
        self.font = SimpleNamespace(
            Font=lambda _name, _size: SimpleNamespace(
                render=lambda *_arguments: IntegrationSurface((1, 1)),
            ),
        )
        self.time = SimpleNamespace(
            Clock=lambda: SimpleNamespace(tick=lambda _rate: None),
        )
        self.draw = SimpleNamespace(
            rect=lambda *arguments: self.rectangles.append(arguments),
            circle=lambda *arguments: self.circles.append(arguments),
        )
        self.transform = SimpleNamespace(
            smoothscale=lambda _surface, size: IntegrationSurface(size),
        )

    def _get_events(self) -> list[object]:
        events = list(self.events)
        self.events.clear()
        return events

    def init(self) -> None:
        return None

    def quit(self) -> None:
        return None


class Plan2RuntimeIntegrationTest(unittest.TestCase):
    def test_gui_api_and_cli_pointing_share_session_geometry_and_overlay_authority(self) -> None:
        discovery = IntegrationDiscovery(2)
        factory = IntegrationCaptureFactory()
        runtime = CameraRuntime(
            discovery,
            factory,
            read_wait_seconds=0.001,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,
        )
        pygame_module = IntegrationPygame()
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(
                window_resolution=Resolution(800, 600),
                projector_resolution=Resolution(1000, 700),
            ),
            pygame_module=pygame_module,
            frame_surface_converter=lambda _frame, _pygame: pygame_module.window_surface,
        )

        def make_calibration(translation_x: float) -> SimpleNamespace:
            return SimpleNamespace(
                camera_to_projector=(
                    (1.0, 0.0, translation_x),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                valid_region=CoordinateBounds(0, 0, 640, 480),
            )

        with TestClient(create_app(service)) as client:
            try:
                assert factory.get_capture('device-0').first_frame.wait(1)
                assert factory.get_capture('device-1').first_frame.wait(1)
                runtime.set_calibration(
                    'camera-0',
                    CalibrationStatus.CALIBRATED,
                    make_calibration(0),
                )
                runtime.set_calibration(
                    'camera-1',
                    CalibrationStatus.CALIBRATED,
                    make_calibration(100),
                )

                def request_sender(
                    method: str,
                    url: str,
                    payload: dict[str, object] | None,
                    timeout_seconds: float,
                ) -> ServiceResponse:
                    del timeout_seconds
                    response = client.request(method, urlsplit(url).path, json=payload)
                    return ServiceResponse(
                        response.status_code,
                        response.headers.get('content-type', ''),
                        response.content,
                    )

                cli_client = MultiVisionClient(
                    'http://service.test',
                    request_sender=request_sender,
                )
                with redirect_stdout(io.StringIO()):
                    assert cli_main(
                        ['cameras', 'rename', 'camera-1', 'side'],
                        cli_client,
                    ) == 0

                display_runtime.render_once()
                layout = display_runtime.preview_layouts['camera-1']
                assert layout.logical_name == 'side', f'{layout=}'
                assert layout.preview_transform is not None
                native_point = Point2D(100, 120)
                content_bounds = layout.preview_transform.content_bounds
                preview_point = Point2D(
                    content_bounds.left + native_point.x * layout.preview_transform.scale,
                    content_bounds.top + native_point.y * layout.preview_transform.scale,
                )
                pygame_module.events.append(
                    SimpleNamespace(
                        type=pygame_module.MOUSEBUTTONDOWN,
                        button=1,
                        pos=(
                            round(layout.preview_bounds.left + preview_point.x),
                            round(layout.preview_bounds.top + preview_point.y),
                        ),
                    ),
                )
                display_runtime.process_events()
                gui_overlay = service.overlay
                assert gui_overlay is not None
                assert gui_overlay.camera_id == 'camera-1', f'{gui_overlay=}'
                assert gui_overlay.logical_name == 'side', f'{gui_overlay=}'
                assert gui_overlay.camera_point == native_point, f'{gui_overlay=}'
                assert gui_overlay.projector_point == Point2D(200, 120), f'{gui_overlay=}'

                display_runtime.render_once()
                assert pygame_module.circles[-1][2] == (200, 120), f'{pygame_module.circles=}'

                api_side_response = client.post(
                    '/overlay/point',
                    json={'camera': 'side', 'x': 100, 'y': 120},
                )
                assert api_side_response.status_code == 200, api_side_response.text
                assert api_side_response.json() == gui_overlay.to_data(), (
                    f'{api_side_response.json()=}, {gui_overlay=}'
                )

                api_other_response = client.post(
                    '/overlay/point',
                    json={'camera': 'camera-0', 'x': 100, 'y': 120},
                )
                assert api_other_response.status_code == 200, api_other_response.text
                assert api_other_response.json()['camera_id'] == 'camera-0'
                assert api_other_response.json()['projector_point'] == [100.0, 120.0]

                with redirect_stdout(io.StringIO()):
                    assert cli_main(
                        ['point', '--camera', 'camera-1', '--x', '100', '--y', '120'],
                        cli_client,
                    ) == 0
                assert service.overlay is not None
                assert service.overlay.camera_id == 'camera-1'
                assert service.overlay.projector_point == Point2D(200, 120)

                with redirect_stdout(io.StringIO()):
                    assert cli_main(['overlay', 'clear'], cli_client) == 0
                assert service.overlay is None
            finally:
                display_runtime.shutdown()

        assert discovery.call_count == 1, f'{discovery.call_count=}'
        assert factory.opened_device_ids == ['device-0', 'device-1'], (
            f'{factory.opened_device_ids=}'
        )

    def test_simultaneous_handles_retain_frames_and_manage_closed_slots(self) -> None:
        discovery = IntegrationDiscovery(5)
        factory = IntegrationCaptureFactory()
        runtime = CameraRuntime(
            discovery,
            factory,
            read_wait_seconds=0.001,
        )
        runtime.start()

        try:
            initial_captures = [
                factory.get_capture(f'device-{capture_index}')
                for capture_index in range(4)
            ]
            assert all(
                capture.first_frame.wait(1)
                for capture in initial_captures
            ), f'{initial_captures=}'
            assert factory.opened_device_ids == [
                'device-0',
                'device-1',
                'device-2',
                'device-3',
            ], f'{factory.opened_device_ids=}'
            assert discovery.call_count == 1, f'{discovery.call_count=}'
            assert [
                camera.state
                for camera in runtime.get_session_cameras()
            ] == [
                SessionCameraState.OPEN,
                SessionCameraState.OPEN,
                SessionCameraState.OPEN,
                SessionCameraState.OPEN,
                SessionCameraState.CLOSED,
            ]

            first_frame = runtime.snapshot('camera-0')
            time.sleep(0.01)
            retained_frame = runtime.snapshot('camera-0')
            assert retained_frame.frame_counter > first_frame.frame_counter, (
                f'{first_frame=}, {retained_frame=}'
            )
            assert all(capture.read_count > 0 for capture in initial_captures), (
                f'{initial_captures=}'
            )

            runtime.close_camera('camera-0')
            closed_read_count = initial_captures[0].read_count
            assert initial_captures[0].is_released
            time.sleep(0.01)
            assert initial_captures[0].read_count == closed_read_count
            with self.assertRaises(CameraUnavailableError):
                runtime.snapshot('camera-0')

            runtime.open_camera('camera-4')
            reopened_capture = factory.get_capture('device-4')
            assert reopened_capture.first_frame.wait(1), 'reopened frame was not captured'
            assert runtime.snapshot('camera-4').data.startswith('device-4-')
            assert factory.opened_device_ids == [
                'device-0',
                'device-1',
                'device-2',
                'device-3',
                'device-4',
            ], f'{factory.opened_device_ids=}'
            assert discovery.call_count == 1, f'{discovery.call_count=}'
        finally:
            runtime.shutdown()

        assert all(
            capture.is_released
            for captures in factory.captures.values()
            for capture in captures
        ), f'{factory.captures=}'

    def test_disconnect_is_explicit_and_api_does_not_substitute_another_camera(self) -> None:
        discovery = IntegrationDiscovery(2)
        factory = IntegrationCaptureFactory()
        service = MultiVisionService(
            Configuration(),
            discovery=discovery,
            capture_factory=factory,
        )

        with TestClient(create_app(service)) as client:
            first_capture = factory.get_capture('device-0')
            second_capture = factory.get_capture('device-1')
            assert first_capture.first_frame.wait(1), 'first camera did not capture'
            assert second_capture.first_frame.wait(1), 'second camera did not capture'

            first_capture.disconnect()
            deadline = time.monotonic() + 1
            while True:
                camera_data = client.get('/cameras').json()
                first_camera = next(
                    camera
                    for camera in camera_data
                    if camera['slot'] == 'camera-0'
                )
                if first_camera['state'] == SessionCameraState.UNAVAILABLE.value:
                    break
                assert time.monotonic() < deadline, f'{camera_data=}'
                time.sleep(0.001)

            assert first_capture.is_released
            assert first_camera['capture_index'] == 0, f'{first_camera=}'
            assert first_camera['frame_metadata'] is None, f'{first_camera=}'
            assert first_camera['runtime_status'] == RuntimeStatus.UNAVAILABLE.value
            assert client.get('/cameras/camera-1/snapshot').json()['data'].startswith(
                'device-1-',
            )
            snapshot_response = client.get('/cameras/camera-0/snapshot')
            assert snapshot_response.status_code == 503
            assert snapshot_response.json()['error']['code'] == 'CAMERA_UNAVAILABLE'
            assert discovery.call_count == 1, f'{discovery.call_count=}'
            assert factory.opened_device_ids == ['device-0', 'device-1'], (
                f'{factory.opened_device_ids=}'
            )
            assert not second_capture.is_released

        assert second_capture.is_released

    def test_cli_management_requests_update_the_running_service_without_restart(self) -> None:
        discovery = IntegrationDiscovery(5)
        factory = IntegrationCaptureFactory()
        service = MultiVisionService(
            Configuration(),
            discovery=discovery,
            capture_factory=factory,
        )

        with TestClient(create_app(service)) as client:
            def request_sender(
                method: str,
                url: str,
                payload: dict[str, object] | None,
                timeout_seconds: float,
            ) -> ServiceResponse:
                del timeout_seconds
                path = urlsplit(url).path
                response = client.request(method, path, json=payload)
                return ServiceResponse(
                    response.status_code,
                    response.headers.get('content-type', ''),
                    response.content,
                )

            cli_client = MultiVisionClient(
                'http://service.test',
                request_sender=request_sender,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                assert cli_main(['cameras', 'list'], cli_client) == 0
                assert cli_main(
                    ['cameras', 'rename', 'camera-0', 'overhead'],
                    cli_client,
                ) == 0
                assert cli_main(['cameras', 'close', 'camera-0'], cli_client) == 0
                assert cli_main(['cameras', 'open', 'camera-4'], cli_client) == 0
            assert len(output.getvalue()) > 0, f'{output.getvalue()=}'

            cameras_response = client.get('/cameras')
            assert cameras_response.status_code == 200
            cameras = cameras_response.json()
            assert cameras[0]['slot'] == 'camera-0'
            assert cameras[0]['name'] == 'overhead'
            assert cameras[0]['state'] == SessionCameraState.CLOSED.value
            assert cameras[1]['state'] == SessionCameraState.OPEN.value
            assert cameras[4]['state'] == SessionCameraState.OPEN.value
            assert service.is_running
            assert discovery.call_count == 1, f'{discovery.call_count=}'
            assert factory.opened_device_ids == [
                'device-0',
                'device-1',
                'device-2',
                'device-3',
                'device-4',
            ], (
                f'{factory.opened_device_ids=}'
            )

        assert all(
            capture.is_released
            for captures in factory.captures.values()
            for capture in captures
        ), f'{factory.captures=}'


if __name__ == '__main__':
    unittest.main()
