import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import (
    CALIBRATION_PATTERN_EDGE_MARGIN_PIXELS,
    MultiVisionService,
)
from multivision.camera import CameraRuntime
from multivision.cli import MultiVisionClient, ServiceResponse, main as cli_main
from multivision.config import Configuration, ProjectorOutputDescriptor
from multivision.display import BLACK, WHITE, DisplayConfiguration, PygameDisplayRuntime
from multivision.fiducials import DetectedMarker
from multivision.geometry import CoordinateBounds, Point2D, project_point
from multivision.metric_target import METRIC_TARGET
from multivision.pattern import CalibrationPattern, build_calibration_pattern
from multivision.types import DeviceInfo, Resolution


PROJECTOR_RESOLUTION = Resolution(640, 480)
CAMERA_RESOLUTION = Resolution(640, 480)
IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class IntegrationCapture:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.is_released = False
        self.read_count = 0

    def is_opened(self) -> bool:
        return not self.is_released

    def get_native_resolution(self) -> Resolution:
        return CAMERA_RESOLUTION

    def read(self) -> tuple[bool, str]:
        if self.is_released:
            return False, ''
        self.read_count += 1
        return True, f'{self.device_id}-{self.read_count}'

    def release(self) -> None:
        self.is_released = True


class IntegrationDiscovery:
    def __init__(self) -> None:
        self.devices = [
            DeviceInfo(
                f'device-{camera_index}',
                f'Camera {camera_index}',
                camera_index,
                CAMERA_RESOLUTION,
            )
            for camera_index in range(2)
        ]
        self.call_count = 0

    def discover_devices(self) -> list[DeviceInfo]:
        self.call_count += 1
        return list(self.devices)


class IntegrationCaptureFactory:
    def __init__(self) -> None:
        self.captures: list[IntegrationCapture] = []

    def open_capture(self, device: DeviceInfo) -> IntegrationCapture:
        capture = IntegrationCapture(device.device_id)
        self.captures.append(capture)
        return capture


class IntegrationDetector:
    dictionary_name = METRIC_TARGET.marker_family

    def __init__(self, pattern: CalibrationPattern) -> None:
        self.pattern = pattern
        self.mode = 'pattern'
        self.metric_source_to_camera = IDENTITY_MATRIX
        self.detected_modes: list[str] = []

    def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
        self.detected_modes.append(self.mode)
        markers = self.pattern.markers if self.mode == 'pattern' else METRIC_TARGET.markers
        return tuple(
            DetectedMarker(
                marker.marker_id,
                tuple(
                    project_point(corner, self.metric_source_to_camera)
                    for corner in marker.corners
                ) if self.mode == 'metric' else marker.corners,
            )
            for marker in markers
        )


class IntegrationSleepInhibitor:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class IntegrationSurface:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.fills: list[tuple[int, int, int]] = []
        self.blits: list[tuple[object, tuple[int, int]]] = []
        self.commands: list[tuple[str, tuple[object, ...]]] = []

    def fill(self, colour: tuple[int, int, int]) -> None:
        self.fills.append(colour)
        self.commands.append(('fill', (colour,)))

    def blit(self, surface: object, position: tuple[int, int]) -> None:
        self.blits.append((surface, position))
        self.commands.append(('blit', (surface, position)))


class IntegrationProjectorOutput:
    def __init__(self) -> None:
        self.presented_commands: list[list[tuple[str, tuple[object, ...]]]] = []
        self.shutdown_called = False

    def present(self, surface: IntegrationSurface) -> None:
        self.presented_commands.append(list(surface.commands))
        surface.commands.clear()

    def shutdown(self) -> None:
        self.shutdown_called = True


class IntegrationPygame:
    MOUSEBUTTONDOWN = 3
    QUIT = 1
    KEYDOWN = 2
    K_ESCAPE = 27
    Surface = IntegrationSurface

    def __init__(self) -> None:
        self.events: list[object] = []
        self.draw_calls: list[tuple[str, tuple[object, ...]]] = []
        self.rendered_text: list[str] = []
        self.window_surface = IntegrationSurface((800, 600))
        self.display = SimpleNamespace(
            set_mode=lambda _size: self.window_surface,
            set_caption=lambda _caption: None,
            flip=lambda: None,
        )
        self.event = SimpleNamespace(get=self._get_events)
        self.font = SimpleNamespace(
            Font=lambda _name, _size: SimpleNamespace(render=self._render_text),
        )
        self.time = SimpleNamespace(
            Clock=lambda: SimpleNamespace(tick=lambda _rate: None),
        )
        self.draw = SimpleNamespace(
            rect=lambda *arguments: self._record_draw('rect', arguments),
            polygon=lambda *arguments: self._record_draw('polygon', arguments),
            line=lambda *arguments: self._record_draw('line', arguments),
            circle=lambda *arguments: self._record_draw('circle', arguments),
        )
        self.transform = SimpleNamespace(
            smoothscale=lambda _surface, size: IntegrationSurface(size),
        )

    def _get_events(self) -> list[object]:
        events = list(self.events)
        self.events.clear()
        return events

    def _render_text(
        self,
        text: str,
        _antialias: bool,
        _colour: tuple[int, int, int],
    ) -> IntegrationSurface:
        self.rendered_text.append(text)
        return IntegrationSurface((max(1, len(text)), 18))

    def _record_draw(self, name: str, arguments: tuple[object, ...]) -> None:
        self.draw_calls.append((name, arguments))
        surface = arguments[0]
        if isinstance(surface, IntegrationSurface):
            surface.commands.append((name, arguments[1:]))

    def init(self) -> None:
        return None

    def quit(self) -> None:
        return None


class Plan4IntegrationTest(unittest.TestCase):
    def test_running_service_metric_workflow_and_projector_layers(self) -> None:
        discovery = IntegrationDiscovery()
        capture_factory = IntegrationCaptureFactory()
        configuration = Configuration(projector_resolution=PROJECTOR_RESOLUTION)
        detector = IntegrationDetector(
            # The service creates the same deterministic pattern for this descriptor.
            # Passing it explicitly keeps the display and detector seams independent.
            pattern=_build_pattern(PROJECTOR_RESOLUTION),
        )
        runtime = CameraRuntime(
            discovery,
            capture_factory,
            read_wait_seconds=0.001,
        )
        service = MultiVisionService(
            configuration,
            camera_runtime=runtime,
            detector=detector,  # type: ignore[arg-type]
            sleep_inhibitor=IntegrationSleepInhibitor(),
        )
        pygame_module = IntegrationPygame()
        projector_output = IntegrationProjectorOutput()
        display_runtime = PygameDisplayRuntime(
            service,  # type: ignore[arg-type]
            DisplayConfiguration(
                window_resolution=Resolution(800, 600),
                projector_resolution=PROJECTOR_RESOLUTION,
            ),
            calibration_pattern=service.calibration_pattern,
            pygame_module=pygame_module,
            marker_image_renderer=lambda _family, _marker_id, size, _pygame: (
                IntegrationSurface((size, size))
            ),
            frame_surface_converter=lambda _frame, _pygame: IntegrationSurface(
                (640, 480),
            ),
            projector_output=projector_output,
        )

        with TestClient(create_app(service)) as client:
            try:
                camera_pattern_payload = _camera_correspondence_payload(
                    service.calibration_pattern,
                    IDENTITY_MATRIX,
                )
                camera_one_payload = _camera_correspondence_payload(
                    service.calibration_pattern,
                    ((0.9, 0.0, 32.0), (0.0, 0.9, 24.0), (0.0, 0.0, 1.0)),
                )
                detector.mode = 'pattern'
                with (
                    patch('multivision.application.CALIBRATION_PATTERN_SETTLE_SECONDS', 0.0),
                    patch(
                        'multivision.application.CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS',
                        0.0,
                    ),
                ):
                    calibration_response = _request_in_thread(
                        lambda: client.post(
                            '/calibration',
                            json={'camera': 'camera-0'},
                        ),
                    )
                    _wait_until(service, 'calibration_pattern_visible')
                    display_runtime.run(max_frames=1)
                    assert display_runtime.projector_surface is not None
                    assert display_runtime.projector_surface.fills[-1] == WHITE
                    assert len(display_runtime.projector_surface.blits) == len(
                        service.calibration_pattern.markers
                    )
                    calibration_response = calibration_response()
                assert calibration_response.status_code == 200, calibration_response.text
                assert calibration_response.json()['status'] == 'UNVERIFIED'

                camera_zero_verify = client.post(
                    '/calibration/verify',
                    json={
                        'camera': 'camera-0',
                        'correspondences': camera_pattern_payload,
                    },
                )
                camera_one_calibrate = client.post(
                    '/calibration',
                    json={
                        'camera': 'camera-1',
                        'correspondences': camera_one_payload,
                    },
                )
                camera_one_verify = client.post(
                    '/calibration/verify',
                    json={
                        'camera': 'camera-1',
                        'correspondences': camera_one_payload,
                    },
                )
                assert camera_zero_verify.status_code == 200, camera_zero_verify.text
                assert camera_one_calibrate.status_code == 200, camera_one_calibrate.text
                assert camera_one_verify.status_code == 200, camera_one_verify.text
                assert camera_zero_verify.json()['status'] == 'CALIBRATED'
                assert camera_one_verify.json()['status'] == 'CALIBRATED'

                camera_zero_record = service.get_calibration_records()['camera-0']
                camera_one_record = service.get_calibration_records()['camera-1']
                assert camera_zero_record.camera_to_projector != (
                    camera_one_record.camera_to_projector
                )
                assert not hasattr(camera_zero_record, 'metric_calibration')
                assert not hasattr(camera_one_record, 'mm_per_pixel')

                first_area = client.post(
                    '/cameras/camera-0/area',
                    json={'enabled': True},
                )
                second_area = client.post(
                    '/cameras/camera-1/area',
                    json={'enabled': True},
                )
                assert first_area.status_code == 200, first_area.text
                assert second_area.status_code == 200, second_area.text

                detector.mode = 'metric'
                with patch('multivision.application.METRIC_CAPTURE_SETTLE_SECONDS', 0.0):
                    metric_response = _request_in_thread(
                        lambda: client.post(
                            '/metric/calibration',
                            json={'camera': 'camera-0'},
                        ),
                    )
                    _wait_until(service, 'metric_capture_active')
                    projector_draw_count = len(
                        [
                            call
                            for call in pygame_module.draw_calls
                            if call[1][0] is display_runtime.projector_surface
                        ]
                    )
                    display_runtime.run(max_frames=1)
                    assert display_runtime.projector_surface is not None
                    assert display_runtime.projector_surface.fills[-1] == BLACK
                    assert len(
                        [
                            call
                            for call in pygame_module.draw_calls
                            if call[1][0] is display_runtime.projector_surface
                        ]
                    ) == projector_draw_count
                    metric_response = metric_response()
                    assert metric_response.status_code == 200, metric_response.text
                    assert metric_response.json()['state'] == 'CALIBRATED'
                    assert detector.detected_modes[-3:] == [
                        'metric',
                        'metric',
                        'metric',
                    ]

                first_metric_record = service.metric_calibration
                assert first_metric_record is not None
                first_projector_to_surface = first_metric_record.projector_to_surface
                first_surface_to_projector = first_metric_record.surface_to_projector
                assert first_projector_to_surface is not None
                assert first_surface_to_projector is not None

                # Recalibrating through the second camera must reproduce the
                # shared projector/surface transform, not create camera-owned
                # metric state.
                detector.metric_source_to_camera = (
                    (0.9, 0.0, 32.0),
                    (0.0, 0.9, 24.0),
                    (0.0, 0.0, 1.0),
                )
                with patch('multivision.application.METRIC_CAPTURE_SETTLE_SECONDS', 0.0):
                    second_metric_response = _request_in_thread(
                        lambda: client.post(
                            '/metric/calibration',
                            json={'camera': 'camera-1'},
                        ),
                    )
                    _wait_until(service, 'metric_capture_active')
                    display_runtime.run(max_frames=1)
                    second_metric_response = second_metric_response()
                assert second_metric_response.status_code == 200, (
                    second_metric_response.text
                )
                second_metric_record = service.metric_calibration
                assert second_metric_record is not None
                assert second_metric_record.observation_camera_slot == 'camera-1'
                assert second_metric_record.projector_to_surface is not None
                assert second_metric_record.surface_to_projector is not None
                for first_row, second_row in zip(
                    first_projector_to_surface,
                    second_metric_record.projector_to_surface,
                ):
                    for first_value, second_value in zip(first_row, second_row):
                        assert abs(first_value - second_value) < 1e-4, (
                            f'{first_value=}, {second_value=}'
                        )
                for first_row, second_row in zip(
                    first_surface_to_projector,
                    second_metric_record.surface_to_projector,
                ):
                    for first_value, second_value in zip(first_row, second_row):
                        assert abs(first_value - second_value) < 1e-4, (
                            f'{first_value=}, {second_value=}'
                        )

                cli_client = _make_cli_client(client)
                with redirect_stdout(io.StringIO()):
                    assert cli_main(['metric', 'status'], cli_client) == 0
                status_response = client.get('/metric/calibration/status')
                assert status_response.status_code == 200
                assert status_response.json()['state'] == 'CALIBRATED'
                assert status_response.json()['calibration']['observation_camera_slot'] == (
                    'camera-1'
                )

                first_ruler = _run_cli(
                    cli_client,
                    [
                        'metric',
                        'ruler',
                        '--from-mm',
                        '50,50',
                        '--to-mm',
                        '150,50',
                        '--unit',
                        'cm',
                        '--observed-length',
                        '9.5',
                        '--observed-unit',
                        'cm',
                    ],
                )
                assert first_ruler == 0
                first_ruler_record = service.metric_ruler
                assert first_ruler_record is not None
                assert first_ruler_record.label == '10.0 cm'

                point_response = client.post(
                    '/overlay/point',
                    json={'camera': 'camera-1', 'x': 200.0, 'y': 150.0},
                )
                assert point_response.status_code == 200, point_response.text
                point_overlay = service.overlay
                assert point_overlay is not None

                second_ruler = _run_cli(
                    cli_client,
                    [
                        'metric',
                        'ruler',
                        '--from-mm',
                        '80,60',
                        '--to-mm',
                        '180,60',
                        '--unit',
                        'mm',
                    ],
                )
                assert second_ruler == 0
                replacement_ruler = service.metric_ruler
                assert replacement_ruler is not None
                assert replacement_ruler is not first_ruler_record
                assert replacement_ruler.label == '100.0 mm'
                assert service.overlay is point_overlay

                validation_status = client.get('/metric/calibration/status').json()
                assert validation_status['calibration'][
                    'latest_physical_validation_error_mm'
                ] == 5.0
                assert len(validation_status['calibration']['validation_records']) == 1
                assert validation_status['calibration']['fit_error_mm'] != 5.0

                display_runtime.run(max_frames=1)
                normal_commands = projector_output.presented_commands[-1]
                normal_draw_names = [
                    name
                    for name, _arguments in normal_commands
                    if name in {'polygon', 'line', 'circle'}
                ]
                assert normal_draw_names == [
                    'polygon',
                    'polygon',
                    'polygon',
                    'polygon',
                    'line',
                    *(['line'] * len(replacement_ruler.ticks)),
                    'circle',
                ], f'{normal_commands=}'
                assert [
                    arguments[0]
                    for name, arguments in normal_commands
                    if name == 'polygon'
                ][:2] == [
                    tuple(first_area.json()['area_colour']),
                    tuple(second_area.json()['area_colour']),
                ]
                assert pygame_module.rendered_text[-1] == '100.0 mm'

                clear_response = client.delete('/metric/ruler')
                assert clear_response.status_code == 200
                assert service.metric_ruler is None
                assert service.metric_calibration is not None
                assert service.overlay is point_overlay

                service.set_metric_ruler((50.0, 50.0), (150.0, 50.0))
                descriptor = ProjectorOutputDescriptor(
                    Resolution(800, 600),
                    'projector-reconfigured',
                )
                service.update_projector_descriptor(descriptor)
                stale_status = client.get('/metric/calibration/status')
                stale_cameras = client.get('/cameras').json()
                assert stale_status.json()['state'] == 'STALE'
                assert all(
                    camera['calibration_status'] == 'STALE'
                    for camera in stale_cameras
                ), f'{stale_cameras=}'
                assert service.metric_ruler is None
                assert service.overlay is None

                display_runtime.run(max_frames=1)
                stale_commands = projector_output.presented_commands[-1]
                assert display_runtime.projector_surface is not None
                assert display_runtime.projector_surface.size == (800, 600)
                assert [
                    name
                    for name, _arguments in stale_commands
                    if name in {'polygon', 'line', 'circle'}
                ] == []
                assert [name for name, _arguments in stale_commands] == ['fill']
            finally:
                display_runtime.shutdown()

        assert discovery.call_count == 1, f'{discovery.call_count=}'
        assert len(capture_factory.captures) == 2, f'{capture_factory.captures=}'
        assert all(capture.is_released for capture in capture_factory.captures)
        assert projector_output.shutdown_called


def _build_pattern(projector_resolution: Resolution) -> CalibrationPattern:
    margin = CALIBRATION_PATTERN_EDGE_MARGIN_PIXELS
    return build_calibration_pattern(
        projector_resolution,
        usable_area=CoordinateBounds(
            margin,
            margin,
            projector_resolution.width - margin,
            projector_resolution.height - margin,
        ),
    )


def _camera_correspondence_payload(
    pattern: CalibrationPattern,
    source_to_camera: tuple[tuple[float, float, float], ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for marker in pattern.markers:
        for corner_index, projector_point in enumerate(marker.corners):
            camera_point = project_point(projector_point, source_to_camera)
            payload.append(
                {
                    'marker_id': marker.marker_id,
                    'corner_index': corner_index,
                    'projector': {
                        'x': projector_point.x,
                        'y': projector_point.y,
                    },
                    'camera': {
                        'x': camera_point.x,
                        'y': camera_point.y,
                    },
                },
            )
    return payload


def _make_cli_client(client: TestClient) -> MultiVisionClient:
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

    return MultiVisionClient('http://service.test', request_sender=request_sender)


def _run_cli(client: MultiVisionClient, arguments: list[str]) -> int:
    with redirect_stdout(io.StringIO()):
        return cli_main(arguments, client)


def _request_in_thread(
    request: Callable[[], Any],
) -> Callable[[], Any]:
    result: list[Any] = []
    errors: list[BaseException] = []

    def run_request() -> None:
        try:
            result.append(request())
        except BaseException as ex:  # noqa: BLE001 (The test thread reports endpoint failures.)
            errors.append(ex)

    thread = threading.Thread(target=run_request)
    thread.start()

    def finish_request() -> Any:
        thread.join(10.0)
        assert not thread.is_alive(), 'Integration request did not finish'
        assert errors == [], f'{errors=}'
        assert len(result) == 1, f'{result=}'
        return result[0]

    return finish_request


def _wait_until(service: MultiVisionService, attribute_name: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if getattr(service, attribute_name):
            return
        time.sleep(0.001)
    assert getattr(service, attribute_name), f'{attribute_name=} never became true'


if __name__ == '__main__':
    unittest.main()
