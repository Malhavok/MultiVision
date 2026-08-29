import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.camera import CameraRuntime
from multivision.config import Configuration, load_configuration, save_configuration
from multivision.fiducials import FiducialCorrespondence
from multivision.geometry import Point2D
from multivision.persistence import CalibrationStore
from multivision.pattern import build_calibration_pattern
from multivision.types import CalibrationStatus, DeviceInfo, Resolution


class FakeCapture:
    def __init__(self) -> None:
        self.is_released = False

    def is_opened(self) -> bool:
        return not self.is_released

    def get_native_resolution(self) -> Resolution:
        return Resolution(1000, 700)

    def read(self) -> tuple[bool, str]:
        return not self.is_released, 'fake-frame'

    def release(self) -> None:
        self.is_released = True


class FakeDiscovery:
    def __init__(self, devices: list[DeviceInfo]) -> None:
        self.devices = devices

    def discover_devices(self) -> list[DeviceInfo]:
        return self.devices


class FakeCaptureFactory:
    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.open_count = 0

    def open_capture(self, _device: DeviceInfo) -> FakeCapture:
        self.open_count += 1
        return self.capture


class FailingService:
    def __init__(self) -> None:
        self.shutdown_count = 0

    def start(self) -> None:
        raise RuntimeError('startup failed')

    def shutdown(self) -> None:
        self.shutdown_count += 1


class MalformedStatusRuntime:
    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_statuses(self) -> list[object]:
        return [object()]


class ApiTest(unittest.TestCase):
    def test_service_start_uses_session_slots_instead_of_persisted_bindings(self) -> None:
        capture = FakeCapture()
        service = MultiVisionService(
            Configuration(),
            discovery=FakeDiscovery(
                [DeviceInfo('current-device-id', 'Current camera', 2, Resolution(1000, 700))],
            ),
            capture_factory=FakeCaptureFactory(capture),
        )

        service.start()
        try:
            statuses = service.get_camera_statuses()
            assert [status.logical_name for status in statuses] == ['camera-0'], f'{statuses=}'
            assert statuses[0].device_id is None, f'{statuses=}'
        finally:
            service.shutdown()

        assert capture.is_released

    def test_session_camera_management_is_available_without_service_restart(self) -> None:
        class ReopeningCaptureFactory:
            def __init__(self) -> None:
                self.captures: list[FakeCapture] = []

            def open_capture(self, _device: DeviceInfo) -> FakeCapture:
                capture = FakeCapture()
                self.captures.append(capture)
                return capture

        factory = ReopeningCaptureFactory()
        service = MultiVisionService(
            Configuration(),
            discovery=FakeDiscovery(
                [
                    DeviceInfo('device-a', 'Camera A', 0, Resolution(1000, 700)),
                    DeviceInfo('device-b', 'Camera B', 1, Resolution(1000, 700)),
                ],
            ),
            capture_factory=factory,
        )

        with TestClient(create_app(service)) as client:
            cameras_response = client.get('/cameras')
            rename_response = client.post(
                '/cameras/camera-0/rename',
                json={'name': 'overhead'},
            )
            duplicate_response = client.post(
                '/cameras/camera-1/rename',
                json={'name': 'overhead'},
            )
            unknown_response = client.post('/cameras/camera-9/close')
            close_response = client.post('/cameras/camera-0/close')
            invalid_transition_response = client.post('/cameras/camera-0/close')
            open_response = client.post('/cameras/camera-0/open')

        assert cameras_response.status_code == 200
        assert [camera['slot'] for camera in cameras_response.json()] == [
            'camera-0',
            'camera-1',
        ]
        assert rename_response.status_code == 200
        assert rename_response.json()['name'] == 'overhead'
        assert duplicate_response.status_code == 409
        assert duplicate_response.json()['error']['code'] == 'DUPLICATE_CAMERA_NAME'
        assert unknown_response.status_code == 404
        assert unknown_response.json()['error']['code'] == 'CAMERA_SLOT_NOT_FOUND'
        assert close_response.status_code == 200
        assert close_response.json()['state'] == 'CLOSED'
        assert invalid_transition_response.status_code == 409
        assert invalid_transition_response.json()['error']['code'] == 'CAMERA_STATE_ERROR'
        assert open_response.status_code == 200
        assert open_response.json()['state'] == 'OPEN'
        assert len(factory.captures) == 3, f'{len(factory.captures)=}'

    def test_injected_calibration_store_is_the_configuration_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / 'config.json'
            store = CalibrationStore(config_path)
            save_configuration(Configuration(), config_path)

            service = MultiVisionService(calibration_store=store)
            assert service.configuration == Configuration(), f'{service.configuration=}'
            assert load_configuration(config_path) == Configuration(), f'{config_path=}'

            with self.assertRaises(ValueError):
                MultiVisionService(
                    Configuration(),
                    config_path=Path(temporary_directory) / 'other.json',
                    calibration_store=store,
                )

    def test_session_calibration_is_slot_local_and_not_persisted(self) -> None:
        capture = FakeCapture()
        configuration = Configuration(projector_resolution=Resolution(1000, 700))
        pattern = build_calibration_pattern(configuration.projector_resolution)
        correspondences = [
            FiducialCorrespondence(
                marker.marker_id,
                corner_index,
                Point2D(corner.x, corner.y),
                Point2D(corner.x, corner.y),
            )
            for marker in pattern.markers
            for corner_index, corner in enumerate(marker.corners)
        ]
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device-a', 'Camera A', 0, Resolution(1000, 700))]),
            FakeCaptureFactory(capture),
            read_wait_seconds=0.001,
        )

        with TemporaryDirectory() as temporary_directory:
            store = CalibrationStore(Path(temporary_directory) / 'state.json')
            service = MultiVisionService(
                configuration,
                camera_runtime=runtime,
                calibration_store=store,
                calibration_pattern=pattern,
            )
            service.start()
            try:
                record = service.calibrate('camera-0', correspondences)
                assert record.camera_id == 'camera-0', f'{record=}'
                assert service.get_calibration_records() == {'camera-0': record}
                assert not store.path.exists(), f'{store.path=}'
                assert service.verify('camera-0', correspondences) is CalibrationStatus.CALIBRATED
                service.rename_camera('camera-0', 'overhead')
                assert service.get_calibration_metrics('overhead') == record.metrics
                assert service.verify('overhead', correspondences) is CalibrationStatus.CALIBRATED
                overlay = service.point_from_camera('overhead', (500, 300))
                assert overlay.camera_id == 'camera-0', f'{overlay=}'
            finally:
                service.shutdown()

    def test_startup_failure_still_shuts_down_the_injected_service(self) -> None:
        service = FailingService()

        with self.assertRaisesRegex(RuntimeError, 'startup failed'):
            with TestClient(create_app(service)):  # type: ignore[arg-type]
                pass

        assert service.shutdown_count == 1, f'{service.shutdown_count=}'

    def test_malformed_runtime_status_is_an_explicit_error_response(self) -> None:
        service = MultiVisionService(
            Configuration(),
            camera_runtime=MalformedStatusRuntime(),  # type: ignore[arg-type]
        )

        with TestClient(create_app(service)) as client:
            response = client.get('/cameras')

        assert response.status_code == 503
        assert response.json()['error']['code'] == 'CAMERA_UNAVAILABLE'

    def test_huge_coordinates_are_validation_errors(self) -> None:
        configuration = Configuration()
        camera_runtime = CameraRuntime(
            FakeDiscovery([]),
            FakeCaptureFactory(FakeCapture()),
        )
        service = MultiVisionService(configuration, camera_runtime=camera_runtime)
        huge_coordinate = '9' * 4000

        with TestClient(create_app(service)) as client:
            response = client.post(
                '/overlay/point',
                content=(
                    '{"camera":"overhead","x":'
                    + huge_coordinate
                    + ',"y":1}'
                ),
                headers={'content-type': 'application/json'},
            )

        assert response.status_code == 422
        assert response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'

if __name__ == '__main__':
    unittest.main()
