import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.camera import CameraRuntime
from multivision.config import Configuration, load_configuration, save_configuration
from multivision.errors import CalibrationError
from multivision.fiducials import FiducialCorrespondence
from multivision.persistence import CalibrationRegistry, CalibrationStore
from multivision.pattern import build_calibration_pattern
from multivision.types import DeviceInfo, Resolution


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


class NonFiniteFrameCapture(FakeCapture):
    def read(self) -> tuple[bool, object]:
        return not self.is_released, {'value': float('nan')}


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


class FailingCalibrationStore(CalibrationStore):
    def save(self, _calibration: object) -> None:
        raise CalibrationError('disk full')


class ApiTest(unittest.TestCase):
    def test_camera_binding_endpoint_persists_the_stable_device_id(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / 'config.json'
            service = MultiVisionService(
                Configuration(),
                config_path=config_path,
            )
            with TestClient(create_app(service)) as client:
                response = client.post(
                    '/cameras/overhead/binding',
                    json={'device_id': 'stable-camera-id'},
                )

            assert response.status_code == 200, response.text
            assert response.json() == {
                'bound': True,
                'camera': 'overhead',
                'device_id': 'stable-camera-id',
                'restart_required': True,
            }
            assert config_path.exists()
            assert load_configuration(config_path).camera_bindings == {
                'overhead': 'stable-camera-id',
            }

    def test_injected_calibration_store_is_the_configuration_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / 'config.json'
            store = CalibrationStore(config_path)
            save_configuration(
                Configuration(camera_bindings={'existing': 'camera-a'}),
                config_path,
            )

            service = MultiVisionService(calibration_store=store)
            service.bind_camera('overhead', 'camera-b')

            assert load_configuration(config_path).camera_bindings == {
                'existing': 'camera-a',
                'overhead': 'camera-b',
            }

            with self.assertRaises(ValueError):
                MultiVisionService(
                    Configuration(),
                    config_path=Path(temporary_directory) / 'other.json',
                    calibration_store=store,
                )

    def test_fake_camera_endpoints_use_one_service_owned_handle(self) -> None:
        capture = FakeCapture()
        configuration = Configuration(
            camera_bindings={'overhead': 'device-a'},
            projector_resolution=Resolution(1000, 700),
        )
        factory = FakeCaptureFactory(capture)
        camera_runtime = CameraRuntime(
            FakeDiscovery(
                [DeviceInfo('device-a', 'Fake camera', 0, Resolution(1000, 700))],
            ),
            factory,
            configuration.camera_bindings,
            read_wait_seconds=0.001,
        )
        with TemporaryDirectory() as temporary_directory:
            service = MultiVisionService(
                configuration,
                calibration_store=CalibrationStore(Path(temporary_directory) / 'state.json'),
                calibration_registry=CalibrationRegistry(
                    calibration_version=configuration.calibration_version,
                    projector_resolution=configuration.projector_resolution,
                ),
                camera_runtime=camera_runtime,
            )
            with TestClient(create_app(service)) as client:
                health_response = client.get('/health')
                cameras_response = client.get('/cameras')
                snapshot_response = client.get('/cameras/overhead/snapshot')
                point_response = client.post(
                    '/overlay/point',
                    json={'camera': 'overhead', 'x': 10, 'y': 20},
                )

            assert health_response.status_code == 200
            assert health_response.json()['available_camera_count'] == 1
            assert cameras_response.json()[0]['runtime_status'] == 'AVAILABLE'
            assert snapshot_response.status_code == 200
            assert snapshot_response.json()['data'] == 'fake-frame'
            assert point_response.status_code == 422
            assert point_response.json()['error']['code'] == 'CALIBRATION_UNCALIBRATED'
            assert capture.is_released
            assert factory.open_count == 1

    def test_failed_calibration_persistence_does_not_publish_a_record(self) -> None:
        capture = FakeCapture()
        configuration = Configuration(
            camera_bindings={'overhead': 'device-a'},
            projector_resolution=Resolution(1000, 700),
        )
        camera_runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device-a', 'Fake camera', 0, Resolution(1000, 700))]),
            FakeCaptureFactory(capture),
            configuration.camera_bindings,
            read_wait_seconds=0.001,
        )
        pattern = build_calibration_pattern(configuration.projector_resolution)
        correspondences = tuple(
            FiducialCorrespondence(
                marker.marker_id,
                corner_index,
                corner,
                corner,
            )
            for marker in pattern.markers
            for corner_index, corner in enumerate(marker.corners)
        )
        with TemporaryDirectory() as temporary_directory:
            service = MultiVisionService(
                configuration,
                calibration_store=FailingCalibrationStore(
                    Path(temporary_directory) / 'state.json',
                ),
                camera_runtime=camera_runtime,
                calibration_pattern=pattern,
            )
            service.start()
            try:
                with self.assertRaises(CalibrationError):
                    service.calibrate('overhead', correspondences)
                assert service.get_calibration_records() == {}
            finally:
                service.shutdown()

    def test_calibration_and_verification_accept_deterministic_fake_correspondences(self) -> None:
        capture = FakeCapture()
        configuration = Configuration(
            camera_bindings={'overhead': 'device-a'},
            projector_resolution=Resolution(1000, 700),
        )
        factory = FakeCaptureFactory(capture)
        camera_runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device-a', 'Fake camera', 0, Resolution(1000, 700))]),
            factory,
            configuration.camera_bindings,
            read_wait_seconds=0.001,
        )
        pattern = build_calibration_pattern(configuration.projector_resolution)
        correspondences = [
            {
                'marker_id': marker.marker_id,
                'corner_index': corner_index,
                'projector': {'x': corner.x, 'y': corner.y},
                'camera': {'x': corner.x, 'y': corner.y},
            }
            for marker in pattern.markers
            for corner_index, corner in enumerate(marker.corners)
        ]
        with TemporaryDirectory() as temporary_directory:
            service = MultiVisionService(
                configuration,
                calibration_store=CalibrationStore(Path(temporary_directory) / 'state.json'),
                camera_runtime=camera_runtime,
                calibration_pattern=pattern,
            )
            with TestClient(create_app(service)) as client:
                calibration_response = client.post(
                    '/calibration',
                    json={'camera': 'overhead', 'correspondences': correspondences},
                )
                verification_response = client.post(
                    '/calibration/verify',
                    json={'camera': 'overhead', 'correspondences': correspondences},
                )
                status_response = client.get('/calibration/status')
                point_response = client.post(
                    '/overlay/point',
                    json={'camera': 'overhead', 'x': 500, 'y': 300},
                )
                clear_response = client.delete('/overlay')

            assert calibration_response.status_code == 200, calibration_response.text
            assert calibration_response.json()['status'] == 'UNVERIFIED'
            assert verification_response.status_code == 200, verification_response.text
            assert verification_response.json()['status'] == 'CALIBRATED'
            assert status_response.json()['cameras']['overhead'] == 'CALIBRATED'
            assert point_response.status_code == 200, point_response.text
            projector_point = point_response.json()['projector_point']
            assert abs(projector_point[0] - 500) < 1e-6
            assert abs(projector_point[1] - 300) < 1e-6
            assert clear_response.status_code == 200
            assert clear_response.json() == {'cleared': True}
            assert factory.open_count == 1
            assert capture.is_released

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

    def test_non_finite_snapshot_data_is_an_explicit_capture_error(self) -> None:
        configuration = Configuration(camera_bindings={'overhead': 'device-a'})
        capture = NonFiniteFrameCapture()
        camera_runtime = CameraRuntime(
            FakeDiscovery(
                [DeviceInfo('device-a', 'Fake camera', 0, Resolution(1000, 700))],
            ),
            FakeCaptureFactory(capture),
            configuration.camera_bindings,
            read_wait_seconds=0.001,
        )
        service = MultiVisionService(configuration, camera_runtime=camera_runtime)

        with TestClient(create_app(service)) as client:
            response = client.get('/cameras/overhead/snapshot')

        assert response.status_code == 503
        assert response.json()['error']['code'] == 'FRAME_UNAVAILABLE'

    def test_snapshot_encoding_failure_is_an_explicit_capture_error(self) -> None:
        class UnencodableFrame:
            shape = (1, 1, 3)

        capture = FakeCapture()
        capture.read = lambda: (True, UnencodableFrame())  # type: ignore[method-assign]
        configuration = Configuration(camera_bindings={'overhead': 'device-a'})
        camera_runtime = CameraRuntime(
            FakeDiscovery(
                [DeviceInfo('device-a', 'Fake camera', 0, Resolution(1000, 700))],
            ),
            FakeCaptureFactory(capture),
            configuration.camera_bindings,
            read_wait_seconds=0.001,
        )
        service = MultiVisionService(configuration, camera_runtime=camera_runtime)
        failing_cv2 = SimpleNamespace(
            imencode=lambda *_arguments: (_ for _ in ()).throw(
                RuntimeError('encoding failed'),
            ),
        )

        with patch.dict(sys.modules, {'cv2': failing_cv2}):
            with TestClient(create_app(service)) as client:
                response = client.get('/cameras/overhead/snapshot')

        assert response.status_code == 503
        assert response.json()['error']['code'] == 'FRAME_UNAVAILABLE'

    def test_huge_coordinates_are_validation_errors(self) -> None:
        configuration = Configuration(camera_bindings={'overhead': 'missing'})
        camera_runtime = CameraRuntime(
            FakeDiscovery([]),
            FakeCaptureFactory(FakeCapture()),
            configuration.camera_bindings,
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

    def test_no_physical_camera_returns_explicit_errors_and_validation(self) -> None:
        configuration = Configuration(camera_bindings={'overhead': 'missing'})
        camera_runtime = CameraRuntime(
            FakeDiscovery([]),
            FakeCaptureFactory(FakeCapture()),
            configuration.camera_bindings,
        )
        service = MultiVisionService(configuration, camera_runtime=camera_runtime)

        with TestClient(create_app(service)) as client:
            cameras_response = client.get('/cameras')
            snapshot_response = client.get('/cameras/overhead/snapshot')
            calibration_response = client.post('/calibration', json={})
            invalid_point_response = client.post(
                '/overlay/point',
                json={'camera': 'overhead', 'x': 'not-a-number', 'y': 1},
            )

        assert cameras_response.status_code == 200
        assert cameras_response.json()[0]['runtime_status'] == 'UNAVAILABLE'
        assert snapshot_response.status_code == 503
        assert snapshot_response.json()['error']['code'] == 'CAMERA_UNAVAILABLE'
        assert calibration_response.status_code == 503
        assert calibration_response.json()['error']['code'] == 'CAMERA_UNAVAILABLE'
        assert invalid_point_response.status_code == 422
        assert invalid_point_response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'


if __name__ == '__main__':
    unittest.main()
