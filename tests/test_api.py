import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.camera import CameraRuntime
from multivision.config import Configuration, load_configuration, save_configuration
from multivision.fiducials import FiducialCorrespondence
from multivision.geometry import Point2D
from multivision.persistence import CalibrationStore
from multivision.pattern import build_calibration_pattern
from multivision.service import RedCircleOverlay
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
            name_area_response = client.post(
                '/cameras/overhead/area',
                json={'enabled': False},
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
        assert name_area_response.status_code == 404
        assert name_area_response.json()['error']['code'] == 'CAMERA_SLOT_NOT_FOUND'
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

    def test_area_endpoint_validates_state_and_exposes_ordered_area_data(self) -> None:
        runtime = CameraRuntime(
            FakeDiscovery(
                [DeviceInfo('device-a', 'Camera A', 0, Resolution(1000, 700))],
            ),
            FakeCaptureFactory(FakeCapture()),
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,
        )

        with TestClient(create_app(service)) as client:
            initial_response = client.get('/cameras')
            malformed_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': 1},
            )
            uncalibrated_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            unknown_response = client.post(
                '/cameras/camera-9/area',
                json={'enabled': True},
            )
            runtime.set_calibration(
                'camera-0',
                CalibrationStatus.CALIBRATED,
                SimpleNamespace(
                    camera_to_projector=(
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    valid_region=((0, 0), (1000, 0), (1000, 700), (0, 700)),
                ),
            )
            enabled_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            retry_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            status_response = client.get('/cameras/camera-0/status')
            disabled_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': False},
            )
            disabled_retry_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': False},
            )
            runtime.set_calibration(
                'camera-0',
                CalibrationStatus.CALIBRATED,
                SimpleNamespace(
                    camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 0)),
                    valid_region=((0, 0), (100, 0), (100, 100)),
                ),
            )
            before_invalid_request = client.get('/cameras/camera-0/status').json()
            invalid_polygon_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            after_invalid_request = client.get('/cameras/camera-0/status').json()
            close_response = client.post('/cameras/camera-0/close')
            unavailable_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )

        def without_volatile_frame_fields(response: object) -> dict[str, object]:
            data = response.json()  # type: ignore[union-attr]
            return {
                key: value
                for key, value in data.items()
                if key not in {'frame_counter', 'frame_metadata'}
            }

        assert initial_response.status_code == 200
        initial_camera = initial_response.json()[0]
        assert initial_camera['area_enabled'] is False, f'{initial_camera=}'
        assert initial_camera['available_area'] is None, f'{initial_camera=}'
        assert malformed_response.status_code == 422
        assert malformed_response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'
        assert uncalibrated_response.status_code == 422
        assert uncalibrated_response.json()['error']['code'] == 'CALIBRATION_UNCALIBRATED'
        assert unknown_response.status_code == 404
        assert unknown_response.json()['error']['code'] == 'CAMERA_SLOT_NOT_FOUND'
        assert enabled_response.status_code == 200
        enabled_camera = enabled_response.json()
        assert enabled_camera['slot'] == 'camera-0', f'{enabled_camera=}'
        assert enabled_camera['name'] == 'camera-0', f'{enabled_camera=}'
        assert enabled_camera['lifecycle'] == 'OPEN', f'{enabled_camera=}'
        assert enabled_camera['calibration'] == 'CALIBRATED', f'{enabled_camera=}'
        assert enabled_camera['area_enabled'] is True, f'{enabled_camera=}'
        assert enabled_camera['area_colour'] == [70, 190, 255], f'{enabled_camera=}'
        assert enabled_camera['available_area'] == [
            [0.0, 0.0],
            [1000.0, 0.0],
            [1000.0, 700.0],
            [0.0, 700.0],
        ], f'{enabled_camera=}'
        assert without_volatile_frame_fields(retry_response) == without_volatile_frame_fields(
            enabled_response,
        ), f'{retry_response.json()=}'
        assert without_volatile_frame_fields(status_response) == without_volatile_frame_fields(
            enabled_response,
        ), f'{status_response.json()=}'
        assert disabled_response.json()['area_enabled'] is False
        assert disabled_response.json()['available_area'] is None
        assert without_volatile_frame_fields(disabled_retry_response) == (
            without_volatile_frame_fields(disabled_response)
        )

        assert invalid_polygon_response.status_code == 422
        assert invalid_polygon_response.json()['error']['code'] == 'AVAILABLE_AREA_INVALID'
        assert after_invalid_request == before_invalid_request, (
            f'{after_invalid_request=}, {before_invalid_request=}'
        )
        assert close_response.status_code == 200
        assert unavailable_response.status_code == 503
        assert unavailable_response.json()['error']['code'] == 'CAMERA_UNAVAILABLE'

    def test_area_api_preserves_schema_independence_and_pointing_state(self) -> None:
        class ReopeningCaptureFactory:
            def __init__(self) -> None:
                self.captures: list[FakeCapture] = []

            def open_capture(self, _device: DeviceInfo) -> FakeCapture:
                capture = FakeCapture()
                self.captures.append(capture)
                return capture

        factory = ReopeningCaptureFactory()
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo('device-a', 'Camera A', 0, Resolution(1000, 700)),
                    DeviceInfo('device-b', 'Camera B', 1, Resolution(1000, 700)),
                ],
            ),
            factory,
            read_wait_seconds=0.001,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,
        )
        calibration_values = [
            SimpleNamespace(
                camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
            ),
            SimpleNamespace(
                camera_to_projector=((1, 0, 100), (0, 1, 0), (0, 0, 1)),
                valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
            ),
        ]
        overlay_after_point: RedCircleOverlay | None = None
        overlay_after_rename: RedCircleOverlay | None = None
        overlay_after_disable: RedCircleOverlay | None = None

        with TestClient(create_app(service)) as client:
            initial_response = client.get('/cameras')
            expected_fields = {
                'camera',
                'slot',
                'name',
                'device_id',
                'capture_index',
                'state',
                'lifecycle',
                'runtime_status',
                'calibration_status',
                'calibration',
                'native_resolution',
                'frame_counter',
                'frame_metadata',
                'area_enabled',
                'area_colour',
                'available_area',
                'error_message',
            }
            for slot_index, calibration in enumerate(calibration_values):
                runtime.set_calibration(
                    f'camera-{slot_index}',
                    CalibrationStatus.CALIBRATED,
                    calibration,
                )

            point_response = client.post(
                '/overlay/point',
                json={'camera': 'camera-0', 'x': 100, 'y': 100},
            )
            overlay_after_point = service.overlay
            first_area_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            first_retry_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': True},
            )
            second_area_response = client.post(
                '/cameras/camera-1/area',
                json={'enabled': True},
            )
            rename_response = client.post(
                '/cameras/camera-0/rename',
                json={'name': 'overhead'},
            )
            overlay_after_rename = service.overlay
            disabled_response = client.post(
                '/cameras/camera-0/area',
                json={'enabled': False},
            )
            overlay_after_disable = service.overlay
            close_response = client.post('/cameras/camera-0/close')
            reopen_response = client.post('/cameras/camera-0/open')
            remaining_camera_response = client.get('/cameras/camera-1/status')

        assert initial_response.status_code == 200
        initial_cameras = initial_response.json()
        assert [camera['slot'] for camera in initial_cameras] == [
            'camera-0',
            'camera-1',
        ], f'{initial_cameras=}'
        assert all(set(camera) == expected_fields for camera in initial_cameras), (
            f'{initial_cameras=}'
        )
        assert point_response.status_code == 200, f'{point_response.json()=}'
        assert point_response.json()['projector_point'] == [100.0, 100.0]

        assert first_area_response.status_code == 200
        first_area = first_area_response.json()
        assert set(first_area) == expected_fields, f'{first_area=}'
        assert first_area['area_enabled'] is True, f'{first_area=}'
        assert first_area['available_area'] is not None, f'{first_area=}'
        retry_area = first_retry_response.json()
        assert all(
            retry_area[field] == first_area[field]
            for field in (
                'slot',
                'name',
                'lifecycle',
                'calibration',
                'area_enabled',
                'area_colour',
                'available_area',
            )
        ), f'{retry_area=}, {first_area=}'
        assert second_area_response.status_code == 200
        second_area = second_area_response.json()
        assert second_area['available_area'] != first_area['available_area']
        assert second_area['area_colour'] != first_area['area_colour']
        assert overlay_after_point is not None
        assert overlay_after_point.projector_point == Point2D(100, 100)

        assert rename_response.status_code == 200
        renamed_camera = rename_response.json()
        assert renamed_camera['slot'] == 'camera-0'
        assert renamed_camera['name'] == 'overhead'
        assert renamed_camera['area_enabled'] is True
        assert renamed_camera['available_area'] == first_area['available_area']
        assert overlay_after_rename is not None
        assert overlay_after_rename.logical_name == 'overhead'

        assert disabled_response.status_code == 200
        assert disabled_response.json()['area_enabled'] is False
        assert disabled_response.json()['available_area'] is None
        assert overlay_after_disable is not None
        assert overlay_after_disable.projector_point == Point2D(100, 100)

        assert close_response.status_code == 200
        assert close_response.json()['state'] == 'CLOSED'
        assert close_response.json()['area_enabled'] is False
        assert close_response.json()['available_area'] is None
        assert service.overlay is None
        assert reopen_response.status_code == 200
        assert reopen_response.json()['state'] == 'OPEN'
        assert reopen_response.json()['calibration'] == 'UNCALIBRATED'
        assert reopen_response.json()['area_enabled'] is False
        assert reopen_response.json()['available_area'] is None
        assert remaining_camera_response.json()['area_enabled'] is True
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
