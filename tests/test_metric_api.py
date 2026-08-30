import math
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.calibration import CalibrationMetrics
from multivision.config import Configuration, ProjectorOutputDescriptor
from multivision.errors import InvalidCalibrationStateError
from multivision.geometry import Point2D
from multivision.metric import (
    MetricCalibrationMetrics,
    MetricCalibrationRecord,
    MetricCalibrationResult,
    MetricCalibrationStatus,
    MetricHomographyPair,
    MetricValidationRecord,
)
from multivision.metric_target import METRIC_TARGET
from multivision.persistence import PersistedCalibration
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
)


class RecordingMetricService:
    def __init__(self) -> None:
        self.projector_output_descriptor = ProjectorOutputDescriptor(
            Resolution(640, 480),
        )
        self.metric_calibration: MetricCalibrationRecord | None = None
        self.calibration_calls: list[tuple[str, object]] = []
        self.ruler_calls: list[tuple[Point2D, Point2D, object]] = []
        self.validation_calls: list[tuple[object, ...]] = []
        self.clear_calibration_calls = 0
        self.clear_ruler_calls = 0
        self.fail_calibration = False
        self.ruler = SimpleNamespace(
            length_mm=100.0,
            to_data=lambda: {
                'surface_start': [10.0, 10.0],
                'surface_end': [110.0, 10.0],
                'length_mm': 100.0,
                'unit': 'mm',
                'length': 100.0,
                'label': '100.0 mm',
            },
        )

    def calibrate_metric(self, camera: str, correspondences: object) -> MetricCalibrationRecord:
        if self.fail_calibration:
            error = InvalidCalibrationStateError('selected camera calibration is stale')
            error.code = 'CALIBRATION_STALE'
            raise error
        self.calibration_calls.append((camera, correspondences))
        self.metric_calibration = _calibration_record()
        return self.metric_calibration

    def get_metric_status(self) -> MetricCalibrationStatus:
        return (
            MetricCalibrationStatus.UNCALIBRATED
            if self.metric_calibration is None
            else self.metric_calibration.state
        )

    def get_metric_status_snapshot(
        self,
    ) -> tuple[MetricCalibrationStatus, MetricCalibrationRecord | None]:
        return self.get_metric_status(), self.metric_calibration

    def clear_metric_calibration(self) -> None:
        self.clear_calibration_calls += 1
        self.metric_calibration = None

    def set_metric_ruler(
        self,
        surface_start: Point2D,
        surface_end: Point2D,
        unit: object,
    ) -> SimpleNamespace:
        self.ruler_calls.append((surface_start, surface_end, unit))
        return self.ruler

    def clear_metric_ruler(self) -> None:
        self.clear_ruler_calls += 1

    def set_metric_ruler_with_validation(
        self,
        surface_start: Point2D,
        surface_end: Point2D,
        unit: object,
        observed_length: object | None = None,
        observed_unit: object = 'mm',
    ) -> tuple[SimpleNamespace, MetricValidationRecord | None]:
        ruler = self.set_metric_ruler(surface_start, surface_end, unit)
        validation = None
        if observed_length is not None:
            validation = self.record_physical_validation(
                ruler.length_mm,
                observed_length,
                'mm',
                observed_unit,
            )
        return ruler, validation

    def record_physical_validation(
        self,
        *arguments: object,
    ) -> MetricValidationRecord:
        self.validation_calls.append(arguments)
        validation = MetricValidationRecord(100.0, 101.6, 1.6, 1.0)
        assert self.metric_calibration is not None
        self.metric_calibration = self.metric_calibration._replace(
            validation_records=(validation,),
            latest_physical_validation_error_mm=validation.absolute_error_mm,
        )
        return validation


class FailClosedRulerService(RecordingMetricService):
    def set_metric_ruler(
        self,
        surface_start: Point2D,
        surface_end: Point2D,
        unit: object,
    ) -> SimpleNamespace:
        status = self.get_metric_status()
        if status is MetricCalibrationStatus.CALIBRATED:
            return super().set_metric_ruler(surface_start, surface_end, unit)
        error = InvalidCalibrationStateError(
            f'Metric calibration is {status.value.lower()}',
        )
        error.code = (
            'METRIC_STALE'
            if status is MetricCalibrationStatus.STALE
            else 'METRIC_UNAVAILABLE'
        )
        raise error


class ApiMetricRuntime:
    def __init__(self) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    'device-0',
                    'Camera 0',
                    capture_index=0,
                    native_resolution=Resolution(640, 480),
                ),
            ],
        )

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        return CameraStatus(
            slot_id,
            camera.device_info.device_id if camera.device_info is not None else None,
            RuntimeStatus.AVAILABLE,
            camera.calibration_status,
            Resolution(640, 480),
        )


def _camera_calibration() -> PersistedCalibration:
    identity_matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    resolution = Resolution(640, 480)
    return PersistedCalibration(
        'camera-0',
        resolution,
        resolution,
        1,
        identity_matrix,
        identity_matrix,
        CalibrationMetrics(20, 80, 80, 1.0, 0.0, 0.0, 0.8),
        1.0,
        ((0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)),
        projector_output_descriptor=ProjectorOutputDescriptor(resolution),
    )


def _calibration_record() -> MetricCalibrationRecord:
    descriptor = ProjectorOutputDescriptor(Resolution(640, 480))
    metrics = MetricCalibrationMetrics(20, 80, 80, 1.0, 0.25, 0.5, 0.75)
    return MetricCalibrationRecord(
        MetricCalibrationStatus.CALIBRATED,
        descriptor,
        MetricHomographyPair.from_projector_to_surface(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        'camera-0',
        'device-0',
        1,
        1.0,
        METRIC_TARGET.format_name,
        METRIC_TARGET.format_version,
        METRIC_TARGET.marker_family,
        metrics,
        1.0,
    )


def _assert_json_finite(value: Any) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f'{value=}'
    elif isinstance(value, list):
        for element in value:
            _assert_json_finite(element)
    elif isinstance(value, dict):
        for element in value.values():
            _assert_json_finite(element)


def test_metric_api_delegates_and_serialises_json_safe_data() -> None:
    service = RecordingMetricService()
    correspondence = {
        'marker_id': 0,
        'corner_index': 0,
        'surface': {'x': 8.0, 'y': 40.0},
        'camera': {'x': 20.0, 'y': 30.0},
    }

    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        calibration_response = client.post(
            '/metric/calibration',
            json={'camera': 'camera-0', 'correspondences': [correspondence]},
        )
        status_response = client.get('/metric/calibration/status')
        ruler_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': 10.0, 'y': 10.0},
                'to': {'x': 110.0, 'y': 10.0},
                'unit': 'cm',
                'observed_length': 10.16,
                'observed_unit': 'cm',
            },
        )
        optional_observation_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': 20.0, 'y': 20.0},
                'to': {'x': 120.0, 'y': 20.0},
            },
        )

    calibration_data = calibration_response.json()
    status_data = status_response.json()
    ruler_data = ruler_response.json()
    optional_observation_data = optional_observation_response.json()
    assert calibration_response.status_code == 200
    assert service.calibration_calls[0][0] == 'camera-0'
    assert calibration_data['state'] == 'CALIBRATED'
    assert calibration_data['projector_to_surface'] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert calibration_data['surface_to_projector'] == calibration_data[
        'projector_to_surface'
    ]
    assert calibration_data['target']['marker_count'] == 20
    assert calibration_data['target']['reference_segment']['length_mm'] == 100.0
    assert status_data['calibration']['target_version'] == METRIC_TARGET.format_version
    assert status_data['applicable'] is True
    assert ruler_response.status_code == 200
    assert optional_observation_response.status_code == 200
    assert service.ruler_calls == [
        (Point2D(10.0, 10.0), Point2D(110.0, 10.0), 'cm'),
        (Point2D(20.0, 20.0), Point2D(120.0, 20.0), 'mm'),
    ]
    assert service.validation_calls == [(100.0, 10.16, 'mm', 'cm')]
    assert ruler_data['observed_length_mm'] == 101.6
    assert ruler_data['absolute_error_mm'] == 1.6
    assert ruler_data['validation']['requested_length_mm'] == 100.0
    assert optional_observation_data['observed_length'] is None
    assert optional_observation_data['validation'] is None
    _assert_json_finite(calibration_data)
    _assert_json_finite(status_data)
    _assert_json_finite(ruler_data)
    _assert_json_finite(optional_observation_data)


def test_metric_status_keeps_current_descriptor_when_record_is_stale() -> None:
    service = RecordingMetricService()
    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        service.metric_calibration = _calibration_record()
        service.metric_calibration = service.metric_calibration._replace(
            state=MetricCalibrationStatus.STALE,
        )
        service.projector_output_descriptor = ProjectorOutputDescriptor(
            Resolution(800, 600),
            'projector-b',
        )
        response = client.get('/metric/calibration/status')

    status = response.json()
    assert status['projector_output_descriptor'] == {
        'projector_resolution': {'width': 800, 'height': 600},
        'output_identity': 'projector-b',
    }, f'{status=}'
    assert status['calibration']['projector_output_descriptor'] == {
        'projector_resolution': {'width': 640, 'height': 480},
        'output_identity': 'default',
    }, f'{status=}'


def test_metric_api_rejects_malformed_bodies_without_service_calls() -> None:
    service = RecordingMetricService()
    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        calibration_response = client.post(
            '/metric/calibration',
            json={'camera': 'camera-0', 'unexpected': True},
        )
        ruler_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': 1.0, 'y': 1.0},
                'to': {'x': 2.0, 'y': 1.0},
                'unit': 'yards',
                'observed_length': 'not-a-number',
            },
        )

    assert calibration_response.status_code == 422
    assert calibration_response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'
    assert ruler_response.status_code == 422
    assert ruler_response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'
    assert service.calibration_calls == []
    assert service.ruler_calls == []


def test_metric_ruler_fails_closed_for_unavailable_and_stale_calibration() -> None:
    cases = (
        (None, MetricCalibrationStatus.UNCALIBRATED, 'METRIC_UNAVAILABLE'),
        (
            _calibration_record()._replace(state=MetricCalibrationStatus.STALE),
            MetricCalibrationStatus.STALE,
            'METRIC_STALE',
        ),
    )
    for record, expected_status, expected_error_code in cases:
        service = FailClosedRulerService()
        service.metric_calibration = record
        with TestClient(create_app(service, manage_lifecycle=False)) as client:
            status_response = client.get('/metric/calibration/status')
            ruler_response = client.post(
                '/metric/ruler',
                json={
                    'from': {'x': 10.0, 'y': 10.0},
                    'to': {'x': 110.0, 'y': 10.0},
                },
            )

        assert status_response.json()['state'] == expected_status.value
        assert status_response.json()['error_code'] == expected_error_code
        assert ruler_response.status_code == 422
        assert ruler_response.json()['error']['code'] == expected_error_code
        assert service.ruler_calls == [], f'{service.ruler_calls=}'


def test_failed_metric_requests_preserve_existing_metric_state() -> None:
    service = RecordingMetricService()
    service.metric_calibration = _calibration_record()
    metric_before = service.metric_calibration
    ruler_before = service.ruler
    service.fail_calibration = True

    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        failed_calibration_response = client.post(
            '/metric/calibration',
            json={'camera': 'camera-0'},
        )
        malformed_ruler_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': 10.0, 'y': 10.0},
                'to': {'x': 110.0, 'y': 10.0},
                'unit': 'yards',
            },
        )
        status_response = client.get('/metric/calibration/status')

    assert failed_calibration_response.status_code == 422
    assert failed_calibration_response.json()['error']['code'] == 'CALIBRATION_STALE'
    assert malformed_ruler_response.status_code == 422
    assert malformed_ruler_response.json()['error']['code'] == 'REQUEST_VALIDATION_ERROR'
    assert service.metric_calibration == metric_before
    assert service.ruler is ruler_before
    assert service.calibration_calls == []
    assert service.ruler_calls == []
    assert status_response.json()['state'] == MetricCalibrationStatus.CALIBRATED.value


def test_metric_api_failed_requests_preserve_camera_calibration_and_overlay_state() -> None:
    runtime = ApiMetricRuntime()
    resolution = Resolution(640, 480)
    descriptor = ProjectorOutputDescriptor(resolution)
    camera_calibration = _camera_calibration()
    runtime.registry.set_calibration(
        'camera-0',
        CalibrationStatus.CALIBRATED,
        camera_calibration,
    )
    service = MultiVisionService(
        Configuration(projector_resolution=resolution),
        camera_runtime=runtime,  # type: ignore[arg-type]
    )
    identity_matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    service.metric_registry.register(
        MetricCalibrationResult(
            MetricHomographyPair.from_projector_to_surface(identity_matrix),
            MetricCalibrationMetrics(20, 80, 80, 1.0, 0.0, 0.0, 0.75),
            resolution,
            METRIC_TARGET.format_name,
            METRIC_TARGET.format_version,
            METRIC_TARGET.marker_family,
            'camera-0',
        ),
        descriptor,
        'camera-0',
        camera_calibration,
    )
    ruler_before = service.set_metric_ruler((100.0, 100.0), (200.0, 100.0))
    overlay_before = service.point_from_camera('camera-0', (20.0, 20.0))
    metric_before = service.metric_calibration
    camera_before = runtime.registry.get('camera-0')

    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        malformed_calibration_response = client.post(
            '/metric/calibration',
            json={'camera': 'camera-0', 'unexpected': True},
        )
        failed_ruler_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': -1.0, 'y': 100.0},
                'to': {'x': 200.0, 'y': 100.0},
            },
        )
        malformed_measurement_response = client.post(
            '/metric/ruler',
            json={
                'from': {'x': 100.0, 'y': 100.0},
                'to': {'x': 200.0, 'y': 100.0},
                'observed_length': 0,
            },
        )

    assert malformed_calibration_response.status_code == 422
    assert malformed_calibration_response.json()['error']['code'] == (
        'REQUEST_VALIDATION_ERROR'
    )
    assert failed_ruler_response.status_code == 422
    assert failed_ruler_response.json()['error']['code'] == 'POINT_OUTSIDE_PROJECTOR_BOUNDS'
    assert malformed_measurement_response.status_code == 422
    assert malformed_measurement_response.json()['error']['code'] == (
        'REQUEST_VALIDATION_ERROR'
    )
    assert service.metric_calibration == metric_before
    assert service.metric_ruler is ruler_before
    assert runtime.registry.get('camera-0') == camera_before
    assert service.overlay == overlay_before


def test_metric_api_returns_structured_service_errors_and_clear_is_idempotent() -> None:
    service = RecordingMetricService()
    service.fail_calibration = True
    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        failure_response = client.post(
            '/metric/calibration',
            json={'camera': 'camera-1'},
        )
        first_clear = client.delete('/metric/calibration')
        second_clear = client.delete('/metric/calibration')
        first_ruler_clear = client.delete('/metric/ruler')
        second_ruler_clear = client.delete('/metric/ruler')

    assert failure_response.status_code == 422
    assert failure_response.json()['error']['code'] == 'CALIBRATION_STALE'
    assert first_clear.json() == {'cleared': True}
    assert second_clear.json() == {'cleared': True}
    assert first_ruler_clear.json() == {'cleared': True}
    assert second_ruler_clear.json() == {'cleared': True}
    assert service.clear_calibration_calls == 2
    assert service.clear_ruler_calls == 2