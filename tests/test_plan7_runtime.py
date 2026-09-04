"""Deterministic Plan7 integration coverage; hardware coexistence is not claimed."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from multivision.application import MultiVisionService
from multivision.config import Configuration, FiducialGroup
from multivision.fiducials import (
    DetectedMarker,
    FiducialObservation,
    build_fiducial_observation,
)
from multivision.geometry import Point2D
from multivision.metric import MetricCalibrationStatus
from multivision.overlays import ArrowRequest
from multivision.server import ApiServerRuntime
from multivision.session import SessionCameraRegistry
from multivision.spatial import SpatialState, SpatialTracker
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
)


class _Clock:
    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds

    def __call__(self) -> float:
        return self.seconds


class _Runtime:
    def __init__(self) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [DeviceInfo('device-0', 'Camera 0', 0, Resolution(640, 480))],
        )
        self.frame = Frame(object(), 1, 0.0)
        self.snapshot_calls = 0
        self.direct_snapshot_calls = 0
        self.is_started = False

    def start(self) -> None:
        self.is_started = True

    def shutdown(self) -> None:
        self.is_started = False

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        return CameraStatus(
            slot_id,
            'device-0',
            RuntimeStatus.AVAILABLE,
            camera.calibration_status,
            Resolution(640, 480),
            self.frame.frame_counter,
        )

    def snapshot_latest_frames(self) -> dict[str, Frame]:
        self.snapshot_calls += 1
        return {'camera-0': self.frame}

    def snapshot(self, _slot_id: str) -> Frame:
        self.direct_snapshot_calls += 1
        return self.frame

    def close_camera(self, slot_id: str) -> object:
        return self.registry.close(slot_id)

    def set_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: object,
    ) -> object:
        return self.registry.set_calibration(slot_id, calibration_status, calibration)


class _MetricAuthority:
    state = MetricCalibrationStatus.CALIBRATED
    is_usable = True
    projector_to_surface = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    surface_to_projector = projector_to_surface

    def __init__(self, descriptor: object) -> None:
        self.projector_output_descriptor = descriptor


class _SleepInhibitor:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


def _make_service(
    runtime: _Runtime,
    detector_calls: list[str],
) -> MultiVisionService:
    configuration = Configuration(
        fiducial_groups={
            'alpha': FiducialGroup('DICT_5X5_1000', 10.0),
            'beta': FiducialGroup('DICT_5X5_1000', 10.0),
        },
        preview_mode='off',
    )
    calibration = SimpleNamespace(
        camera_to_projector=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        camera_resolution=Resolution(640, 480),
        projector_output_descriptor=configuration.projector_output_descriptor,
        version=1,
        camera_id='camera-0',
    )
    runtime.set_calibration('camera-0', CalibrationStatus.CALIBRATED, calibration)
    service = MultiVisionService(
        configuration,
        camera_runtime=runtime,  # type: ignore[arg-type]
        sleep_inhibitor=_SleepInhibitor(),
        tag_detector_factory=lambda dictionary: SimpleNamespace(
            detect=lambda _frame: (
                detector_calls.append(dictionary)
                or (
                    DetectedMarker(
                        4,
                        (
                            Point2D(10, 10),
                            Point2D(20, 10),
                            Point2D(20, 20),
                            Point2D(10, 20),
                        ),
                    ),
                )
            ),
        ),
    )
    authority = _MetricAuthority(configuration.projector_output_descriptor)
    service.metric_calibration_registry = SimpleNamespace(
        is_usable=lambda _descriptor: True,
        get_record=lambda: authority,
    )
    return service


def test_tracking_worker_uses_retained_frames_and_all_configured_groups() -> None:
    runtime = _Runtime()
    detector_calls: list[str] = []
    service = _make_service(runtime, detector_calls)

    service.run_tracking_cycle()

    assert runtime.snapshot_calls == 1, f'{runtime.snapshot_calls=}'
    assert runtime.direct_snapshot_calls == 0, f'{runtime.direct_snapshot_calls=}'
    assert detector_calls == ['DICT_5X5_1000', 'DICT_5X5_1000'], f'{detector_calls=}'
    assert set(service.spatial_state.selected_observations) == {
        ('alpha', 4),
        ('beta', 4),
    }, f'{service.spatial_state=}'


def test_newer_tracking_cycle_wins_when_cycles_overlap() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    old_data = object()
    new_data = object()
    runtime.frame = Frame(old_data, 1, 0.0)
    detector_started = threading.Event()
    release_detector = threading.Event()

    class OverlappingDetector:
        def detect(self, frame_data: object) -> tuple[DetectedMarker, ...]:
            if frame_data is old_data:
                detector_started.set()
                release_detector.wait(1.0)
                x_pos = 10.0
            else:
                x_pos = 30.0
            return (
                DetectedMarker(
                    4,
                    (
                        Point2D(x_pos, 10),
                        Point2D(x_pos + 10, 10),
                        Point2D(x_pos + 10, 20),
                        Point2D(x_pos, 20),
                    ),
                ),
            )

    detector = OverlappingDetector()
    service.tag_detector_factory = lambda _dictionary: detector
    first_error: list[Exception] = []
    first_thread = threading.Thread(
        target=lambda: _run_tracking(service, first_error),
    )
    first_thread.start()
    assert detector_started.wait(1.0), f'{detector_started.is_set()=}'

    runtime.frame = Frame(new_data, 2, 0.0)
    second_error: list[Exception] = []
    second_thread = threading.Thread(
        target=lambda: _run_tracking(service, second_error),
    )
    second_thread.start()
    second_thread.join(1.0)
    assert not second_thread.is_alive(), 'newer tracking cycle did not finish'

    release_detector.set()
    first_thread.join(1.0)
    assert not first_thread.is_alive(), 'older tracking cycle did not finish'
    assert first_error == [], f'{first_error=}'
    assert second_error == [], f'{second_error=}'
    selected_observation = service.spatial_state.get_observation('alpha', 4)
    assert selected_observation is not None, f'{service.spatial_state=}'
    assert selected_observation.camera.centre.x == 35.0, f'{selected_observation=}'


def test_tracking_lifecycle_stops_worker_and_invalidates_before_camera_close() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    service.start()
    try:
        deadline = time.monotonic() + 1.0
        while service.tracking_thread is not None and service.tracking_thread.is_alive():
            if len(service.spatial_state.selected_observations) == 2:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert len(service.spatial_state.selected_observations) == 2, f'{service.spatial_state=}'
        service.close_camera('camera-0')
        assert len(service.spatial_state.selected_observations) == 0, f'{service.spatial_state=}'
    finally:
        service.shutdown()
    assert service.tracking_thread is not None
    assert not service.tracking_thread.is_alive()
    assert not runtime.is_started


def test_tracking_fails_closed_when_runtime_status_disconnects() -> None:
    runtime = _Runtime()
    detector_calls: list[str] = []
    service = _make_service(runtime, detector_calls)

    service.run_tracking_cycle()
    assert len(service.spatial_state.selected_observations) == 2, f'{service.spatial_state=}'

    with patch.object(runtime, 'get_status', side_effect=RuntimeError('device lost')):
        service.run_tracking_cycle()

    assert len(service.spatial_state.selected_observations) == 0, f'{service.spatial_state=}'
    assert len(detector_calls) == 2, f'{detector_calls=}'


def test_tracking_fails_closed_for_stale_runtime_calibration_status() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    service.run_tracking_cycle()
    assert len(service.spatial_state.selected_observations) == 2, f'{service.spatial_state=}'

    stale_status = runtime.get_status('camera-0')._replace(
        calibration_status=CalibrationStatus.STALE,
    )
    with patch.object(runtime, 'get_status', return_value=stale_status):
        service.run_tracking_cycle()

    assert len(service.spatial_state.selected_observations) == 0, f'{service.spatial_state=}'


def test_failed_camera_recalibration_invalidates_spatial_state_before_capture() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    service.run_tracking_cycle()
    previous_state = service.spatial_state

    with patch.object(
        service,
        '_get_correspondences_for_operation',
        side_effect=ValueError('capture failed'),
    ):
        with pytest.raises(ValueError):
            service.calibrate('camera-0', ())

    assert service.spatial_state is not previous_state
    assert len(service.spatial_state.selected_observations) == 0, f'{service.spatial_state=}'


def test_tag_inspection_does_not_hold_camera_management_lock_during_detection() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    detector_started = threading.Event()
    release_detector = threading.Event()

    class BlockingDetector:
        def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
            detector_started.set()
            release_detector.wait(1.0)
            return ()

    service.tag_detector_factory = lambda _dictionary: BlockingDetector()
    service._get_tag_projection_status = lambda _slot: None  # type: ignore[method-assign]
    inspection_error: list[Exception] = []

    inspection_thread = threading.Thread(
        target=lambda: _run_inspection(service, inspection_error),
    )
    inspection_thread.start()
    assert detector_started.wait(1.0), f'{detector_started.is_set()=}'

    close_thread = threading.Thread(target=service.close_camera, args=('camera-0',))
    close_thread.start()
    close_thread.join(0.2)
    assert not close_thread.is_alive(), 'camera lifecycle is blocked by tag detection'

    release_detector.set()
    inspection_thread.join(1.0)
    assert not inspection_thread.is_alive(), 'tag inspection did not finish'
    assert inspection_error == [], f'{inspection_error=}'


def test_tracking_drops_detector_results_when_runtime_disconnects_during_detection() -> None:
    runtime = _Runtime()
    service = _make_service(runtime, [])
    service.run_tracking_cycle()
    detector_started = threading.Event()
    release_detector = threading.Event()

    class BlockingDetector:
        def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
            detector_started.set()
            release_detector.wait(1.0)
            return (
                DetectedMarker(
                    4,
                    (
                        Point2D(10, 10),
                        Point2D(20, 10),
                        Point2D(20, 20),
                        Point2D(10, 20),
                    ),
                ),
            )

    service.tag_detector_factory = lambda _dictionary: BlockingDetector()
    tracking_error: list[Exception] = []
    tracking_thread = threading.Thread(
        target=lambda: _run_tracking(service, tracking_error),
    )
    tracking_thread.start()
    assert detector_started.wait(1.0), f'{detector_started.is_set()=}'

    with patch.object(runtime, 'get_status', side_effect=RuntimeError('device lost')):
        release_detector.set()
        tracking_thread.join(1.0)

    assert not tracking_thread.is_alive(), 'tracking did not finish'
    assert tracking_error == [], f'{tracking_error=}'
    assert len(service.spatial_state.selected_observations) == 0, f'{service.spatial_state=}'


def _run_tracking(
    service: MultiVisionService,
    tracking_error: list[Exception],
) -> None:
    try:
        service.run_tracking_cycle()
    except Exception as ex:  # noqa: BLE001 (The thread reports test failures.)
        tracking_error.append(ex)


def _run_inspection(
    service: MultiVisionService,
    inspection_error: list[Exception],
) -> None:
    try:
        service.inspect_tags('camera-0')
    except Exception as ex:  # noqa: BLE001 (The thread reports test failures.)
        inspection_error.append(ex)


def test_dynamic_arrow_snapshot_retains_intent_through_grace_and_recovers() -> None:
    clock = _Clock()
    runtime = _Runtime()
    service = _make_service(runtime, [])
    authority = service.metric_calibration_registry.get_record()
    tracker = SpatialTracker(
        grace_period_seconds=5.0,
        metric_calibration=authority,
        clock=clock,
    )
    initial_state = tracker.update(
        (
            _spatial_observation('alpha', 7, 10.0, 1, 0.0, authority),
            _spatial_observation('beta', 7, 40.0, 1, 0.0, authority),
        ),
    )
    service.update_spatial_state(initial_state)
    request = ArrowRequest(
        start={
            'type': 'fiducial',
            'group': 'alpha',
            'id': 7,
            'local_offset': {'x': 5, 'y': 0, 'unit': 'mm'},
            'follow_rotation': True,
        },
        end={'type': 'fiducial', 'group': 'beta', 'id': 7},
        geometry_space='surface_mm',
        head_length={'value': 5, 'unit': 'mm'},
        head_width={'value': 3, 'unit': 'mm'},
        style={'colour': '#ffffff', 'intensity': 0.5},
    )

    created = service.create_overlay(request)
    visible_snapshot = service.get_render_snapshot()

    assert created.is_dynamic
    assert created.is_resolved
    assert visible_snapshot.registry_snapshot[0].request == request
    assert visible_snapshot.overlays[0].is_resolved
    assert created.materialised_primitives.segments[0].start == Point2D(15, 100)
    assert visible_snapshot.overlays[0].materialised_primitives.segments[0].start == Point2D(20, 100)
    assert len(visible_snapshot.protected_projector_regions) == 2

    clock.seconds = 5.0
    expired_state = tracker.update(())
    service.update_spatial_state(expired_state)
    hidden_snapshot = service.get_render_snapshot()

    assert hidden_snapshot.registry_snapshot[0].request == request
    assert not hidden_snapshot.overlays[0].is_resolved
    assert hidden_snapshot.overlays[0].materialised_primitives.segments == ()

    clock.seconds = 5.001
    recovered_state = tracker.update(
        (
            _spatial_observation('alpha', 7, 20.0, 2, 5.001, authority),
            _spatial_observation('beta', 7, 40.0, 2, 5.001, authority),
        ),
    )
    service.update_spatial_state(recovered_state)
    recovered_snapshot = service.get_render_snapshot()

    assert recovered_snapshot.overlays[0].is_resolved
    assert recovered_snapshot.overlays[0].request == request
    assert recovered_snapshot.spatial_state is recovered_state


def test_dynamic_zero_length_arrow_is_retained_as_unresolved() -> None:
    clock = _Clock()
    runtime = _Runtime()
    service = _make_service(runtime, [])
    authority = service.metric_calibration_registry.get_record()
    tracker = SpatialTracker(
        metric_calibration=authority,
        clock=clock,
    )
    service.update_spatial_state(
        tracker.update((_spatial_observation('alpha', 7, 10.0, 1, 0.0, authority),)),
    )
    request = ArrowRequest(
        start={'type': 'fiducial', 'group': 'alpha', 'id': 7},
        end={'type': 'surface', 'x': 10, 'y': 100, 'unit': 'mm'},
        geometry_space='surface_mm',
        head_length={'value': 5, 'unit': 'mm'},
        head_width={'value': 3, 'unit': 'mm'},
    )

    entry = service.create_overlay(request)
    snapshot = service.get_render_snapshot()

    assert entry.unresolved
    assert snapshot.registry_snapshot[0].request == request
    assert snapshot.overlays[0].unresolved
    assert snapshot.overlays[0].materialised_primitives.segments == ()


def test_runtime_batch_is_atomic_over_http_in_a_running_api_process() -> None:
    runtime = _Runtime()
    service = MultiVisionService(Configuration(), camera_runtime=runtime)  # type: ignore[arg-type]
    port = _reserve_port()
    api_runtime = ApiServerRuntime(service, port=port, startup_timeout_seconds=5.0)
    service.start()
    try:
        api_runtime.start()
        base_url = f'http://127.0.0.1:{port}'
        overlay_id = '123e4567-e89b-42d3-a456-426614174000'
        line_request = {
            'kind': 'line',
            'id': overlay_id,
            'start': {'type': 'projector', 'x': 2, 'y': 2, 'unit': 'px'},
            'end': {'type': 'projector', 'x': 30, 'y': 20, 'unit': 'px'},
        }
        status, created = _http_json(
            f'{base_url}/overlays/batch',
            {
                'operations': [
                    {'op': 'create', 'request': line_request},
                ],
            },
        )
        assert status == 200, f'{status=}, {created=}'
        assert created['overlays'][0]['id'] == overlay_id

        status, failed = _http_json(
            f'{base_url}/overlays/batch',
            {
                'operations': [
                    {
                        'op': 'create',
                        'request': {
                            **line_request,
                            'id': '123e4567-e89b-42d3-a456-426614174001',
                        },
                    },
                    {
                        'op': 'remove',
                        'selector': '123e4567-e89b-42d3-a456-426614174002',
                    },
                ],
            },
        )
        assert status == 404, f'{status=}, {failed=}'
        assert failed['error']['code'] == 'OVERLAY_NOT_FOUND', f'{failed=}'

        status, listing = _http_json(f'{base_url}/overlays')
        assert status == 200, f'{status=}, {listing=}'
        assert [entry['id'] for entry in listing] == [overlay_id], f'{listing=}'
    finally:
        try:
            api_runtime.shutdown()
        finally:
            service.shutdown()


def _spatial_observation(
    group: str,
    marker_id: int,
    x_pos: float,
    frame_counter: int,
    timestamp: float,
    metric_calibration: object,
) -> FiducialObservation:
    return build_fiducial_observation(
        DetectedMarker(
            marker_id,
            (
                Point2D(x_pos - 5, 95),
                Point2D(x_pos + 5, 95),
                Point2D(x_pos + 5, 105),
                Point2D(x_pos - 5, 105),
            ),
        ),
        group,
        10.0,
        camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        projector_to_surface=metric_calibration,
        camera_slot='camera-0',
        frame_counter=frame_counter,
        received_monotonic_seconds=timestamp,
    )


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(('127.0.0.1', 0))
        return int(socket_handle.getsockname()[1])


def _http_json(
    url: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, object]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode('utf-8'),
        headers={} if payload is None else {'Content-Type': 'application/json'},
        method='GET' if payload is None else 'POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read().decode('utf-8'))
