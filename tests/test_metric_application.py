import math
import threading

import numpy
import time
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

from multivision.application import (
    MultiVisionService,
    _select_stable_frame_window,
)
from multivision.config import Configuration, ProjectorOutputDescriptor
from multivision.errors import (
    CalibrationError,
    CameraUnavailableError,
    InvalidCalibrationStateError,
    PointOutsideProjectorError,
)
from multivision.fiducials import (
    DetectedMarker,
    MetricTargetCorrespondence,
    MetricTargetCorrespondences,
)
from multivision.geometry import Point2D
from multivision.metric import (
    MetricCalibrationMetrics,
    MetricCalibrationResult,
    MetricCalibrationStatus,
    MetricHomographyPair,
)
from multivision.metric_target import METRIC_TARGET
from multivision.overlays import ProjectorCoverageGridRequest
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
)


IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class MetricApplicationTest(unittest.TestCase):
    def test_one_metric_record_is_shared_by_all_camera_slots(self) -> None:
        runtime = MetricSessionRuntime(camera_count=2)
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        for camera_id in ('camera-0', 'camera-1'):
            runtime.registry.set_calibration(
                camera_id,
                CalibrationStatus.CALIBRATED,
                _camera_calibration(configuration),
            )
        service = MultiVisionService(configuration, camera_runtime=runtime)

        with patch('multivision.application.calibrate_metric_homography') as estimator:
            estimator.side_effect = [
                _metric_result(configuration.projector_resolution, 'camera-0'),
                _metric_result(configuration.projector_resolution, 'camera-1'),
            ]
            first_record = service.calibrate_metric(
                'camera-0',
                _build_correspondences(0.0, 'camera-0'),
            )
            second_record = service.calibrate_metric(
                'camera-1',
                _build_correspondences(0.0, 'camera-1'),
            )

        assert service.metric_calibration is second_record
        assert service.metric_registry.get_record() is second_record
        assert first_record is not second_record
        assert second_record.observation_camera_slot == 'camera-1'
        assert second_record.observation_camera_id == 'camera-1'
        assert all(
            not hasattr(camera, 'metric_calibration')
            and not hasattr(camera, 'mm_per_pixel')
            for camera in runtime.registry.get_cameras()
        ), f'{runtime.registry.get_cameras()=}'

    def test_calibration_stage_reports_metric_completion_separately(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            _camera_calibration(configuration),
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)

        assert service.get_calibration_stage().value == 'CALIBRATED'
        with patch('multivision.application.calibrate_metric_homography') as estimator:
            estimator.return_value = _metric_result(configuration.projector_resolution)
            service.calibrate_metric('camera-0', _build_correspondences(0.0))

        assert service.get_calibration_stage().value == 'METRIC_CALIBRATED'

    def test_projector_coverage_grid_uses_the_current_metric_output_footprint(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )

        entry = service.create_projector_coverage_grid(
            ProjectorCoverageGridRequest(
                name='projector-grid',
                spacing={'value': 35, 'unit': 'mm'},
            ),
        )

        assert entry.request.origin.x == -30.0, f'{entry=}'
        assert entry.request.origin.y == -5.0, f'{entry=}'
        assert entry.request.extent.width.value == 700.0, f'{entry=}'
        assert entry.request.extent.height.value == 490.0, f'{entry=}'
        assert entry.request.spacing.value == 35.0, f'{entry=}'
        assert len(service.list_overlays()) == 1, f'{service.list_overlays()=}'

    def test_metric_capture_selects_a_white_balance_stable_frame_window(self) -> None:
        frames = tuple(
            Frame(
                numpy.full((4, 4, 3), values, dtype=numpy.uint8),
                frame_counter,
                1.0,
            )
            for frame_counter, values in enumerate(
                (
                    (100, 100, 100),
                    (120, 80, 100),
                    (100, 100, 100),
                    (101, 100, 100),
                    (100, 101, 100),
                ),
                start=1,
            )
        )

        selected = _select_stable_frame_window(frames, 0.01)

        assert [frame.frame_counter for frame in selected] == [3, 4, 5], (
            f'{selected=}'
        )

    def test_metric_capture_requires_a_calibration_for_the_current_output(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(
            projector_resolution=Resolution(640, 480),
            projector_output_identity='projector-current',
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=IDENTITY_MATRIX,
                projector_output_descriptor=ProjectorOutputDescriptor(
                    configuration.projector_resolution,
                    'projector-other',
                ),
                camera_resolution=Resolution(640, 480),
                version=configuration.calibration_version,
                timestamp=1.0,
            ),
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)

        with self.assertRaises(InvalidCalibrationStateError) as raised:
            service.calibrate_metric('camera-0', _build_correspondences(0.0))

        assert raised.exception.code == 'CALIBRATION_STALE'
        assert service.metric_state is MetricCalibrationStatus.UNCALIBRATED
        assert service.metric_calibration is None
        assert not service.metric_capture_active

    def test_injected_metric_frames_are_strictly_validated_before_estimation(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        malformed_frame = MetricTargetCorrespondences(
            tuple(
                correspondence._replace(marker_id=99)
                if correspondence.marker_id == 0
                else correspondence
                for correspondence in _build_correspondences(0.0).correspondences
            ),
            'camera-0',
        )
        estimator_called = False

        def fake_estimator(*_args: object, **_kwargs: object) -> MetricCalibrationResult:
            nonlocal estimator_called
            estimator_called = True
            return _metric_result(configuration.projector_resolution)

        with patch('multivision.application.calibrate_metric_homography', fake_estimator):
            with self.assertRaises(CalibrationError):
                service.calibrate_metric(
                    'camera-0',
                    (malformed_frame, malformed_frame, malformed_frame),
                )

        assert not estimator_called
        assert service.metric_calibration is None
        assert not service.metric_capture_active

    def test_metric_capture_averages_stable_frames_and_commits_atomically(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        old_result = _metric_result(configuration.projector_resolution)
        old_record = service.metric_registry.register(
            old_result,
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )
        service._metric_ruler = object()
        camera_before = runtime.registry.get('camera-0')
        observed: list[MetricTargetCorrespondences] = []

        def fake_estimator(
            correspondences: MetricTargetCorrespondences,
            *_args: object,
            **_kwargs: object,
        ) -> MetricCalibrationResult:
            observed.append(correspondences)
            assert service.metric_capture_active
            assert service.metric_calibration is None
            assert service.metric_ruler is None
            return _metric_result(configuration.projector_resolution)

        frames = (
            _build_correspondences(0.0),
            MetricTargetCorrespondences(
                tuple(reversed(_build_correspondences(0.5).correspondences)),
            ),
            _build_correspondences(1.0),
        )
        with patch('multivision.application.calibrate_metric_homography', fake_estimator):
            record = service.calibrate_metric('camera-0', frames)

        assert observed[0].correspondences[0].camera_position.x == 0.5 + 8.0
        assert record is service.metric_calibration
        assert record.state is MetricCalibrationStatus.CALIBRATED
        assert record.observation_camera_slot == 'camera-0'
        assert old_record is not record
        assert runtime.registry.get('camera-0') == camera_before
        assert not service.metric_capture_active

    def test_metric_capture_precondition_failure_clears_old_record(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )

        runtime.registry.close('camera-0')
        with self.assertRaises(CameraUnavailableError):
            service.calibrate_metric('camera-0', _build_correspondences(0.0))

        assert service.metric_calibration is None
        assert service.metric_state is MetricCalibrationStatus.UNCALIBRATED
        assert service.metric_ruler is None

    def test_metric_capture_estimation_failure_is_atomic(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )
        service._metric_ruler = object()
        camera_before = runtime.registry.get('camera-0')

        def fail_estimation(*_args: object, **_kwargs: object) -> MetricCalibrationResult:
            raise CalibrationError('synthetic estimator failure')

        with patch(
            'multivision.application.calibrate_metric_homography',
            fail_estimation,
        ):
            with self.assertRaises(CalibrationError):
                service.calibrate_metric('camera-0', _build_correspondences(0.0))

        assert service.metric_calibration is None
        assert service.metric_state is MetricCalibrationStatus.UNCALIBRATED
        assert service.metric_ruler is None
        assert not service.metric_capture_active
        assert runtime.registry.get('camera-0') == camera_before

    def test_metric_capture_cannot_commit_after_an_explicit_reset(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        estimation_started = threading.Event()
        release_estimation = threading.Event()
        errors: list[BaseException] = []

        def blocked_estimation(
            *_args: object,
            **_kwargs: object,
        ) -> MetricCalibrationResult:
            estimation_started.set()
            assert release_estimation.wait(1.0)
            return _metric_result(configuration.projector_resolution)

        def calibrate_metric() -> None:
            try:
                service.calibrate_metric('camera-0', _build_correspondences(0.0))
            except BaseException as ex:  # noqa: BLE001 (Test thread must report failures).
                errors.append(ex)

        with patch(
            'multivision.application.calibrate_metric_homography',
            blocked_estimation,
        ):
            calibration_thread = threading.Thread(target=calibrate_metric)
            calibration_thread.start()
            assert estimation_started.wait(1.0)
            service.clear_metric_calibration()
            release_estimation.set()
            calibration_thread.join(1.0)

        assert not calibration_thread.is_alive()
        assert len(errors) == 1, f'{errors=}'
        assert isinstance(errors[0], CalibrationError), f'{errors=}'
        assert service.metric_calibration is None
        assert service.metric_state is MetricCalibrationStatus.UNCALIBRATED
        assert not service.metric_capture_active

    def test_metric_capture_rejects_movement_and_leaves_no_old_record(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )
        service._metric_ruler = object()
        camera_before = runtime.registry.get('camera-0')

        with self.assertRaises(CalibrationError):
            service.calibrate_metric(
                'camera-0',
                (
                    _build_correspondences(0.0),
                    _build_correspondences(3.0),
                    _build_correspondences(6.0),
                ),
            )

        assert service.metric_calibration is None
        assert service.metric_state is MetricCalibrationStatus.UNCALIBRATED
        assert service.metric_ruler is None
        assert not service.metric_capture_active
        assert runtime.registry.get('camera-0') == camera_before

        disagreeing_frame = MetricTargetCorrespondences(
            tuple(
                correspondence._replace(marker_id=99)
                if correspondence.marker_id == 0
                else correspondence
                for correspondence in _build_correspondences(0.0).correspondences
            ),
            'camera-0',
        )
        with self.assertRaises(CalibrationError):
            service.calibrate_metric(
                'camera-0',
                (
                    _build_correspondences(0.0),
                    disagreeing_frame,
                    _build_correspondences(0.0),
                ),
            )
        assert service.metric_calibration is None
        assert not service.metric_capture_active

    def test_metric_capture_waits_for_main_thread_blank_acknowledgement(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            _camera_calibration(configuration),
        )
        detector = StaticMetricDetector()
        service = MultiVisionService(
            configuration,
            camera_runtime=runtime,
            detector=detector,  # type: ignore[arg-type]
        )
        errors: list[BaseException] = []

        def capture_metric() -> None:
            try:
                service.calibrate_metric('camera-0')
            except BaseException as ex:  # noqa: BLE001 (Test thread must report failures).
                errors.append(ex)

        with patch('multivision.application.METRIC_CAPTURE_SETTLE_SECONDS', 0.0):
            capture_thread = threading.Thread(target=capture_metric)
            capture_thread.start()
            assert _wait_for(lambda: service.metric_capture_active)
            assert runtime.snapshot_count == 0
            assert detector.call_count == 0

            service.mark_metric_capture_presented()
            capture_thread.join(10.0)

        assert not capture_thread.is_alive()
        assert errors == [], f'{errors=}'
        assert runtime.snapshot_count == 3
        assert detector.call_count == 3
        assert service.metric_state is MetricCalibrationStatus.CALIBRATED

    def test_metric_capture_rejects_reusing_one_retained_camera_frame(self) -> None:
        runtime = MetricSessionRuntime(repeat_frame_counter=True)
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            _camera_calibration(configuration),
        )
        service = MultiVisionService(
            configuration,
            camera_runtime=runtime,
            detector=StaticMetricDetector(),  # type: ignore[arg-type]
        )
        service._begin_metric_capture()
        service.mark_metric_capture_presented()
        try:
            with patch('multivision.application.METRIC_CAPTURE_SETTLE_SECONDS', 0.0):
                with self.assertRaises(CalibrationError):
                    service._get_metric_capture_frames('camera-0', None)
        finally:
            service._finish_metric_capture()

        assert not service.metric_capture_active

    def test_metric_capture_rejects_skipped_camera_frames(self) -> None:
        runtime = MetricSessionRuntime(frame_counters=(1, 3, 4))
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            _camera_calibration(configuration),
        )
        service = MultiVisionService(
            configuration,
            camera_runtime=runtime,
            detector=StaticMetricDetector(),  # type: ignore[arg-type]
        )
        service._begin_metric_capture()
        service.mark_metric_capture_presented()
        try:
            with patch('multivision.application.METRIC_CAPTURE_SETTLE_SECONDS', 0.0):
                with self.assertRaises(CalibrationError):
                    service._get_metric_capture_frames('camera-0', None)
        finally:
            service._finish_metric_capture()

        assert not service.metric_capture_active

    def test_physical_validation_is_append_only_and_latest_error_is_explicit(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )
        before = service.metric_calibration
        assert before is not None

        first_validation = service.record_physical_validation(
            10.0,
            9.0,
            'cm',
            'cm',
        )
        second_validation = service.record_physical_validation(
            1.0,
            0.5,
            'in',
            'in',
        )
        after = service.metric_calibration

        assert first_validation is not None
        assert second_validation is not None
        assert after is not None
        assert after.validation_records == (first_validation, second_validation)
        assert after.latest_physical_validation_error_mm == (
            second_validation.absolute_error_mm
        )
        assert after.latest_physical_validation_error_mm != before.fit_error_mm
        assert after.projector_to_surface == before.projector_to_surface

    def test_physical_validation_is_optional_and_does_not_replace_metric_geometry(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )
        before = service.metric_calibration
        assert before is not None

        assert service.record_physical_validation(100.0) is None
        assert service.metric_calibration == before
        validation = service.record_physical_validation(100.0, 10.0, 'mm', 'cm')
        after = service.metric_calibration
        assert validation is not None
        assert validation.requested_length_mm == 100.0, f'{validation=}'
        assert validation.observed_length_mm == 100.0, f'{validation=}'
        assert validation.absolute_error_mm == 0.0, f'{validation=}'
        assert after is not None
        assert after.projector_to_surface == before.projector_to_surface
        assert after.latest_physical_validation_error_mm == 0.0
        assert after.validation_records == (validation,)

    def test_metric_ruler_replacement_clear_and_point_overlay_coexist(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        camera_calibration = _camera_calibration(configuration)
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            camera_calibration,
        )
        service = MultiVisionService(configuration, camera_runtime=runtime)
        service.metric_registry.register(
            _metric_result(configuration.projector_resolution),
            configuration.projector_output_descriptor,
            'camera-0',
            camera_calibration,
        )

        first_ruler = service.set_metric_ruler((100.0, 100.0), (200.0, 100.0), 'mm')
        point_overlay = service.point_from_camera('camera-0', (50.0, 50.0))
        second_ruler = service.set_metric_ruler((120.0, 120.0), (220.0, 120.0), 'cm')

        assert service.metric_ruler is second_ruler
        assert second_ruler is not first_ruler
        assert second_ruler.label == '10.0 cm', f'{second_ruler=}'
        assert service.overlay is point_overlay, f'{service.overlay=}'
        with self.assertRaises(PointOutsideProjectorError):
            service.set_metric_ruler((3.9, 50.0), (80.0, 50.0), 'mm')
        assert service.metric_ruler is second_ruler, f'{service.metric_ruler=}'
        service.clear_metric_ruler()
        assert service.metric_ruler is None
        assert service.overlay is point_overlay, f'{service.overlay=}'
        assert service.metric_calibration is not None
        assert service.metric_state is MetricCalibrationStatus.CALIBRATED

    def test_metric_ruler_requires_current_metric_calibration(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        service = MultiVisionService(configuration, camera_runtime=runtime)

        with self.assertRaises(InvalidCalibrationStateError):
            service.set_metric_ruler((100.0, 100.0), (200.0, 100.0), 'mm')

        with self.assertRaises(InvalidCalibrationStateError):
            service.set_metric_ruler((math.nan, 100.0), (200.0, 100.0), 'mm')

    def test_metric_capture_requires_an_open_available_calibrated_output(self) -> None:
        runtime = MetricSessionRuntime()
        configuration = Configuration(projector_resolution=Resolution(640, 480))
        service = MultiVisionService(configuration, camera_runtime=runtime)

        with self.assertRaises(InvalidCalibrationStateError):
            service.calibrate_metric('camera-0', _build_correspondences(0.0))

        runtime.registry.close('camera-0')
        with self.assertRaises(CameraUnavailableError):
            service.calibrate_metric('camera-0', _build_correspondences(0.0))

        runtime.registry.open('camera-0')
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(camera_to_projector=IDENTITY_MATRIX),
        )
        with self.assertRaises(InvalidCalibrationStateError):
            service.calibrate_metric('camera-0', _build_correspondences(0.0))


class MetricSessionRuntime:
    def __init__(
        self,
        repeat_frame_counter: bool = False,
        frame_counters: tuple[int, ...] | None = None,
        camera_count: int = 1,
    ) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    f'device-{camera_index}',
                    f'Camera {camera_index}',
                    capture_index=camera_index,
                )
                for camera_index in range(camera_count)
            ],
        )
        self.snapshot_count = 0
        self.repeat_frame_counter = repeat_frame_counter
        self.frame_counters = frame_counters

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_statuses(self) -> list[CameraStatus]:
        return [
            self.get_status(camera.slot_id)
            for camera in self.registry.get_cameras()
        ]

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        return CameraStatus(
            slot_id,
            camera.device_info.device_id if camera.device_info is not None else None,
            RuntimeStatus.AVAILABLE,
            camera.calibration_status,
            Resolution(640, 480),
        )

    def snapshot(self, _slot_id: str) -> Frame:
        self.snapshot_count += 1
        if self.frame_counters is not None:
            frame_counter = self.frame_counters[self.snapshot_count - 1]
        else:
            frame_counter = 1 if self.repeat_frame_counter else self.snapshot_count
        return Frame(object(), frame_counter, time.time())


class StaticMetricDetector:
    def __init__(self) -> None:
        self.call_count = 0

    def detect(self, _frame: object) -> tuple[DetectedMarker, ...]:
        self.call_count += 1
        return tuple(
            DetectedMarker(marker.marker_id, marker.corners)
            for marker in METRIC_TARGET.markers
        )


def _camera_calibration(configuration: Configuration) -> SimpleNamespace:
    return SimpleNamespace(
        camera_to_projector=IDENTITY_MATRIX,
        projector_output_descriptor=configuration.projector_output_descriptor,
        camera_resolution=Resolution(640, 480),
        version=configuration.calibration_version,
        timestamp=1.0,
    )


def _build_correspondences(
    offset_x: float,
    camera_id: str = 'camera-0',
) -> MetricTargetCorrespondences:
    return MetricTargetCorrespondences(
        tuple(
            MetricTargetCorrespondence(
                marker.marker_id,
                corner_index,
                surface_point,
                Point2D(surface_point.x + offset_x, surface_point.y),
            )
            for marker in METRIC_TARGET.markers
            for corner_index, surface_point in enumerate(marker.corners)
        ),
        camera_id,
    )


def _metric_result(
    projector_resolution: Resolution,
    camera_id: str = 'camera-0',
) -> MetricCalibrationResult:
    return MetricCalibrationResult(
        MetricHomographyPair.from_projector_to_surface(IDENTITY_MATRIX),
        MetricCalibrationMetrics(20, 80, 80, 1.0, 0.0, 0.0, 0.75),
        projector_resolution,
        METRIC_TARGET.format_name,
        METRIC_TARGET.format_version,
        METRIC_TARGET.marker_family,
        camera_id,
    )


def _wait_for(
    predicate: Callable[[], bool],
    timeout_seconds: float = 1.0,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


if __name__ == '__main__':
    unittest.main()
