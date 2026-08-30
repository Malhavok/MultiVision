import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from multivision.application import (
    AREA_COLOURS,
    MultiVisionService,
    _aggregate_camera_correspondences,
)
from multivision.calibration import CalibrationMetrics, CalibrationResult
from multivision.errors import (
    CalibrationError,
    CameraUnavailableError,
    InvalidAvailableAreaError,
    InvalidCalibrationStateError,
)
from multivision.fiducials import (
    CameraCorrespondences,
    FiducialCorrespondence,
)
from multivision.config import Configuration
from multivision.geometry import HomographyPair, Point2D
from multivision.overlays import (
    LineRequest,
    ProjectorMaterialisation,
)
from multivision.persistence import PersistedCalibration
from multivision.service import PointOverlayService
from multivision.session import FrameMetadata, SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
)


class CameraCaptureAggregationTest(unittest.TestCase):
    def test_common_tags_are_relative_to_median_frame_count(self) -> None:
        def make_frame(extra_ids: tuple[int, ...], offset: float) -> CameraCorrespondences:
            values = []
            for marker_id in extra_ids:
                for corner_index in range(4):
                    projector_position = Point2D(
                        marker_id * 30 + (corner_index % 2) * 10,
                        (corner_index // 2) * 10,
                    )
                    values.append(
                        FiducialCorrespondence(
                            marker_id,
                            corner_index,
                            projector_position,
                            Point2D(
                                projector_position.x + offset,
                                projector_position.y + offset,
                            ),
                        ),
                    )
            return CameraCorrespondences(tuple(values), 'camera-0')

        aggregated, noise = _aggregate_camera_correspondences(
            (
                make_frame((0, 1, 2, 3), 0.0),
                make_frame((0, 1, 4, 5), 0.5),
                make_frame((0, 1, 6, 7), 1.0),
            ),
        )

        assert aggregated.unique_marker_ids == (0, 1)
        assert len(aggregated.correspondences) == 8
        assert noise is not None
        assert noise.median_sigma_pixels > 0


class FakeSessionRuntime:
    def __init__(self, capture_indexes: tuple[int, ...] = (0, 1)) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    f'device-{capture_index}',
                    f'Camera {capture_index}',
                    capture_index=capture_index,
                )
                for capture_index in capture_indexes
            ],
        )

    def get_session_cameras(self) -> list[object]:
        return list(reversed(self.registry.get_cameras()))

    def rename_camera(self, slot_id: str, display_name: str) -> object:
        return self.registry.rename(slot_id, display_name)

    def close_camera(self, slot_id: str) -> object:
        return self.registry.close(slot_id)

    def open_camera(self, slot_id: str) -> object:
        return self.registry.open(slot_id)


class CalibrationSessionRuntime(FakeSessionRuntime):
    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        runtime_status = {
            'OPEN': RuntimeStatus.AVAILABLE,
            'CLOSED': RuntimeStatus.STOPPED,
            'UNAVAILABLE': RuntimeStatus.UNAVAILABLE,
        }[camera.state.value]
        return CameraStatus(
            slot_id,
            None,
            runtime_status,
            camera.calibration_status,
            Resolution(640, 480),
        )

    def get_statuses(self) -> list[CameraStatus]:
        return [
            self.get_status(camera.slot_id)
            for camera in self.registry.get_cameras()
        ]

    def set_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: object,
    ) -> object:
        return self.registry.set_calibration(
            slot_id,
            calibration_status,
            calibration,
        )

    def set_area_enabled(self, slot_id: str, area_enabled: bool) -> object:
        return self.registry.set_area_enabled(slot_id, area_enabled)


class FakePointService:
    def __init__(self) -> None:
        self.renamed: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    def rename_overlay_camera(self, camera_id: str, logical_name: str) -> None:
        self.renamed.append((camera_id, logical_name))

    def clear_overlay_for_camera(self, camera_id: str) -> None:
        self.cleared.append(camera_id)


class MultiVisionServiceAreaTest(unittest.TestCase):
    def test_area_is_derived_on_enable_and_disabled_without_calibration_mutation(self) -> None:
        runtime = CalibrationSessionRuntime()
        calibration = SimpleNamespace(
            camera_to_projector=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_region=((-10, 10), (500, 10), (500, 400), (-10, 400)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=PointOverlayService(
                runtime,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                Resolution(1000, 700),
            ),
        )

        initial_area = service.get_camera_area('camera-0')
        assert initial_area.area_enabled is False, f'{initial_area=}'
        assert initial_area.available_area is None, f'{initial_area=}'
        calculated_area = service.calculate_available_area('camera-0')
        assert calculated_area == (
            Point2D(0.0, 0.0),
            Point2D(640.0, 0.0),
            Point2D(640.0, 480.0),
            Point2D(0.0, 480.0),
        ), f'{calculated_area=}'

        enabled_area = service.set_area_enabled('camera-0', True)
        assert enabled_area.area_enabled is True, f'{enabled_area=}'
        assert enabled_area.available_area == calculated_area, f'{enabled_area=}'
        assert runtime.registry.get('camera-0').calibration == calibration
        overlay = service.point_from_camera('camera-0', (600, 450))
        assert overlay.camera_point == Point2D(600, 450), f'{overlay=}'

        disabled_area = service.set_area_enabled('camera-0', False)
        assert disabled_area.area_enabled is False, f'{disabled_area=}'
        assert disabled_area.available_area is None, f'{disabled_area=}'
        assert runtime.registry.get('camera-0').calibration == calibration

    def test_independent_areas_preserve_pointing_and_rename_overlay_semantics(self) -> None:
        runtime = CalibrationSessionRuntime()
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
            ),
        )
        runtime.registry.set_calibration(
            'camera-1',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=((1, 0, 100), (0, 1, 0), (0, 0, 1)),
                valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
            ),
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=PointOverlayService(
                runtime,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                Resolution(1000, 700),
            ),
        )

        first_overlay = service.point_from_camera('camera-0', (100, 100))
        first_area = service.set_area_enabled('camera-0', True)
        second_area = service.set_area_enabled('camera-1', True)

        assert first_area.available_area is not None
        assert second_area.available_area is not None
        assert first_area.available_area != second_area.available_area, (
            f'{first_area=}, {second_area=}'
        )
        assert first_area.area_colour != second_area.area_colour
        assert service.overlay is first_overlay, f'{service.overlay=}'

        renamed_camera = service.rename_camera('camera-0', 'overhead')
        renamed_area = service.get_camera_area('camera-0')
        assert renamed_camera.display_name == 'overhead', f'{renamed_camera=}'
        assert renamed_area.display_name == 'overhead', f'{renamed_area=}'
        assert renamed_area.area_enabled is True, f'{renamed_area=}'
        assert renamed_area.available_area == first_area.available_area
        assert service.overlay is not None
        assert service.overlay.logical_name == 'overhead'

        service.set_area_enabled('camera-0', False)
        assert service.overlay is not None
        assert service.overlay.projector_point == Point2D(100, 100)
        assert service.get_camera_area('camera-1').area_enabled is True

        second_overlay = service.point_from_camera('camera-1', (100, 100))
        assert second_overlay.camera_id == 'camera-1', f'{second_overlay=}'
        assert second_overlay.projector_point == Point2D(200, 100), f'{second_overlay=}'

    def test_renderable_areas_receive_distinct_slot_ordered_colours(self) -> None:
        runtime = CalibrationSessionRuntime((0, 1, 2, 3, 4))
        runtime.registry.close('camera-1')
        runtime.registry.open('camera-4')
        calibration = SimpleNamespace(
            camera_to_projector=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
        )
        for slot_id in ('camera-0', 'camera-2', 'camera-3', 'camera-4'):
            runtime.registry.set_calibration(
                slot_id,
                CalibrationStatus.CALIBRATED,
                calibration,
            )
            runtime.registry.set_area_enabled(slot_id, True)

        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )

        enabled_areas = [
            area
            for area in service.get_camera_areas()
            if area.area_enabled and area.available_area is not None
        ]
        assert [area.slot_id for area in enabled_areas] == [
            'camera-0',
            'camera-2',
            'camera-3',
            'camera-4',
        ], f'{enabled_areas=}'
        assert [area.area_colour for area in enabled_areas] == list(AREA_COLOURS)

    def test_invalidating_runtime_status_clears_area_without_silent_restoration(self) -> None:
        runtime = CalibrationSessionRuntime()
        calibration = SimpleNamespace(
            camera_to_projector=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_region=((0, 0), (500, 0), (500, 400), (0, 400)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )
        service.set_area_enabled('camera-0', True)
        unavailable_status = CameraStatus(
            'camera-0',
            None,
            RuntimeStatus.ERROR,
            CalibrationStatus.CALIBRATED,
            Resolution(640, 480),
            error_message='capture failed',
        )

        original_get_status = runtime.get_status

        def get_status(slot_id: str) -> CameraStatus:
            if slot_id == 'camera-0':
                return unavailable_status
            return original_get_status(slot_id)

        with patch.object(runtime, 'get_status', side_effect=get_status):
            invalidated_area = service.get_camera_area('camera-0')

        assert invalidated_area.area_enabled is False, f'{invalidated_area=}'
        assert invalidated_area.available_area is None, f'{invalidated_area=}'
        assert runtime.registry.get('camera-0').area_enabled is False
        assert service.get_camera_area('camera-0').area_enabled is False

    def test_recalibration_clears_area_before_attempt_and_after_failure(self) -> None:
        runtime = CalibrationSessionRuntime()
        calibration = SimpleNamespace(
            camera_to_projector=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_region=((0, 0), (500, 0), (500, 400), (0, 400)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=FakePointService(),  # type: ignore[arg-type]
        )
        service.set_area_enabled('camera-0', True)
        assert service.get_camera_area('camera-0').available_area is not None

        observed_areas: list[object] = []

        def fail_calibration(*_args: object, **_kwargs: object) -> CalibrationResult:
            observed_areas.append(service.get_camera_area('camera-0').available_area)
            raise CalibrationError('calibration failed')

        with patch('multivision.application.calibrate_homography', fail_calibration):
            with self.assertRaises(CalibrationError):
                service.calibrate('camera-0', ())

        assert observed_areas == [None], f'{observed_areas=}'
        failed_area = service.get_camera_area('camera-0')
        assert failed_area.area_enabled is False, f'{failed_area=}'
        assert failed_area.available_area is None, f'{failed_area=}'

    def test_failed_enable_does_not_change_calibration_overlay_or_lifecycle(self) -> None:
        runtime = CalibrationSessionRuntime()
        point_service = FakePointService()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=point_service,  # type: ignore[arg-type]
        )
        initial_camera = runtime.registry.get('camera-0')

        with self.assertRaises(InvalidCalibrationStateError):
            service.set_area_enabled('camera-0', True)

        failed_camera = runtime.registry.get('camera-0')
        assert failed_camera == initial_camera, f'{failed_camera=}, {initial_camera=}'
        assert point_service.renamed == [], f'{point_service.renamed=}'
        assert point_service.cleared == [], f'{point_service.cleared=}'

        invalid_calibration = SimpleNamespace(
            camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 0)),
            valid_region=((0, 0), (100, 0), (100, 100)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            invalid_calibration,
        )
        before_invalid_request = runtime.registry.get('camera-0')
        with self.assertRaises(InvalidAvailableAreaError):
            service.set_area_enabled('camera-0', True)
        after_invalid_request = runtime.registry.get('camera-0')
        assert after_invalid_request == before_invalid_request, (
            f'{after_invalid_request=}, {before_invalid_request=}'
        )
        assert after_invalid_request.state is SessionCameraState.OPEN
        assert after_invalid_request.area_enabled is False

        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            invalid_calibration,
        )
        runtime.registry.close('camera-0')
        closed_camera = runtime.registry.get('camera-0')
        with self.assertRaises(CameraUnavailableError):
            service.set_area_enabled('camera-0', True)
        assert runtime.registry.get('camera-0') == closed_camera, f'{closed_camera=}'

    def test_recalibration_clears_area_before_availability_failure(self) -> None:
        runtime = CalibrationSessionRuntime()
        calibration = SimpleNamespace(
            camera_to_projector=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_region=((0, 0), (500, 0), (500, 400), (0, 400)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=FakePointService(),  # type: ignore[arg-type]
        )
        service.set_area_enabled('camera-0', True)
        unavailable_status = CameraStatus(
            'camera-0',
            None,
            RuntimeStatus.ERROR,
            CalibrationStatus.CALIBRATED,
            Resolution(640, 480),
            error_message='capture failed',
        )

        with patch.object(runtime, 'get_status', return_value=unavailable_status):
            with self.assertRaises(CameraUnavailableError):
                service.calibrate('camera-0', ())

        camera = runtime.registry.get('camera-0')
        assert camera.area_enabled is False, f'{camera=}'


class MultiVisionServiceCameraManagementTest(unittest.TestCase):
    def test_management_is_deterministic_and_preserves_rename_state(self) -> None:
        runtime = FakeSessionRuntime()
        runtime.registry.set_frame_metadata(
            'camera-0',
            FrameMetadata(9, 10.0, Resolution(640, 480)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            'transform',
        )
        point_service = FakePointService()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=point_service,  # type: ignore[arg-type]
        )

        assert [camera.slot_id for camera in service.get_session_cameras()] == [
            'camera-0',
            'camera-1',
        ]
        renamed_camera = service.rename_camera('camera-0', 'overhead')

        assert renamed_camera.display_name == 'overhead', f'{renamed_camera=}'
        assert renamed_camera.frame_metadata == FrameMetadata(9, 10.0, Resolution(640, 480))
        assert renamed_camera.calibration_status is CalibrationStatus.CALIBRATED
        assert renamed_camera.calibration == 'transform'
        assert point_service.renamed == [('camera-0', 'overhead')]

    def test_close_reopen_and_disconnect_remove_enabled_area_state(self) -> None:
        runtime = CalibrationSessionRuntime()
        calibration = SimpleNamespace(
            camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            valid_region=((0, 0), (640, 0), (640, 480), (0, 480)),
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=FakePointService(),  # type: ignore[arg-type]
        )

        service.set_area_enabled('camera-0', True)
        service.close_camera('camera-0')
        closed_area = service.get_camera_area('camera-0')
        assert closed_area.area_enabled is False, f'{closed_area=}'
        assert closed_area.available_area is None, f'{closed_area=}'

        service.open_camera('camera-0')
        reopened_area = service.get_camera_area('camera-0')
        assert reopened_area.area_enabled is False, f'{reopened_area=}'
        assert reopened_area.available_area is None, f'{reopened_area=}'

        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service.set_area_enabled('camera-0', True)
        runtime.registry.mark_unavailable('camera-0', 'disconnected')
        disconnected_area = service.get_camera_area('camera-0')
        assert disconnected_area.area_enabled is False, f'{disconnected_area=}'
        assert disconnected_area.available_area is None, f'{disconnected_area=}'

    def test_overlay_camera_names_are_stored_as_stable_slot_ids(self) -> None:
        runtime = CalibrationSessionRuntime(capture_indexes=(0,))
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            SimpleNamespace(
                camera_to_projector=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            ),
        )
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )
        service.rename_camera('camera-0', 'overhead')

        entry = service.create_overlay(
            LineRequest(
                start={'space': 'camera_px', 'camera': 'overhead', 'x': 1, 'y': 1},
                end={'space': 'projector_px', 'x': 10, 'y': 10},
            ),
        )
        service.rename_camera('camera-0', 'side')

        entries = service.list_overlays()
        assert entry.request.start.camera == 'camera-0', f'{entry=}'
        assert entry.camera_dependencies == ('camera-0',), f'{entry=}'
        assert entries == [entry], f'{entries=}'

    def test_overlay_creation_prunes_stale_camera_dependencies(self) -> None:
        runtime = CalibrationSessionRuntime()
        service = MultiVisionService(
            Configuration(projector_resolution=Resolution(1000, 700)),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )
        camera_request = LineRequest(
            start={'space': 'camera_px', 'camera': 'camera-0', 'x': 1, 'y': 1},
            end={'space': 'projector_px', 'x': 10, 'y': 10},
        )
        stale_entry = service.overlay_registry.create(
            camera_request,
            ProjectorMaterialisation(),
        )
        runtime.registry.mark_unavailable('camera-0', 'disconnected')

        service.create_overlay(
            LineRequest(
                start={'space': 'projector_px', 'x': 1, 'y': 1},
                end={'space': 'projector_px', 'x': 10, 'y': 10},
            ),
        )

        entries = service.overlay_registry.list()
        assert stale_entry.id not in {entry.id for entry in entries}, f'{entries=}'
        assert len(entries) == 1, f'{entries=}'

    def test_close_and_reopen_clear_only_the_changed_camera_spatial_state(self) -> None:
        runtime = FakeSessionRuntime()
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            'transform',
        )
        point_service = FakePointService()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=point_service,  # type: ignore[arg-type]
        )

        closed_camera = service.close_camera('camera-0')
        reopened_camera = service.open_camera('camera-0')

        assert closed_camera.calibration_status is CalibrationStatus.UNCALIBRATED
        assert closed_camera.calibration is None
        assert reopened_camera.calibration_status is CalibrationStatus.UNCALIBRATED
        assert reopened_camera.calibration is None
        assert point_service.cleared == ['camera-0', 'camera-0']

    def test_verification_capture_does_not_block_main_thread_status_reads(self) -> None:
        resolution = Resolution(640, 480)
        configuration = Configuration(projector_resolution=resolution)
        runtime = CalibrationSessionRuntime((0,))
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        calibration = PersistedCalibration(
            'camera-0',
            resolution,
            resolution,
            1,
            identity,
            identity,
            CalibrationMetrics(20, 80, 80, 1.0, 0.0, 0.0, 1.0),
            1.0,
            ((0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)),
            projector_output_descriptor=configuration.projector_output_descriptor,
        )
        runtime.registry.set_calibration(
            'camera-0',
            CalibrationStatus.CALIBRATED,
            calibration,
        )
        service = MultiVisionService(
            configuration,
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=FakePointService(),  # type: ignore[arg-type]
        )
        correspondences = CameraCorrespondences(
            tuple(
                FiducialCorrespondence(
                    marker.marker_id,
                    marker.corner_index,
                    marker.projector_position,
                    marker.projector_position,
                )
                for marker in service.calibration_pattern.marker_corners
            ),
            'camera-0',
        )
        capture_started = threading.Event()
        release_capture = threading.Event()
        verification_errors: list[BaseException] = []
        verification_results: list[CalibrationStatus] = []

        def blocked_capture(*_args: object, **_kwargs: object) -> CameraCorrespondences:
            capture_started.set()
            assert release_capture.wait(1), 'verification capture was not released'
            return correspondences

        def verify_camera() -> None:
            try:
                verification_results.append(service.verify('camera-0'))
            except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                verification_errors.append(ex)

        status_read_finished = threading.Event()

        def read_statuses() -> None:
            service.get_camera_statuses()
            status_read_finished.set()

        with patch.object(service, '_get_correspondences_for_operation', blocked_capture):
            verification_thread = threading.Thread(target=verify_camera)
            verification_thread.start()
            assert capture_started.wait(1), 'verification capture did not start'

            status_thread = threading.Thread(target=read_statuses)
            status_thread.start()
            assert status_read_finished.wait(0.5), (
                'main-thread status reads were blocked during verification capture'
            )

            release_capture.set()
            verification_thread.join(1)
            status_thread.join(1)

        assert not verification_thread.is_alive(), 'verification did not finish'
        assert not status_thread.is_alive(), 'status read did not finish'
        assert verification_errors == [], f'{verification_errors=}'
        assert verification_results == [CalibrationStatus.CALIBRATED], (
            f'{verification_results=}'
        )

    def test_late_calibration_cannot_restore_state_after_close_and_reopen(self) -> None:
        runtime = CalibrationSessionRuntime()
        point_service = FakePointService()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            point_service=point_service,  # type: ignore[arg-type]
        )
        calibration_result = CalibrationResult(
            HomographyPair(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            (
                Point2D(0.0, 0.0),
                Point2D(640.0, 0.0),
                Point2D(640.0, 480.0),
            ),
            CalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 0.5),
        )
        calibration_started = threading.Event()
        release_calibration = threading.Event()
        calibration_errors: list[BaseException] = []

        def blocked_calibration(*_args: object, **_kwargs: object) -> CalibrationResult:
            calibration_started.set()
            assert release_calibration.wait(1), 'calibration was not released'
            return calibration_result

        def calibrate_camera() -> None:
            try:
                service.calibrate('camera-0', ())
            except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                calibration_errors.append(ex)

        close_and_reopen_finished = threading.Event()

        def close_and_reopen_camera() -> None:
            service.close_camera('camera-0')
            service.open_camera('camera-0')
            close_and_reopen_finished.set()

        with patch('multivision.application.calibrate_homography', blocked_calibration):
            calibration_thread = threading.Thread(target=calibrate_camera)
            calibration_thread.start()
            assert calibration_started.wait(1), 'calibration did not start'

            lifecycle_thread = threading.Thread(target=close_and_reopen_camera)
            lifecycle_thread.start()
            assert not close_and_reopen_finished.wait(0.05), (
                'camera lifecycle changed during calibration'
            )

            release_calibration.set()
            calibration_thread.join(1)
            lifecycle_thread.join(1)

        assert not calibration_thread.is_alive(), 'calibration did not finish'
        assert not lifecycle_thread.is_alive(), 'camera lifecycle did not finish'
        assert calibration_errors == [], f'{calibration_errors=}'
        camera = runtime.registry.get('camera-0')
        assert camera.calibration_status is CalibrationStatus.UNCALIBRATED, f'{camera=}'
        assert camera.calibration is None, f'{camera=}'


if __name__ == '__main__':
    unittest.main()
