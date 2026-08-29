import threading
import unittest
from unittest.mock import patch

from multivision.application import MultiVisionService
from multivision.calibration import CalibrationMetrics, CalibrationResult
from multivision.config import Configuration
from multivision.geometry import HomographyPair, Point2D
from multivision.session import FrameMetadata, SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
)


class FakeSessionRuntime:
    def __init__(self) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo('device-1', 'Camera 1', capture_index=1),
                DeviceInfo('device-0', 'Camera 0', capture_index=0),
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


class FakePointService:
    def __init__(self) -> None:
        self.renamed: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    def rename_overlay_camera(self, camera_id: str, logical_name: str) -> None:
        self.renamed.append((camera_id, logical_name))

    def clear_overlay_for_camera(self, camera_id: str) -> None:
        self.cleared.append(camera_id)


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
