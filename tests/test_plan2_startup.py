import json
import tempfile
import unittest
from pathlib import Path

from multivision.application import MultiVisionService
from multivision.config import load_configuration
from multivision.types import CalibrationStatus, DeviceInfo, Resolution


class FakeCapture:
    def __init__(self) -> None:
        self.is_released = False

    def is_opened(self) -> bool:
        return not self.is_released

    def get_native_resolution(self) -> Resolution:
        return Resolution(640, 480)

    def read(self) -> tuple[bool, str]:
        return not self.is_released, 'frame'

    def release(self) -> None:
        self.is_released = True


class FakeDiscovery:
    def discover_devices(self) -> list[DeviceInfo]:
        return [DeviceInfo('current-device', 'Current camera', 7, Resolution(640, 480))]


class FakeCaptureFactory:
    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def open_capture(self, _device: DeviceInfo) -> FakeCapture:
        return self.capture


class Plan2StartupTest(unittest.TestCase):
    def test_legacy_camera_state_is_ignored_and_non_camera_configuration_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / 'config.json'
            config_path.write_text(
                json.dumps(
                    {
                        'camera_bindings': ['legacy-data-is-not-session-state'],
                        'calibrations': {'legacy-camera': {'old': 'transform'}},
                        'projector_resolution': {'width': 1600, 'height': 900},
                        'calibration_thresholds': {'min_unique_tags': 6},
                    },
                ),
                encoding='utf-8',
            )
            assert load_configuration(config_path).projector_resolution == Resolution(1600, 900)

            capture = FakeCapture()
            service = MultiVisionService(
                config_path=config_path,
                discovery=FakeDiscovery(),
                capture_factory=FakeCaptureFactory(capture),
            )

            service.start()
            try:
                cameras = service.get_session_cameras()
                assert [camera.slot_id for camera in cameras] == ['camera-0'], f'{cameras=}'
                assert cameras[0].display_name == 'camera-0', f'{cameras=}'
                assert cameras[0].calibration_status is CalibrationStatus.UNCALIBRATED
                assert cameras[0].calibration is None
                assert service.configuration.projector_resolution == Resolution(1600, 900)
                assert service.configuration.calibration_thresholds.min_unique_tags == 6
                assert service.get_calibration_records() == {}
            finally:
                service.shutdown()

            assert capture.is_released


if __name__ == '__main__':
    unittest.main()
