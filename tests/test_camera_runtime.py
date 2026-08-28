import threading
import time
import unittest

from multivision.camera import CameraRuntime
from multivision.errors import (
    CameraUnavailableError,
    FrameCaptureError,
    HardwareError,
)
from multivision.types import (
    DeviceInfo,
    Resolution,
    RuntimeStatus,
)


class FakeCapture:
    def __init__(
        self,
        frame_prefix: str,
        native_resolution: Resolution = Resolution(1920, 1080),
    ) -> None:
        self.frame_prefix = frame_prefix
        self.native_resolution = native_resolution
        self.read_count = 0
        self.is_released = False
        self.first_frame = threading.Event()

    def is_opened(self) -> bool:
        return not self.is_released

    def get_native_resolution(self) -> Resolution:
        return self.native_resolution

    def read(self) -> tuple[bool, object]:
        if self.is_released:
            return False, None
        self.read_count += 1
        self.first_frame.set()
        return True, f'{self.frame_prefix}-{self.read_count}'

    def release(self) -> None:
        self.is_released = True


class FakeDiscovery:
    def __init__(self, devices: list[DeviceInfo]) -> None:
        self.devices = devices

    def discover_devices(self) -> list[DeviceInfo]:
        return self.devices


class FakeCaptureFactory:
    def __init__(self, captures: dict[str, FakeCapture]) -> None:
        self.captures = captures
        self.opened_device_ids: list[str] = []

    def open_capture(self, device: DeviceInfo) -> FakeCapture:
        self.opened_device_ids.append(device.device_id)
        return self.captures[device.device_id]


class CameraRuntimeTest(unittest.TestCase):
    def test_snapshots_use_one_persistent_handle_and_shutdown_releases_it(self) -> None:
        captures = {
            'overhead-device': FakeCapture('overhead', Resolution(1920, 1080)),
            'side-device': FakeCapture('side', Resolution(1280, 720)),
        }
        factory = FakeCaptureFactory(captures)
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo(
                        'overhead-device',
                        'Overhead',
                        capture_index=0,
                        native_resolution=Resolution(1920, 1080),
                    ),
                    DeviceInfo(
                        'side-device',
                        'Side',
                        capture_index=1,
                        native_resolution=Resolution(1280, 720),
                    ),
                ],
            ),
            factory,
            {'overhead': 'overhead-device', 'side-left': 'side-device'},
            read_wait_seconds=0.001,
        )

        runtime.start()
        assert captures['overhead-device'].first_frame.wait(1), 'overhead frame was not captured'
        assert captures['side-device'].first_frame.wait(1), 'side frame was not captured'

        first_snapshot = runtime.snapshot('overhead')
        time.sleep(0.01)
        second_snapshot = runtime.snapshot('overhead')

        assert factory.opened_device_ids == [
            'overhead-device',
            'side-device',
        ], f'{factory.opened_device_ids=}'
        assert first_snapshot.data.startswith(
            'overhead-',
        ), f'{first_snapshot=}'
        assert second_snapshot.frame_counter > first_snapshot.frame_counter, f'{second_snapshot=}'
        assert runtime.get_status('overhead').runtime_status is RuntimeStatus.AVAILABLE
        assert runtime.get_status('overhead').frame_counter == second_snapshot.frame_counter

        runtime.shutdown()

        assert all(capture.is_released for capture in captures.values()), f'{captures=}'
        assert all(
            status.runtime_status is RuntimeStatus.STOPPED
            for status in runtime.get_statuses()
        ), f'{runtime.get_statuses()=}'

    def test_discovered_resolution_supports_captures_without_a_resolution_method(self) -> None:
        class BasicCapture:
            def __init__(self) -> None:
                self.is_released = False
                self.first_frame = threading.Event()

            def is_opened(self) -> bool:
                return not self.is_released

            def read(self) -> tuple[bool, str]:
                self.first_frame.set()
                return not self.is_released, 'frame'

            def release(self) -> None:
                self.is_released = True

        capture = BasicCapture()
        runtime = CameraRuntime(
            FakeDiscovery(
                [DeviceInfo('device', 'Camera', native_resolution=Resolution(640, 480))],
            ),
            FakeCaptureFactory({'device': capture}),  # type: ignore[arg-type]
            {'overhead': 'device'},
            read_wait_seconds=0.001,
        )

        runtime.start()

        assert capture.first_frame.wait(1), 'capture did not start reading'
        assert runtime.get_status('overhead').native_resolution == Resolution(640, 480)
        runtime.shutdown()
        assert capture.is_released

    def test_malformed_capture_result_is_an_error_and_is_retried(self) -> None:
        class RecoveringCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('recovering')
                self.attempt_count = 0

            def read(self) -> tuple[bool, object]:
                self.attempt_count += 1
                if self.attempt_count == 1:
                    return [True, 'malformed']  # type: ignore[return-value]
                return super().read()

        capture = RecoveringCapture()
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera')]),
            FakeCaptureFactory({'device': capture}),
            {'overhead': 'device'},
            read_wait_seconds=0.001,
        )

        runtime.start()

        assert capture.first_frame.wait(1), 'recovery frame was not captured'
        snapshot = runtime.snapshot('overhead')
        assert snapshot.data == 'recovering-1', f'{snapshot=}'
        assert capture.attempt_count >= 2, f'{capture.attempt_count=}'
        assert runtime.get_status('overhead').runtime_status is RuntimeStatus.AVAILABLE

        runtime.shutdown()

    def test_shutdown_releases_capture_and_waits_for_read_to_finish(self) -> None:
        class BlockingCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('blocking')
                self.read_finished = threading.Event()
                self.release_started = threading.Event()

            def read(self) -> tuple[bool, object]:
                self.first_frame.set()
                self.release_started.wait(1)
                self.read_finished.set()
                return False, None

            def release(self) -> None:
                self.release_started.set()
                super().release()

        capture = BlockingCapture()
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera')]),
            FakeCaptureFactory({'device': capture}),
            {'overhead': 'device'},
        )
        runtime.start()
        assert capture.first_frame.wait(1), 'capture did not start reading'

        shutdown_thread = threading.Thread(target=runtime.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(1)

        assert not shutdown_thread.is_alive(), 'shutdown did not finish'
        assert capture.release_started.is_set(), 'shutdown did not release the capture'
        assert capture.read_finished.is_set(), 'shutdown did not stop the read worker'
        assert capture.is_released

    def test_shutdown_reports_a_worker_that_does_not_stop(self) -> None:
        class NeverEndingCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('never-ending')
                self.stop_read = threading.Event()

            def read(self) -> tuple[bool, object]:
                self.first_frame.set()
                self.stop_read.wait(10)
                return False, None

        capture = NeverEndingCapture()
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera')]),
            FakeCaptureFactory({'device': capture}),
            {'overhead': 'device'},
            worker_shutdown_timeout_seconds=0.01,
        )
        runtime.start()
        assert capture.first_frame.wait(1), 'capture did not start reading'

        shutdown_started_at = time.monotonic()
        with self.assertRaises(HardwareError):
            runtime.shutdown()
        shutdown_duration_seconds = time.monotonic() - shutdown_started_at
        assert shutdown_duration_seconds < 1, f'{shutdown_duration_seconds=}'

        capture.stop_read.set()
        runtime.shutdown()

    def test_invalid_logical_names_are_reported_as_unavailable(self) -> None:
        runtime = CameraRuntime(
            FakeDiscovery([]),
            FakeCaptureFactory({}),
            {'overhead': 'device'},
        )

        for logical_name in [None, [], '']:
            with self.subTest(logical_name=logical_name):
                with self.assertRaises(CameraUnavailableError):
                    runtime.snapshot(logical_name)  # type: ignore[arg-type]

    def test_missing_camera_is_unavailable_and_open_error_is_explicit(self) -> None:
        failing_capture = FakeCapture('failing')

        class FailingFactory(FakeCaptureFactory):
            def open_capture(self, device: DeviceInfo) -> FakeCapture:
                self.opened_device_ids.append(device.device_id)
                raise RuntimeError('open failed')

        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo('missing-device', 'Missing', is_available=False),
                    DeviceInfo('failing-device', 'Failing'),
                ],
            ),
            FailingFactory({'failing-device': failing_capture}),
            {'missing': 'missing-device', 'failing': 'failing-device', 'absent': 'absent-device'},
        )

        runtime.start()

        assert runtime.get_status('missing').runtime_status is RuntimeStatus.UNAVAILABLE
        assert runtime.get_status('absent').runtime_status is RuntimeStatus.UNAVAILABLE
        assert runtime.get_status('failing').runtime_status is RuntimeStatus.ERROR
        with self.assertRaises(CameraUnavailableError):
            runtime.snapshot('missing')
        with self.assertRaises(FrameCaptureError):
            runtime.snapshot('failing')

        runtime.shutdown()


if __name__ == '__main__':
    unittest.main()
