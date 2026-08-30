import threading
import time
import unittest

from multivision.camera import CameraRuntime
from multivision.discovery import OpenCVCaptureIndexDiscovery
from multivision.errors import (
    CameraStateError,
    CameraUnavailableError,
    HardwareError,
)
from multivision.hardware import OpenCVCaptureDeviceFactory
from multivision.types import (
    CalibrationStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
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
    def test_runtime_adopts_startup_probes_without_opening_selected_devices_twice(self) -> None:
        class RawCapture:
            def __init__(self, capture_index: int) -> None:
                self.capture_index = capture_index
                self.is_released = False
                self.read_count = 0

            def isOpened(self) -> bool:
                return not self.is_released

            def get(self, _property_id: int) -> int:
                return 640 if _property_id == 3 else 480

            def read(self) -> tuple[bool, str]:
                self.read_count += 1
                return True, f'frame-{self.capture_index}-{self.read_count}'

            def release(self) -> None:
                self.is_released = True

        probes: dict[int, RawCapture] = {}
        opened_indexes: list[int] = []

        def open_probe(capture_index: int) -> RawCapture:
            opened_indexes.append(capture_index)
            probe = RawCapture(capture_index)
            probes[capture_index] = probe
            return probe

        discovery = OpenCVCaptureIndexDiscovery(
            max_capture_index=5,
            capture_opener=open_probe,
        )
        runtime = CameraRuntime(
            discovery,
            OpenCVCaptureDeviceFactory(),
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            assert [camera.capture_index for camera in runtime.get_session_cameras()] == [
                0,
                1,
                2,
                3,
                4,
            ], f'{runtime.get_session_cameras()=}'
            assert opened_indexes == [0, 1, 2, 3, 4], f'{opened_indexes=}'
            assert probes[4].is_released
            assert all(not probes[index].is_released for index in range(4))
        finally:
            runtime.shutdown()

        assert all(probe.is_released for probe in probes.values()), f'{probes=}'

    def test_session_discovery_failure_is_explicit(self) -> None:
        class FailingDiscovery:
            def discover_devices(self) -> list[DeviceInfo]:
                raise HardwareError('malformed OpenCV probe')

        runtime = CameraRuntime(
            FailingDiscovery(),  # type: ignore[arg-type]
            FakeCaptureFactory({}),
        )

        with self.assertRaises(HardwareError):
            runtime.start()

        runtime.shutdown()

    def test_session_start_ignores_persisted_bindings_and_keeps_startup_slots(self) -> None:
        captures = {
            'device-a': FakeCapture('a', Resolution(1920, 1080)),
            'device-b': FakeCapture('b', Resolution(1280, 720)),
        }
        factory = FakeCaptureFactory(captures)
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo(
                        'device-a',
                        'Camera A',
                        capture_index=3,
                        native_resolution=Resolution(1920, 1080),
                    ),
                    DeviceInfo(
                        'device-b',
                        'Camera B',
                        capture_index=1,
                        native_resolution=Resolution(1280, 720),
                    ),
                ],
            ),
            factory,
            read_wait_seconds=0.001,
        )

        runtime.start()

        assert captures['device-a'].first_frame.wait(1), 'camera-1 frame was not captured'
        assert captures['device-b'].first_frame.wait(1), 'camera-0 frame was not captured'
        assert factory.opened_device_ids == [
            'device-b',
            'device-a',
        ], f'{factory.opened_device_ids=}'
        assert [status.logical_name for status in runtime.get_statuses()] == [
            'camera-0',
            'camera-1',
        ]
        assert all(status.device_id is None for status in runtime.get_statuses())

        cameras = runtime.get_session_cameras()
        assert [camera.capture_index for camera in cameras] == [1, 3], f'{cameras=}'
        assert all(camera.state is SessionCameraState.OPEN for camera in cameras), f'{cameras=}'
        assert [
            camera.device_info.native_resolution
            for camera in cameras
            if camera.device_info is not None
        ] == [
            Resolution(1280, 720),
            Resolution(1920, 1080),
        ]
        runtime.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')

        runtime.shutdown()

        assert all(
            camera.state is SessionCameraState.CLOSED
            and camera.frame_metadata is None
            and camera.calibration_status is CalibrationStatus.UNCALIBRATED
            and camera.calibration is None
            for camera in runtime.get_session_cameras()
        ), f'{runtime.get_session_cameras()=}'

    def test_session_disconnect_releases_only_its_handle_and_never_falls_back(self) -> None:
        class DisconnectingCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('disconnecting')
                self.disconnect_observed = threading.Event()

            def read(self) -> tuple[bool, object]:
                self.read_count += 1
                self.first_frame.set()
                if self.read_count == 2:
                    self.disconnect_observed.set()
                    return False, None
                return True, f'{self.frame_prefix}-{self.read_count}'

        class CountingDiscovery(FakeDiscovery):
            def __init__(self, devices: list[DeviceInfo]) -> None:
                super().__init__(devices)
                self.call_count = 0

            def discover_devices(self) -> list[DeviceInfo]:
                self.call_count += 1
                return super().discover_devices()

        disconnecting_capture = DisconnectingCapture()
        other_capture = FakeCapture('other')
        devices = [
            DeviceInfo('disconnected-device', 'Disconnected', capture_index=0),
            DeviceInfo('other-device', 'Other', capture_index=1),
        ]
        discovery = CountingDiscovery(devices)
        factory = FakeCaptureFactory(
            {
                'disconnected-device': disconnecting_capture,
                'other-device': other_capture,
            },
        )
        runtime = CameraRuntime(
            discovery,
            factory,
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            assert disconnecting_capture.disconnect_observed.wait(1), (
                'the disconnected camera did not report a read failure'
            )
            deadline = time.monotonic() + 1
            while runtime.get_status('camera-0').runtime_status is not RuntimeStatus.UNAVAILABLE:
                assert time.monotonic() < deadline, 'camera was not marked unavailable'
                time.sleep(0.001)

            assert disconnecting_capture.is_released
            failed_read_count = disconnecting_capture.read_count
            time.sleep(0.01)
            assert disconnecting_capture.read_count == failed_read_count
            assert runtime.get_session_cameras()[0].state is SessionCameraState.UNAVAILABLE
            assert runtime.get_session_cameras()[0].frame_metadata is None
            status = runtime.get_status('camera-0')
            assert status.frame_counter == 0
            assert status.calibration_status is CalibrationStatus.UNCALIBRATED, f'{status=}'
            with self.assertRaises(CameraUnavailableError):
                runtime.snapshot('camera-0')
            assert discovery.call_count == 1, f'{discovery.call_count=}'
            assert factory.opened_device_ids == [
                'disconnected-device',
                'other-device',
            ], f'{factory.opened_device_ids=}'
            assert other_capture.read_count > 0
        finally:
            runtime.shutdown()

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
            read_wait_seconds=0.001,
        )

        runtime.start()
        assert captures['overhead-device'].first_frame.wait(1), 'overhead frame was not captured'
        assert captures['side-device'].first_frame.wait(1), 'side frame was not captured'

        first_snapshot = runtime.snapshot('camera-0')
        time.sleep(0.01)
        second_snapshot = runtime.snapshot('camera-0')

        assert factory.opened_device_ids == [
            'overhead-device',
            'side-device',
        ], f'{factory.opened_device_ids=}'
        assert first_snapshot.data.startswith(
            'overhead-',
        ), f'{first_snapshot=}'
        assert second_snapshot.frame_counter > first_snapshot.frame_counter, f'{second_snapshot=}'
        assert runtime.get_status('camera-0').runtime_status is RuntimeStatus.AVAILABLE
        assert runtime.get_status('camera-0').frame_counter == second_snapshot.frame_counter

        runtime.shutdown()

        assert all(capture.is_released for capture in captures.values()), f'{captures=}'
        assert all(
            status.runtime_status is RuntimeStatus.STOPPED
            for status in runtime.get_statuses()
        ), f'{runtime.get_statuses()=}'

    def test_consecutive_frames_are_not_reconstructed_from_latest_snapshot(self) -> None:
        capture = FakeCapture('camera', Resolution(640, 480))
        runtime = CameraRuntime(
            FakeDiscovery([
                DeviceInfo(
                    'device-0',
                    'Camera 0',
                    capture_index=0,
                    native_resolution=Resolution(640, 480),
                ),
            ]),
            FakeCaptureFactory({'device-0': capture}),
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            assert capture.first_frame.wait(1), 'camera frame was not captured'
            frames = runtime.get_consecutive_frames('camera-0', 3, 1.0)
        finally:
            runtime.shutdown()

        assert [frame.frame_counter for frame in frames] == [
            frames[0].frame_counter,
            frames[0].frame_counter + 1,
            frames[0].frame_counter + 2,
        ], f'{frames=}'
        assert [frame.data for frame in frames] == [
            f'camera-{frame.frame_counter}'
            for frame in frames
        ], f'{frames=}'

    def test_session_open_does_not_publish_open_without_a_handle(self) -> None:
        class BlockingReopenFactory:
            def __init__(self) -> None:
                self.open_count = 0
                self.open_started = threading.Event()
                self.allow_open = threading.Event()

            def open_capture(self, _device: DeviceInfo) -> FakeCapture:
                self.open_count += 1
                if self.open_count == 3:
                    self.open_started.set()
                    assert self.allow_open.wait(1), 'reopen was not released'
                return FakeCapture('camera')

        factory = BlockingReopenFactory()
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo('device-0', 'Camera 0', capture_index=0),
                    DeviceInfo('device-1', 'Camera 1', capture_index=1),
                ],
            ),
            factory,
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            runtime.close_camera('camera-0')
            open_finished = threading.Event()

            def reopen() -> None:
                runtime.open_camera('camera-0')
                open_finished.set()

            reopen_thread = threading.Thread(target=reopen)
            reopen_thread.start()
            assert factory.open_started.wait(1), 'reopen did not reach the capture factory'

            inventory_finished = threading.Event()
            observed_states: list[SessionCameraState] = []

            def read_inventory() -> None:
                observed_states.append(runtime.get_session_cameras()[0].state)
                inventory_finished.set()

            inventory_thread = threading.Thread(target=read_inventory)
            inventory_thread.start()
            assert not inventory_finished.wait(0.05), (
                'inventory observed a half-open camera transition'
            )

            factory.allow_open.set()
            assert open_finished.wait(1), 'reopen did not finish'
            inventory_thread.join(1)
            assert not inventory_thread.is_alive(), 'inventory read did not finish'
            assert observed_states == [SessionCameraState.OPEN], f'{observed_states=}'
            reopen_thread.join(1)
            assert not reopen_thread.is_alive(), 'reopen thread did not finish'
        finally:
            factory.allow_open.set()
            runtime.shutdown()

    def test_session_close_and_open_reuses_startup_slot_without_rediscovery(self) -> None:
        class ReopeningFactory:
            def __init__(self) -> None:
                self.opened_device_ids: list[str] = []
                self.captures: list[FakeCapture] = []

            def open_capture(self, device: DeviceInfo) -> FakeCapture:
                self.opened_device_ids.append(device.device_id)
                capture = FakeCapture(device.device_id)
                self.captures.append(capture)
                return capture

        class CountingDiscovery(FakeDiscovery):
            def __init__(self, devices: list[DeviceInfo]) -> None:
                super().__init__(devices)
                self.call_count = 0

            def discover_devices(self) -> list[DeviceInfo]:
                self.call_count += 1
                return super().discover_devices()

        devices = [
            DeviceInfo(
                f'device-{capture_index}',
                f'Camera {capture_index}',
                capture_index=capture_index,
                native_resolution=Resolution(640, 480),
            )
            for capture_index in range(5)
        ]
        discovery = CountingDiscovery(devices)
        factory = ReopeningFactory()
        runtime = CameraRuntime(
            discovery,
            factory,
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            assert all(
                capture.first_frame.wait(1)
                for capture in factory.captures[:4]
            ), f'{factory.captures=}'
            assert factory.opened_device_ids == [
                'device-0',
                'device-1',
                'device-2',
                'device-3',
            ], f'{factory.opened_device_ids=}'
            assert discovery.call_count == 1, f'{discovery.call_count=}'
            first_frame = runtime.snapshot('camera-0')
            time.sleep(0.01)
            assert runtime.snapshot('camera-0').frame_counter > first_frame.frame_counter

            first_capture = factory.captures[0]
            runtime.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
            assert runtime.get_status('camera-0').calibration_status is CalibrationStatus.CALIBRATED
            runtime.close_camera('camera-0')
            closed_read_count = first_capture.read_count
            assert first_capture.is_released
            assert runtime.get_session_cameras()[0].state is SessionCameraState.CLOSED
            assert runtime.get_session_cameras()[0].frame_metadata is None
            assert (
                runtime.get_status('camera-0').calibration_status
                is CalibrationStatus.UNCALIBRATED
            )
            time.sleep(0.01)
            assert first_capture.read_count == closed_read_count

            runtime.open_camera('camera-4')
            reopened_capture = factory.captures[4]
            assert reopened_capture.first_frame.wait(1), 'reopened camera did not capture a frame'
            reopened_camera = runtime.get_session_cameras()[4]
            assert reopened_camera.state is SessionCameraState.OPEN
            assert reopened_camera.capture_index == 4
            assert reopened_camera.frame_metadata is not None
            assert reopened_camera.frame_metadata.frame_counter > 0
            assert runtime.get_status('camera-4').runtime_status is RuntimeStatus.AVAILABLE
            assert runtime.snapshot('camera-4').frame_counter > 0
            assert factory.opened_device_ids == [
                'device-0',
                'device-1',
                'device-2',
                'device-3',
                'device-4',
            ], f'{factory.opened_device_ids=}'
            assert discovery.call_count == 1, f'{discovery.call_count=}'
        finally:
            runtime.shutdown()

        assert all(capture.is_released for capture in factory.captures), f'{factory.captures=}'

    def test_session_reopen_resets_runtime_calibration_status(self) -> None:
        class ReopeningFactory:
            def __init__(self) -> None:
                self.captures: list[FakeCapture] = []

            def open_capture(self, _device: DeviceInfo) -> FakeCapture:
                capture = FakeCapture('camera')
                self.captures.append(capture)
                return capture

        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera', capture_index=0)]),
            ReopeningFactory(),
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            runtime.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
            runtime.close_camera('camera-0')
            assert (
                runtime.get_status('camera-0').calibration_status
                is CalibrationStatus.UNCALIBRATED
            )

            runtime.open_camera('camera-0')
            assert (
                runtime.get_status('camera-0').calibration_status
                is CalibrationStatus.UNCALIBRATED
            )
        finally:
            runtime.shutdown()

    def test_session_open_failure_is_explicit_and_unavailable_slots_cannot_open(self) -> None:
        class FailingOpenCapture(FakeCapture):
            def is_opened(self) -> bool:
                return False

        class SelectiveFactory:
            def __init__(self) -> None:
                self.opened_device_ids: list[str] = []
                self.failed_capture: FailingOpenCapture | None = None

            def open_capture(self, device: DeviceInfo) -> FakeCapture:
                self.opened_device_ids.append(device.device_id)
                if device.device_id == 'device-4':
                    self.failed_capture = FailingOpenCapture('device-4')
                    return self.failed_capture
                return FakeCapture(device.device_id)

        factory = SelectiveFactory()
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo(
                        f'device-{capture_index}',
                        f'Camera {capture_index}',
                        capture_index=capture_index,
                        native_resolution=Resolution(640, 480),
                        is_available=capture_index != 5,
                    )
                    for capture_index in range(6)
                ],
            ),
            factory,
            read_wait_seconds=0.001,
        )

        runtime.start()
        try:
            with self.assertRaises(CameraStateError):
                runtime.open_camera('camera-5')
            assert 'device-5' not in factory.opened_device_ids, f'{factory.opened_device_ids=}'

            runtime.close_camera('camera-0')
            with self.assertRaises(CameraUnavailableError):
                runtime.open_camera('camera-4')
            assert runtime.get_session_cameras()[4].state is SessionCameraState.UNAVAILABLE
            assert factory.failed_capture is not None
            assert factory.failed_capture.is_released
        finally:
            runtime.shutdown()

    def test_discovered_resolution_is_used_when_capture_metadata_is_missing(self) -> None:
        class BasicCapture:
            def __init__(self) -> None:
                self.is_released = False
                self.first_frame = threading.Event()

            def is_opened(self) -> bool:
                return not self.is_released

            def get_native_resolution(self) -> None:
                return None

            def read(self) -> tuple[bool, str]:
                self.first_frame.set()
                return not self.is_released, 'frame'

            def release(self) -> None:
                self.is_released = True

        capture = BasicCapture()
        runtime = CameraRuntime(
            FakeDiscovery(
                [
                    DeviceInfo(
                        'device',
                        'Camera',
                        capture_index=0,
                        native_resolution=Resolution(640, 480),
                    ),
                ],
            ),
            FakeCaptureFactory({'device': capture}),  # type: ignore[arg-type]
            read_wait_seconds=0.001,
        )

        runtime.start()

        assert capture.first_frame.wait(1), 'capture did not start reading'
        assert runtime.get_status('camera-0').native_resolution == Resolution(640, 480)
        runtime.shutdown()
        assert capture.is_released

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
            FakeDiscovery([DeviceInfo('device', 'Camera', capture_index=0)]),
            FakeCaptureFactory({'device': capture}),
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
            FakeDiscovery([DeviceInfo('device', 'Camera', capture_index=0)]),
            FakeCaptureFactory({'device': capture}),
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

    def test_status_reads_wait_for_an_atomic_close_transition(self) -> None:
        class BlockingReleaseCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('blocking-release')
                self.release_started = threading.Event()
                self.allow_release = threading.Event()

            def release(self) -> None:
                self.release_started.set()
                self.allow_release.wait(1)
                super().release()

        capture = BlockingReleaseCapture()
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera', capture_index=0)]),
            FakeCaptureFactory({'device': capture}),
            read_wait_seconds=0.001,
        )
        runtime.start()
        assert capture.first_frame.wait(1), 'capture did not start reading'

        close_thread = threading.Thread(target=lambda: runtime.close_camera('camera-0'))
        close_thread.start()
        assert capture.release_started.wait(1), 'close did not release the capture'

        status_finished = threading.Event()
        observed_statuses: list[RuntimeStatus] = []

        def read_status() -> None:
            observed_statuses.append(runtime.get_status('camera-0').runtime_status)
            status_finished.set()

        status_thread = threading.Thread(target=read_status)
        status_thread.start()
        assert not status_finished.wait(0.05), 'status read observed a half-closed camera'

        capture.allow_release.set()
        close_thread.join(1)
        status_thread.join(1)
        assert not close_thread.is_alive(), 'close did not finish'
        assert not status_thread.is_alive(), 'status read did not finish'
        assert observed_statuses == [RuntimeStatus.STOPPED], f'{observed_statuses=}'
        runtime.shutdown()

    def test_close_timeout_keeps_worker_for_later_cleanup(self) -> None:
        class BlockingCapture(FakeCapture):
            def __init__(self) -> None:
                super().__init__('blocking-close')
                self.stop_read = threading.Event()
                self.read_finished = threading.Event()

            def read(self) -> tuple[bool, object]:
                self.first_frame.set()
                self.stop_read.wait(10)
                self.read_finished.set()
                return False, None

        capture = BlockingCapture()
        runtime = CameraRuntime(
            FakeDiscovery([DeviceInfo('device', 'Camera', capture_index=0)]),
            FakeCaptureFactory({'device': capture}),
            worker_shutdown_timeout_seconds=0.01,
        )
        runtime.start()
        assert capture.first_frame.wait(1), 'capture did not start reading'

        with self.assertRaises(HardwareError):
            runtime.close_camera('camera-0')
        assert capture.is_released
        assert runtime.get_status('camera-0').runtime_status is RuntimeStatus.STOPPED
        with self.assertRaises(HardwareError):
            runtime.open_camera('camera-0')

        capture.stop_read.set()
        runtime.shutdown()
        assert capture.read_finished.is_set(), 'close worker was not cleaned up'

    def test_invalid_logical_names_are_reported_as_unavailable(self) -> None:
        runtime = CameraRuntime(
            FakeDiscovery([]),
            FakeCaptureFactory({}),
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
                    DeviceInfo('failing-device', 'Failing', capture_index=1),
                ],
            ),
            FailingFactory({'failing-device': failing_capture}),
        )

        runtime.start()

        assert runtime.get_status('camera-0').runtime_status is RuntimeStatus.UNAVAILABLE
        assert runtime.get_status('camera-1').runtime_status is RuntimeStatus.UNAVAILABLE
        with self.assertRaises(CameraUnavailableError):
            runtime.snapshot('camera-0')
        with self.assertRaises(CameraUnavailableError):
            runtime.snapshot('camera-1')

        runtime.shutdown()


if __name__ == '__main__':
    unittest.main()
