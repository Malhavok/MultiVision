import unittest

from multivision.discovery import OpenCVCaptureIndexDiscovery
from multivision.errors import HardwareError
from multivision.session import SessionCameraRegistry
from multivision.types import Resolution


class FakeProbe:
    def __init__(self, native_resolution: Resolution | None = Resolution(640, 480)) -> None:
        self.native_resolution = native_resolution
        self.is_released = False

    def isOpened(self) -> bool:
        return not self.is_released

    def get_native_resolution(self) -> Resolution | None:
        return self.native_resolution

    def release(self) -> None:
        self.is_released = True


class CaptureIndexDiscoveryTest(unittest.TestCase):
    def test_discovers_a_fixed_index_snapshot_with_session_metadata(self) -> None:
        probes: dict[int, FakeProbe] = {}
        opened_indexes: list[int] = []
        probe_resolution = Resolution(640, 480)

        def open_probe(capture_index: int) -> FakeProbe:
            opened_indexes.append(capture_index)
            probe = FakeProbe(probe_resolution)
            probes[capture_index] = probe
            return probe

        discovery = OpenCVCaptureIndexDiscovery(
            max_capture_index=4,
            capture_opener=open_probe,
        )

        devices = discovery.discover_devices()
        second_result = discovery.discover_devices()

        assert opened_indexes == [0, 1, 2, 3], f'{opened_indexes=}'
        assert all(
            device.native_resolution == Resolution(640, 480)
            for device in second_result
        ), f'{second_result=}'
        assert devices == second_result, f'{devices=}, {second_result=}'
        assert [device.capture_index for device in devices] == [0, 1, 2, 3]
        assert all(
            device.native_resolution == Resolution(640, 480)
            and device.metadata == {
                'capture_index': device.capture_index,
                'probe': 'opencv',
                'native_width': 640,
                'native_height': 480,
            }
            for device in devices
        ), f'{devices=}'
        assert [device.device_id for device in devices] == [
            'capture-index-0',
            'capture-index-1',
            'capture-index-2',
            'capture-index-3',
        ]
        assert all(not device.is_stable_id for device in devices), f'{devices=}'
        assert all(probe.is_released for probe in probes.values()), f'{probes=}'

    def test_skips_closed_indexes_but_rejects_malformed_probe_state(self) -> None:
        class ClosedProbe(FakeProbe):
            def isOpened(self) -> bool:
                return False

        probes = {
            0: ClosedProbe(),
            1: FakeProbe(),
        }
        discovery = OpenCVCaptureIndexDiscovery(
            max_capture_index=2,
            capture_opener=lambda capture_index: probes[capture_index],
        )

        devices = discovery.discover_devices()
        assert [device.capture_index for device in devices] == [1], f'{devices=}'
        assert probes[0].is_released
        assert probes[1].is_released

        class MalformedProbe(FakeProbe):
            def isOpened(self) -> str:
                return 'yes'

        discovery = OpenCVCaptureIndexDiscovery(
            max_capture_index=1,
            capture_opener=lambda _capture_index: MalformedProbe(),
        )
        with self.assertRaises(HardwareError):
            discovery.discover_devices()

    def test_malformed_probe_metadata_is_an_explicit_error(self) -> None:
        discovery = OpenCVCaptureIndexDiscovery(
            max_capture_index=1,
            capture_opener=lambda _capture_index: FakeProbe(
                native_resolution=('bad', 'metadata'),  # type: ignore[arg-type]
            ),
        )

        with self.assertRaises(HardwareError):
            discovery.discover_devices()

    def test_new_processes_use_their_own_capture_index_snapshot(self) -> None:
        class ClosedProbe(FakeProbe):
            def isOpened(self) -> bool:
                return False

        def build_discovery(
            available_indexes: set[int],
        ) -> OpenCVCaptureIndexDiscovery:
            def open_probe(capture_index: int) -> FakeProbe:
                if capture_index in available_indexes:
                    return FakeProbe()
                return ClosedProbe()

            return OpenCVCaptureIndexDiscovery(
                max_capture_index=10,
                capture_opener=open_probe,
            )

        first_process_devices = build_discovery({2, 7}).discover_devices()
        second_process_devices = build_discovery({0, 4}).discover_devices()

        assert [device.capture_index for device in first_process_devices] == [2, 7], (
            f'{first_process_devices=}'
        )
        assert [device.capture_index for device in second_process_devices] == [0, 4], (
            f'{second_process_devices=}'
        )
        assert [device.device_id for device in first_process_devices] != [
            device.device_id for device in second_process_devices
        ], f'{first_process_devices=}, {second_process_devices=}'

        first_process_slots = SessionCameraRegistry.from_devices(first_process_devices)
        second_process_slots = SessionCameraRegistry.from_devices(second_process_devices)
        first_cameras = first_process_slots.get_cameras()
        second_cameras = second_process_slots.get_cameras()
        assert [camera.slot_id for camera in first_cameras] == [
            'camera-0',
            'camera-1',
        ], f'{first_cameras=}'
        assert [camera.slot_id for camera in second_cameras] == [
            'camera-0',
            'camera-1',
        ], f'{second_cameras=}'
        assert [camera.capture_index for camera in first_cameras] == [2, 7], (
            f'{first_cameras=}'
        )
        assert [camera.capture_index for camera in second_cameras] == [0, 4], (
            f'{second_cameras=}'
        )


if __name__ == '__main__':
    unittest.main()
