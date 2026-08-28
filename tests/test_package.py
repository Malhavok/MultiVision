import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multivision.config import (
    CalibrationThresholds,
    Configuration,
    load_configuration,
    save_configuration,
)
from multivision.discovery import MacOSDeviceDiscovery
from multivision.errors import (
    CameraOpenError,
    ConfigurationError,
    FrameCaptureError,
    HardwareError,
)
from multivision.hardware import (
    OpenCVCaptureDevice,
    OpenCVCaptureDeviceFactory,
)
from multivision.types import (
    CalibrationStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
)


class PackageTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        package = importlib.import_module('multivision')
        assert package.__name__ == 'multivision'
        assert RuntimeStatus.UNAVAILABLE.value == 'UNAVAILABLE'
        assert CalibrationStatus.UNVERIFIED.value == 'UNVERIFIED'

    def test_configuration_round_trip(self) -> None:
        configuration = Configuration(
            camera_bindings={
                'overhead': 'device-overhead',
                'side-left': 'device-left',
                'side-right': 'device-right',
                'macbook': 'device-macbook',
            },
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            save_configuration(configuration, path)
            loaded_configuration = load_configuration(path)

        assert loaded_configuration == configuration, f'{loaded_configuration=}, {configuration=}'

    def test_malformed_configuration_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            path.write_text('{ malformed', encoding='utf-8')

            with self.assertRaises(ConfigurationError):
                load_configuration(path)

    def test_missing_configuration_uses_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            configuration = load_configuration(Path(temporary_directory) / 'missing.json')

        assert configuration.projector_resolution == Resolution(1920, 1080), f'{configuration=}'
        assert configuration.camera_bindings == {}, f'{configuration=}'

    def test_configuration_paths_and_values_are_rejected_at_save_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            configuration = Configuration(camera_bindings={'overhead': 'device-id'})
            configuration.camera_bindings['overhead'] = None  # type: ignore[assignment]

            with self.assertRaises(ConfigurationError):
                save_configuration(configuration, path)
            with self.assertRaises(ConfigurationError):
                load_configuration(str(path))  # type: ignore[arg-type]
            with self.assertRaises(ConfigurationError):
                save_configuration(configuration, str(path))  # type: ignore[arg-type]

        with self.assertRaises(ConfigurationError):
            save_configuration(object(), Path('config.json'))  # type: ignore[arg-type]

    def test_malformed_configuration_is_rejected(self) -> None:
        malformed_values = [
            [],
            {'camera_bindings': []},
            {'projector_resolution': {'width': 0}},
            {'calibration_thresholds': {'min_inlier_ratio': True}},
            {'calibration_version': 0},
        ]
        for malformed_value in malformed_values:
            with self.subTest(malformed_value=malformed_value):
                with self.assertRaises(ConfigurationError):
                    Configuration.from_data(malformed_value)

    def test_non_finite_thresholds_are_rejected(self) -> None:
        for invalid_value in [math.nan, math.inf, -math.inf]:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ConfigurationError):
                    CalibrationThresholds(max_mean_reprojection_error=invalid_value)

    def test_save_is_valid_json_even_when_replaced(self) -> None:
        configuration = Configuration(camera_bindings={'side-left': 'device-b'})
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            save_configuration(configuration, path)
            saved_data = json.loads(path.read_text(encoding='utf-8'))
            temporary_files = list(path.parent.glob(f'.{path.name}.*'))

        assert saved_data == configuration.to_data(), f'{saved_data=}'
        assert temporary_files == [], f'{temporary_files=}'

    def test_capture_boundary_rejects_failures_and_malformed_results(self) -> None:
        class FailingCapture:
            def isOpened(self) -> bool:
                return True

            def read(self) -> tuple[bool, object]:
                raise RuntimeError('device disconnected')

            def release(self) -> None:
                return None

        class MalformedCapture(FailingCapture):
            def read(self) -> tuple[bool, object]:
                return (True,)

        for capture in [FailingCapture(), MalformedCapture()]:
            with self.subTest(capture=capture.__class__.__name__):
                with self.assertRaises(FrameCaptureError):
                    OpenCVCaptureDevice(capture).read()

        class ReleasingCapture(FailingCapture):
            def release(self) -> None:
                raise RuntimeError('release failed')

        with self.assertRaises(HardwareError):
            OpenCVCaptureDevice(ReleasingCapture()).release()

    def test_capture_factory_rejects_invalid_device_before_opencv(self) -> None:
        factory = OpenCVCaptureDeviceFactory()
        for capture_index in [-1, True, None]:
            device = DeviceInfo('stable-id', 'Camera', capture_index=capture_index)
            with self.subTest(capture_index=capture_index):
                with self.assertRaises(CameraOpenError):
                    factory.open_capture(device)

    def test_capture_factory_rejects_malformed_open_state(self) -> None:
        class FakeOpenCVCapture:
            def isOpened(self) -> str:
                return 'yes'

            def release(self) -> None:
                return None

        fake_cv2 = SimpleNamespace(VideoCapture=lambda *_arguments: FakeOpenCVCapture())
        device = DeviceInfo('stable-id', 'Camera', capture_index=0)
        with patch.dict(sys.modules, {'cv2': fake_cv2}):
            with self.assertRaises(CameraOpenError):
                OpenCVCaptureDeviceFactory().open_capture(device)

    def test_capture_factory_rejects_missing_discovered_backend(self) -> None:
        video_capture_calls: list[tuple[object, ...]] = []

        class FakeOpenCVCapture:
            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                return None

        def video_capture(*arguments: object) -> FakeOpenCVCapture:
            video_capture_calls.append(arguments)
            return FakeOpenCVCapture()

        fake_cv2 = SimpleNamespace(VideoCapture=video_capture)
        device = DeviceInfo(
            'stable-id',
            'Camera',
            capture_index=3,
            backend_name='avfoundation',
        )
        with patch.dict(sys.modules, {'cv2': fake_cv2}):
            with self.assertRaises(CameraOpenError):
                OpenCVCaptureDeviceFactory().open_capture(device)

        assert video_capture_calls == [], f'{video_capture_calls=}'

    def test_capture_factory_uses_discovered_backend(self) -> None:
        capture_arguments: list[tuple[int, int]] = []

        class FakeOpenCVCapture:
            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                return None

        def video_capture(*arguments: int) -> FakeOpenCVCapture:
            capture_arguments.append(arguments)
            return FakeOpenCVCapture()

        fake_cv2 = SimpleNamespace(
            CAP_AVFOUNDATION=42,
            VideoCapture=video_capture,
        )
        device = DeviceInfo(
            'stable-id',
            'Camera',
            capture_index=3,
            backend_name='avfoundation',
        )
        with patch.dict(sys.modules, {'cv2': fake_cv2}):
            capture = OpenCVCaptureDeviceFactory().open_capture(device)
            capture.release()

        assert capture_arguments == [(3, 42)], f'{capture_arguments=}'

    def test_discovery_exposes_platform_metadata_and_stable_ids(self) -> None:
        profiler_data = {
            'SPCameraDataType': [
                {
                    '_name': 'Continuity Camera',
                    'spcamera_device_unique_id': 'iphone-camera-id',
                    'spcamera_device_type': 'Continuity Camera',
                    'spcamera_video_control': {'auto_exposure': True},
                },
                {
                    '_name': 'MacBook Camera',
                    'spcamera_model_id': 'built-in-camera-model',
                },
            ],
        }
        completed_process = type(
            'CompletedProcess',
            (),
            {'returncode': 0, 'stdout': json.dumps(profiler_data)},
        )()

        discovery = MacOSDeviceDiscovery(
            command_runner=lambda *args, **kwargs: completed_process,
            platform_name='darwin',
        )

        devices = discovery.discover_devices()

        assert [device.device_id for device in devices] == [
            'iphone-camera-id',
            'unstable-macos-name:MacBook Camera',
        ], f'{devices=}'
        assert devices[0].metadata == profiler_data['SPCameraDataType'][0], f'{devices=}'
        assert devices[0].native_resolution is None, f'{devices=}'
        assert devices[0].backend_name == 'avfoundation', f'{devices=}'
        assert devices[0].is_stable_id is True, f'{devices=}'
        assert devices[1].is_stable_id is False, f'{devices=}'
        assert [device.capture_index for device in devices] == [0, 1]

    def test_discovery_accepts_system_profiler_hyphenated_unique_ids(self) -> None:
        profiler_data = {
            'SPCameraDataType': [
                {
                    '_name': 'GENERAL WEBCAM',
                    'spcamera_unique-id': 'uvc-camera-id',
                },
            ],
        }
        completed_process = type(
            'CompletedProcess',
            (),
            {'returncode': 0, 'stdout': json.dumps(profiler_data)},
        )()

        discovery = MacOSDeviceDiscovery(
            command_runner=lambda *args, **kwargs: completed_process,
            platform_name='darwin',
        )

        devices = discovery.discover_devices()

        assert devices[0].device_id == 'uvc-camera-id', f'{devices=}'
        assert devices[0].is_stable_id is True, f'{devices=}'

    def test_discovery_resolves_native_indexes_from_one_avfoundation_snapshot(self) -> None:
        profiler_data = {
            'SPCameraDataType': [
                {'_name': 'Camera A', 'spcamera_unique-id': 'device-a'},
                {'_name': 'Camera B', 'spcamera_unique-id': 'device-b'},
            ],
        }
        completed_process = type(
            'CompletedProcess',
            (),
            {'returncode': 0, 'stdout': json.dumps(profiler_data)},
        )()

        with (
            patch(
                'multivision.discovery.subprocess.run',
                return_value=completed_process,
            ),
            patch(
                'multivision.discovery._resolve_avfoundation_capture_indices',
                return_value={'device-a': 1, 'device-b': 0},
            ) as resolve_capture_indices,
        ):
            devices = MacOSDeviceDiscovery(platform_name='darwin').discover_devices()

        assert [device.capture_index for device in devices] == [1, 0], f'{devices=}'
        resolve_capture_indices.assert_called_once_with()

    def test_discovery_uses_native_index_resolution_for_real_commands(self) -> None:
        profiler_data = {
            'SPCameraDataType': [
                {'_name': 'Camera A', 'spcamera_device_unique_id': 'device-a'},
                {'_name': 'Camera B', 'spcamera_device_unique_id': 'device-b'},
            ],
        }
        completed_process = type(
            'CompletedProcess',
            (),
            {'returncode': 0, 'stdout': json.dumps(profiler_data)},
        )()
        discovery = MacOSDeviceDiscovery(
            command_runner=lambda *args, **kwargs: completed_process,
            platform_name='darwin',
            capture_index_resolver=lambda device_id: {
                'device-a': 1,
                'device-b': 0,
            }.get(device_id),
        )

        devices = discovery.discover_devices()

        assert [device.capture_index for device in devices] == [1, 0], f'{devices=}'

    def test_malformed_system_profiler_results_are_explicit_errors(self) -> None:
        malformed_results = [None, object()]
        for malformed_result in malformed_results:
            with self.subTest(malformed_result=malformed_result.__class__.__name__):
                discovery = MacOSDeviceDiscovery(
                    command_runner=lambda *args, **kwargs: malformed_result,
                    platform_name='darwin',
                )
                with self.assertRaises(HardwareError):
                    discovery.discover_devices()

        def failing_command(*args: object, **kwargs: object) -> object:
            raise RuntimeError('system_profiler failed')

        discovery = MacOSDeviceDiscovery(
            command_runner=failing_command,
            platform_name='darwin',
        )
        with self.assertRaises(HardwareError):
            discovery.discover_devices()
