import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multivision.config import (
    CalibrationThresholds,
    Configuration,
    MetricCalibrationThresholds,
    ProjectorOutputDescriptor,
    load_configuration,
    save_configuration,
)
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
from multivision.pattern import (
    APRILTAG_FAMILIES,
    DEFAULT_TAG_DICTIONARY,
    DICT_5X5_1000,
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
        configuration = Configuration()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            save_configuration(configuration, path)
            loaded_configuration = load_configuration(path)

        assert loaded_configuration == configuration, f'{loaded_configuration=}, {configuration=}'

    def test_tag_dictionary_defaults_and_round_trips(self) -> None:
        assert Configuration().tag_dictionary == DEFAULT_TAG_DICTIONARY
        assert Configuration().to_data()['tag_dictionary'] == DICT_5X5_1000
        for dictionary_name in sorted(APRILTAG_FAMILIES | {DICT_5X5_1000}):
            configuration = Configuration(tag_dictionary=dictionary_name)
            round_tripped = Configuration.from_data(configuration.to_data())
            assert round_tripped == configuration, f'{round_tripped=}, {configuration=}'

    def test_unsupported_tag_dictionary_is_explicitly_rejected(self) -> None:
        for dictionary_name in [None, '', 'DICT_UNKNOWN', 1, True]:
            with self.subTest(dictionary_name=dictionary_name):
                with self.assertRaises(ConfigurationError):
                    Configuration(tag_dictionary=dictionary_name)  # type: ignore[arg-type]
                with self.assertRaises(ConfigurationError):
                    Configuration.from_data({'tag_dictionary': dictionary_name})

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

    def test_configuration_paths_and_values_are_rejected_at_save_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            configuration = Configuration()

            with self.assertRaises(ConfigurationError):
                load_configuration(str(path))  # type: ignore[arg-type]
            with self.assertRaises(ConfigurationError):
                save_configuration(configuration, str(path))  # type: ignore[arg-type]

        with self.assertRaises(ConfigurationError):
            save_configuration(object(), Path('config.json'))  # type: ignore[arg-type]

    def test_malformed_configuration_is_rejected(self) -> None:
        malformed_values = [
            [],
            {'projector_resolution': {'width': 0}},
            {'calibration_thresholds': {'min_inlier_ratio': True}},
            {'metric_calibration_thresholds': []},
            {'projector_output_identity': None},
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

    def test_configuration_keeps_existing_positional_arguments(self) -> None:
        configuration = Configuration(
            Resolution(1600, 900),
            CalibrationThresholds(min_unique_tags=6),
            3,
        )

        assert configuration.projector_resolution == Resolution(1600, 900), f'{configuration=}'
        assert configuration.calibration_thresholds.min_unique_tags == 6, f'{configuration=}'
        assert configuration.calibration_version == 3, f'{configuration=}'

    def test_metric_configuration_round_trip_and_projector_descriptor(self) -> None:
        configuration = Configuration(
            projector_resolution=Resolution(1600, 900),
            projector_output_identity='table-projector',
            metric_calibration_thresholds=MetricCalibrationThresholds(
                ransac_reprojection_threshold_mm=1.5,
                max_capture_corner_jitter_pixels=1.0,
                max_mean_fit_error_mm=1.0,
                max_fit_error_mm=3.0,
                min_inlier_ratio=0.75,
                min_unique_target_fiducials=5,
                min_spatial_coverage=0.6,
            ),
        )

        loaded_configuration = Configuration.from_data(configuration.to_data())

        assert loaded_configuration == configuration, f'{loaded_configuration=}'
        assert configuration.projector_output_descriptor == ProjectorOutputDescriptor(
            Resolution(1600, 900),
            'table-projector',
        )

    def test_metric_configuration_rejects_invalid_values(self) -> None:
        invalid_values = {
            'ransac_reprojection_threshold_mm': [0, -1, math.nan, math.inf],
            'max_capture_white_balance_delta': [0, -1, 1.1, math.nan, math.inf],
            'max_capture_corner_jitter_pixels': [0, -1, math.nan, math.inf],
            'max_mean_fit_error_mm': [-1, math.nan, math.inf],
            'max_fit_error_mm': [-1, math.nan, math.inf],
            'min_inlier_ratio': [-1, 1.1, math.nan, math.inf],
            'min_unique_target_fiducials': [0, -1, True],
            'min_capture_marker_ratio': [-1, 0, 1.1, math.nan, math.inf],
            'min_spatial_coverage': [-1, 1.1, math.nan, math.inf],
        }
        for field_name, values in invalid_values.items():
            for invalid_value in values:
                with self.subTest(field_name=field_name, invalid_value=invalid_value):
                    with self.assertRaises(ConfigurationError):
                        MetricCalibrationThresholds(**{field_name: invalid_value})

        for invalid_identity in ['', None, 1, True]:
            with self.subTest(invalid_identity=invalid_identity):
                with self.assertRaises(ConfigurationError):
                    Configuration(
                        projector_output_identity=invalid_identity,  # type: ignore[arg-type]
                    )

    def test_save_preserves_unrelated_configuration_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            path.write_text(
                json.dumps({'unrelated_setting': {'enabled': True}}),
                encoding='utf-8',
            )

            save_configuration(Configuration(), path)
            saved_data = json.loads(path.read_text(encoding='utf-8'))

        assert saved_data['unrelated_setting'] == {'enabled': True}, f'{saved_data=}'

    def test_save_is_valid_json_even_when_replaced(self) -> None:
        configuration = Configuration()
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

    def test_capture_boundary_switches_exact_black_frames_to_fallback(self) -> None:
        class BlackCapture:
            def __init__(self) -> None:
                self.is_released = False

            def isOpened(self) -> bool:
                return not self.is_released

            def read(self) -> tuple[bool, object]:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self) -> None:
                self.is_released = True

        class FallbackCapture:
            def __init__(self) -> None:
                self.is_released = False
                self.read_count = 0

            def is_opened(self) -> bool:
                return not self.is_released

            def read(self) -> tuple[bool, object]:
                self.read_count += 1
                return True, np.full((2, 2, 3), 7, dtype=np.uint8)

            def release(self) -> None:
                self.is_released = True

        original_capture = BlackCapture()
        fallback_capture = FallbackCapture()
        capture = OpenCVCaptureDevice(
            original_capture,
            fallback_opener=lambda: fallback_capture,
        )

        for _ in range(2):
            success, frame = capture.read()
            assert success is True, f'{success=}'
            assert frame is not None and frame.max() == 0, f'{frame=}'
            assert fallback_capture.read_count == 0, f'{fallback_capture.read_count=}'

        success, frame = capture.read()

        assert success is True, f'{success=}'
        assert frame is not None and frame.max() == 7, f'{frame=}'
        assert original_capture.is_released, f'{original_capture.is_released=}'
        assert fallback_capture.read_count == 1, f'{fallback_capture.read_count=}'
        capture.release()
        assert fallback_capture.is_released, f'{fallback_capture.is_released=}'

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
