import json
import tempfile
import threading
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multivision.calibration import CalibrationMetrics, CalibrationResult
from multivision.config import Configuration, load_configuration, save_configuration
from multivision.errors import CalibrationError, GeometryError
from multivision.fiducials import FiducialCorrespondence
from multivision.geometry import HomographyPair, Point2D
from multivision.persistence import (
    CalibrationRegistry,
    CalibrationStore,
    InvalidCalibrationStateError,
    PersistedCalibration,
)
from multivision.types import CalibrationStatus, Resolution


class CalibrationPersistenceTest(unittest.TestCase):
    def test_calibration_round_trip_preserves_required_metadata(self) -> None:
        calibration = _calibration()

        with tempfile.TemporaryDirectory() as temporary_directory:
            store = CalibrationStore(Path(temporary_directory) / 'calibrations.json')
            store.save(calibration)
            loaded = store.load()

        assert loaded == {'camera-a': calibration}, f'{loaded=}'
        saved_data = calibration.to_data()
        assert saved_data['camera_id'] == 'camera-a', f'{saved_data=}'
        assert saved_data['camera_resolution'] == {'width': 1280, 'height': 720}
        assert saved_data['projector_resolution'] == {'width': 1920, 'height': 1080}
        assert saved_data['version'] == 3
        assert saved_data['timestamp'] == 123.5
        assert 'metrics' in saved_data, f'{saved_data=}'
        assert 'projector_to_camera' in saved_data, f'{saved_data=}'
        assert 'camera_to_projector' in saved_data, f'{saved_data=}'

    def test_calibration_store_preserves_the_shared_configuration_document(self) -> None:
        calibration = _calibration()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'config.json'
            save_configuration(Configuration(camera_bindings={'overhead': 'camera-a'}), path)
            CalibrationStore(path).save(calibration)
            configuration = load_configuration(path)
            saved_data = json.loads(path.read_text(encoding='utf-8'))

        assert configuration.camera_bindings == {'overhead': 'camera-a'}, f'{configuration=}'
        assert 'camera-a' in saved_data['calibrations'], f'{saved_data=}'

    def test_restart_loaded_calibration_is_unverified(self) -> None:
        calibration = _calibration()
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = CalibrationStore(Path(temporary_directory) / 'calibrations.json')
            store.save(calibration)
            registry = CalibrationRegistry.from_store(
                store,
                calibration_version=3,
                projector_resolution=Resolution(1920, 1080),
            )

            assert registry.get_status('camera-a') is CalibrationStatus.UNVERIFIED
            with self.assertRaises(InvalidCalibrationStateError):
                registry.project_camera_to_projector('camera-a', Point2D(20, 20))

    def test_new_calibrations_require_verification_before_spatial_use(self) -> None:
        registry = CalibrationRegistry(
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )

        registry.register(
            _calibration_result(),
            camera_resolution=Resolution(1280, 720),
            projector_resolution=Resolution(1920, 1080),
        )

        assert registry.get_status('camera-a') is CalibrationStatus.UNVERIFIED
        with self.assertRaises(InvalidCalibrationStateError):
            registry.project_camera_to_projector('camera-a', Point2D(20, 20))

    def test_saved_calibration_requires_verification_before_spatial_use(self) -> None:
        registry = CalibrationRegistry(
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )
        calibration = _calibration()

        with tempfile.TemporaryDirectory() as temporary_directory:
            registry.save(
                CalibrationStore(Path(temporary_directory) / 'calibrations.json'),
                calibration,
            )

        assert registry.get_status('camera-a') is CalibrationStatus.UNVERIFIED

    def test_malformed_verification_is_stale_and_valid_retry_calibrates(self) -> None:
        registry = CalibrationRegistry(
            {'camera-a': _calibration()},
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )
        malformed = _verification_correspondences()
        malformed = malformed[:3] + (
            malformed[3]._replace(marker_id=[]),  # type: ignore[arg-type]
        )

        assert registry.verify(
            'camera-a',
            malformed,
            camera_resolution=Resolution(1280, 720),
        ) is CalibrationStatus.STALE
        assert registry.verify(
            'camera-a',
            _verification_correspondences(),
            camera_resolution=Resolution(1280, 720),
        ) is CalibrationStatus.CALIBRATED

    def test_falsey_invalid_verification_thresholds_are_not_defaulted(self) -> None:
        registry = CalibrationRegistry(
            {'camera-a': _calibration()},
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )

        with self.assertRaises(CalibrationError):
            registry.verify(
                'camera-a',
                _verification_correspondences(),
                camera_resolution=Resolution(1280, 720),
                thresholds=(),  # type: ignore[arg-type]
            )

    def test_resolution_and_verification_failures_stale_calibration(self) -> None:
        registry = CalibrationRegistry(
            {'camera-a': _calibration()},
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )
        correspondences = _verification_correspondences()

        assert registry.get_status('camera-a', Resolution(640, 480)) is CalibrationStatus.STALE
        assert registry.verify(
            'camera-a',
            correspondences,
            camera_resolution=Resolution(640, 480),
        ) is CalibrationStatus.STALE

        assert registry.verify(
            'camera-a',
            tuple(
                correspondence._replace(camera_position=Point2D(1000, 1000))
                for correspondence in correspondences
            ),
            camera_resolution=Resolution(1280, 720),
        ) is CalibrationStatus.STALE
        with self.assertRaises(InvalidCalibrationStateError):
            registry.project_camera_to_projector(
                'camera-a',
                Point2D(20, 20),
                camera_resolution=Resolution(1280, 720),
            )

    def test_successful_verification_allows_spatial_operation(self) -> None:
        registry = CalibrationRegistry(
            {'camera-a': _calibration()},
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )

        status = registry.verify(
            'camera-a',
            _verification_correspondences(),
            camera_resolution=Resolution(1280, 720),
        )

        assert status is CalibrationStatus.CALIBRATED
        assert registry.project_camera_to_projector(
            'camera-a',
            Point2D(20, 20),
            camera_resolution=Resolution(1280, 720),
        ) == Point2D(20, 20)

    def test_falsey_projector_resolution_cannot_bypass_stale_state(self) -> None:
        registry = CalibrationRegistry(
            {'camera-a': _calibration()},
            calibration_version=3,
            projector_resolution=Resolution(1920, 1080),
        )
        assert registry.verify(
            'camera-a',
            _verification_correspondences(),
            camera_resolution=Resolution(1280, 720),
        ) is CalibrationStatus.CALIBRATED

        assert registry.get_status('camera-a', projector_resolution=()) is CalibrationStatus.STALE
        with self.assertRaises(GeometryError):
            registry.project_camera_to_projector(
                'camera-a',
                Point2D(20, 20),
                projector_resolution=(),
            )

    def test_non_numeric_valid_region_is_rejected(self) -> None:
        calibration_data = _calibration().to_data()
        calibration_data['valid_region'] = [
            ['0', '0'],
            ['100', '0'],
            ['100', '100'],
        ]

        with self.assertRaises(CalibrationError):
            PersistedCalibration.from_data(calibration_data)

    def test_zero_area_valid_region_is_rejected(self) -> None:
        calibration_data = _calibration().to_data()
        calibration_data['valid_region'] = [
            [0, 0],
            [100, 0],
            [0, 0],
        ]

        with self.assertRaises(CalibrationError):
            PersistedCalibration.from_data(calibration_data)

    def test_malformed_metric_relationships_are_rejected(self) -> None:
        invalid_metrics = (
            CalibrationMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0),
            CalibrationMetrics(1, 4, 5, 1.0, 0.0, 0.0, 1.0),
            CalibrationMetrics(2, 4, 4, 1.0, 2.0, 1.0, 1.0),
            CalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 2.0),
            CalibrationMetrics(4, 16, 16, 0.5, 0.0, 0.0, 1.0),
        )
        for metrics in invalid_metrics:
            with self.subTest(metrics=metrics):
                with self.assertRaises(CalibrationError):
                    PersistedCalibration(
                        camera_id='camera-a',
                        camera_resolution=Resolution(1280, 720),
                        projector_resolution=Resolution(1920, 1080),
                        version=3,
                        projector_to_camera=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                        camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                        metrics=metrics,
                        timestamp=123.5,
                        valid_region=(
                            Point2D(0, 0),
                            Point2D(100, 0),
                            Point2D(100, 100),
                            Point2D(0, 100),
                        ),
                    )

    def test_uncalibrated_camera_fails_closed(self) -> None:
        registry = CalibrationRegistry()

        assert registry.get_status('missing') is CalibrationStatus.UNCALIBRATED
        with self.assertRaises(GeometryError):
            registry.project_camera_to_projector('missing', Point2D(1, 1))

    def test_concurrent_store_saves_do_not_lose_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = _PausingCalibrationStore(Path(temporary_directory) / 'calibrations.json')
            errors: list[BaseException] = []

            def save_first() -> None:
                try:
                    store.save(_calibration('camera-a'))
                except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                    errors.append(ex)

            def save_second() -> None:
                try:
                    store.save(_calibration('camera-b'))
                except BaseException as ex:  # noqa: BLE001 (test thread cleanup).
                    errors.append(ex)

            first_thread = threading.Thread(target=save_first)
            second_thread = threading.Thread(target=save_second)
            first_thread.start()
            assert store.first_write_started.wait(1), f'{store.first_write_started=}'
            second_thread.start()
            try:
                assert not store.second_write_started.wait(0.1), (
                    f'{store.second_write_started=}'
                )
            finally:
                store.release_first_write.set()
            first_thread.join(2)
            second_thread.join(2)

            assert not first_thread.is_alive(), f'{first_thread=}'
            assert not second_thread.is_alive(), f'{second_thread=}'
            assert errors == [], f'{errors=}'
            assert set(store.load()) == {'camera-a', 'camera-b'}, f'{store.load()=}'


def _calibration(camera_id: str = 'camera-a') -> PersistedCalibration:
    return PersistedCalibration(
        camera_id=camera_id,
        camera_resolution=Resolution(1280, 720),
        projector_resolution=Resolution(1920, 1080),
        version=3,
        projector_to_camera=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        metrics=CalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 1.0),
        timestamp=123.5,
        valid_region=(Point2D(0, 0), Point2D(100, 0), Point2D(100, 100), Point2D(0, 100)),
    )


def _calibration_result() -> CalibrationResult:
    return CalibrationResult(
        HomographyPair.from_projector_to_camera(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ),
        (Point2D(0, 0), Point2D(100, 0), Point2D(100, 100)),
        CalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 1.0),
        'camera-a',
    )


class _PausingCalibrationStore(CalibrationStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.first_write_started = threading.Event()
        self.release_first_write = threading.Event()
        self.second_write_started = threading.Event()
        self._write_count = 0
        self._write_count_lock = threading.Lock()

    def _write_document(self, data: Mapping[str, Any]) -> None:
        with self._write_count_lock:
            self._write_count += 1
            write_count = self._write_count
        if write_count == 1:
            self.first_write_started.set()
            if not self.release_first_write.wait(2):
                raise AssertionError('Timed out waiting to release the first save')
        else:
            self.second_write_started.set()
        super()._write_document(data)


def _verification_correspondences() -> tuple[FiducialCorrespondence, ...]:
    return tuple(
        FiducialCorrespondence(
            marker_id,
            0,
            Point2D(20 + marker_id * 10, 20 + marker_id * 10),
            Point2D(20 + marker_id * 10, 20 + marker_id * 10),
        )
        for marker_id in range(4)
    )


if __name__ == '__main__':
    unittest.main()
