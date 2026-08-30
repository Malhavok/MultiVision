"""Persistent calibration records and fail-closed verification state."""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from multivision.calibration import (
    CalibrationMetrics,
    CalibrationResult,
    validate_correspondences_against_pattern,
)
from multivision.config import (
    DEFAULT_CONFIG_PATH,
    CalibrationThresholds,
    ProjectorOutputDescriptor,
    _get_configuration_file_lock,
)
from multivision.errors import (
    CalibrationError,
    ConfigurationError,
    GeometryError,
    InvalidCalibrationStateError,
    InvalidHomographyError,
)
from multivision.fiducials import CameraCorrespondences, FiducialCorrespondence
from multivision.geometry import (
    CoordinateBounds,
    MatrixLike,
    Point2D,
    RegionLike,
    invert_homography,
    is_finite_point,
    is_point_in_resolution,
    project_camera_to_projector,
    project_point,
    validate_homography,
    validate_point_in_region,
)
from multivision.pattern import CalibrationPattern
from multivision.types import (
    CalibrationStatus,
    Resolution,
    is_finite_real,
    is_valid_resolution,
)


@dataclass(frozen=True, init=False)
class PersistedCalibration:
    """A complete calibration record, independent of its current trust state."""

    camera_id: str
    camera_resolution: Resolution
    projector_resolution: Resolution
    projector_output_descriptor: ProjectorOutputDescriptor
    version: int
    projector_to_camera: tuple[tuple[float, float, float], ...]
    camera_to_projector: tuple[tuple[float, float, float], ...]
    metrics: CalibrationMetrics
    timestamp: float
    valid_region: tuple[Point2D, ...]

    def __init__(
        self,
        camera_id: str,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        version: int,
        projector_to_camera: MatrixLike,
        camera_to_projector: MatrixLike,
        metrics: CalibrationMetrics,
        timestamp: float,
        valid_region: RegionLike,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
        projector_output_identity: str | None = None,
    ) -> None:
        _validate_camera_id(camera_id)
        _validate_resolution(camera_resolution, 'camera_resolution')
        _validate_resolution(projector_resolution, 'projector_resolution')
        checked_projector_output_descriptor = _normalise_projector_output_descriptor(
            projector_resolution,
            projector_output_descriptor,
            projector_output_identity,
        )
        _validate_version(version)
        try:
            checked_projector_to_camera = validate_homography(projector_to_camera)
            checked_camera_to_projector = validate_homography(camera_to_projector)
        except (TypeError, ValueError, InvalidHomographyError) as ex:
            raise CalibrationError('Calibration contains an invalid homography') from ex
        if not _are_inverse_matrices(checked_projector_to_camera, checked_camera_to_projector):
            raise CalibrationError('Calibration homography matrices are not inverses')
        _validate_metrics(metrics)
        if not is_finite_real(timestamp):
            raise CalibrationError('Calibration timestamp must be a finite number')
        object.__setattr__(self, 'camera_id', camera_id)
        object.__setattr__(self, 'camera_resolution', camera_resolution)
        object.__setattr__(self, 'projector_resolution', projector_resolution)
        object.__setattr__(
            self,
            'projector_output_descriptor',
            checked_projector_output_descriptor,
        )
        object.__setattr__(self, 'version', version)
        object.__setattr__(self, 'projector_to_camera', checked_projector_to_camera)
        object.__setattr__(self, 'camera_to_projector', checked_camera_to_projector)
        object.__setattr__(self, 'metrics', metrics)
        object.__setattr__(self, 'timestamp', float(timestamp))
        object.__setattr__(self, 'valid_region', _normalise_region(valid_region))

    @property
    def calibration_version(self) -> int:
        return self.version

    @property
    def projector_output_identity(self) -> str:
        return self.projector_output_descriptor.output_identity

    @classmethod
    def from_result(
        cls,
        result: CalibrationResult,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        version: int = 1,
        timestamp: float | None = None,
        camera_id: str | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
        projector_output_identity: str | None = None,
    ) -> 'PersistedCalibration':
        if not isinstance(result, CalibrationResult):
            raise CalibrationError('result must be CalibrationResult')
        resolved_camera_id = camera_id if camera_id is not None else result.camera_id
        if resolved_camera_id is None:
            raise CalibrationError('A stable camera ID is required for persistence')
        return cls(
            resolved_camera_id,
            camera_resolution,
            projector_resolution,
            version,
            result.projector_to_camera,
            result.camera_to_projector,
            result.metrics,
            time.time() if timestamp is None else timestamp,
            result.valid_region,
            projector_output_descriptor,
            projector_output_identity,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> 'PersistedCalibration':
        if not isinstance(data, Mapping):
            raise CalibrationError('Calibration record must be an object')
        required_fields = (
            'camera_id',
            'camera_resolution',
            'projector_resolution',
            'projector_to_camera',
            'camera_to_projector',
            'metrics',
            'timestamp',
            'valid_region',
        )
        if any(field_name not in data for field_name in required_fields):
            raise CalibrationError('Calibration record is missing required fields')
        version = data.get('version', data.get('calibration_version'))
        if version is None:
            raise CalibrationError('Calibration record is missing required fields')
        projector_resolution = _parse_resolution(
            data['projector_resolution'],
            'projector_resolution',
        )
        projector_output_descriptor = _parse_projector_output_descriptor_data(
            data.get('projector_output_descriptor'),
            projector_resolution,
            data.get('projector_output_identity'),
        )
        return cls(
            data['camera_id'],
            _parse_resolution(data['camera_resolution'], 'camera_resolution'),
            projector_resolution,
            version,
            data['projector_to_camera'],
            data['camera_to_projector'],
            _parse_metrics(data['metrics']),
            data['timestamp'],
            _parse_region(data['valid_region']),
            projector_output_descriptor,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'camera_resolution': _resolution_to_data(self.camera_resolution),
            'projector_resolution': _resolution_to_data(self.projector_resolution),
            'projector_output_descriptor': {
                'projector_resolution': _resolution_to_data(
                    self.projector_output_descriptor.projector_resolution,
                ),
                'output_identity': self.projector_output_descriptor.output_identity,
            },
            'version': self.version,
            'projector_to_camera': [list(row) for row in self.projector_to_camera],
            'camera_to_projector': [list(row) for row in self.camera_to_projector],
            'metrics': _metrics_to_data(self.metrics),
            'timestamp': self.timestamp,
            'valid_region': [_point_to_data(point) for point in self.valid_region],
        }


def _parse_calibrations(data: Mapping[str, Any]) -> dict[str, PersistedCalibration]:
    raw_calibrations = data.get('calibrations', {})
    if not isinstance(raw_calibrations, Mapping):
        raise CalibrationError('Calibration file must contain a calibrations object')
    if any(not isinstance(camera_id, str) for camera_id in raw_calibrations):
        raise CalibrationError('Calibration keys must be stable camera IDs')

    records: dict[str, PersistedCalibration] = {}
    for camera_id in sorted(raw_calibrations):
        record = PersistedCalibration.from_data(raw_calibrations[camera_id])
        if record.camera_id != camera_id:
            raise CalibrationError('Calibration key does not match camera_id')
        records[camera_id] = record
    return records


class CalibrationStore:
    """Read and atomically update calibration records in one JSON file."""

    def __init__(self, path: pathlib.Path | None = None) -> None:
        resolved_path = DEFAULT_CONFIG_PATH if path is None else path
        if not isinstance(resolved_path, pathlib.Path):
            raise CalibrationError('Calibration path must be a pathlib.Path')
        self.path = resolved_path
        self._lock = _get_configuration_file_lock(resolved_path)

    def load(self) -> dict[str, PersistedCalibration]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, PersistedCalibration]:
        return _parse_calibrations(self._read_document())

    def save(self, calibration: PersistedCalibration) -> None:
        if not isinstance(calibration, PersistedCalibration):
            raise CalibrationError('calibration must be PersistedCalibration')
        with self._lock:
            data = self._read_document()
            records = _parse_calibrations(data)
            records[calibration.camera_id] = calibration
            data['calibrations'] = {
                camera_id: records[camera_id].to_data()
                for camera_id in sorted(records)
            }
            self._write_document(data)

    def _read_document(self) -> dict[str, Any]:
        try:
            raw_data = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError) as ex:
            raise CalibrationError(f'Could not read calibrations at {self.path}') from ex
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as ex:
            raise CalibrationError(f'Could not read calibrations at {self.path}') from ex
        if not isinstance(data, dict):
            raise CalibrationError('Calibration file must contain an object')
        return data

    def _write_document(self, data: Mapping[str, Any]) -> None:
        temporary_path: pathlib.Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialised_data = json.dumps(
                data,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + '\n'
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=self.path.parent,
                prefix=f'.{self.path.name}.',
                delete=False,
            ) as temporary_file:
                temporary_path = pathlib.Path(temporary_file.name)
                temporary_file.write(serialised_data)
            temporary_path.replace(self.path)
        except (OSError, TypeError, ValueError) as ex:
            raise CalibrationError(f'Could not write calibrations at {self.path}') from ex
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def delete(self, camera_id: str) -> None:
        _validate_camera_id(camera_id)
        with self._lock:
            data = self._read_document()
            records = _parse_calibrations(data)
            if camera_id not in records:
                return
            del records[camera_id]
            data['calibrations'] = {
                record_camera_id: records[record_camera_id].to_data()
                for record_camera_id in sorted(records)
            }
            self._write_document(data)


class CalibrationRegistry:
    """Own calibration trust states and reject all untrusted spatial requests."""

    def __init__(
        self,
        calibrations: Mapping[str, PersistedCalibration] | None = None,
        calibration_version: int = 1,
        projector_resolution: Resolution | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> None:
        _validate_version(calibration_version)
        if projector_resolution is not None:
            _validate_resolution(projector_resolution, 'projector_resolution')
        if projector_output_descriptor is not None and not isinstance(
            projector_output_descriptor,
            ProjectorOutputDescriptor,
        ):
            raise CalibrationError(
                'projector_output_descriptor must be ProjectorOutputDescriptor',
            )
        if (
            projector_output_descriptor is not None
            and projector_resolution is not None
            and projector_output_descriptor.projector_resolution != projector_resolution
        ):
            raise CalibrationError(
                'projector_output_descriptor resolution does not match projector_resolution',
            )
        if calibrations is None:
            initial_calibrations: Mapping[str, PersistedCalibration] = {}
        elif not isinstance(calibrations, Mapping):
            raise CalibrationError('calibrations must be keyed by stable camera ID')
        else:
            initial_calibrations = calibrations
        if any(
            not isinstance(camera_id, str)
            or not isinstance(calibration, PersistedCalibration)
            or calibration.camera_id != camera_id
            for camera_id, calibration in initial_calibrations.items()
        ):
            raise CalibrationError('calibrations must be keyed by stable camera ID')
        self._calibrations = {
            camera_id: initial_calibrations[camera_id]
            for camera_id in sorted(initial_calibrations)
        }
        self._calibration_version = calibration_version
        self._projector_resolution = projector_resolution
        self._projector_output_descriptor = projector_output_descriptor
        self._statuses = {
            camera_id: CalibrationStatus.UNVERIFIED
            for camera_id in self._calibrations
        }
        self._lock = threading.RLock()

    @classmethod
    def from_store(
        cls,
        store: CalibrationStore,
        calibration_version: int = 1,
        projector_resolution: Resolution | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> 'CalibrationRegistry':
        if not isinstance(store, CalibrationStore):
            raise CalibrationError('store must be CalibrationStore')
        return cls(
            store.load(),
            calibration_version,
            projector_resolution,
            projector_output_descriptor,
        )

    @property
    def calibrations(self) -> dict[str, PersistedCalibration]:
        """Return a compatibility snapshot of the registry records."""
        return self.get_records()

    def get_record(self, camera_id: str) -> PersistedCalibration | None:
        """Return one calibration record without exposing mutable storage."""
        _validate_camera_id(camera_id)
        with self._lock:
            return self._calibrations.get(camera_id)

    def get_records(self) -> dict[str, PersistedCalibration]:
        """Return a stable snapshot of all calibration records."""
        with self._lock:
            return dict(self._calibrations)

    def load(self, store: CalibrationStore) -> None:
        """Load records as unverified, even when they were trusted before restart."""
        loaded_calibrations = store.load()
        with self._lock:
            self._calibrations = loaded_calibrations
            self._statuses = {
                camera_id: CalibrationStatus.UNVERIFIED
                for camera_id in loaded_calibrations
            }

    def save(self, store: CalibrationStore, calibration: PersistedCalibration) -> None:
        if not isinstance(store, CalibrationStore):
            raise CalibrationError('store must be CalibrationStore')
        if not isinstance(calibration, PersistedCalibration):
            raise CalibrationError('calibration must be PersistedCalibration')
        store.save(calibration)
        with self._lock:
            self._calibrations[calibration.camera_id] = calibration
            self._statuses[calibration.camera_id] = CalibrationStatus.UNVERIFIED

    def register(
        self,
        result: CalibrationResult,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        version: int | None = None,
        timestamp: float | None = None,
        camera_id: str | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> PersistedCalibration:
        record = PersistedCalibration.from_result(
            result,
            camera_resolution,
            projector_resolution,
            self._calibration_version if version is None else version,
            timestamp,
            camera_id,
            projector_output_descriptor or self._projector_output_descriptor,
        )
        with self._lock:
            self._calibrations[record.camera_id] = record
            self._statuses[record.camera_id] = CalibrationStatus.UNVERIFIED
        return record

    def update_projector_descriptor(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
    ) -> None:
        """Set the current output descriptor used for applicability checks."""
        if not isinstance(projector_output_descriptor, ProjectorOutputDescriptor):
            raise CalibrationError(
                'projector_output_descriptor must be ProjectorOutputDescriptor',
            )
        with self._lock:
            self._projector_output_descriptor = projector_output_descriptor
            self._projector_resolution = projector_output_descriptor.projector_resolution
            for camera_id, calibration in self._calibrations.items():
                if calibration.projector_output_descriptor != projector_output_descriptor:
                    self._statuses[camera_id] = CalibrationStatus.STALE

    def get_status_error_code(
        self,
        camera_id: str,
        camera_resolution: Resolution | None = None,
        projector_resolution: Resolution | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> str:
        """Return the stable failure code for a calibration applicability check."""
        _validate_camera_id(camera_id)
        _validate_optional_projector_descriptor(projector_output_descriptor)
        with self._lock:
            record = self._calibrations.get(camera_id)
            if record is None:
                return 'CALIBRATION_UNCALIBRATED'
            effective_projector_resolution = (
                projector_resolution
                if projector_resolution is not None
                else self._projector_resolution
            )
            applicability_error_code = _applicability_error_code(
                record,
                self._calibration_version,
                camera_resolution,
                effective_projector_resolution,
                projector_output_descriptor
                if projector_output_descriptor is not None
                else self._projector_output_descriptor,
            )
            if applicability_error_code is not None:
                return applicability_error_code
            return {
                CalibrationStatus.UNVERIFIED: 'CALIBRATION_UNVERIFIED',
                CalibrationStatus.STALE: 'CALIBRATION_STALE',
                CalibrationStatus.CALIBRATED: 'CALIBRATED',
            }.get(self._statuses[camera_id], 'CALIBRATION_INVALID')

    def get_status(
        self,
        camera_id: str,
        camera_resolution: Resolution | None = None,
        projector_resolution: Resolution | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> CalibrationStatus:
        _validate_camera_id(camera_id)
        _validate_optional_projector_descriptor(projector_output_descriptor)
        with self._lock:
            record = self._calibrations.get(camera_id)
            if record is None:
                return CalibrationStatus.UNCALIBRATED
            effective_projector_resolution = (
                projector_resolution
                if projector_resolution is not None
                else self._projector_resolution
            )
            if _applicability_error_code(
                record,
                self._calibration_version,
                camera_resolution,
                effective_projector_resolution,
                projector_output_descriptor
                if projector_output_descriptor is not None
                else self._projector_output_descriptor,
            ) is not None:
                self._statuses[camera_id] = CalibrationStatus.STALE
            return self._statuses[camera_id]

    def verify(
        self,
        camera_id: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence],
        camera_resolution: Resolution | None = None,
        projector_resolution: Resolution | None = None,
        thresholds: CalibrationThresholds | None = None,
        pattern: CalibrationPattern | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> CalibrationStatus:
        _validate_camera_id(camera_id)
        _validate_optional_projector_descriptor(projector_output_descriptor)
        checked_thresholds = (
            thresholds
            if thresholds is not None
            else CalibrationThresholds()
        )
        if not isinstance(checked_thresholds, CalibrationThresholds):
            raise CalibrationError('thresholds must be CalibrationThresholds')
        with self._lock:
            record = self._calibrations.get(camera_id)
            if record is None:
                return CalibrationStatus.UNCALIBRATED
            effective_projector_resolution = (
                projector_resolution
                if projector_resolution is not None
                else self._projector_resolution
            )
            if _applicability_error_code(
                record,
                self._calibration_version,
                camera_resolution,
                effective_projector_resolution,
                projector_output_descriptor
                if projector_output_descriptor is not None
                else self._projector_output_descriptor,
            ) is not None:
                self._statuses[camera_id] = CalibrationStatus.STALE
                return self._statuses[camera_id]
            is_verified = _passes_verification(
                record,
                correspondences,
                checked_thresholds,
                pattern,
            )
            self._statuses[camera_id] = (
                CalibrationStatus.CALIBRATED
                if is_verified
                else CalibrationStatus.STALE
            )
            return self._statuses[camera_id]

    def project_camera_to_projector(
        self,
        camera_id: str,
        point: Sequence[Real],
        camera_resolution: Resolution | None = None,
        projector_resolution: Resolution | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> Point2D:
        _validate_camera_id(camera_id)
        _validate_optional_projector_descriptor(projector_output_descriptor)
        with self._lock:
            status = self.get_status(
                camera_id,
                camera_resolution,
                projector_resolution,
                projector_output_descriptor,
            )
            if status is not CalibrationStatus.CALIBRATED:
                error = InvalidCalibrationStateError(
                    f'Camera {camera_id!r} calibration is {status.value}',
                )
                error.code = self.get_status_error_code(
                    camera_id,
                    camera_resolution,
                    projector_resolution,
                    projector_output_descriptor,
                )
                raise error
            record = self._calibrations[camera_id]
            effective_projector_resolution = (
                projector_resolution
                if projector_resolution is not None
                else self._projector_resolution
                if self._projector_resolution is not None
                else record.projector_resolution
            )
            camera_bounds = CoordinateBounds(
                0.0,
                0.0,
                float(record.camera_resolution.width),
                float(record.camera_resolution.height),
            )
            return project_camera_to_projector(
                point,
                record.camera_to_projector,
                calibrated_region=camera_bounds,
                projector_resolution=effective_projector_resolution,
            )


def _passes_verification(
    record: PersistedCalibration,
    correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence],
    thresholds: CalibrationThresholds,
    pattern: CalibrationPattern | None = None,
) -> bool:
    if (
        isinstance(correspondences, CameraCorrespondences)
        and correspondences.camera_id is not None
        and correspondences.camera_id != record.camera_id
    ):
        return False
    values = (
        correspondences.correspondences
        if isinstance(correspondences, CameraCorrespondences)
        else correspondences
    )
    try:
        checked_values = tuple(values)
    except Exception as ex:  # noqa: BLE001 (verification input is an external boundary).
        return False
    if len(checked_values) < 4:
        return False

    seen_corners: set[tuple[int, int]] = set()
    for correspondence in checked_values:
        if not isinstance(correspondence, FiducialCorrespondence):
            return False
        if (
            not isinstance(correspondence.marker_id, int)
            or isinstance(correspondence.marker_id, bool)
            or correspondence.marker_id < 0
            or not isinstance(correspondence.corner_index, int)
            or isinstance(correspondence.corner_index, bool)
            or not 0 <= correspondence.corner_index < 4
            or not is_finite_point(correspondence.projector_position)
            or not is_finite_point(correspondence.camera_position)
            or not is_point_in_resolution(
                correspondence.projector_position,
                record.projector_resolution,
            )
            or not is_point_in_resolution(
                correspondence.camera_position,
                record.camera_resolution,
            )
        ):
            return False
        corner_key = (correspondence.marker_id, correspondence.corner_index)
        if corner_key in seen_corners:
            return False
        seen_corners.add(corner_key)

    marker_ids = {value.marker_id for value in checked_values}
    if len(marker_ids) < thresholds.min_unique_tags:
        return False
    if pattern is not None:
        if not isinstance(pattern, CalibrationPattern):
            return False
        try:
            validate_correspondences_against_pattern(checked_values, pattern)
        except (CalibrationError, ValueError):
            return False

    errors: list[float] = []
    for correspondence in checked_values:
        try:
            predicted_camera = project_point(
                correspondence.projector_position,
                record.projector_to_camera,
            )
        except GeometryError:
            return False
        error = math.hypot(
            predicted_camera.x - correspondence.camera_position.x,
            predicted_camera.y - correspondence.camera_position.y,
        )
        if not math.isfinite(error):
            return False
        errors.append(error)
    return (
        len(errors) > 0
        and sum(errors) / len(errors) <= thresholds.max_mean_reprojection_error
        and max(errors) <= thresholds.max_reprojection_error
    )


def _are_inverse_matrices(
    projector_to_camera: tuple[tuple[float, float, float], ...],
    camera_to_projector: tuple[tuple[float, float, float], ...],
) -> bool:
    try:
        expected_inverse = validate_homography(invert_homography(projector_to_camera))
    except (InvalidHomographyError, TypeError, ValueError):
        return False
    return all(
        abs(expected_inverse[row_idx][column_idx] - camera_to_projector[row_idx][column_idx])
        <= 1e-6
        for row_idx in range(3)
        for column_idx in range(3)
    )


def _validate_optional_projector_descriptor(
    projector_output_descriptor: ProjectorOutputDescriptor | None,
) -> None:
    if projector_output_descriptor is not None and not isinstance(
        projector_output_descriptor,
        ProjectorOutputDescriptor,
    ):
        raise CalibrationError(
            'projector_output_descriptor must be ProjectorOutputDescriptor',
        )


def _applicability_error_code(
    record: PersistedCalibration,
    calibration_version: int,
    camera_resolution: Resolution | None,
    projector_resolution: Resolution | None,
    projector_output_descriptor: ProjectorOutputDescriptor | None = None,
) -> str | None:
    if record.version != calibration_version:
        return 'CALIBRATION_STALE'
    if camera_resolution is not None and camera_resolution != record.camera_resolution:
        return 'CAMERA_RESOLUTION_CHANGED'
    if projector_resolution is not None and projector_resolution != record.projector_resolution:
        return 'PROJECTOR_RESOLUTION_CHANGED'
    if (
        projector_output_descriptor is not None
        and projector_output_descriptor != record.projector_output_descriptor
    ):
        if (
            projector_output_descriptor.projector_resolution
            != record.projector_output_descriptor.projector_resolution
        ):
            return 'PROJECTOR_RESOLUTION_CHANGED'
        return 'PROJECTOR_OUTPUT_CHANGED'
    return None


def _normalise_projector_output_descriptor(
    projector_resolution: Resolution,
    projector_output_descriptor: ProjectorOutputDescriptor | None,
    projector_output_identity: str | None,
) -> ProjectorOutputDescriptor:
    if projector_output_descriptor is not None and not isinstance(
        projector_output_descriptor,
        ProjectorOutputDescriptor,
    ):
        raise CalibrationError(
            'projector_output_descriptor must be ProjectorOutputDescriptor',
        )
    if (
        projector_output_descriptor is not None
        and projector_output_identity is not None
        and projector_output_descriptor.output_identity != projector_output_identity
    ):
        raise CalibrationError('Projector output descriptor values disagree')
    if projector_output_descriptor is not None:
        if projector_output_descriptor.projector_resolution != projector_resolution:
            raise CalibrationError(
                'Projector output descriptor resolution does not match calibration',
            )
        return projector_output_descriptor
    try:
        return ProjectorOutputDescriptor(
            projector_resolution,
            'default' if projector_output_identity is None else projector_output_identity,
        )
    except ConfigurationError as ex:
        raise CalibrationError('Calibration projector descriptor is invalid') from ex


def _parse_projector_output_descriptor_data(
    data: Any,
    projector_resolution: Resolution,
    output_identity: Any,
) -> ProjectorOutputDescriptor:
    if data is None:
        if output_identity is not None and not isinstance(output_identity, str):
            raise CalibrationError('projector_output_identity must be a string')
        try:
            return _normalise_projector_output_descriptor(
                projector_resolution,
                None,
                output_identity,
            )
        except (ConfigurationError, TypeError, ValueError) as ex:
            raise CalibrationError('Calibration projector descriptor is invalid') from ex
    if not isinstance(data, Mapping):
        raise CalibrationError('Calibration projector descriptor must be an object')
    descriptor_resolution_data = data.get(
        'projector_resolution',
        data.get('resolution'),
    )
    descriptor_resolution = (
        projector_resolution
        if descriptor_resolution_data is None
        else _parse_resolution(
            descriptor_resolution_data,
            'projector_output_descriptor.projector_resolution',
        )
    )
    descriptor_identity = data.get(
        'output_identity',
        data.get('identity', 'default'),
    )
    try:
        return _normalise_projector_output_descriptor(
            projector_resolution,
            ProjectorOutputDescriptor(descriptor_resolution, descriptor_identity),
            output_identity,
        )
    except (ConfigurationError, TypeError, ValueError) as ex:
        raise CalibrationError('Calibration projector descriptor is invalid') from ex


def _normalise_region(region: RegionLike) -> tuple[Point2D, ...]:
    try:
        raw_points = tuple(region)
    except (TypeError, ValueError):
        raise CalibrationError('Calibration valid_region must be a point sequence') from None
    points: list[Point2D] = []
    for raw_point in raw_points:
        if isinstance(raw_point, (str, bytes, bytearray, Mapping)):
            raise CalibrationError('Calibration valid_region must contain point pairs')
        try:
            coordinates = tuple(raw_point)
        except (TypeError, ValueError):
            raise CalibrationError('Calibration valid_region must contain point pairs') from None
        if len(coordinates) != 2:
            raise CalibrationError('Calibration valid_region must contain point pairs')
        if not all(is_finite_real(coordinate) for coordinate in coordinates):
            raise CalibrationError('Calibration valid_region must contain point pairs')
        points.append(Point2D(float(coordinates[0]), float(coordinates[1])))
    if (
        len(points) < 3
        or len(set(points)) < 3
        or any(not is_finite_point(point) for point in points)
    ):
        raise CalibrationError('Calibration valid_region must be a finite polygon')
    try:
        validate_point_in_region(points[0], points)
    except GeometryError:
        raise CalibrationError('Calibration valid_region must have non-zero area') from None
    return tuple(points)


def _parse_region(data: Any) -> tuple[Point2D, ...]:
    if isinstance(data, (str, bytes, bytearray, Mapping)):
        raise CalibrationError('Calibration valid_region must be a point sequence')
    return _normalise_region(data)


def _validate_camera_id(camera_id: object) -> None:
    if not isinstance(camera_id, str) or len(camera_id) == 0:
        raise CalibrationError('camera_id must be a non-empty stable ID')


def _validate_resolution(resolution: object, field_name: str) -> None:
    if not is_valid_resolution(resolution):
        raise CalibrationError(f'{field_name} must be a positive integer resolution')


def _validate_version(version: object) -> None:
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise CalibrationError('Calibration version must be a positive integer')


def _validate_metrics(metrics: object) -> None:
    if not isinstance(metrics, CalibrationMetrics):
        raise CalibrationError('Calibration metrics must be CalibrationMetrics')
    if (
        not isinstance(metrics.unique_tag_count, int)
        or not isinstance(metrics.correspondence_corner_count, int)
        or not isinstance(metrics.ransac_inlier_count, int)
        or isinstance(metrics.unique_tag_count, bool)
        or isinstance(metrics.correspondence_corner_count, bool)
        or isinstance(metrics.ransac_inlier_count, bool)
        or min(
            metrics.unique_tag_count,
            metrics.correspondence_corner_count,
            metrics.ransac_inlier_count,
        ) < 0
        or not all(
            is_finite_real(value)
            for value in (
                metrics.inlier_ratio,
                metrics.mean_reprojection_error,
                metrics.max_reprojection_error,
                metrics.spatial_coverage,
            )
        )
        or metrics.unique_tag_count == 0
        or metrics.correspondence_corner_count < 4
        or metrics.ransac_inlier_count < 4
        or metrics.ransac_inlier_count > metrics.correspondence_corner_count
        or metrics.unique_tag_count * 4 > metrics.correspondence_corner_count
        or not math.isclose(
            metrics.inlier_ratio,
            metrics.ransac_inlier_count / metrics.correspondence_corner_count,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or metrics.inlier_ratio < 0
        or metrics.inlier_ratio > 1
        or metrics.mean_reprojection_error < 0
        or metrics.max_reprojection_error < 0
        or metrics.mean_reprojection_error > metrics.max_reprojection_error
        or metrics.spatial_coverage <= 0
        or metrics.spatial_coverage > 1
    ):
        raise CalibrationError('Calibration metrics are invalid')


def _parse_resolution(data: Any, field_name: str) -> Resolution:
    if not isinstance(data, Mapping):
        raise CalibrationError(f'{field_name} must be an object')
    resolution = Resolution(data.get('width'), data.get('height'))
    _validate_resolution(resolution, field_name)
    return resolution


def _resolution_to_data(resolution: Resolution) -> dict[str, int]:
    return {'width': resolution.width, 'height': resolution.height}


def _parse_metrics(data: Any) -> CalibrationMetrics:
    if not isinstance(data, Mapping):
        raise CalibrationError('Calibration metrics must be an object')
    try:
        return CalibrationMetrics(
            data['unique_tag_count'],
            data['correspondence_corner_count'],
            data['ransac_inlier_count'],
            data['inlier_ratio'],
            data['mean_reprojection_error'],
            data['max_reprojection_error'],
            data['spatial_coverage'],
        )
    except (KeyError, TypeError, ValueError):
        raise CalibrationError('Calibration metrics are malformed') from None


def _metrics_to_data(metrics: CalibrationMetrics) -> dict[str, int | float]:
    return {
        'unique_tag_count': metrics.unique_tag_count,
        'correspondence_corner_count': metrics.correspondence_corner_count,
        'ransac_inlier_count': metrics.ransac_inlier_count,
        'inlier_ratio': metrics.inlier_ratio,
        'mean_reprojection_error': metrics.mean_reprojection_error,
        'max_reprojection_error': metrics.max_reprojection_error,
        'spatial_coverage': metrics.spatial_coverage,
    }


def _point_to_data(point: Point2D) -> list[float]:
    return [point.x, point.y]


__all__ = [
    'CalibrationRegistry',
    'CalibrationStore',
    'InvalidCalibrationStateError',
    'PersistedCalibration',
]
