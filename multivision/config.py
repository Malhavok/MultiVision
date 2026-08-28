"""Central, JSON-backed MultiVision configuration."""

import json
import math
import pathlib
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from multivision.errors import ConfigurationError
from multivision.types import (
    Resolution,
    is_valid_resolution,
)


if sys.platform == 'darwin':
    DEFAULT_CONFIG_PATH = (
        pathlib.Path.home()
        / 'Library'
        / 'Application Support'
        / 'MultiVision'
        / 'config.json'
    )
else:
    DEFAULT_CONFIG_PATH = pathlib.Path.home() / '.config' / 'multivision' / 'config.json'

_CONFIG_FILE_LOCKS: dict[pathlib.Path, threading.RLock] = {}
_CONFIG_FILE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class CalibrationThresholds:
    max_mean_reprojection_error: float = 5.0
    max_reprojection_error: float = 10.0
    min_inlier_ratio: float = 0.5
    min_unique_tags: int = 4
    min_spatial_coverage: float = 0.1
    valid_region_margin: float = 0.05

    def __post_init__(self) -> None:
        _validate_thresholds(self)


@dataclass(frozen=True)
class Configuration:
    camera_bindings: dict[str, str] = field(default_factory=dict)
    projector_resolution: Resolution = field(
        default_factory=lambda: Resolution(1920, 1080),
    )
    calibration_thresholds: CalibrationThresholds = field(
        default_factory=CalibrationThresholds,
    )
    calibration_version: int = 1

    def __post_init__(self) -> None:
        validate_camera_bindings(self.camera_bindings)
        _validate_resolution(self.projector_resolution, 'projector_resolution')
        if not isinstance(self.calibration_thresholds, CalibrationThresholds):
            raise ConfigurationError('calibration_thresholds must be CalibrationThresholds')
        _validate_positive_integer(self.calibration_version, 'calibration_version')

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> 'Configuration':
        if not isinstance(data, Mapping):
            raise ConfigurationError('The configuration root must be an object')

        camera_bindings_data = data.get('camera_bindings', {})
        if not isinstance(camera_bindings_data, Mapping):
            raise ConfigurationError('camera_bindings must be an object')

        camera_bindings = dict(camera_bindings_data)
        projector_resolution = _parse_resolution(data.get('projector_resolution', {}))
        thresholds = _parse_thresholds(data.get('calibration_thresholds', {}))
        calibration_version = data.get('calibration_version', 1)

        return cls(
            camera_bindings=camera_bindings,
            projector_resolution=projector_resolution,
            calibration_thresholds=thresholds,
            calibration_version=calibration_version,
        )

    def to_data(self) -> dict[str, Any]:
        validate_camera_bindings(self.camera_bindings)
        return {
            'camera_bindings': {
                logical_name: self.camera_bindings[logical_name]
                for logical_name in sorted(self.camera_bindings)
            },
            'projector_resolution': {
                'width': self.projector_resolution.width,
                'height': self.projector_resolution.height,
            },
            'calibration_thresholds': {
                'max_mean_reprojection_error': (
                    self.calibration_thresholds.max_mean_reprojection_error
                ),
                'max_reprojection_error': self.calibration_thresholds.max_reprojection_error,
                'min_inlier_ratio': self.calibration_thresholds.min_inlier_ratio,
                'min_unique_tags': self.calibration_thresholds.min_unique_tags,
                'min_spatial_coverage': self.calibration_thresholds.min_spatial_coverage,
                'valid_region_margin': self.calibration_thresholds.valid_region_margin,
            },
            'calibration_version': self.calibration_version,
        }


def load_configuration(config_path: pathlib.Path | None = None) -> Configuration:
    """Load the one MultiVision configuration file, using defaults if absent."""
    config_path = _resolve_config_path(config_path)
    try:
        raw_configuration = config_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return Configuration()
    except (OSError, UnicodeDecodeError) as ex:
        raise ConfigurationError(f'Could not read configuration at {config_path}') from ex

    try:
        data = json.loads(raw_configuration)
    except json.JSONDecodeError as ex:
        raise ConfigurationError(f'Could not read configuration at {config_path}') from ex

    return Configuration.from_data(data)


def save_configuration(
    configuration: Configuration,
    config_path: pathlib.Path | None = None,
) -> None:
    """Save the central configuration as readable JSON."""
    if not isinstance(configuration, Configuration):
        raise ConfigurationError('configuration must be a Configuration')
    config_path = _resolve_config_path(config_path)
    temporary_path: pathlib.Path | None = None
    with _get_configuration_file_lock(config_path):
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            data = _load_existing_document_for_update(config_path)
            data.update(configuration.to_data())
            serialised_configuration = json.dumps(
                data,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + '\n'
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=config_path.parent,
                prefix=f'.{config_path.name}.',
                delete=False,
            ) as temporary_file:
                temporary_path = pathlib.Path(temporary_file.name)
                temporary_file.write(serialised_configuration)
            temporary_path.replace(config_path)
        except (OSError, TypeError, ValueError) as ex:
            raise ConfigurationError(f'Could not write configuration at {config_path}') from ex
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


def _get_configuration_file_lock(config_path: pathlib.Path) -> threading.RLock:
    lock_path = config_path.resolve()
    with _CONFIG_FILE_LOCKS_GUARD:
        lock = _CONFIG_FILE_LOCKS.get(lock_path)
        if lock is None:
            lock = threading.RLock()
            _CONFIG_FILE_LOCKS[lock_path] = lock
        return lock


def _resolve_config_path(config_path: pathlib.Path | None) -> pathlib.Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    if not isinstance(config_path, pathlib.Path):
        raise ConfigurationError('config_path must be a pathlib.Path')
    return config_path


def _parse_resolution(data: Any) -> Resolution:
    if not isinstance(data, Mapping):
        raise ConfigurationError('projector_resolution must be an object')

    width = data.get('width', 1920)
    height = data.get('height', 1080)
    _validate_positive_integer(width, 'projector_resolution.width')
    _validate_positive_integer(height, 'projector_resolution.height')
    return Resolution(width, height)


def _parse_thresholds(data: Any) -> CalibrationThresholds:
    if not isinstance(data, Mapping):
        raise ConfigurationError('calibration_thresholds must be an object')

    defaults = CalibrationThresholds()
    values = {
        'max_mean_reprojection_error': data.get(
            'max_mean_reprojection_error',
            defaults.max_mean_reprojection_error,
        ),
        'max_reprojection_error': data.get(
            'max_reprojection_error',
            defaults.max_reprojection_error,
        ),
        'min_inlier_ratio': data.get('min_inlier_ratio', defaults.min_inlier_ratio),
        'min_unique_tags': data.get('min_unique_tags', defaults.min_unique_tags),
        'min_spatial_coverage': data.get(
            'min_spatial_coverage',
            defaults.min_spatial_coverage,
        ),
        'valid_region_margin': data.get(
            'valid_region_margin',
            defaults.valid_region_margin,
        ),
    }

    return CalibrationThresholds(**values)


def _load_existing_document_for_update(config_path: pathlib.Path) -> dict[str, Any]:
    try:
        raw_data = config_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as ex:
        raise ConfigurationError(f'Could not read configuration at {config_path}') from ex
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as ex:
        raise ConfigurationError(f'Could not read configuration at {config_path}') from ex
    if not isinstance(data, dict):
        raise ConfigurationError(f'Configuration at {config_path} must be an object')
    return data


def validate_camera_bindings(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ConfigurationError('camera_bindings must be an object')

    for logical_name, device_id in value.items():
        if not isinstance(logical_name, str) or len(logical_name) == 0:
            raise ConfigurationError('camera binding names must be non-empty strings')
        if not isinstance(device_id, str) or len(device_id) == 0:
            raise ConfigurationError(
                f'camera binding for {logical_name!r} must contain a non-empty device ID',
            )


def _validate_resolution(value: Any, field_name: str) -> None:
    if not is_valid_resolution(value):
        raise ConfigurationError(f'{field_name} must be a positive Resolution')


def _validate_thresholds(value: Any) -> None:
    for field_name in (
        'max_mean_reprojection_error',
        'max_reprojection_error',
        'min_inlier_ratio',
        'min_spatial_coverage',
        'valid_region_margin',
    ):
        field_value = getattr(value, field_name)
        if not isinstance(field_value, (int, float)) or isinstance(field_value, bool):
            raise ConfigurationError(f'{field_name} must be a number')
        if not math.isfinite(field_value):
            raise ConfigurationError(f'{field_name} must be finite')
        if field_value < 0:
            raise ConfigurationError(f'{field_name} must not be negative')

    _validate_positive_integer(value.min_unique_tags, 'min_unique_tags')
    if value.min_inlier_ratio > 1 or value.min_spatial_coverage > 1:
        raise ConfigurationError('ratio and coverage thresholds must be between 0 and 1')
    if value.valid_region_margin > 1:
        raise ConfigurationError('valid_region_margin must be between 0 and 1')


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f'{field_name} must be a positive integer')
