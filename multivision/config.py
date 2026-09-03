"""Central, JSON-backed MultiVision configuration."""

import json
import math
import pathlib
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from multivision.errors import ConfigurationError
from multivision.overlays import OverlayConfiguration
from multivision.pattern import (
    DEFAULT_TAG_DICTIONARY,
    validate_tag_dictionary,
)
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
    max_capture_white_balance_delta: float = 0.01
    max_capture_p95_sigma_pixels: float = 5.0
    min_inlier_ratio: float = 0.5
    min_unique_tags: int = 2
    min_spatial_coverage: float = 0.1
    valid_region_margin: float = 0.05

    def __post_init__(self) -> None:
        _validate_thresholds(self)


@dataclass(frozen=True)
class MetricCalibrationThresholds:
    ransac_reprojection_threshold_mm: float = 3.0
    max_capture_white_balance_delta: float = 0.01
    max_capture_corner_jitter_pixels: float = 2.0
    max_mean_fit_error_mm: float = 2.0
    max_fit_error_mm: float = 5.0
    min_inlier_ratio: float = 0.5
    min_unique_target_fiducials: int = 4
    min_capture_marker_ratio: float = 0.8
    min_spatial_coverage: float = 0.5

    def __post_init__(self) -> None:
        _validate_metric_thresholds(self)


@dataclass(frozen=True)
class ProjectorOutputDescriptor:
    projector_resolution: Resolution
    output_identity: str = 'default'

    def __post_init__(self) -> None:
        _validate_resolution(
            self.projector_resolution,
            'projector_output.projector_resolution',
        )
        _validate_output_identity(self.output_identity)


@dataclass(frozen=True)
class FiducialGroup:
    dictionary: str
    marker_size_mm: float

    def __post_init__(self) -> None:
        _validate_tag_dictionary(self.dictionary)
        _validate_finite_positive_number(self.marker_size_mm, 'marker_size_mm')

    @classmethod
    def from_data(
        cls: type['FiducialGroup'],
        data: Mapping[str, Any],
    ) -> 'FiducialGroup':
        if not isinstance(data, Mapping):
            raise ConfigurationError('fiducial group must be an object')
        if set(data) != {'dictionary', 'marker_size_mm'}:
            raise ConfigurationError(
                'fiducial group must contain only dictionary and marker_size_mm',
            )
        try:
            return cls(data['dictionary'], data['marker_size_mm'])
        except KeyError as ex:
            raise ConfigurationError(
                f'fiducial group is missing {ex.args[0]}',
            ) from ex
        except (TypeError, ValueError) as ex:
            raise ConfigurationError(str(ex)) from ex


@dataclass(frozen=True)
class Configuration:
    projector_resolution: Resolution = field(
        default_factory=lambda: Resolution(1920, 1080),
    )
    calibration_thresholds: CalibrationThresholds = field(
        default_factory=CalibrationThresholds,
    )
    calibration_version: int = 1
    projector_output_identity: str = 'default'
    metric_calibration_thresholds: MetricCalibrationThresholds = field(
        default_factory=MetricCalibrationThresholds,
    )
    overlay_limits: OverlayConfiguration = field(default_factory=OverlayConfiguration)
    tag_dictionary: str = DEFAULT_TAG_DICTIONARY
    fiducial_groups: Mapping[str, FiducialGroup] = field(default_factory=dict)
    fiducial_history_length: int = 8
    fiducial_tracking_rate_hz: float = 30.0
    fiducial_grace_period_seconds: float = 5.0
    fiducial_protection_margin_mm: float = 5.0
    max_batch_operations: int = 100
    preview_mode: str = 'active'
    preview_low_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        _validate_resolution(self.projector_resolution, 'projector_resolution')
        _validate_output_identity(self.projector_output_identity)
        if not isinstance(self.calibration_thresholds, CalibrationThresholds):
            raise ConfigurationError('calibration_thresholds must be CalibrationThresholds')
        if not isinstance(
            self.metric_calibration_thresholds,
            MetricCalibrationThresholds,
        ):
            raise ConfigurationError(
                'metric_calibration_thresholds must be MetricCalibrationThresholds',
            )
        if not isinstance(self.overlay_limits, OverlayConfiguration):
            raise ConfigurationError('overlay_limits must be OverlayConfiguration')
        _validate_positive_integer(self.calibration_version, 'calibration_version')
        _validate_tag_dictionary(self.tag_dictionary)
        object.__setattr__(
            self,
            'fiducial_groups',
            _normalise_fiducial_groups(self.fiducial_groups),
        )
        _validate_bounded_positive_integer(
            self.fiducial_history_length,
            'fiducial_history_length',
            32,
        )
        _validate_bounded_number(
            self.fiducial_tracking_rate_hz,
            'fiducial_tracking_rate_hz',
            1.0,
            60.0,
        )
        _validate_bounded_number(
            self.fiducial_grace_period_seconds,
            'fiducial_grace_period_seconds',
            0.1,
            60.0,
        )
        _validate_bounded_number(
            self.fiducial_protection_margin_mm,
            'fiducial_protection_margin_mm',
            0.1,
            1000.0,
        )
        _validate_bounded_positive_integer(
            self.max_batch_operations,
            'max_batch_operations',
            1000,
        )
        _validate_preview_mode(self.preview_mode)
        _validate_bounded_number(
            self.preview_low_rate_hz,
            'preview_low_rate_hz',
            1.0,
            15.0,
        )

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor:
        return ProjectorOutputDescriptor(
            self.projector_resolution,
            self.projector_output_identity,
        )

    @classmethod
    def from_data(
        cls: type['Configuration'],
        data: Mapping[str, Any],
    ) -> 'Configuration':
        if not isinstance(data, Mapping):
            raise ConfigurationError('The configuration root must be an object')

        projector_resolution = _parse_resolution(data.get('projector_resolution', {}))
        projector_output_identity = data.get('projector_output_identity', 'default')
        thresholds = _parse_thresholds(data.get('calibration_thresholds', {}))
        metric_thresholds = _parse_metric_thresholds(
            data.get('metric_calibration_thresholds', {}),
        )
        overlay_limits = _parse_overlay_configuration(
            data.get('overlay_limits', {}),
        )
        calibration_version = data.get('calibration_version', 1)
        tag_dictionary = data.get('tag_dictionary', DEFAULT_TAG_DICTIONARY)
        fiducial_groups = _parse_fiducial_groups(data.get('fiducial_groups', {}))

        return cls(
            projector_resolution=projector_resolution,
            projector_output_identity=projector_output_identity,
            calibration_thresholds=thresholds,
            metric_calibration_thresholds=metric_thresholds,
            overlay_limits=overlay_limits,
            calibration_version=calibration_version,
            tag_dictionary=tag_dictionary,
            fiducial_groups=fiducial_groups,
            fiducial_history_length=data.get('fiducial_history_length', 8),
            fiducial_tracking_rate_hz=data.get('fiducial_tracking_rate_hz', 30.0),
            fiducial_grace_period_seconds=data.get('fiducial_grace_period_seconds', 5.0),
            fiducial_protection_margin_mm=data.get(
                'fiducial_protection_margin_mm',
                5.0,
            ),
            max_batch_operations=data.get('max_batch_operations', 100),
            preview_mode=data.get('preview_mode', 'active'),
            preview_low_rate_hz=data.get('preview_low_rate_hz', 10.0),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            'projector_resolution': {
                'width': self.projector_resolution.width,
                'height': self.projector_resolution.height,
            },
            'projector_output_identity': self.projector_output_identity,
            'calibration_thresholds': {
                'max_mean_reprojection_error': (
                    self.calibration_thresholds.max_mean_reprojection_error
                ),
                'max_reprojection_error': self.calibration_thresholds.max_reprojection_error,
                'max_capture_white_balance_delta': (
                    self.calibration_thresholds.max_capture_white_balance_delta
                ),
                'max_capture_p95_sigma_pixels': (
                    self.calibration_thresholds.max_capture_p95_sigma_pixels
                ),
                'min_inlier_ratio': self.calibration_thresholds.min_inlier_ratio,
                'min_unique_tags': self.calibration_thresholds.min_unique_tags,
                'min_spatial_coverage': self.calibration_thresholds.min_spatial_coverage,
                'valid_region_margin': self.calibration_thresholds.valid_region_margin,
            },
            'metric_calibration_thresholds': {
                'ransac_reprojection_threshold_mm': (
                    self.metric_calibration_thresholds.ransac_reprojection_threshold_mm
                ),
                'max_capture_white_balance_delta': (
                    self.metric_calibration_thresholds.max_capture_white_balance_delta
                ),
                'max_capture_corner_jitter_pixels': (
                    self.metric_calibration_thresholds.max_capture_corner_jitter_pixels
                ),
                'max_mean_fit_error_mm': self.metric_calibration_thresholds.max_mean_fit_error_mm,
                'max_fit_error_mm': self.metric_calibration_thresholds.max_fit_error_mm,
                'min_inlier_ratio': self.metric_calibration_thresholds.min_inlier_ratio,
                'min_unique_target_fiducials': (
                    self.metric_calibration_thresholds.min_unique_target_fiducials
                ),
                'min_capture_marker_ratio': (
                    self.metric_calibration_thresholds.min_capture_marker_ratio
                ),
                'min_spatial_coverage': self.metric_calibration_thresholds.min_spatial_coverage,
            },
            'overlay_limits': self.overlay_limits.to_data(),
            'calibration_version': self.calibration_version,
            'tag_dictionary': self.tag_dictionary,
            'fiducial_groups': {
                group_name: {
                    'dictionary': group.dictionary,
                    'marker_size_mm': group.marker_size_mm,
                }
                for group_name, group in self.fiducial_groups.items()
            },
            'fiducial_history_length': self.fiducial_history_length,
            'fiducial_tracking_rate_hz': self.fiducial_tracking_rate_hz,
            'fiducial_grace_period_seconds': self.fiducial_grace_period_seconds,
            'fiducial_protection_margin_mm': self.fiducial_protection_margin_mm,
            'max_batch_operations': self.max_batch_operations,
            'preview_mode': self.preview_mode,
            'preview_low_rate_hz': self.preview_low_rate_hz,
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


def _parse_metric_thresholds(data: Any) -> MetricCalibrationThresholds:
    if not isinstance(data, Mapping):
        raise ConfigurationError('metric_calibration_thresholds must be an object')

    defaults = MetricCalibrationThresholds()
    values = {
        'ransac_reprojection_threshold_mm': data.get(
            'ransac_reprojection_threshold_mm',
            defaults.ransac_reprojection_threshold_mm,
        ),
        'max_capture_white_balance_delta': data.get(
            'max_capture_white_balance_delta',
            defaults.max_capture_white_balance_delta,
        ),
        'max_capture_corner_jitter_pixels': data.get(
            'max_capture_corner_jitter_pixels',
            defaults.max_capture_corner_jitter_pixels,
        ),
        'max_mean_fit_error_mm': data.get(
            'max_mean_fit_error_mm',
            defaults.max_mean_fit_error_mm,
        ),
        'max_fit_error_mm': data.get(
            'max_fit_error_mm',
            defaults.max_fit_error_mm,
        ),
        'min_inlier_ratio': data.get('min_inlier_ratio', defaults.min_inlier_ratio),
        'min_unique_target_fiducials': data.get(
            'min_unique_target_fiducials',
            defaults.min_unique_target_fiducials,
        ),
        'min_capture_marker_ratio': data.get(
            'min_capture_marker_ratio',
            defaults.min_capture_marker_ratio,
        ),
        'min_spatial_coverage': data.get(
            'min_spatial_coverage',
            defaults.min_spatial_coverage,
        ),
    }
    return MetricCalibrationThresholds(**values)


def _parse_fiducial_groups(data: Any) -> dict[str, FiducialGroup]:
    if not isinstance(data, Mapping):
        raise ConfigurationError('fiducial_groups must be an object')

    groups: dict[str, FiducialGroup] = {}
    for group_name, group_data in data.items():
        _validate_group_name(group_name)
        groups[group_name] = FiducialGroup.from_data(group_data)
    return groups


def _normalise_fiducial_groups(
    groups: Mapping[str, FiducialGroup],
) -> Mapping[str, FiducialGroup]:
    if not isinstance(groups, Mapping):
        raise ConfigurationError('fiducial_groups must be an object')

    normalised: dict[str, FiducialGroup] = {}
    for group_name, group in groups.items():
        _validate_group_name(group_name)
        if isinstance(group, FiducialGroup):
            normalised[group_name] = group
            continue
        if isinstance(group, Mapping):
            normalised[group_name] = FiducialGroup.from_data(group)
            continue
        raise ConfigurationError(
            f'fiducial_groups[{group_name!r}] must be a FiducialGroup',
        )
    return MappingProxyType(normalised)


def _parse_overlay_configuration(data: Any) -> OverlayConfiguration:
    try:
        return OverlayConfiguration.from_data(data)
    except (TypeError, ValueError) as ex:
        raise ConfigurationError(str(ex)) from ex


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
        'max_capture_white_balance_delta': data.get(
            'max_capture_white_balance_delta',
            defaults.max_capture_white_balance_delta,
        ),
        'max_capture_p95_sigma_pixels': data.get(
            'max_capture_p95_sigma_pixels',
            defaults.max_capture_p95_sigma_pixels,
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


def _validate_resolution(value: Any, field_name: str) -> None:
    if not is_valid_resolution(value):
        raise ConfigurationError(f'{field_name} must be a positive Resolution')


def _validate_thresholds(value: Any) -> None:
    _validate_finite_non_negative_numbers(
        value,
        (
            'max_mean_reprojection_error',
            'max_reprojection_error',
            'max_capture_white_balance_delta',
            'max_capture_p95_sigma_pixels',
            'min_inlier_ratio',
            'min_spatial_coverage',
            'valid_region_margin',
        ),
    )

    _validate_positive_integer(value.min_unique_tags, 'min_unique_tags')
    if value.max_capture_p95_sigma_pixels <= 0:
        raise ConfigurationError('max_capture_p95_sigma_pixels must be positive')
    if (
        value.max_capture_white_balance_delta <= 0
        or value.max_capture_white_balance_delta > 1
    ):
        raise ConfigurationError(
            'max_capture_white_balance_delta must be between 0 and 1',
        )
    if value.min_inlier_ratio > 1 or value.min_spatial_coverage > 1:
        raise ConfigurationError('ratio and coverage thresholds must be between 0 and 1')
    if value.valid_region_margin > 1:
        raise ConfigurationError('valid_region_margin must be between 0 and 1')


def _validate_metric_thresholds(value: MetricCalibrationThresholds) -> None:
    _validate_finite_non_negative_numbers(
        value,
        (
            'ransac_reprojection_threshold_mm',
            'max_capture_white_balance_delta',
            'max_capture_corner_jitter_pixels',
            'max_mean_fit_error_mm',
            'max_fit_error_mm',
            'min_inlier_ratio',
            'min_capture_marker_ratio',
            'min_spatial_coverage',
        ),
    )

    if value.ransac_reprojection_threshold_mm <= 0:
        raise ConfigurationError(
            'ransac_reprojection_threshold_mm must be positive',
        )
    if value.max_capture_white_balance_delta <= 0 or value.max_capture_white_balance_delta > 1:
        raise ConfigurationError(
            'max_capture_white_balance_delta must be between 0 and 1',
        )
    if value.max_capture_corner_jitter_pixels <= 0:
        raise ConfigurationError(
            'max_capture_corner_jitter_pixels must be positive',
        )
    _validate_positive_integer(
        value.min_unique_target_fiducials,
        'min_unique_target_fiducials',
    )
    if value.min_capture_marker_ratio <= 0:
        raise ConfigurationError('min_capture_marker_ratio must be positive')
    if (
        value.min_inlier_ratio > 1
        or value.min_capture_marker_ratio > 1
        or value.min_spatial_coverage > 1
    ):
        raise ConfigurationError('ratio and coverage thresholds must be between 0 and 1')


def _validate_finite_non_negative_numbers(
    value: Any,
    field_names: tuple[str, ...],
) -> None:
    for field_name in field_names:
        field_value = getattr(value, field_name)
        if not isinstance(field_value, (int, float)) or isinstance(field_value, bool):
            raise ConfigurationError(f'{field_name} must be a number')
        try:
            is_finite = math.isfinite(field_value)
        except (OverflowError, TypeError, ValueError):
            is_finite = False
        if not is_finite:
            raise ConfigurationError(f'{field_name} must be finite')
        if field_value < 0:
            raise ConfigurationError(f'{field_name} must not be negative')


def _validate_group_name(value: Any) -> None:
    if not isinstance(value, str) or len(value.strip()) == 0:
        raise ConfigurationError('fiducial group names must be non-empty strings')


def _validate_bounded_positive_integer(value: Any, field_name: str, maximum: int) -> None:
    _validate_positive_integer(value, field_name)
    if value > maximum:
        raise ConfigurationError(f'{field_name} must not exceed {maximum}')


def _validate_bounded_number(
    value: Any,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    try:
        is_finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        is_finite = False
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not is_finite
        or value < minimum
        or value > maximum
    ):
        raise ConfigurationError(
            f'{field_name} must be finite and between {minimum} and {maximum}',
        )


def _validate_finite_positive_number(value: Any, field_name: str) -> None:
    try:
        is_finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        is_finite = False
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not is_finite
        or value <= 0
    ):
        raise ConfigurationError(f'{field_name} must be a finite positive number')


def _validate_preview_mode(value: Any) -> None:
    if not isinstance(value, str) or value not in {'active', 'low_rate', 'off'}:
        raise ConfigurationError('preview_mode must be active, low_rate or off')


def _validate_tag_dictionary(value: Any) -> None:
    try:
        validate_tag_dictionary(value)
    except ValueError as ex:
        raise ConfigurationError(str(ex)) from ex


def _validate_output_identity(value: Any) -> None:
    if not isinstance(value, str) or len(value) == 0:
        raise ConfigurationError('projector_output_identity must be a non-empty string')


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f'{field_name} must be a positive integer')
