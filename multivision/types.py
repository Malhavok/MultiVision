import copy
import math
from enum import Enum
from numbers import Real
from typing import (
    Any,
    NamedTuple,
)


class RuntimeStatus(str, Enum):
    STARTING = 'STARTING'
    AVAILABLE = 'AVAILABLE'
    UNAVAILABLE = 'UNAVAILABLE'
    ERROR = 'ERROR'
    STOPPED = 'STOPPED'


class SessionCameraState(str, Enum):
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
    UNAVAILABLE = 'UNAVAILABLE'


class CalibrationStatus(str, Enum):
    UNCALIBRATED = 'UNCALIBRATED'
    UNVERIFIED = 'UNVERIFIED'
    CALIBRATED = 'CALIBRATED'
    STALE = 'STALE'


class CalibrationScope(str, Enum):
    GLOBAL = 'global'
    LOCAL = 'local'
    LOCAL_LOW_CONFIDENCE = 'local_low_confidence'


class CalibrationStage(str, Enum):
    UNCALIBRATED = 'UNCALIBRATED'
    UNVERIFIED = 'UNVERIFIED'
    CALIBRATED = 'CALIBRATED'
    METRIC_CALIBRATED = 'METRIC_CALIBRATED'
    STALE = 'STALE'


class Resolution(NamedTuple):
    width: int
    height: int


def is_finite_real(value: object) -> bool:
    """Return whether a value is a finite, non-boolean real number."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def is_valid_resolution(value: object) -> bool:
    """Return whether a resolution has positive integer dimensions."""
    return (
        isinstance(value, Resolution)
        and isinstance(value.width, int)
        and not isinstance(value.width, bool)
        and value.width > 0
        and isinstance(value.height, int)
        and not isinstance(value.height, bool)
        and value.height > 0
    )


class DeviceInfo(NamedTuple):
    device_id: str
    name: str
    capture_index: int | None = None
    native_resolution: Resolution | None = None
    backend_name: str | None = None
    metadata: dict[str, Any] | None = None
    is_available: bool = True
    error_message: str | None = None
    is_stable_id: bool = True


def copy_device_info(device: DeviceInfo) -> DeviceInfo:
    """Copy a device record without sharing its mutable metadata."""
    return device._replace(
        metadata=None if device.metadata is None else copy.deepcopy(device.metadata),
    )


class Frame(NamedTuple):
    data: Any
    frame_counter: int
    captured_at_seconds: float


class CameraStatus(NamedTuple):
    logical_name: str
    device_id: str | None
    runtime_status: RuntimeStatus
    calibration_status: CalibrationStatus
    native_resolution: Resolution | None = None
    frame_counter: int = 0
    error_message: str | None = None
