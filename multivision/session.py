"""In-memory camera slots and lifecycle state for one MultiVision session."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, replace

from multivision.errors import (
    ActiveCameraLimitError,
    CameraSlotNotFoundError,
    CameraStateError,
    DuplicateCameraNameError,
    SessionCameraError,
)
from multivision.types import (
    CalibrationStatus,
    DeviceInfo,
    Resolution,
    SessionCameraState,
    copy_device_info,
    is_finite_real,
    is_valid_resolution,
)


MAX_ACTIVE_CAMERAS = 4


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata for the latest usable frame retained by a session camera."""

    frame_counter: int = 0
    captured_at_seconds: float | None = None
    native_resolution: Resolution | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_counter, int)
            or isinstance(self.frame_counter, bool)
            or self.frame_counter < 0
        ):
            raise SessionCameraError('frame_counter must be a non-negative integer')
        if (
            self.captured_at_seconds is not None
            and not is_finite_real(self.captured_at_seconds)
        ):
            raise SessionCameraError(
                'captured_at_seconds must be a finite number or None',
            )
        if (
            self.native_resolution is not None
            and not is_valid_resolution(self.native_resolution)
        ):
            raise SessionCameraError(
                'native_resolution must be a positive Resolution or None',
            )


@dataclass
class SessionCamera:
    """Mutable state associated with one immutable session slot."""

    slot_id: str
    display_name: str
    capture_index: int | None
    device_info: DeviceInfo | None = None
    state: SessionCameraState = SessionCameraState.OPEN
    frame_metadata: FrameMetadata | None = None
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    calibration: object | None = None
    error_message: str | None = None
    lifecycle_generation: int = 0

    def __post_init__(self) -> None:
        _validate_slot_id(self.slot_id)
        _validate_display_name(self.display_name)
        _validate_capture_index(self.capture_index)
        _validate_device_info(self.device_info)
        _validate_camera_state(self.state)
        _validate_frame_metadata(self.frame_metadata)
        _validate_calibration_status(self.calibration_status)
        _validate_camera_runtime_state(
            self.state,
            self.frame_metadata,
            self.calibration_status,
            self.calibration,
        )
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise SessionCameraError('error_message must be a string or None')
        if (
            not isinstance(self.lifecycle_generation, int)
            or isinstance(self.lifecycle_generation, bool)
            or self.lifecycle_generation < 0
        ):
            raise SessionCameraError(
                'lifecycle_generation must be a non-negative integer',
            )
        if self.state is SessionCameraState.OPEN and self.capture_index is None:
            raise SessionCameraError('an open camera must have a capture index')

    def __setattr__(self, attribute_name: str, value: object) -> None:
        if attribute_name == 'slot_id' and hasattr(self, 'slot_id'):
            raise AttributeError('slot_id is immutable for the session lifetime')
        super().__setattr__(attribute_name, value)

    def __delattr__(self, attribute_name: str) -> None:
        if attribute_name == 'slot_id':
            raise AttributeError('slot_id is immutable for the session lifetime')
        super().__delattr__(attribute_name)


class SessionCameraRegistry:
    """Keep session slots ordered and enforce their lifecycle invariants."""

    def __init__(self, devices: Iterable[DeviceInfo] = ()) -> None:
        self._cameras = self._build_cameras(devices)

    @classmethod
    def from_devices(
        cls,
        devices: Iterable[DeviceInfo],
    ) -> 'SessionCameraRegistry':
        return cls(devices)

    @classmethod
    def from_capture_indexes(
        cls,
        capture_indexes: Iterable[int],
    ) -> 'SessionCameraRegistry':
        try:
            devices = tuple(
                DeviceInfo(
                    device_id=f'capture-index-{capture_index}',
                    name=f'Camera {capture_index}',
                    capture_index=capture_index,
                    is_stable_id=False,
                )
                for capture_index in capture_indexes
            )
        except TypeError as ex:
            raise SessionCameraError('capture_indexes must be iterable') from ex
        return cls(devices)

    def get_cameras(self) -> list[SessionCamera]:
        """Return every slot in deterministic session-slot order."""
        return [self._snapshot_camera(camera) for camera in self._cameras]

    def get(self, slot_id: str) -> SessionCamera:
        return self._snapshot_camera(self._get(slot_id))

    def rename(self, slot_id: str, display_name: str) -> SessionCamera:
        camera = self._get(slot_id)
        _validate_display_name(display_name)
        for other_camera in self._cameras:
            if (
                other_camera.slot_id != slot_id
                and other_camera.display_name == display_name
            ):
                raise DuplicateCameraNameError(
                    f'Camera display name {display_name!r} is already in use',
                )
        camera.display_name = display_name
        return self._snapshot_camera(camera)

    def close(self, slot_id: str) -> SessionCamera:
        camera = self._get(slot_id)
        if camera.state is not SessionCameraState.OPEN:
            raise CameraStateError(
                f'Camera {slot_id!r} cannot be closed from {camera.state}',
            )
        camera.state = SessionCameraState.CLOSED
        camera.lifecycle_generation += 1
        self._clear_runtime_state(camera)
        return self._snapshot_camera(camera)

    def open(self, slot_id: str) -> SessionCamera:
        camera = self._get(slot_id)
        if camera.state is SessionCameraState.UNAVAILABLE:
            raise CameraStateError(f'Camera {slot_id!r} is unavailable')
        if camera.state is SessionCameraState.OPEN:
            raise CameraStateError(f'Camera {slot_id!r} is already open')
        if camera.capture_index is None:
            raise CameraStateError(f'Camera {slot_id!r} has no capture index')
        if self.active_count >= MAX_ACTIVE_CAMERAS:
            raise ActiveCameraLimitError(
                f'Opening {slot_id!r} would exceed the '
                f'{MAX_ACTIVE_CAMERAS}-camera active limit',
            )
        camera.state = SessionCameraState.OPEN
        camera.error_message = None
        camera.lifecycle_generation += 1
        self._clear_runtime_state(camera)
        return self._snapshot_camera(camera)

    def mark_unavailable(
        self,
        slot_id: str,
        error_message: str | None = None,
    ) -> SessionCamera:
        camera = self._get(slot_id)
        if error_message is not None and not isinstance(error_message, str):
            raise SessionCameraError('error_message must be a string or None')
        camera.state = SessionCameraState.UNAVAILABLE
        camera.error_message = error_message
        camera.lifecycle_generation += 1
        self._clear_runtime_state(camera)
        return self._snapshot_camera(camera)

    def set_frame_metadata(
        self,
        slot_id: str,
        frame_metadata: FrameMetadata,
    ) -> SessionCamera:
        camera = self._get(slot_id)
        _validate_frame_metadata(frame_metadata)
        if camera.state is not SessionCameraState.OPEN:
            raise CameraStateError(
                f'Camera {slot_id!r} cannot receive a frame while {camera.state}',
            )
        camera.frame_metadata = frame_metadata
        return self._snapshot_camera(camera)

    def set_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: object | None = None,
    ) -> SessionCamera:
        camera = self._get(slot_id)
        _validate_calibration_status(calibration_status)
        if camera.state is not SessionCameraState.OPEN:
            raise CameraStateError(
                f'Camera {slot_id!r} cannot be calibrated while {camera.state}',
            )
        _validate_camera_runtime_state(
            camera.state,
            camera.frame_metadata,
            calibration_status,
            calibration,
        )
        camera.calibration_status = calibration_status
        camera.calibration = copy.deepcopy(calibration)
        return self._snapshot_camera(camera)

    def _get(self, slot_id: str) -> SessionCamera:
        for camera in self._cameras:
            if camera.slot_id == slot_id:
                return camera
        raise CameraSlotNotFoundError(f'Unknown session camera slot {slot_id!r}')

    @staticmethod
    def _snapshot_camera(camera: SessionCamera) -> SessionCamera:
        device_info = camera.device_info
        if device_info is not None:
            device_info = copy_device_info(device_info)
        return replace(
            camera,
            device_info=device_info,
            calibration=copy.deepcopy(camera.calibration),
        )

    @property
    def active_count(self) -> int:
        return sum(
            camera.state is SessionCameraState.OPEN
            for camera in self._cameras
        )

    def _build_cameras(self, devices: Iterable[DeviceInfo]) -> list[SessionCamera]:
        try:
            checked_devices = tuple(devices)
        except TypeError as ex:
            raise SessionCameraError('devices must be iterable') from ex
        if any(not isinstance(device, DeviceInfo) for device in checked_devices):
            raise SessionCameraError('devices must contain DeviceInfo values')

        for device in checked_devices:
            _validate_device(device)

        device_ids = [device.device_id for device in checked_devices]
        if len(device_ids) != len(set(device_ids)):
            raise SessionCameraError('session cameras must have unique device IDs')

        ordered_devices = sorted(checked_devices, key=_device_sort_key)
        capture_indexes = [
            device.capture_index
            for device in ordered_devices
            if device.capture_index is not None
        ]
        if len(capture_indexes) != len(set(capture_indexes)):
            raise SessionCameraError('session cameras must have unique capture indexes')

        cameras: list[SessionCamera] = []
        active_camera_count = 0
        for slot_index, device in enumerate(ordered_devices):
            device = copy_device_info(device)
            state = _initial_camera_state(
                device,
                active_camera_count,
            )
            if state is SessionCameraState.OPEN:
                active_camera_count += 1
            cameras.append(
                SessionCamera(
                    slot_id=f'camera-{slot_index}',
                    display_name=f'camera-{slot_index}',
                    capture_index=device.capture_index,
                    device_info=device,
                    state=state,
                    error_message=device.error_message,
                ),
            )
        return cameras

    @staticmethod
    def _clear_calibration(camera: SessionCamera) -> None:
        camera.calibration_status = CalibrationStatus.UNCALIBRATED
        camera.calibration = None

    @staticmethod
    def _clear_runtime_state(camera: SessionCamera) -> None:
        camera.frame_metadata = None
        SessionCameraRegistry._clear_calibration(camera)


def _initial_camera_state(
    device: DeviceInfo,
    active_camera_count: int,
) -> SessionCameraState:
    if not device.is_available or device.capture_index is None:
        return SessionCameraState.UNAVAILABLE
    if active_camera_count >= MAX_ACTIVE_CAMERAS:
        return SessionCameraState.CLOSED
    return SessionCameraState.OPEN


def _device_sort_key(device: DeviceInfo) -> tuple[bool, int, str]:
    capture_index = device.capture_index
    return (
        capture_index is None,
        capture_index if capture_index is not None else 0,
        device.device_id,
    )


def _validate_slot_id(slot_id: object) -> None:
    if not isinstance(slot_id, str) or len(slot_id) == 0:
        raise SessionCameraError('slot_id must be a non-empty string')
    prefix, separator, index = slot_id.rpartition('-')
    if prefix != 'camera' or separator == '' or not index.isdigit():
        raise SessionCameraError("slot_id must use the 'camera-N' format")


def _validate_display_name(display_name: object) -> None:
    if not isinstance(display_name, str) or len(display_name.strip()) == 0:
        raise SessionCameraError('display_name must be a non-empty string')


def _validate_capture_index(capture_index: object) -> None:
    if capture_index is not None and (
        not isinstance(capture_index, int)
        or isinstance(capture_index, bool)
        or capture_index < 0
    ):
        raise SessionCameraError('capture_index must be a non-negative integer or None')


def _validate_device_info(device_info: object) -> None:
    if device_info is None:
        return
    if not isinstance(device_info, DeviceInfo):
        raise SessionCameraError('device_info must be DeviceInfo or None')
    _validate_device(device_info)


def _validate_camera_state(state: object) -> None:
    if not isinstance(state, SessionCameraState):
        raise SessionCameraError('state must be OPEN, CLOSED or UNAVAILABLE')


def _validate_frame_metadata(frame_metadata: object) -> None:
    if frame_metadata is not None and not isinstance(frame_metadata, FrameMetadata):
        raise SessionCameraError('frame_metadata must be FrameMetadata or None')


def _validate_calibration_status(status: object) -> None:
    if not isinstance(status, CalibrationStatus):
        raise SessionCameraError('calibration_status must be CalibrationStatus')


def _validate_camera_runtime_state(
    state: SessionCameraState,
    frame_metadata: FrameMetadata | None,
    calibration_status: CalibrationStatus,
    calibration: object | None,
) -> None:
    if state is not SessionCameraState.OPEN and frame_metadata is not None:
        raise SessionCameraError(
            'closed or unavailable cameras cannot retain frame metadata',
        )
    if state is not SessionCameraState.OPEN and (
        calibration_status is not CalibrationStatus.UNCALIBRATED
        or calibration is not None
    ):
        raise SessionCameraError(
            'closed or unavailable cameras cannot retain calibration',
        )
    if calibration_status is CalibrationStatus.UNCALIBRATED and calibration is not None:
        raise SessionCameraError(
            'uncalibrated cameras cannot retain a calibration record',
        )
    if calibration_status is not CalibrationStatus.UNCALIBRATED and calibration is None:
        raise SessionCameraError(
            'calibrated-state cameras must retain a calibration record',
        )


def _validate_device(device: DeviceInfo) -> None:
    if not isinstance(device.device_id, str) or len(device.device_id) == 0:
        raise SessionCameraError('device_id must be a non-empty string')
    if not isinstance(device.name, str) or len(device.name) == 0:
        raise SessionCameraError('device name must be a non-empty string')
    _validate_capture_index(device.capture_index)
    if (
        device.native_resolution is not None
        and not is_valid_resolution(device.native_resolution)
    ):
        raise SessionCameraError('native_resolution must be a positive Resolution or None')
    if device.backend_name is not None and not isinstance(device.backend_name, str):
        raise SessionCameraError('backend_name must be a string or None')
    if device.metadata is not None and not isinstance(device.metadata, dict):
        raise SessionCameraError('metadata must be a dictionary or None')
    if not isinstance(device.is_available, bool):
        raise SessionCameraError('is_available must be a bool')
    if device.error_message is not None and not isinstance(device.error_message, str):
        raise SessionCameraError('error_message must be a string or None')
    if not isinstance(device.is_stable_id, bool):
        raise SessionCameraError('is_stable_id must be a bool')


__all__ = [
    'FrameMetadata',
    'MAX_ACTIVE_CAMERAS',
    'SessionCamera',
    'SessionCameraRegistry',
]
