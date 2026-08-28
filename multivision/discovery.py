"""Camera discovery and binding-status reporting."""

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any

from multivision.errors import HardwareError
from multivision.hardware import DeviceDiscovery
from multivision.types import (
    DeviceInfo,
    is_valid_resolution,
)

CommandRunner = Callable[..., Any]
CaptureIndexResolver = Callable[[str], int | None]

_STABLE_ID_KEYS = (
    'spcamera_device_unique_id',
    'spcamera_device_uid',
    'spcamera_unique_id',
    'device_unique_id',
    'device_uid',
    'unique_id',
)


def _system_camera_metadata(command_runner: CommandRunner) -> list[dict[str, Any]]:
    try:
        result = command_runner(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if getattr(result, 'returncode', 1) != 0:
            raise HardwareError('macOS camera metadata command failed')
        output = getattr(result, 'stdout', None)
        if not isinstance(output, (str, bytes, bytearray)):
            raise HardwareError('macOS camera metadata output was malformed')
        profile = json.loads(output)
    except HardwareError:
        raise
    except Exception as ex:  # noqa: BLE001 (Discovery is a hardware boundary).
        raise HardwareError('Could not read macOS camera metadata') from ex

    if not isinstance(profile, dict):
        raise HardwareError('macOS camera metadata root was malformed')
    entries = profile.get('SPCameraDataType', [])
    if not isinstance(entries, list):
        raise HardwareError('macOS camera metadata entries were malformed')
    if any(not isinstance(entry, dict) for entry in entries):
        raise HardwareError('macOS camera metadata contained a malformed device')
    return entries


def _device_id_from_entry(entry: Mapping[str, Any], name: str) -> tuple[str, bool]:
    for key in _STABLE_ID_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and len(value) > 0:
            return value, True
    return f'unstable-macos-name:{name}', False


class MacOSDeviceDiscovery:
    """Discover macOS cameras using native metadata without opening them."""

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        platform_name: str | None = None,
        capture_index_resolver: CaptureIndexResolver | None = None,
    ) -> None:
        self.command_runner = subprocess.run if command_runner is None else command_runner
        self.platform_name = sys.platform if platform_name is None else platform_name
        self._capture_index_resolver = (
            _resolve_avfoundation_capture_index
            if command_runner is None
            else capture_index_resolver
        )

    def discover_devices(self) -> list[DeviceInfo]:
        if self.platform_name != 'darwin':
            return []

        devices: list[DeviceInfo] = []
        capture_index_resolver = self._capture_index_resolver
        for profiler_index, entry in enumerate(
            _system_camera_metadata(self.command_runner),
        ):
            name = entry.get('_name')
            if not isinstance(name, str) or len(name) == 0:
                name = f'Camera {profiler_index + 1}'
            device_id, is_stable_id = _device_id_from_entry(entry, name)
            capture_index = profiler_index
            is_available = True
            error_message: str | None = None
            if capture_index_resolver is not None:
                capture_index = capture_index_resolver(device_id)
                if capture_index is None:
                    is_available = False
                    error_message = (
                        f'Could not resolve stable device ID {device_id!r} '
                        'to an AVFoundation capture index'
                    )
                elif (
                    not isinstance(capture_index, int)
                    or isinstance(capture_index, bool)
                    or capture_index < 0
                ):
                    raise HardwareError(
                        'AVFoundation capture-index resolver returned an invalid index',
                    )
            device = DeviceInfo(
                device_id=device_id,
                name=name,
                capture_index=capture_index,
                backend_name='avfoundation',
                metadata=dict(entry),
                is_available=is_available,
                error_message=error_message,
                is_stable_id=is_stable_id,
            )
            devices.append(device)
        return devices


class PlatformDeviceDiscovery:
    """Select the platform implementation without leaking it to callers."""

    def __init__(
        self,
        platform_name: str | None = None,
        macos_discovery: DeviceDiscovery | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.platform_name = sys.platform if platform_name is None else platform_name
        self._macos_discovery = macos_discovery
        if self.platform_name == 'darwin' and self._macos_discovery is None:
            self._macos_discovery = MacOSDeviceDiscovery(
                command_runner=command_runner,
                platform_name=self.platform_name,
            )

    def discover_devices(self) -> list[DeviceInfo]:
        if self.platform_name != 'darwin':
            return []
        assert self._macos_discovery is not None, 'macOS discovery must be configured'
        return self._macos_discovery.discover_devices()


def _resolve_avfoundation_capture_index(device_id: str) -> int | None:
    try:
        import AVFoundation
    except ImportError as ex:
        raise HardwareError(
            'PyObjC AVFoundation support is required to resolve macOS camera IDs',
        ) from ex

    media_type = getattr(AVFoundation, 'AVMediaTypeVideo', None)
    capture_device_class = getattr(AVFoundation, 'AVCaptureDevice', None)
    list_devices = getattr(capture_device_class, 'devicesWithMediaType_', None)
    if media_type is None or not callable(list_devices):
        raise HardwareError('AVFoundation does not provide video-device enumeration')

    try:
        devices = list_devices(media_type)
    except Exception as ex:  # noqa: BLE001 (AVFoundation is a hardware boundary).
        raise HardwareError('Could not enumerate AVFoundation camera devices') from ex

    matching_index: int | None = None
    for capture_index, device in enumerate(devices):
        get_unique_id = getattr(device, 'uniqueID', None)
        unique_id = get_unique_id() if callable(get_unique_id) else None
        if not isinstance(unique_id, str) or len(unique_id) == 0:
            continue
        if unique_id != device_id:
            continue
        if matching_index is not None:
            raise HardwareError(f'AVFoundation returned duplicate camera ID {device_id!r}')
        matching_index = capture_index
    return matching_index


def _group_discovered_devices(
    discovered_devices: Any,
) -> dict[str, list[DeviceInfo]]:
    if not isinstance(discovered_devices, list):
        raise HardwareError('Discovery returned a malformed device list')

    devices_by_id: dict[str, list[DeviceInfo]] = {}
    for device in discovered_devices:
        _validate_discovered_device(device)
        devices_by_id.setdefault(device.device_id, []).append(device)
    return devices_by_id


def _validate_discovered_device(device: DeviceInfo) -> None:
    if not isinstance(device, DeviceInfo):
        raise HardwareError('Discovery returned malformed device information')
    if not isinstance(device.device_id, str) or len(device.device_id) == 0:
        raise HardwareError('Discovery returned a device without an ID')
    if not isinstance(device.name, str) or len(device.name) == 0:
        raise HardwareError('Discovery returned a device without a name')
    if (
        device.capture_index is not None
        and (
            not isinstance(device.capture_index, int)
            or isinstance(device.capture_index, bool)
            or device.capture_index < 0
        )
    ):
        raise HardwareError('Discovery returned a device with an invalid capture index')
    if device.backend_name is not None and not isinstance(device.backend_name, str):
        raise HardwareError('Discovery returned a device with an invalid backend name')
    if device.metadata is not None and not isinstance(device.metadata, dict):
        raise HardwareError('Discovery returned a device with invalid metadata')
    if (
        not isinstance(device.is_available, bool)
        or not isinstance(device.is_stable_id, bool)
    ):
        raise HardwareError('Discovery returned malformed device status')
    if (
        device.error_message is not None
        and not isinstance(device.error_message, str)
    ):
        raise HardwareError('Discovery returned a device with an invalid error message')
    if (
        device.native_resolution is not None
        and not is_valid_resolution(device.native_resolution)
    ):
        raise HardwareError('Discovery returned a malformed resolution')


__all__ = [
    'MacOSDeviceDiscovery',
    'PlatformDeviceDiscovery',
]
