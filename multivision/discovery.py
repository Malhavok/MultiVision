"""Session-local camera discovery."""

import sys
from collections.abc import Callable
from typing import Any

from multivision.errors import HardwareError
from multivision.hardware import DeviceDiscovery
from multivision.session import MAX_ACTIVE_CAMERAS
from multivision.types import (
    DeviceInfo,
    Resolution,
    copy_device_info,
    is_finite_real,
    is_valid_resolution,
)

CaptureOpener = Callable[[int], Any]


class OpenCVCaptureIndexDiscovery:
    """Probe current OpenCV indexes once and retain that session snapshot."""

    def __init__(
        self,
        max_capture_index: int = 10,
        capture_opener: CaptureOpener | None = None,
        backend: int | None = None,
    ) -> None:
        if (
            not isinstance(max_capture_index, int)
            or isinstance(max_capture_index, bool)
            or max_capture_index <= 0
        ):
            raise ValueError('max_capture_index must be a positive integer')
        if backend is not None and (
            not isinstance(backend, int) or isinstance(backend, bool)
        ):
            raise ValueError('backend must be an integer or None')
        self.max_capture_index = max_capture_index
        self._capture_opener = capture_opener
        self.backend = backend
        self._snapshot: tuple[DeviceInfo, ...] | None = None
        self._retain_probes = False
        self._probed_captures: dict[int, Any] = {}

    def prepare_for_runtime(self) -> None:
        """Keep the selected startup probes for the owning runtime to adopt."""
        self._retain_probes = True

    def take_capture(self, capture_index: int) -> Any | None:
        """Transfer one retained startup probe to the persistent capture owner."""
        return self._probed_captures.pop(capture_index, None)

    def release_unused_captures(self) -> None:
        """Release startup probes that were not selected for persistent capture."""
        captures = list(self._probed_captures.items())
        self._probed_captures.clear()
        release_error: HardwareError | None = None
        for capture_index, capture in captures:
            try:
                _release_probe_capture(capture, capture_index)
            except HardwareError as ex:
                if release_error is None:
                    release_error = ex
        if release_error is not None:
            raise release_error

    def discover_devices(self) -> list[DeviceInfo]:
        """Return the fixed startup inventory without remapping its slots."""
        if self._snapshot is None:
            devices: list[DeviceInfo] = []
            try:
                for capture_index in range(self.max_capture_index):
                    device = self._probe_capture_index(
                        capture_index,
                        retain_capture=(
                            self._retain_probes
                            and len(devices) < MAX_ACTIVE_CAMERAS
                        ),
                    )
                    if device is not None:
                        devices.append(device)
            except BaseException:
                self.release_unused_captures()
                raise
            self._snapshot = tuple(devices)
        return [copy_device_info(device) for device in self._snapshot]

    def _probe_capture_index(
        self,
        capture_index: int,
        retain_capture: bool = False,
    ) -> DeviceInfo | None:
        capture = self._open_capture(capture_index)
        if capture is None:
            raise HardwareError(
                f'OpenCV returned no capture probe for index {capture_index}',
            )

        keep_capture = False
        try:
            is_opened = getattr(capture, 'isOpened', None)
            if not callable(is_opened):
                raise HardwareError(
                    f'OpenCV capture probe {capture_index} has no isOpened method',
                )
            try:
                opened = is_opened()
            except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
                raise HardwareError(
                    f'Could not probe OpenCV capture index {capture_index}',
                ) from ex
            if not isinstance(opened, bool):
                raise HardwareError(
                    f'OpenCV capture probe {capture_index} returned a malformed open state',
                )
            if not opened:
                return None

            native_resolution, resolution_metadata = _probe_native_resolution(
                capture,
                capture_index,
            )
            metadata: dict[str, Any] = {
                'capture_index': capture_index,
                'probe': 'opencv',
            }
            metadata.update(resolution_metadata)
            device = DeviceInfo(
                device_id=f'capture-index-{capture_index}',
                name=f'Camera {capture_index}',
                capture_index=capture_index,
                native_resolution=native_resolution,
                backend_name='opencv',
                metadata=metadata,
                is_available=True,
                is_stable_id=False,
            )
            if retain_capture:
                self._probed_captures[capture_index] = capture
                keep_capture = True
            return device
        finally:
            if not keep_capture:
                _release_probe_capture(capture, capture_index)

    def _open_capture(self, capture_index: int) -> Any:
        if self._capture_opener is not None:
            try:
                return self._capture_opener(capture_index)
            except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
                raise HardwareError(
                    f'Could not open OpenCV capture index {capture_index}',
                ) from ex

        try:
            import cv2
        except ImportError as ex:
            raise HardwareError('OpenCV is not installed') from ex

        try:
            if self.backend is None:
                return cv2.VideoCapture(capture_index)
            return cv2.VideoCapture(capture_index, self.backend)
        except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
            raise HardwareError(
                f'Could not open OpenCV capture index {capture_index}',
            ) from ex


class PlatformDeviceDiscovery:
    """Select the session-local OpenCV inventory for the current platform."""

    def __init__(
        self,
        platform_name: str | None = None,
        opencv_discovery: DeviceDiscovery | None = None,
    ) -> None:
        self.platform_name = sys.platform if platform_name is None else platform_name
        self._discovery = opencv_discovery
        if self._discovery is None and self.platform_name == 'darwin':
            self._discovery = OpenCVCaptureIndexDiscovery()

    def discover_devices(self) -> list[DeviceInfo]:
        if self._discovery is None:
            return []
        return self._discovery.discover_devices()


def _probe_native_resolution(
    capture: Any,
    capture_index: int,
) -> tuple[Resolution | None, dict[str, int | float]]:
    get_native_resolution = getattr(capture, 'get_native_resolution', None)
    if callable(get_native_resolution):
        try:
            native_resolution = get_native_resolution()
        except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
            raise HardwareError(
                f'Could not read metadata for OpenCV capture index {capture_index}',
            ) from ex
        if native_resolution is not None and not is_valid_resolution(native_resolution):
            raise HardwareError(
                f'OpenCV capture index {capture_index} returned a malformed resolution',
            )
        if native_resolution is None:
            return None, {}
        return native_resolution, {
            'native_width': native_resolution.width,
            'native_height': native_resolution.height,
        }

    get_property = getattr(capture, 'get', None)
    if not callable(get_property):
        return None, {}
    try:
        import cv2

        width = get_property(cv2.CAP_PROP_FRAME_WIDTH)
        height = get_property(cv2.CAP_PROP_FRAME_HEIGHT)
    except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
        raise HardwareError(
            f'Could not read metadata for OpenCV capture index {capture_index}',
        ) from ex
    if not isinstance(width, (int, float)) or isinstance(width, bool):
        raise HardwareError(
            f'OpenCV capture index {capture_index} returned a malformed width',
        )
    if not isinstance(height, (int, float)) or isinstance(height, bool):
        raise HardwareError(
            f'OpenCV capture index {capture_index} returned a malformed height',
        )
    if not is_finite_real(width) or not is_finite_real(height):
        raise HardwareError(
            f'OpenCV capture index {capture_index} returned a non-finite resolution',
        )
    if width <= 0 or height <= 0:
        return None, {}

    native_resolution = Resolution(round(width), round(height))
    if not is_valid_resolution(native_resolution):
        raise HardwareError(
            f'OpenCV capture index {capture_index} returned a malformed resolution',
        )
    return native_resolution, {
        'native_width': native_resolution.width,
        'native_height': native_resolution.height,
    }


def _release_probe_capture(capture: Any, capture_index: int) -> None:
    release = getattr(capture, 'release', None)
    if not callable(release):
        raise HardwareError(
            f'OpenCV capture probe {capture_index} has no release method',
        )
    try:
        release()
    except Exception as ex:  # noqa: BLE001 (OpenCV is a hardware boundary).
        raise HardwareError(
            f'Could not release OpenCV capture probe {capture_index}',
        ) from ex


__all__ = [
    'OpenCVCaptureIndexDiscovery',
    'PlatformDeviceDiscovery',
]
