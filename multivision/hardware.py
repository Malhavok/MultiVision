"""Small seams between MultiVision and physical camera hardware."""

import math
from typing import (
    Any,
    Protocol,
)

from multivision.errors import (
    CameraOpenError,
    FrameCaptureError,
    HardwareError,
)
from multivision.types import (
    DeviceInfo,
    Resolution,
)


class CaptureDevice(Protocol):
    """The part of a camera handle owned by the runtime."""

    def is_opened(self) -> bool:
        ...

    def read(self) -> tuple[bool, Any]:
        ...

    def release(self) -> None:
        ...


class DeviceDiscovery(Protocol):
    """Enumerate devices without exposing platform details to the service."""

    def discover_devices(self) -> list[DeviceInfo]:
        ...


class CaptureDeviceFactory(Protocol):
    """Open a capture handle for a discovered device."""

    def open_capture(self, device: DeviceInfo) -> CaptureDevice:
        ...


class OpenCVCaptureDevice:
    def __init__(self, capture: Any) -> None:
        self._capture = capture

    def is_opened(self) -> bool:
        try:
            return bool(self._capture.isOpened())
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise CameraOpenError('Could not query the camera state') from ex

    def read(self) -> tuple[bool, Any]:
        try:
            result = self._capture.read()
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise FrameCaptureError('Could not read from the camera') from ex
        if not isinstance(result, tuple) or len(result) != 2:
            raise FrameCaptureError('The camera returned a malformed frame result')
        success, frame = result
        if not isinstance(success, bool):
            raise FrameCaptureError('The camera returned a malformed frame result')
        return success, frame

    def get_native_resolution(self) -> Resolution | None:
        try:
            import cv2

            width = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise HardwareError('Could not query the camera resolution') from ex

        if (
            not isinstance(width, (int, float))
            or isinstance(width, bool)
            or not isinstance(height, (int, float))
            or isinstance(height, bool)
        ):
            return None
        if not math.isfinite(width) or not math.isfinite(height):
            return None
        if width <= 0 or height <= 0:
            return None

        rounded_resolution = Resolution(round(width), round(height))
        if rounded_resolution.width <= 0 or rounded_resolution.height <= 0:
            return None
        return rounded_resolution

    def release(self) -> None:
        try:
            self._capture.release()
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise HardwareError('Could not release the camera') from ex


class OpenCVCaptureDeviceFactory:
    """OpenCV-backed factory kept behind the capture seam."""

    def __init__(self, backend: int | None = None) -> None:
        self.backend = backend

    def open_capture(self, device: DeviceInfo) -> CaptureDevice:
        if not isinstance(device, DeviceInfo):
            raise CameraOpenError('Device must be discovered device information')
        if not isinstance(device.device_id, str) or len(device.device_id) == 0:
            raise CameraOpenError('Device must have a non-empty device ID')
        if not isinstance(device.name, str) or len(device.name) == 0:
            raise CameraOpenError(f'Device {device.device_id!r} has no name')
        capture_index = device.capture_index
        if (
            not isinstance(capture_index, int)
            or isinstance(capture_index, bool)
            or capture_index < 0
        ):
            raise CameraOpenError(
                f'Device {device.device_id!r} has an invalid capture index',
            )

        try:
            import cv2
        except ImportError as ex:
            raise CameraOpenError('OpenCV is not installed') from ex

        capture: Any | None = None
        try:
            backend = self.backend
            if backend is None and device.backend_name == 'avfoundation':
                backend = getattr(cv2, 'CAP_AVFOUNDATION', None)
                if backend is None:
                    raise CameraOpenError(
                        'OpenCV does not provide the required AVFoundation backend',
                    )
            if backend is None:
                capture = cv2.VideoCapture(capture_index)
            else:
                capture = cv2.VideoCapture(capture_index, backend)
            opened = capture.isOpened()
            if not isinstance(opened, bool):
                raise CameraOpenError('Capture handle returned a malformed open state')
            if not opened:
                raise CameraOpenError(f'Could not open device {device.device_id!r}')
            return OpenCVCaptureDevice(capture)
        except CameraOpenError:
            if capture is not None:
                _release_capture(capture)
            raise
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            if capture is not None:
                _release_capture(capture)
            raise CameraOpenError(f'Could not open device {device.device_id!r}') from ex


def _release_capture(capture: Any) -> None:
    try:
        capture.release()
    except Exception:  # noqa: BLE001 (Cleanup must not hide the opening failure).
        pass
