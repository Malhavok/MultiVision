"""Small seams between MultiVision and physical camera hardware."""

import math
import shutil
import subprocess
import sys
from typing import (
    Any,
    Callable,
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


# Some macOS AVFoundation cameras return a near-black conversion while a native
# AVFoundation client can still provide the usable frame.
OPENCV_BLACK_FRAME_MAX_VALUE = 8


class OpenCVCaptureDevice:
    def __init__(
        self,
        capture: Any,
        fallback_opener: Callable[[], CaptureDevice] | None = None,
    ) -> None:
        self._capture = capture
        self._fallback_capture: CaptureDevice | None = None
        self._fallback_opener = fallback_opener

    def is_opened(self) -> bool:
        if self._fallback_capture is not None:
            return self._fallback_capture.is_opened()
        try:
            return bool(self._capture.isOpened())
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise CameraOpenError('Could not query the camera state') from ex

    def read(self) -> tuple[bool, Any]:
        if self._fallback_capture is not None:
            return self._fallback_capture.read()
        try:
            result = self._capture.read()
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise FrameCaptureError('Could not read from the camera') from ex
        if not isinstance(result, tuple) or len(result) != 2:
            raise FrameCaptureError('The camera returned a malformed frame result')
        success, frame = result
        if not isinstance(success, bool):
            raise FrameCaptureError('The camera returned a malformed frame result')
        if (
            not success
            or not _is_opencv_black_frame(frame)
            or self._fallback_opener is None
        ):
            return success, frame
        self._activate_fallback()
        assert self._fallback_capture is not None
        return self._fallback_capture.read()

    def get_native_resolution(self) -> Resolution | None:
        if self._fallback_capture is not None:
            get_native_resolution = getattr(
                self._fallback_capture,
                'get_native_resolution',
                None,
            )
            if callable(get_native_resolution):
                return get_native_resolution()
            return None
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
        if self._fallback_capture is not None:
            self._fallback_capture.release()
            self._fallback_capture = None
            return
        try:
            self._capture.release()
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            raise HardwareError('Could not release the camera') from ex

    def _activate_fallback(self) -> None:
        fallback_opener = self._fallback_opener
        self._fallback_opener = None
        if fallback_opener is None:
            return
        try:
            self._capture.release()
            self._fallback_capture = fallback_opener()
        except Exception as ex:  # noqa: BLE001 (Fallback is a hardware boundary).
            raise FrameCaptureError('Could not switch to the camera fallback') from ex


class _FfmpegCaptureDevice:
    def __init__(self, process: Any, resolution: Resolution) -> None:
        self._process: Any | None = process
        self._resolution = resolution

    def is_opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def read(self) -> tuple[bool, Any]:
        process = self._process
        if process is None or not self.is_opened() or process.stdout is None:
            raise FrameCaptureError('FFmpeg camera capture is not running')
        frame_size = self._resolution.width * self._resolution.height * 3
        frame_bytes = _read_exact(process.stdout, frame_size)
        if len(frame_bytes) != frame_size:
            raise FrameCaptureError('FFmpeg camera capture ended before a full frame')
        try:
            import numpy as np
        except ImportError as ex:
            raise FrameCaptureError('NumPy is required for FFmpeg camera capture') from ex
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self._resolution.height, self._resolution.width, 3),
        )
        return True, frame.copy()

    def get_native_resolution(self) -> Resolution:
        return self._resolution

    def release(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        finally:
            if process.stdout is not None:
                process.stdout.close()


class OpenCVCaptureDeviceFactory:
    """OpenCV-backed factory kept behind the capture seam."""

    def __init__(self, backend: int | None = None) -> None:
        self.backend = backend

    def adopt_capture(
        self,
        capture: Any,
        device: DeviceInfo | None = None,
    ) -> CaptureDevice:
        """Adopt a discovery probe so startup does not open the device twice."""
        return OpenCVCaptureDevice(capture, _make_ffmpeg_fallback(device))

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
            return OpenCVCaptureDevice(capture, _make_ffmpeg_fallback(device))
        except CameraOpenError:
            if capture is not None:
                _release_capture(capture)
            raise
        except Exception as ex:  # noqa: BLE001 (OpenCV errors cross the hardware boundary).
            if capture is not None:
                _release_capture(capture)
            raise CameraOpenError(f'Could not open device {device.device_id!r}') from ex


def _make_ffmpeg_fallback(
    device: DeviceInfo | None,
) -> Callable[[], CaptureDevice] | None:
    if sys.platform != 'darwin' or device is None:
        return None
    if shutil.which('ffmpeg') is None:
        return None
    if device.capture_index is None or device.native_resolution is None:
        return None
    return lambda: _open_ffmpeg_capture(device)


def _open_ffmpeg_capture(device: DeviceInfo) -> CaptureDevice:
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        raise CameraOpenError('FFmpeg is required for this camera capture fallback')
    assert device.capture_index is not None
    assert device.native_resolution is not None
    resolution = device.native_resolution
    command = [
        ffmpeg_path,
        '-hide_banner',
        '-loglevel',
        'error',
        '-f',
        'avfoundation',
        '-pixel_format',
        'uyvy422',
        '-framerate',
        '30',
        '-video_size',
        f'{resolution.width}x{resolution.height}',
        '-i',
        str(device.capture_index),
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        '-',
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
    except OSError as ex:
        raise CameraOpenError('Could not start FFmpeg camera capture fallback') from ex
    return _FfmpegCaptureDevice(process, resolution)


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining_size = size
    while remaining_size > 0:
        chunk = stream.read(remaining_size)
        if not isinstance(chunk, bytes):
            raise FrameCaptureError('FFmpeg camera capture returned malformed data')
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        remaining_size -= len(chunk)
    return b''.join(chunks)


def _is_opencv_black_frame(frame: Any) -> bool:
    try:
        return (
            frame is not None
            and frame.size > 0
            and bool(frame.max() <= OPENCV_BLACK_FRAME_MAX_VALUE)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _release_capture(capture: Any) -> None:
    try:
        capture.release()
    except Exception:  # noqa: BLE001 (Cleanup must not hide the opening failure).
        pass
