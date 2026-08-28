import math
import threading
import time
from collections.abc import Mapping
from typing import Any

from multivision.config import validate_camera_bindings
from multivision.discovery import _group_discovered_devices
from multivision.errors import (
    CameraUnavailableError,
    FrameCaptureError,
    HardwareError,
)
from multivision.hardware import (
    CaptureDevice,
    CaptureDeviceFactory,
    DeviceDiscovery,
)
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
    is_valid_resolution,
)


_UNSET_NATIVE_RESOLUTION = object()


class CameraRuntime:
    """Own open camera handles and publish their latest usable frames."""

    def __init__(
        self,
        discovery: DeviceDiscovery,
        capture_factory: CaptureDeviceFactory,
        camera_bindings: Mapping[str, str],
        read_wait_seconds: float = 0.01,
        worker_shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        validate_camera_bindings(camera_bindings)
        if not isinstance(read_wait_seconds, (int, float)) or isinstance(
            read_wait_seconds,
            bool,
        ) or not math.isfinite(read_wait_seconds) or read_wait_seconds < 0:
            raise ValueError('read_wait_seconds must be a finite, non-negative number')
        if (
            not isinstance(worker_shutdown_timeout_seconds, (int, float))
            or isinstance(worker_shutdown_timeout_seconds, bool)
            or not math.isfinite(worker_shutdown_timeout_seconds)
            or worker_shutdown_timeout_seconds <= 0
        ):
            raise ValueError(
                'worker_shutdown_timeout_seconds must be a finite positive number',
            )

        self._discovery = discovery
        self._capture_factory = capture_factory
        self._camera_bindings = dict(camera_bindings)
        self._read_wait_seconds = read_wait_seconds
        self._worker_shutdown_timeout_seconds = worker_shutdown_timeout_seconds
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._captures: dict[str, CaptureDevice] = {}
        self._discovered_devices: list[DeviceInfo] = []
        self._logical_names_by_device: dict[str, list[str]] = {}
        self._latest_frames: dict[str, Frame] = {}
        self._threads: list[threading.Thread] = []
        self._statuses: dict[str, CameraStatus] = {
            logical_name: CameraStatus(
                logical_name=logical_name,
                device_id=device_id,
                runtime_status=RuntimeStatus.STARTING,
                calibration_status=CalibrationStatus.UNCALIBRATED,
            )
            for logical_name, device_id in self._camera_bindings.items()
        }
        self._has_started = False
        self._has_stopped = False

    def start(self) -> None:
        """Discover, open and start one worker for every configured device."""
        with self._lifecycle_lock:
            if self._has_stopped:
                raise RuntimeError('The camera runtime has already been shut down')
            if self._has_started:
                return
            self._has_started = True

            try:
                discovered_devices = self._discovery.discover_devices()
            except Exception as ex:  # noqa: BLE001 (Discovery is a hardware boundary).
                self._set_all_errors(f'Could not discover cameras: {ex}')
                return

            try:
                devices_by_id = _group_discovered_devices(discovered_devices)
                with self._lock:
                    self._discovered_devices = list(discovered_devices)
            except HardwareError as ex:
                self._set_all_errors(str(ex))
                return

            devices_to_open: dict[str, DeviceInfo] = {}
            for logical_name, device_id in sorted(self._camera_bindings.items()):
                matching_devices = devices_by_id.get(device_id, [])
                if len(matching_devices) == 0:
                    self._set_status(
                        logical_name,
                        RuntimeStatus.UNAVAILABLE,
                        f'Device {device_id!r} is unavailable',
                    )
                    continue
                if len(matching_devices) > 1:
                    self._set_status(
                        logical_name,
                        RuntimeStatus.ERROR,
                        f'Device ID {device_id!r} is ambiguous',
                    )
                    continue

                device = matching_devices[0]
                if not device.is_stable_id:
                    self._set_status(
                        logical_name,
                        RuntimeStatus.UNAVAILABLE,
                        f'Device {device_id!r} has no stable identifier',
                    )
                    continue
                if not device.is_available:
                    self._set_status(
                        logical_name,
                        RuntimeStatus.UNAVAILABLE,
                        device.error_message or f'Device {device_id!r} is unavailable',
                        device.native_resolution,
                    )
                    continue

                self._logical_names_by_device.setdefault(device_id, []).append(logical_name)
                devices_to_open[device_id] = device

            for device_id, device in devices_to_open.items():
                self._open_device(device_id, device)

            for device_id, capture in self._captures.items():
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(device_id, capture),
                    daemon=True,
                    name=f'multivision-camera-{device_id}',
                )
                self._threads.append(thread)
                thread.start()

    def shutdown(self) -> None:
        """Stop workers and release every handle owned by this runtime."""
        with self._lifecycle_lock:
            if self._has_stopped:
                return
            self._stop_event.set()

            release_error: Exception | None = None
            for capture in list(self._captures.values()):
                try:
                    capture.release()
                except Exception as ex:  # noqa: BLE001 (Cleanup must continue for all handles).
                    if release_error is None:
                        release_error = ex

            shutdown_deadline = (
                time.monotonic() + self._worker_shutdown_timeout_seconds
            )
            unfinished_threads: list[str] = []
            for thread in self._threads:
                remaining_seconds = max(0.0, shutdown_deadline - time.monotonic())
                thread.join(remaining_seconds)
                if thread.is_alive():
                    unfinished_threads.append(thread.name)

            if len(unfinished_threads) > 0:
                error_message = (
                    'Camera workers did not stop within '
                    f'{self._worker_shutdown_timeout_seconds} seconds: '
                    + ', '.join(unfinished_threads)
                )
                with self._lock:
                    for logical_name, status in self._statuses.items():
                        self._statuses[logical_name] = status._replace(
                            runtime_status=RuntimeStatus.ERROR,
                            error_message=error_message,
                        )
                raise HardwareError(error_message)

            with self._lock:
                for logical_name, status in self._statuses.items():
                    self._statuses[logical_name] = status._replace(
                        runtime_status=RuntimeStatus.STOPPED,
                        error_message=(
                            str(release_error) if release_error is not None else None
                        ),
                    )
                self._captures.clear()
                self._threads.clear()
                self._has_stopped = True

            if release_error is not None:
                raise HardwareError(
                    'Could not release one or more camera handles',
                ) from release_error

    def snapshot(self, logical_name: str) -> Frame:
        """Return the latest usable frame without opening or reading a camera."""
        status = self.get_status(logical_name)
        if status.runtime_status in {
            RuntimeStatus.UNAVAILABLE,
            RuntimeStatus.STOPPED,
        }:
            raise CameraUnavailableError(
                status.error_message or f'Camera {logical_name!r} is unavailable',
            )

        with self._lock:
            frame = self._latest_frames.get(status.device_id or '')
        if frame is None:
            raise FrameCaptureError(
                status.error_message or f'Camera {logical_name!r} has no usable frame',
            )
        return frame

    def get_discovered_devices(self) -> list[DeviceInfo]:
        """Return the latest discovery result without opening another device."""
        with self._lock:
            return list(self._discovered_devices)

    def get_status(self, logical_name: str) -> CameraStatus:
        if not isinstance(logical_name, str) or len(logical_name) == 0:
            raise CameraUnavailableError(f'Camera {logical_name!r} is not configured')
        if logical_name not in self._statuses:
            raise CameraUnavailableError(f'Camera {logical_name!r} is not configured')
        with self._lock:
            return self._statuses[logical_name]

    def get_statuses(self) -> list[CameraStatus]:
        with self._lock:
            return [self._statuses[name] for name in sorted(self._statuses)]

    def _open_device(self, device_id: str, device: DeviceInfo) -> None:
        capture: CaptureDevice | None = None
        try:
            capture = self._capture_factory.open_capture(device)
            opened = capture.is_opened()
            if not isinstance(opened, bool):
                raise HardwareError('Capture handle returned a malformed open state')
            if not opened:
                self._release_unowned_capture(capture)
                self._set_device_status(
                    device_id,
                    RuntimeStatus.UNAVAILABLE,
                    f'Device {device_id!r} is unavailable',
                    native_resolution=None,
                )
                return

            native_resolution = _get_capture_resolution(
                capture,
                device.native_resolution,
            )
            self._captures[device_id] = capture
            self._set_device_status(
                device_id,
                RuntimeStatus.AVAILABLE,
                None,
                native_resolution,
            )
        except Exception as ex:  # noqa: BLE001 (Opening is a hardware boundary).
            if capture is not None:
                self._release_unowned_capture(capture)
            self._set_device_status(
                device_id,
                RuntimeStatus.ERROR,
                str(ex),
                native_resolution=None,
            )

    def _capture_loop(
        self,
        device_id: str,
        capture: CaptureDevice,
    ) -> None:
        while not self._stop_event.is_set():
            try:
                success, frame_data = _read_capture_frame(capture)
                if not success or frame_data is None:
                    self._set_device_status(
                        device_id,
                        RuntimeStatus.ERROR,
                        f'Camera {device_id!r} did not provide a usable frame',
                    )
                else:
                    with self._lock:
                        previous_frame = self._latest_frames.get(device_id)
                        frame_counter = (
                            1
                            if previous_frame is None
                            else previous_frame.frame_counter + 1
                        )
                        frame = Frame(frame_data, frame_counter, time.time())
                        self._latest_frames[device_id] = frame
                    self._set_device_status(
                        device_id,
                        RuntimeStatus.AVAILABLE,
                        None,
                        frame_counter=frame_counter,
                    )
            except Exception as ex:  # noqa: BLE001 (A read failure must be observable and retried).
                self._set_device_status(device_id, RuntimeStatus.ERROR, str(ex))

            self._stop_event.wait(self._read_wait_seconds)

    def _set_all_errors(self, error_message: str) -> None:
        for logical_name in sorted(self._camera_bindings):
            self._set_status(logical_name, RuntimeStatus.ERROR, error_message)

    def _set_status(
        self,
        logical_name: str,
        runtime_status: RuntimeStatus,
        error_message: str | None,
        native_resolution: Resolution | None | object = _UNSET_NATIVE_RESOLUTION,
        frame_counter: int = 0,
    ) -> None:
        with self._lock:
            status = self._statuses[logical_name]
            self._statuses[logical_name] = status._replace(
                runtime_status=runtime_status,
                native_resolution=(
                    status.native_resolution
                    if native_resolution is _UNSET_NATIVE_RESOLUTION
                    else native_resolution
                ),
                frame_counter=frame_counter if frame_counter > 0 else status.frame_counter,
                error_message=error_message,
            )

    def _set_device_status(
        self,
        device_id: str,
        runtime_status: RuntimeStatus,
        error_message: str | None,
        native_resolution: Resolution | None | object = _UNSET_NATIVE_RESOLUTION,
        frame_counter: int = 0,
    ) -> None:
        for logical_name in self._logical_names_by_device.get(device_id, []):
            self._set_status(
                logical_name,
                runtime_status,
                error_message,
                native_resolution,
                frame_counter,
            )

    @staticmethod
    def _release_unowned_capture(capture: CaptureDevice) -> None:
        try:
            capture.release()
        except Exception:  # noqa: BLE001 (Cleanup must not hide the original failure).
            pass


def _read_capture_frame(capture: CaptureDevice) -> tuple[bool, Any]:
    result = capture.read()
    if not isinstance(result, tuple) or len(result) != 2:
        raise FrameCaptureError('The camera returned a malformed frame result')
    success, frame_data = result
    if not isinstance(success, bool):
        raise FrameCaptureError('The camera returned a malformed frame result')
    return success, frame_data


def _get_capture_resolution(
    capture: CaptureDevice,
    discovered_resolution: Resolution | None,
) -> Resolution:
    get_native_resolution = getattr(capture, 'get_native_resolution', None)
    if callable(get_native_resolution):
        native_resolution = get_native_resolution()
        if not is_valid_resolution(native_resolution):
            raise HardwareError('Capture handle returned no valid native resolution')
        return native_resolution
    if is_valid_resolution(discovered_resolution):
        return discovered_resolution
    raise HardwareError('Capture handle cannot report native resolution')
