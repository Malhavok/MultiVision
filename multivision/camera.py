import math
import threading
import time
from typing import Any

from multivision.errors import (
    CameraUnavailableError,
    FrameCaptureError,
    HardwareError,
    SessionCameraError,
)
from multivision.hardware import (
    CaptureDevice,
    CaptureDeviceFactory,
    DeviceDiscovery,
)
from multivision.session import (
    FrameMetadata,
    SessionCamera,
    SessionCameraRegistry,
)
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
    copy_device_info,
    is_valid_resolution,
)


_UNSET_NATIVE_RESOLUTION = object()


class CameraRuntime:
    """Own open camera handles and publish their latest usable frames."""

    def __init__(
        self,
        discovery: DeviceDiscovery,
        capture_factory: CaptureDeviceFactory,
        read_wait_seconds: float = 0.01,
        worker_shutdown_timeout_seconds: float = 5.0,
    ) -> None:
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
        self._read_wait_seconds = read_wait_seconds
        self._worker_shutdown_timeout_seconds = worker_shutdown_timeout_seconds
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._captures: dict[str, CaptureDevice] = {}
        self._session_registry: SessionCameraRegistry | None = None
        self._latest_frames: dict[str, Frame] = {}
        self._workers: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._statuses: dict[str, CameraStatus] = {}
        self._has_started = False
        self._has_stopped = False

    def start(self) -> None:
        """Discover the fixed session inventory, open active slots and start workers."""
        with self._lifecycle_lock:
            if self._has_stopped:
                raise RuntimeError('The camera runtime has already been shut down')
            if self._has_started:
                return
            self._has_started = True

            prepare_for_runtime = getattr(self._discovery, 'prepare_for_runtime', None)
            if callable(prepare_for_runtime):
                try:
                    prepare_for_runtime()
                except Exception as ex:  # noqa: BLE001 (Discovery is a hardware boundary).
                    raise HardwareError('Could not prepare camera discovery') from ex

            try:
                discovered_devices = self._discovery.discover_devices()
            except Exception as ex:  # noqa: BLE001 (Discovery is a hardware boundary).
                raise HardwareError(f'Could not discover cameras: {ex}') from ex

            if not isinstance(discovered_devices, list):
                raise HardwareError('Discovery returned a malformed device list')
            self._start_session_cameras(discovered_devices)
            self._start_capture_threads()

    def shutdown(self) -> None:
        """Stop workers and release every handle owned by this runtime."""
        with self._lifecycle_lock:
            if self._has_stopped:
                return
            self._stop_event.set()

            release_error: Exception | None = None
            with self._lock:
                for _thread, stop_event in self._workers.values():
                    stop_event.set()
                captures = list(self._captures.values())
                # Remove handles from the live ownership map before workers
                # observe the stop signal, so a worker already handling a
                # failed read cannot release the same handle a second time.
                self._captures.clear()
            for capture in captures:
                try:
                    capture.release()
                except Exception as ex:  # noqa: BLE001 (Cleanup must continue for all handles).
                    if release_error is None:
                        release_error = ex

            shutdown_deadline = (
                time.monotonic() + self._worker_shutdown_timeout_seconds
            )
            unfinished_threads: list[str] = []
            for thread, _stop_event in list(self._workers.values()):
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
                if self._session_registry is not None:
                    for camera in self._session_registry.get_cameras():
                        if camera.state is SessionCameraState.OPEN:
                            self._session_registry.close(camera.slot_id)
                            self._statuses[camera.slot_id] = self._statuses[
                                camera.slot_id
                            ]._replace(
                                calibration_status=CalibrationStatus.UNCALIBRATED,
                            )
                for logical_name, status in self._statuses.items():
                    self._statuses[logical_name] = status._replace(
                        runtime_status=RuntimeStatus.STOPPED,
                        error_message=(
                            str(release_error) if release_error is not None else None
                        ),
                    )
                self._captures.clear()
                self._workers.clear()
                self._has_stopped = True

            if release_error is not None:
                raise HardwareError(
                    'Could not release one or more camera handles',
                ) from release_error

    def get_session_cameras(self) -> list[SessionCamera]:
        """Return the session inventory when runtime discovery owns identity."""
        with self._lifecycle_lock:
            with self._lock:
                if self._session_registry is None:
                    return []
                return self._session_registry.get_cameras()

    def set_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: object,
    ) -> SessionCamera:
        """Update one session slot's in-memory calibration state."""
        with self._lifecycle_lock:
            self._require_session_runtime()
            assert self._session_registry is not None
            with self._lock:
                camera = self._session_registry.set_calibration(
                    slot_id,
                    calibration_status,
                    calibration,
                )
                self._statuses[slot_id] = self._statuses[slot_id]._replace(
                    calibration_status=calibration_status,
                )
                return camera

    def rename_camera(self, slot_id: str, display_name: str) -> SessionCamera:
        """Rename a session slot without changing its owned runtime state."""
        with self._lifecycle_lock:
            self._require_session_runtime()
            assert self._session_registry is not None
            with self._lock:
                return self._session_registry.rename(slot_id, display_name)

    def open_camera(self, slot_id: str) -> SessionCamera:
        """Acquire the fixed startup handle for a closed session slot."""
        with self._lifecycle_lock:
            self._require_session_runtime()
            assert self._session_registry is not None
            capture: CaptureDevice | None = None
            error_message: str | None = None
            with self._lock:
                worker_entry = self._workers.get(slot_id)
                if worker_entry is not None:
                    worker, _stop_event = worker_entry
                    if worker.is_alive():
                        raise HardwareError(
                            f'Camera worker for {slot_id!r} is still stopping',
                        )
                    self._workers.pop(slot_id)
                camera = self._session_registry.open(slot_id)
                self._reset_session_calibration_status(slot_id)
                assert camera.device_info is not None
                device_info = camera.device_info

            if not self._open_device(camera.slot_id, device_info):
                error_message = self._statuses[camera.slot_id].error_message
                with self._lock:
                    self._session_registry.mark_unavailable(
                        camera.slot_id,
                        error_message,
                    )
                    self._set_status(
                        camera.slot_id,
                        RuntimeStatus.UNAVAILABLE,
                        error_message or f'Camera {camera.slot_id!r} is unavailable',
                        frame_counter=0,
                    )
            else:
                with self._lock:
                    capture = self._captures.get(camera.slot_id)

            if error_message is not None or capture is None:
                raise CameraUnavailableError(
                    error_message or f'Camera {camera.slot_id!r} could not be opened',
                )
            self._start_capture_thread(camera.slot_id, capture)
            return camera

    def close_camera(self, slot_id: str) -> SessionCamera:
        """Release a session handle without changing its startup slot mapping."""
        with self._lifecycle_lock:
            self._require_session_runtime()
            assert self._session_registry is not None
            with self._lock:
                camera = self._session_registry.close(slot_id)
                self._reset_session_calibration_status(slot_id)
                capture = self._captures.pop(slot_id, None)
                worker_entry = self._workers.get(slot_id)
                self._latest_frames.pop(slot_id, None)
            if worker_entry is not None:
                worker, stop_event = worker_entry
                stop_event.set()
            else:
                worker = None

            release_error: Exception | None = None
            if capture is not None:
                # Cleanup must continue before surfacing release errors.
                try:
                    capture.release()
                except Exception as ex:  # noqa: BLE001 (release cleanup).
                    release_error = ex

            if worker is not None:
                worker.join(self._worker_shutdown_timeout_seconds)
            if worker is not None and worker.is_alive():
                error_message = (
                    f'Camera worker for {slot_id!r} did not stop within '
                    f'{self._worker_shutdown_timeout_seconds} seconds'
                )
                self._set_status(
                    slot_id,
                    RuntimeStatus.STOPPED,
                    error_message,
                    frame_counter=0,
                )
                self._reset_session_calibration_status(slot_id)
                raise HardwareError(error_message)
            with self._lock:
                self._workers.pop(slot_id, None)
            self._set_status(
                slot_id,
                RuntimeStatus.STOPPED,
                str(release_error) if release_error is not None else None,
                frame_counter=0,
            )
            if release_error is not None:
                raise HardwareError(
                    f'Could not release camera handle for {slot_id!r}',
                ) from release_error
            return camera

    def snapshot(self, logical_name: str) -> Frame:
        """Return the latest usable frame without opening or reading a camera."""
        with self._lifecycle_lock:
            status = self.get_status(logical_name)
            if status.runtime_status in {
                RuntimeStatus.UNAVAILABLE,
                RuntimeStatus.STOPPED,
            }:
                raise CameraUnavailableError(
                    status.error_message or f'Camera {logical_name!r} is unavailable',
                )

            with self._lock:
                frame = self._latest_frames.get(logical_name)
            if frame is None:
                raise FrameCaptureError(
                    status.error_message or f'Camera {logical_name!r} has no usable frame',
                )
            return frame

    def get_discovered_devices(self) -> list[DeviceInfo]:
        """Return the fixed session snapshot without opening another device."""
        with self._lifecycle_lock:
            with self._lock:
                if self._session_registry is None:
                    return []
                return [
                    copy_device_info(camera.device_info)
                    for camera in self._session_registry.get_cameras()
                    if camera.device_info is not None
                ]

    def get_status(self, logical_name: str) -> CameraStatus:
        if not isinstance(logical_name, str) or len(logical_name) == 0:
            raise CameraUnavailableError(f'Camera {logical_name!r} is not configured')
        with self._lifecycle_lock:
            with self._lock:
                if logical_name not in self._statuses:
                    raise CameraUnavailableError(f'Camera {logical_name!r} is not configured')
                return self._statuses[logical_name]

    def get_statuses(self) -> list[CameraStatus]:
        with self._lifecycle_lock:
            with self._lock:
                if self._session_registry is not None:
                    return [
                        self._statuses[camera.slot_id]
                        for camera in self._session_registry.get_cameras()
                    ]
                return [self._statuses[name] for name in sorted(self._statuses)]

    def _start_session_cameras(self, discovered_devices: list[DeviceInfo]) -> None:
        try:
            try:
                registry = SessionCameraRegistry.from_devices(discovered_devices)
            except SessionCameraError as ex:
                raise HardwareError('Discovery returned invalid session camera data') from ex

            with self._lock:
                self._session_registry = registry
                cameras = registry.get_cameras()
                self._statuses = {
                    camera.slot_id: CameraStatus(
                        logical_name=camera.slot_id,
                        device_id=None,
                        runtime_status={
                            SessionCameraState.UNAVAILABLE: RuntimeStatus.UNAVAILABLE,
                            SessionCameraState.CLOSED: RuntimeStatus.STOPPED,
                            SessionCameraState.OPEN: RuntimeStatus.STARTING,
                        }[camera.state],
                        calibration_status=CalibrationStatus.UNCALIBRATED,
                        native_resolution=(
                            camera.device_info.native_resolution
                            if camera.device_info is not None
                            else None
                        ),
                        error_message=camera.error_message,
                    )
                    for camera in cameras
                }

            for camera in cameras:
                if camera.state is SessionCameraState.UNAVAILABLE:
                    self._set_status(
                        camera.slot_id,
                        RuntimeStatus.UNAVAILABLE,
                        camera.error_message or f'Camera {camera.slot_id!r} is unavailable',
                    )
                    continue
                if camera.state is SessionCameraState.CLOSED:
                    continue
                assert camera.device_info is not None
                if self._open_device(camera.slot_id, camera.device_info):
                    continue
                error_message = self.get_status(camera.slot_id).error_message
                with self._lock:
                    registry.mark_unavailable(camera.slot_id, error_message)
                self._set_status(
                    camera.slot_id,
                    RuntimeStatus.UNAVAILABLE,
                    error_message or f'Camera {camera.slot_id!r} is unavailable',
                )
        finally:
            self._release_unused_discovery_captures()

    def _start_capture_threads(self) -> None:
        for device_id, capture in list(self._captures.items()):
            self._start_capture_thread(device_id, capture)

    def _start_capture_thread(
        self,
        device_id: str,
        capture: CaptureDevice,
    ) -> None:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._capture_loop,
            args=(device_id, capture, stop_event),
            daemon=True,
            name=f'multivision-camera-{device_id}',
        )
        with self._lock:
            self._workers[device_id] = (thread, stop_event)
        thread.start()

    def _open_device(self, device_id: str, device: DeviceInfo) -> bool:
        capture: CaptureDevice | None = None
        try:
            capture = self._open_capture(device)
            opened = capture.is_opened()
            if not isinstance(opened, bool):
                raise HardwareError('Capture handle returned a malformed open state')
            if not opened:
                self._release_unowned_capture(capture)
                self._set_status(
                    device_id,
                    RuntimeStatus.UNAVAILABLE,
                    f'Device {device_id!r} is unavailable',
                    native_resolution=None,
                )
                return False

            native_resolution = _get_capture_resolution(
                capture,
                device.native_resolution,
            )
            with self._lock:
                self._captures[device_id] = capture
            self._set_status(
                device_id,
                RuntimeStatus.AVAILABLE,
                None,
                native_resolution,
            )
            return True
        except Exception as ex:  # noqa: BLE001 (Opening is a hardware boundary).
            if capture is not None:
                self._release_unowned_capture(capture)
            self._set_status(
                device_id,
                RuntimeStatus.ERROR,
                str(ex),
                native_resolution=None,
            )
            return False

    def _capture_loop(
        self,
        device_id: str,
        capture: CaptureDevice,
        stop_event: threading.Event,
    ) -> None:
        while not self._stop_event.is_set() and not stop_event.is_set():
            session_failure_message: str | None = None
            try:
                opened = capture.is_opened()
                if not isinstance(opened, bool):
                    raise HardwareError(
                        'Capture handle returned a malformed open state',
                    )
                if not opened:
                    session_failure_message = (
                        f'Camera {device_id!r} became unavailable: '
                        'the capture handle is closed'
                    )

                if session_failure_message is None:
                    success, frame_data = _read_capture_frame(capture)
                    if not success or frame_data is None:
                        session_failure_message = (
                            f'Camera {device_id!r} became unavailable: '
                            'it did not provide a usable frame'
                        )

                with self._lock:
                    if not self._is_current_capture_locked(
                        device_id,
                        capture,
                        stop_event,
                    ):
                        return
                    if session_failure_message is None:
                        captured_at_seconds = time.time()
                        previous_frame = self._latest_frames.get(device_id)
                        frame_counter = (
                            1
                            if previous_frame is None
                            else previous_frame.frame_counter + 1
                        )
                        self._latest_frames[device_id] = Frame(
                            frame_data,
                            frame_counter,
                            captured_at_seconds,
                        )
                        self._set_status(
                            device_id,
                            RuntimeStatus.AVAILABLE,
                            None,
                            frame_counter=frame_counter,
                        )
                        native_resolution = self._statuses[device_id].native_resolution
                        if native_resolution is not None:
                            self._session_registry.set_frame_metadata(
                                device_id,
                                FrameMetadata(
                                    frame_counter,
                                    captured_at_seconds,
                                    native_resolution,
                                ),
                            )
            except Exception as ex:  # noqa: BLE001 (A read failure is a hardware boundary).
                session_failure_message = (
                    f'Camera {device_id!r} became unavailable: capture failed: {ex}'
                )

            if session_failure_message is not None:
                self._handle_session_capture_failure(
                    device_id,
                    capture,
                    stop_event,
                    session_failure_message,
                )
                return

            stop_event.wait(self._read_wait_seconds)

    def _handle_session_capture_failure(
        self,
        device_id: str,
        capture: CaptureDevice,
        stop_event: threading.Event,
        error_message: str,
    ) -> None:
        with self._lock:
            if not self._is_current_capture_locked(
                device_id,
                capture,
                stop_event,
            ):
                return
            self._captures.pop(device_id)
            self._latest_frames.pop(device_id, None)
            assert self._session_registry is not None
            self._session_registry.mark_unavailable(device_id, error_message)
            self._set_status(
                device_id,
                RuntimeStatus.UNAVAILABLE,
                error_message,
                frame_counter=0,
            )
            self._reset_session_calibration_status(device_id)
            stop_event.set()

        try:
            capture.release()
        except Exception as ex:  # noqa: BLE001 (Disconnect cleanup must not mask unavailability).
            release_error_message = (
                f'{error_message}; could not release the capture handle: {ex}'
            )
            with self._lock:
                assert self._session_registry is not None
                self._session_registry.mark_unavailable(
                    device_id,
                    release_error_message,
                )
                self._set_status(
                    device_id,
                    RuntimeStatus.UNAVAILABLE,
                    release_error_message,
                    frame_counter=0,
                )
                self._reset_session_calibration_status(device_id)

    def _is_current_capture_locked(
        self,
        device_id: str,
        capture: CaptureDevice,
        stop_event: threading.Event,
    ) -> bool:
        return (
            not self._stop_event.is_set()
            and not stop_event.is_set()
            and self._captures.get(device_id) is capture
        )

    def _reset_session_calibration_status(self, slot_id: str) -> None:
        with self._lock:
            self._statuses[slot_id] = self._statuses[slot_id]._replace(
                calibration_status=CalibrationStatus.UNCALIBRATED,
            )

    def _require_session_runtime(self) -> None:
        if self._session_registry is None:
            raise SessionCameraError('Session camera inventory is not available')
        if not self._has_started:
            raise RuntimeError('The camera runtime has not been started')
        if self._has_stopped:
            raise RuntimeError('The camera runtime has already been shut down')

    def _set_status(
        self,
        logical_name: str,
        runtime_status: RuntimeStatus,
        error_message: str | None,
        native_resolution: Resolution | None | object = _UNSET_NATIVE_RESOLUTION,
        frame_counter: int | None = None,
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
                frame_counter=(
                    status.frame_counter
                    if frame_counter is None
                    else frame_counter
                ),
                error_message=error_message,
            )

    def _open_capture(self, device: DeviceInfo) -> CaptureDevice:
        take_capture = getattr(self._discovery, 'take_capture', None)
        if callable(take_capture) and device.capture_index is not None:
            discovered_capture = take_capture(device.capture_index)
            if discovered_capture is not None:
                adopt_capture = getattr(self._capture_factory, 'adopt_capture', None)
                if not callable(adopt_capture):
                    self._release_unowned_capture(discovered_capture)
                    raise HardwareError(
                        'Capture factory cannot adopt the startup discovery probe',
                    )
                try:
                    capture = adopt_capture(discovered_capture)
                except Exception:
                    self._release_unowned_capture(discovered_capture)
                    raise
                if capture is None:
                    self._release_unowned_capture(discovered_capture)
                    raise HardwareError('Capture factory returned no adopted handle')
                return capture
        return self._capture_factory.open_capture(device)

    def _release_unused_discovery_captures(self) -> None:
        release_unused_captures = getattr(
            self._discovery,
            'release_unused_captures',
            None,
        )
        if not callable(release_unused_captures):
            return
        try:
            release_unused_captures()
        except Exception as ex:  # noqa: BLE001 (Discovery is a hardware boundary).
            raise HardwareError('Could not release unused camera probes') from ex

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
        if native_resolution is not None and not is_valid_resolution(native_resolution):
            raise HardwareError('Capture handle returned an invalid native resolution')
        if is_valid_resolution(native_resolution):
            return native_resolution
    if is_valid_resolution(discovered_resolution):
        return discovered_resolution
    raise HardwareError('Capture handle cannot report native resolution')
