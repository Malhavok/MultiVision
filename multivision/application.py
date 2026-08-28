"""Persistent service composition for MultiVision capabilities."""

from __future__ import annotations

import pathlib
import threading
import time
from collections.abc import Sequence

from multivision.calibration import CalibrationMetrics, calibrate_homography
from multivision.camera import CameraRuntime
from multivision.config import (
    Configuration,
    load_configuration,
    save_configuration,
    validate_camera_bindings,
)
from multivision.discovery import PlatformDeviceDiscovery
from multivision.errors import (
    CalibrationError,
    CameraUnavailableError,
    FrameCaptureError,
)
from multivision.geometry import Point2D, PreviewTransform
from multivision.fiducials import (
    CameraCorrespondences,
    FiducialCorrespondence,
    FiducialDetector,
    OpenCVArucoDetector,
    assemble_correspondences,
    detect_fiducials,
)
from multivision.hardware import (
    CaptureDeviceFactory,
    DeviceDiscovery,
    OpenCVCaptureDeviceFactory,
)
from multivision.pattern import CalibrationPattern, build_calibration_pattern
from multivision.persistence import (
    CalibrationRegistry,
    CalibrationStore,
    PersistedCalibration,
)
from multivision.service import PointOverlayService, RedCircleOverlay
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    RuntimeStatus,
    is_valid_resolution,
)


CALIBRATION_PATTERN_WAIT_TIMEOUT_SECONDS = 5.0
CALIBRATION_PATTERN_SETTLE_SECONDS = 0.1


class MultiVisionService:
    """Own the persistent camera, calibration and overlay capabilities."""

    def __init__(
        self,
        configuration: Configuration | None = None,
        *,
        config_path: pathlib.Path | None = None,
        discovery: DeviceDiscovery | None = None,
        capture_factory: CaptureDeviceFactory | None = None,
        camera_runtime: CameraRuntime | None = None,
        calibration_store: CalibrationStore | None = None,
        calibration_registry: CalibrationRegistry | None = None,
        detector: FiducialDetector | None = None,
        calibration_pattern: CalibrationPattern | None = None,
        point_service: PointOverlayService | None = None,
    ) -> None:
        if calibration_store is not None and not isinstance(
            calibration_store,
            CalibrationStore,
        ):
            raise ValueError('calibration_store must be CalibrationStore')
        if config_path is not None and not isinstance(config_path, pathlib.Path):
            raise ValueError('config_path must be a pathlib.Path')
        if calibration_store is not None and config_path is not None:
            if calibration_store.path.resolve() != config_path.resolve():
                raise ValueError(
                    'config_path and calibration_store.path must refer to the same file',
                )

        effective_config_path = (
            calibration_store.path
            if calibration_store is not None and config_path is None
            else config_path
        )
        if configuration is None:
            configuration = load_configuration(effective_config_path)
        if not isinstance(configuration, Configuration):
            raise ValueError('configuration must be Configuration')
        self.configuration = configuration
        self._config_path = effective_config_path

        self.calibration_store = (
            calibration_store
            if calibration_store is not None
            else CalibrationStore(effective_config_path)
        )
        self.calibration_pattern = (
            calibration_pattern
            if calibration_pattern is not None
            else build_calibration_pattern(configuration.projector_resolution)
        )
        if not isinstance(self.calibration_pattern, CalibrationPattern):
            raise ValueError('calibration_pattern must be CalibrationPattern')
        self.calibration_registry = (
            calibration_registry
            if calibration_registry is not None
            else CalibrationRegistry.from_store(
                self.calibration_store,
                calibration_version=configuration.calibration_version,
                projector_resolution=configuration.projector_resolution,
            )
        )

        self.camera_runtime = (
            camera_runtime
            if camera_runtime is not None
            else CameraRuntime(
                discovery if discovery is not None else PlatformDeviceDiscovery(),
                capture_factory
                if capture_factory is not None
                else OpenCVCaptureDeviceFactory(),
                configuration.camera_bindings,
            )
        )
        self.point_service = (
            point_service
            if point_service is not None
            else PointOverlayService(
                self.camera_runtime,
                self.calibration_registry,
                configuration.projector_resolution,
            )
        )
        self.detector = detector
        self._lifecycle_lock = threading.RLock()
        self._calibration_capture_count = 0
        self._calibration_capture_lock = threading.RLock()
        self._calibration_pattern_presented = threading.Event()
        self._is_running = False
        self._has_stopped = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def overlay(self) -> RedCircleOverlay | None:
        return self.point_service.overlay

    @property
    def calibration_pattern_visible(self) -> bool:
        with self._calibration_capture_lock:
            return self._calibration_capture_count > 0

    def mark_calibration_pattern_presented(self) -> None:
        """Acknowledge a successful main-thread pattern presentation."""
        with self._calibration_capture_lock:
            if self._calibration_capture_count > 0:
                self._calibration_pattern_presented.set()

    def start(self) -> None:
        """Start persistent capture ownership once for the service lifetime."""
        with self._lifecycle_lock:
            if self._has_stopped:
                raise RuntimeError('The MultiVision service has already stopped')
            if self._is_running:
                return
            self.camera_runtime.start()
            self._is_running = True

    def shutdown(self) -> None:
        """Stop capture workers and release every camera handle."""
        with self._lifecycle_lock:
            if self._has_stopped:
                return
            shutdown_error: Exception | None = None
            try:
                self.camera_runtime.shutdown()
            except Exception as ex:  # noqa: BLE001 (Cleanup must complete before surfacing errors).
                shutdown_error = ex
            self._is_running = False
            self._has_stopped = True
            if shutdown_error is not None:
                raise shutdown_error

    def get_camera_status(self, logical_name: str) -> CameraStatus:
        runtime_status = self.camera_runtime.get_status(logical_name)
        if not isinstance(runtime_status, CameraStatus):
            raise CameraUnavailableError(
                f'Camera {logical_name!r} returned an invalid status',
            )
        _validate_camera_status(runtime_status, logical_name)
        calibration_status = self._get_calibration_status(runtime_status)
        return runtime_status._replace(calibration_status=calibration_status)

    def get_discovered_devices(self) -> list[DeviceInfo]:
        devices = self.camera_runtime.get_discovered_devices()
        if not isinstance(devices, list) or any(
            not isinstance(device, DeviceInfo)
            for device in devices
        ):
            raise CameraUnavailableError('Camera runtime returned an invalid discovery list')
        return devices

    def get_camera_statuses(self) -> list[CameraStatus]:
        statuses = self.camera_runtime.get_statuses()
        if not isinstance(statuses, list):
            raise CameraUnavailableError('Camera runtime returned an invalid status list')
        checked_statuses: list[CameraStatus] = []
        for status in statuses:
            if not isinstance(status, CameraStatus):
                raise CameraUnavailableError(
                    'Camera runtime returned an invalid camera status',
                )
            _validate_camera_status(status)
            calibration_status = self._get_calibration_status(status)
            checked_statuses.append(status._replace(calibration_status=calibration_status))
        return checked_statuses

    def snapshot(self, logical_name: str) -> Frame:
        """Return the latest frame retained by the persistent camera runtime."""
        frame = self.camera_runtime.snapshot(logical_name)
        if not isinstance(frame, Frame):
            raise FrameCaptureError(
                f'Camera {logical_name!r} returned an invalid frame',
            )
        return frame

    def bind_camera(self, logical_name: str, device_id: str) -> dict[str, str | bool]:
        """Persist a logical camera binding for the next service startup."""
        candidate_bindings = dict(self.configuration.camera_bindings)
        candidate_bindings[logical_name] = device_id
        validate_camera_bindings(candidate_bindings)
        configuration = Configuration(
            camera_bindings=candidate_bindings,
            projector_resolution=self.configuration.projector_resolution,
            calibration_thresholds=self.configuration.calibration_thresholds,
            calibration_version=self.configuration.calibration_version,
        )
        save_configuration(configuration, self._config_path)
        self.configuration = configuration
        return {
            'bound': True,
            'camera': logical_name,
            'device_id': device_id,
            'restart_required': True,
        }

    def calibrate(
        self,
        logical_name: str | None = None,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> PersistedCalibration | dict[str, PersistedCalibration]:
        if logical_name is not None:
            return self._calibrate_camera(logical_name, correspondences)
        if correspondences is not None:
            raise CalibrationError('camera is required when correspondences are supplied')
        statuses = self._get_available_statuses()
        if len(statuses) == 0:
            raise CameraUnavailableError('No configured camera is available for calibration')
        return {
            status.logical_name: self._calibrate_camera(status.logical_name)
            for status in statuses
        }

    def verify(
        self,
        logical_name: str | None = None,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> CalibrationStatus | dict[str, CalibrationStatus]:
        if logical_name is not None:
            return self._verify_camera(logical_name, correspondences)
        if correspondences is not None:
            raise CalibrationError('camera is required when correspondences are supplied')
        statuses = self._get_available_statuses()
        if len(statuses) == 0:
            raise CameraUnavailableError('No configured camera is available for verification')
        return {
            status.logical_name: self._verify_camera(status.logical_name)
            for status in statuses
        }

    def point_from_preview(
        self,
        logical_name: str,
        preview_point: Point2D,
        preview_transform: PreviewTransform,
    ) -> RedCircleOverlay:
        return self.point_service.point_from_preview(
            logical_name,
            preview_point,
            preview_transform,
        )

    def point_from_camera(
        self,
        logical_name: str,
        camera_point: Sequence[float],
    ) -> RedCircleOverlay:
        return self.point_service.point_from_camera(logical_name, camera_point)

    def get_calibration_metrics(self, logical_name: str) -> CalibrationMetrics | None:
        status = self.get_camera_status(logical_name)
        if status.device_id is None:
            return None
        calibration = self.calibration_registry.get_record(status.device_id)
        if calibration is None:
            return None
        return calibration.metrics

    def get_calibration_records(self) -> dict[str, PersistedCalibration]:
        return self.calibration_registry.get_records()

    def clear_overlay(self) -> None:
        self.point_service.clear_overlay()

    def _calibrate_camera(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> PersistedCalibration:
        status = self._require_available_camera(logical_name)
        if status.device_id is None or status.native_resolution is None:
            raise CameraUnavailableError(f'Camera {logical_name!r} has incomplete metadata')
        checked_correspondences = self._get_correspondences_for_operation(
            status,
            correspondences,
        )
        result = calibrate_homography(
            checked_correspondences,
            self.calibration_pattern,
            self.configuration.calibration_thresholds,
            camera_resolution=status.native_resolution,
        )
        record = PersistedCalibration.from_result(
            result,
            status.native_resolution,
            self.configuration.projector_resolution,
            version=self.configuration.calibration_version,
            camera_id=status.device_id,
        )
        self.calibration_registry.save(self.calibration_store, record)
        return record

    def _verify_camera(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> CalibrationStatus:
        status = self._require_available_camera(logical_name)
        if status.device_id is None or status.native_resolution is None:
            raise CameraUnavailableError(f'Camera {logical_name!r} has incomplete metadata')
        checked_correspondences = self._get_correspondences_for_operation(
            status,
            correspondences,
        )
        return self.calibration_registry.verify(
            status.device_id,
            checked_correspondences,
            camera_resolution=status.native_resolution,
            projector_resolution=self.configuration.projector_resolution,
            thresholds=self.configuration.calibration_thresholds,
            pattern=self.calibration_pattern,
        )

    def _get_available_statuses(self) -> list[CameraStatus]:
        return [
            status
            for status in self.get_camera_statuses()
            if status.runtime_status is RuntimeStatus.AVAILABLE
        ]

    def _require_available_camera(self, logical_name: str) -> CameraStatus:
        status = self.get_camera_status(logical_name)
        if status.runtime_status is not RuntimeStatus.AVAILABLE:
            raise CameraUnavailableError(
                status.error_message or f'Camera {logical_name!r} is unavailable',
            )
        return status

    def _get_correspondences_for_operation(
        self,
        status: CameraStatus,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None,
    ) -> CameraCorrespondences:
        if correspondences is not None:
            return self._get_correspondences(status, correspondences)
        with self._calibration_capture_lock:
            self._calibration_pattern_presented.clear()
            self._calibration_capture_count += 1
        try:
            if not self._calibration_pattern_presented.wait(
                CALIBRATION_PATTERN_WAIT_TIMEOUT_SECONDS,
            ):
                raise CalibrationError(
                    'The calibration pattern was not presented by the main-thread display',
                )
            time.sleep(CALIBRATION_PATTERN_SETTLE_SECONDS)
            return self._get_correspondences(status, None)
        finally:
            with self._calibration_capture_lock:
                self._calibration_capture_count -= 1
                if self._calibration_capture_count == 0:
                    self._calibration_pattern_presented.clear()

    def _get_correspondences(
        self,
        status: CameraStatus,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None,
    ) -> CameraCorrespondences:
        assert status.device_id is not None
        if correspondences is not None:
            if isinstance(correspondences, CameraCorrespondences):
                if correspondences.camera_id not in {None, status.device_id}:
                    raise CalibrationError('Correspondences belong to another camera')
                return correspondences._replace(camera_id=status.device_id)
            try:
                values = tuple(correspondences)
            except (TypeError, ValueError) as ex:
                raise CalibrationError('correspondences must be iterable') from ex
            return CameraCorrespondences(values, status.device_id)

        frame = self.snapshot(status.logical_name)
        detector = self.detector
        if detector is None:
            detector = OpenCVArucoDetector()
        detected_markers = detect_fiducials(frame.data, detector)
        return assemble_correspondences(
            detected_markers,
            self.calibration_pattern,
            camera_id=status.device_id,
        )

    def _get_calibration_status(self, status: CameraStatus) -> CalibrationStatus:
        if status.device_id is None:
            return CalibrationStatus.UNCALIBRATED
        calibration_status = self.calibration_registry.get_status(
            status.device_id,
            camera_resolution=status.native_resolution,
            projector_resolution=self.configuration.projector_resolution,
        )
        if not isinstance(calibration_status, CalibrationStatus):
            raise CalibrationError(
                f'Camera {status.logical_name!r} returned an invalid calibration status',
            )
        return calibration_status


def _validate_camera_status(
    status: CameraStatus,
    expected_logical_name: str | None = None,
) -> None:
    if (
        not isinstance(status.logical_name, str)
        or len(status.logical_name) == 0
        or (
            expected_logical_name is not None
            and status.logical_name != expected_logical_name
        )
        or (
            status.device_id is not None
            and (
                not isinstance(status.device_id, str)
                or len(status.device_id) == 0
            )
        )
        or not isinstance(status.runtime_status, RuntimeStatus)
        or not isinstance(status.calibration_status, CalibrationStatus)
        or (
            status.native_resolution is not None
            and not is_valid_resolution(status.native_resolution)
        )
        or not isinstance(status.frame_counter, int)
        or isinstance(status.frame_counter, bool)
        or status.frame_counter < 0
        or (
            status.error_message is not None
            and not isinstance(status.error_message, str)
        )
    ):
        if expected_logical_name is None:
            raise CameraUnavailableError('Camera runtime returned an invalid camera status')
        raise CameraUnavailableError(
            f'Camera {expected_logical_name!r} returned an invalid status',
        )


__all__ = ['MultiVisionService']
