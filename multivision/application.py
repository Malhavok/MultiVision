"""Service composition for MultiVision capabilities."""

from __future__ import annotations

import math
import pathlib
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple

from multivision.calibration import CalibrationMetrics, calibrate_homography
from multivision.camera import CameraRuntime
from multivision.config import (
    Configuration,
    ProjectorOutputDescriptor,
    load_configuration,
)
from multivision.discovery import PlatformDeviceDiscovery
from multivision.errors import (
    CalibrationError,
    CameraSlotNotFoundError,
    CameraUnavailableError,
    FrameCaptureError,
    InvalidAvailableAreaError,
    InvalidCalibrationStateError,
    SessionCameraError,
)
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    PointLike,
    Polygon,
    PreviewTransform,
    calculate_available_projector_area,
    validate_homography,
)
from multivision.fiducials import (
    CameraCorrespondences,
    FiducialCorrespondence,
    FiducialDetector,
    MetricTargetCorrespondence,
    MetricTargetCorrespondences,
    OpenCVArucoDetector,
    assemble_correspondences,
    detect_and_assemble_metric_correspondences,
    detect_fiducials,
)
from multivision.hardware import (
    CaptureDeviceFactory,
    DeviceDiscovery,
    OpenCVCaptureDeviceFactory,
    SleepInhibitor,
    SystemSleepInhibitor,
)
from multivision.metric import (
    MetricCalibrationRecord,
    MetricCalibrationRegistry,
    MetricCalibrationStatus,
    MetricRulerOverlay,
    MetricValidationRecord,
    build_metric_ruler,
    calibrate_metric_homography,
    validate_metric_correspondences,
    validate_positive_length,
)
from multivision.metric_target import METRIC_TARGET
from multivision.pattern import CalibrationPattern, build_calibration_pattern
from multivision.persistence import (
    CalibrationRegistry,
    CalibrationStore,
    PersistedCalibration,
)
from multivision.service import PointOverlayService, RedCircleOverlay
from multivision.session import SessionCamera
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    RuntimeStatus,
    Resolution,
    SessionCameraState,
    is_valid_resolution,
)


CALIBRATION_PATTERN_WAIT_TIMEOUT_SECONDS = 5.0
CALIBRATION_PATTERN_SETTLE_SECONDS = 3.0
CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS = 5.0
METRIC_CAPTURE_WAIT_TIMEOUT_SECONDS = 5.0
METRIC_CAPTURE_SETTLE_SECONDS = 3.0
METRIC_CAPTURE_FRAME_COUNT = 3
AREA_COLOURS = (
    (70, 190, 255),
    (255, 180, 70),
    (180, 100, 255),
    (80, 220, 150),
)


class CameraArea(NamedTuple):
    """Current session-local visibility and derived projector area."""

    slot_id: str
    display_name: str
    area_enabled: bool
    available_area: Polygon | None
    area_colour: tuple[int, int, int]


class MultiVisionService:
    """Own camera, calibration and overlay capabilities for one service run."""

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
        sleep_inhibitor: SleepInhibitor | None = None,
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
        self._projector_output_descriptor = configuration.projector_output_descriptor
        self.metric_calibration_registry = MetricCalibrationRegistry(
            self._projector_output_descriptor,
        )
        # The ruler is deliberately separate from the shared calibration record.
        self._metric_ruler: MetricRulerOverlay | None = None

        self.calibration_store = (
            calibration_store
            if calibration_store is not None
            else CalibrationStore(effective_config_path)
        )
        self.calibration_pattern = (
            calibration_pattern
            if calibration_pattern is not None
            else build_calibration_pattern(
                self._projector_output_descriptor.projector_resolution,
            )
        )
        if not isinstance(self.calibration_pattern, CalibrationPattern):
            raise ValueError('calibration_pattern must be CalibrationPattern')
        self.camera_runtime = (
            camera_runtime
            if camera_runtime is not None
            else CameraRuntime(
                discovery if discovery is not None else PlatformDeviceDiscovery(),
                capture_factory
                if capture_factory is not None
                else OpenCVCaptureDeviceFactory(),
            )
        )
        self.calibration_registry = (
            calibration_registry
            if calibration_registry is not None
            else CalibrationRegistry(
                calibration_version=self.configuration.calibration_version,
                projector_resolution=self._projector_output_descriptor.projector_resolution,
                projector_output_descriptor=self._projector_output_descriptor,
            )
        )
        self.point_service = (
            point_service
            if point_service is not None
            else PointOverlayService(
                self.camera_runtime,
                self.calibration_registry,
                self._projector_output_descriptor.projector_resolution,
                calibration_version=self.configuration.calibration_version,
                projector_output_descriptor=self._projector_output_descriptor,
            )
        )
        self.point_service.projector_output_descriptor = self._projector_output_descriptor
        self.detector = detector
        self.sleep_inhibitor = (
            sleep_inhibitor
            if sleep_inhibitor is not None
            else SystemSleepInhibitor()
        )
        self._lifecycle_lock = threading.RLock()
        self._camera_management_lock = threading.RLock()
        self._calibration_capture_count = 0
        self._calibration_capture_lock = threading.RLock()
        self._calibration_pattern_presented = threading.Event()
        self._metric_capture_count = 0
        self._metric_capture_lock = threading.RLock()
        # Camera-pattern and metric-blank handshakes must never overlap.
        self._spatial_capture_operation_lock = threading.RLock()
        self._metric_capture_generation = 0
        self._metric_blank_presented = threading.Event()
        self._is_running = False
        self._has_stopped = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def projector_output_descriptor(self) -> ProjectorOutputDescriptor:
        with self._camera_management_lock:
            return self._projector_output_descriptor

    @property
    def metric_calibration(self) -> MetricCalibrationRecord | None:
        """Return the one shared session-local metric record, if present."""
        with self._camera_management_lock:
            return self.metric_calibration_registry.record

    @property
    def metric_registry(self) -> MetricCalibrationRegistry:
        """Return the service-owned shared metric registry."""
        return self.metric_calibration_registry

    @property
    def metric_state(self) -> MetricCalibrationStatus:
        """Return the shared metric calibration state for diagnostics."""
        with self._camera_management_lock:
            return self.metric_calibration_registry.state

    @property
    def metric_ruler(self) -> MetricRulerOverlay | None:
        """Return the session-local ruler, if it remains spatially usable."""
        with self._camera_management_lock:
            if self._metric_ruler is None:
                return None
            if not self.metric_calibration_registry.is_usable(
                self._projector_output_descriptor,
            ):
                self._metric_ruler = None
            return self._metric_ruler

    def set_metric_ruler(
        self,
        surface_start: PointLike,
        surface_end: PointLike,
        unit: object = 'mm',
    ) -> MetricRulerOverlay:
        """Replace the ruler only after its complete geometry is validated."""
        ruler, _validation = self.set_metric_ruler_with_validation(
            surface_start,
            surface_end,
            unit,
        )
        return ruler

    def set_metric_ruler_with_validation(
        self,
        surface_start: PointLike,
        surface_end: PointLike,
        unit: object = 'mm',
        observed_length: object | None = None,
        observed_unit: object = 'mm',
    ) -> tuple[MetricRulerOverlay, MetricValidationRecord | None]:
        """Replace a ruler and optionally record its physical observation atomically."""
        with self._camera_management_lock:
            record = self._require_usable_metric_calibration()
            assert record.surface_to_projector is not None
            ruler = build_metric_ruler(
                surface_start,
                surface_end,
                unit,
                record.surface_to_projector,
                self._projector_output_descriptor.projector_resolution,
            )
            validation = None
            if observed_length is not None:
                validation = self.metric_calibration_registry.add_validation_record(
                    ruler.length_mm,
                    observed_length,
                    'mm',
                    observed_unit,
                )
            self._metric_ruler = ruler
            return ruler, validation

    def clear_metric_ruler(self) -> None:
        """Remove the ruler without changing metric or point-overlay state."""
        with self._camera_management_lock:
            self._metric_ruler = None

    def get_metric_status(self) -> MetricCalibrationStatus:
        """Return a fail-closed status for the current projector descriptor."""
        status, _record = self.get_metric_status_snapshot()
        return status

    def get_metric_status_snapshot(
        self,
    ) -> tuple[MetricCalibrationStatus, MetricCalibrationRecord | None]:
        """Return one consistent metric status and record snapshot."""
        with self._camera_management_lock:
            status = self.metric_calibration_registry.get_status(
                self._projector_output_descriptor,
            )
            return status, self.metric_calibration_registry.record

    def clear_metric_calibration(self) -> None:
        """Clear the shared metric transform and its independent ruler state."""
        with self._camera_management_lock:
            self._metric_capture_generation += 1
            self.metric_calibration_registry.clear()
            self._metric_ruler = None

    def record_physical_validation(
        self,
        requested_length: object,
        observed_length: object | None = None,
        requested_unit: object = 'mm',
        observed_unit: object = 'mm',
    ) -> MetricValidationRecord | None:
        """Record an optional physical observation without changing the transform."""
        with self._camera_management_lock:
            self._require_usable_metric_calibration()
            if observed_length is None:
                validate_positive_length(requested_length, requested_unit)
                return None
            return self.metric_calibration_registry.add_validation_record(
                requested_length,
                observed_length,
                requested_unit,
                observed_unit,
            )

    @property
    def overlay(self) -> RedCircleOverlay | None:
        overlay = self.point_service.overlay
        if overlay is None:
            return None
        camera = self._get_session_camera(overlay.camera_id)
        if camera is not None and camera.state is not SessionCameraState.OPEN:
            self.point_service.clear_overlay_for_camera(overlay.camera_id)
            return None
        return overlay

    @property
    def calibration_pattern_visible(self) -> bool:
        with self._calibration_capture_lock:
            return self._calibration_capture_count > 0

    def mark_calibration_pattern_presented(self) -> None:
        """Acknowledge a successful main-thread pattern presentation."""
        with self._calibration_capture_lock:
            if self._calibration_capture_count > 0:
                self._calibration_pattern_presented.set()

    @property
    def metric_capture_active(self) -> bool:
        """Return whether the main-thread projector must remain blank."""
        with self._metric_capture_lock:
            return self._metric_capture_count > 0

    def mark_metric_capture_presented(self) -> None:
        """Acknowledge a main-thread blank projector frame."""
        with self._metric_capture_lock:
            if self._metric_capture_count > 0:
                self._metric_blank_presented.set()

    def start(self) -> None:
        """Start persistent capture ownership once for the service lifetime."""
        with self._lifecycle_lock:
            if self._has_stopped:
                raise RuntimeError('The MultiVision service has already stopped')
            if self._is_running:
                return
            self.camera_runtime.start()
            try:
                self.sleep_inhibitor.start()
            except Exception:
                self.camera_runtime.shutdown()
                raise
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
            try:
                self.sleep_inhibitor.stop()
            except Exception as ex:  # noqa: BLE001 (Cleanup must complete before surfacing errors).
                if shutdown_error is None:
                    shutdown_error = ex
            self._is_running = False
            self._has_stopped = True
            if shutdown_error is not None:
                raise shutdown_error

    def get_camera_status(self, logical_name: str) -> CameraStatus:
        with self._camera_management_lock:
            resolved_slot = self._resolve_camera_reference(logical_name)
            runtime_status = self.camera_runtime.get_status(resolved_slot)
            if not isinstance(runtime_status, CameraStatus):
                raise CameraUnavailableError(
                    f'Camera {logical_name!r} returned an invalid status',
                )
            _validate_camera_status(runtime_status, resolved_slot)
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
        with self._camera_management_lock:
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
        frame = self.camera_runtime.snapshot(self._resolve_camera_reference(logical_name))
        if not isinstance(frame, Frame):
            raise FrameCaptureError(
                f'Camera {logical_name!r} returned an invalid frame',
            )
        return frame

    def get_session_cameras(self) -> list[SessionCamera]:
        """Return the fixed, deterministically ordered session camera inventory."""
        # The display must read snapshots while calibration holds lifecycle writes locked.
        with self._camera_management_lock:
            cameras = self._get_session_cameras()
            if cameras is None:
                return []
            return cameras

    def calculate_available_area(self, camera_reference: str) -> Polygon:
        """Calculate the current calibrated projector area without enabling it."""
        with self._camera_management_lock:
            camera, status = self._get_area_camera(camera_reference)
            self._require_area_enablement(camera, status)
            return self._calculate_available_area(camera, status)

    def get_camera_area(self, camera_reference: str) -> CameraArea:
        """Return the enabled area derived from the camera's current calibration."""
        with self._camera_management_lock:
            camera = self._get_session_camera(camera_reference)
            if camera is None:
                raise CameraSlotNotFoundError(
                    f'Unknown session camera slot {camera_reference!r}',
                )
            areas = self._get_camera_areas_locked()
            return next(area for area in areas if area.slot_id == camera.slot_id)

    def get_camera_areas(self) -> list[CameraArea]:
        """Return current area data in deterministic session-slot order."""
        with self._camera_management_lock:
            return self._get_camera_areas_locked()

    def set_area_enabled(
        self,
        camera_reference: str,
        area_enabled: bool,
    ) -> CameraArea:
        """Atomically enable or disable one camera's derived projector area."""
        if not isinstance(area_enabled, bool):
            raise ValueError('area_enabled must be a bool')
        with self._camera_management_lock:
            camera, status = self._get_area_camera(camera_reference)
            if area_enabled:
                self._require_area_enablement(camera, status)
                # Calculate first – an invalid polygon must not change session state.
                self._calculate_available_area(camera, status)
            updated_camera = self._set_session_area_enabled(
                camera.slot_id,
                area_enabled,
            )
            areas = self._get_camera_areas_locked()
            return next(area for area in areas if area.slot_id == updated_camera.slot_id)

    def rename_camera(self, slot_id: str, display_name: str) -> SessionCamera:
        """Rename one session slot while retaining its live camera state."""
        with self._camera_management_lock:
            camera = self.camera_runtime.rename_camera(slot_id, display_name)
            if not isinstance(camera, SessionCamera):
                raise SessionCameraError(
                    'Camera runtime returned an invalid renamed session camera',
                )
            self.point_service.rename_overlay_camera(
                camera.slot_id,
                camera.display_name,
            )
            return camera

    def close_camera(self, slot_id: str) -> SessionCamera:
        """Close one session slot and remove only its spatial ownership."""
        with self._camera_management_lock:
            try:
                camera = self.camera_runtime.close_camera(slot_id)
                if not isinstance(camera, SessionCamera):
                    raise SessionCameraError(
                        'Camera runtime returned an invalid closed session camera',
                    )
                return camera
            finally:
                self.point_service.clear_overlay_for_camera(slot_id)

    def open_camera(self, slot_id: str) -> SessionCamera:
        """Reopen one closed session slot with fresh spatial state."""
        with self._camera_management_lock:
            try:
                camera = self.camera_runtime.open_camera(slot_id)
                if not isinstance(camera, SessionCamera):
                    raise SessionCameraError(
                        'Camera runtime returned an invalid opened session camera',
                    )
                return camera
            finally:
                self.point_service.clear_overlay_for_camera(slot_id)

    def update_projector_descriptor(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | Resolution,
        output_identity: str = 'default',
    ) -> ProjectorOutputDescriptor:
        """Change the active output and invalidate all dependent geometry."""
        if isinstance(projector_output_descriptor, Resolution):
            checked_descriptor = ProjectorOutputDescriptor(
                projector_output_descriptor,
                output_identity,
            )
        elif isinstance(projector_output_descriptor, ProjectorOutputDescriptor):
            if output_identity != 'default' and (
                output_identity != projector_output_descriptor.output_identity
            ):
                raise ValueError('output_identity disagrees with the descriptor')
            checked_descriptor = projector_output_descriptor
        else:
            raise ValueError(
                'projector_output_descriptor must be ProjectorOutputDescriptor or Resolution',
            )

        with self._camera_management_lock:
            if checked_descriptor == self._projector_output_descriptor:
                return checked_descriptor
            self._metric_capture_generation += 1
            updated_calibration_pattern = build_calibration_pattern(
                checked_descriptor.projector_resolution,
            )
            self.camera_runtime.mark_calibrations_stale(checked_descriptor)
            self.calibration_registry.update_projector_descriptor(checked_descriptor)
            self.metric_calibration_registry.update_projector_descriptor(
                checked_descriptor,
            )
            self._metric_ruler = None
            self.point_service.clear_overlay()
            self._projector_output_descriptor = checked_descriptor
            self.calibration_pattern = updated_calibration_pattern
            self.configuration = replace(
                self.configuration,
                projector_resolution=checked_descriptor.projector_resolution,
                projector_output_identity=checked_descriptor.output_identity,
            )
            self.point_service.projector_resolution = checked_descriptor.projector_resolution
            self.point_service.projector_output_descriptor = checked_descriptor
            return checked_descriptor

    def calibrate_metric(
        self,
        logical_name: str,
        correspondences: (
            MetricTargetCorrespondences
            | Sequence[MetricTargetCorrespondence]
            | Sequence[MetricTargetCorrespondences]
            | None
        ) = None,
    ) -> MetricCalibrationRecord:
        """Capture and atomically publish the shared metric calibration."""
        with self._spatial_capture_operation_lock:
            with self._camera_management_lock:
                # A failed replacement attempt must never leave an older transform
                # available to callers who assume the capture was authoritative.
                self._metric_capture_generation += 1
                capture_generation = self._metric_capture_generation
                self.metric_calibration_registry.clear()
                self._metric_ruler = None
                camera, calibration = self._require_metric_camera(logical_name)
                camera_generation = camera.lifecycle_generation

            # The main-thread display must read this state and acknowledge the blank frame.
            self._begin_metric_capture()
            try:
                frames = self._get_metric_capture_frames(
                    camera.slot_id,
                    correspondences,
                )
                averaged_correspondences = _average_metric_correspondences(
                    frames,
                    self.configuration.metric_calibration_thresholds
                    .max_capture_corner_jitter_pixels,
                    camera.slot_id,
                )
                result = calibrate_metric_homography(
                    averaged_correspondences,
                    getattr(calibration, 'camera_to_projector'),
                    self._projector_output_descriptor.projector_resolution,
                    self.configuration.metric_calibration_thresholds,
                    target=METRIC_TARGET,
                )

                with self._camera_management_lock:
                    if capture_generation != self._metric_capture_generation:
                        raise CalibrationError(
                            'Metric calibration was invalidated during capture',
                        )
                    current_camera, current_calibration = self._require_metric_camera(
                        camera.slot_id,
                    )
                    if (
                        current_camera.lifecycle_generation != camera_generation
                        or not _same_camera_calibration(
                            current_calibration,
                            calibration,
                        )
                    ):
                        raise CalibrationError(
                            'The selected camera changed during metric calibration',
                        )
                    return self.metric_calibration_registry.register(
                        result,
                        self._projector_output_descriptor,
                        observation_camera_slot=camera.slot_id,
                        observation_camera_calibration=calibration,
                    )
            finally:
                self._finish_metric_capture()

    def _require_metric_camera(
        self,
        logical_name: str,
    ) -> tuple[SessionCamera, object]:
        status = self._require_available_camera(logical_name)
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError(
                f'Camera {logical_name!r} has no session identity',
            )
        calibration_status = self._get_calibration_status(status)
        if calibration_status is not CalibrationStatus.CALIBRATED:
            error = InvalidCalibrationStateError(
                f'Camera {session_camera.slot_id!r} calibration is '
                f'{calibration_status.value}',
            )
            error.code = f'CALIBRATION_{calibration_status.value}'
            raise error
        calibration = session_camera.calibration
        if calibration is None:
            raise InvalidCalibrationStateError(
                f'Camera {session_camera.slot_id!r} has no camera calibration',
            )
        if (
            getattr(calibration, 'projector_output_descriptor', None)
            != self._projector_output_descriptor
        ):
            error = InvalidCalibrationStateError(
                f'Camera {session_camera.slot_id!r} calibration is not output-applicable',
            )
            error.code = 'CALIBRATION_STALE'
            raise error
        if (
            status.native_resolution is None
            or (
                getattr(calibration, 'camera_resolution', status.native_resolution)
                != status.native_resolution
            )
        ):
            error = InvalidCalibrationStateError(
                f'Camera {session_camera.slot_id!r} calibration resolution is stale',
            )
            error.code = 'CAMERA_RESOLUTION_CHANGED'
            raise error
        try:
            validate_homography(getattr(calibration, 'camera_to_projector'))
        except (AttributeError, TypeError, ValueError):
            error = InvalidCalibrationStateError(
                f'Camera {session_camera.slot_id!r} calibration transform is invalid',
            )
            error.code = 'CALIBRATION_INVALID'
            raise error
        return session_camera, calibration

    def _require_usable_metric_calibration(self) -> MetricCalibrationRecord:
        record = self.metric_calibration_registry.get_record()
        if record is None:
            error = InvalidCalibrationStateError(
                'Metric calibration is unavailable',
            )
            error.code = 'METRIC_UNAVAILABLE'
            raise error
        if not self.metric_calibration_registry.is_usable(
            self._projector_output_descriptor,
        ):
            error = InvalidCalibrationStateError(
                'Metric calibration is stale or unavailable',
            )
            error.code = (
                'METRIC_STALE'
                if self.metric_calibration_registry.state
                is MetricCalibrationStatus.STALE
                else 'METRIC_UNAVAILABLE'
            )
            raise error
        current_record = self.metric_calibration_registry.get_record()
        if current_record is None:
            error = InvalidCalibrationStateError(
                'Metric calibration is unavailable',
            )
            error.code = 'METRIC_UNAVAILABLE'
            raise error
        return current_record

    def _begin_metric_capture(self) -> None:
        with self._metric_capture_lock:
            self._metric_blank_presented.clear()
            self._metric_capture_count += 1

    def _finish_metric_capture(self) -> None:
        with self._metric_capture_lock:
            self._metric_capture_count -= 1
            if self._metric_capture_count == 0:
                self._metric_blank_presented.clear()

    def _get_metric_capture_frames(
        self,
        slot_id: str,
        correspondences: (
            MetricTargetCorrespondences
            | Sequence[MetricTargetCorrespondence]
            | Sequence[MetricTargetCorrespondences]
            | None
        ),
    ) -> tuple[MetricTargetCorrespondences, ...]:
        injected_frames = _normalise_injected_metric_frames(correspondences, slot_id)
        if injected_frames is not None:
            return injected_frames
        if not self._metric_blank_presented.wait(METRIC_CAPTURE_WAIT_TIMEOUT_SECONDS):
            raise CalibrationError(
                'The metric blank frame was not acknowledged by the main-thread display',
            )
        time.sleep(METRIC_CAPTURE_SETTLE_SECONDS)
        detector = self.detector
        if detector is None:
            detector = OpenCVArucoDetector()
        get_consecutive_frames = getattr(
            self.camera_runtime,
            'get_consecutive_frames',
            None,
        )
        if callable(get_consecutive_frames):
            captured_frames = get_consecutive_frames(
                slot_id,
                METRIC_CAPTURE_FRAME_COUNT,
                METRIC_CAPTURE_WAIT_TIMEOUT_SECONDS,
            )
            if not isinstance(captured_frames, Sequence) or len(captured_frames) != (
                METRIC_CAPTURE_FRAME_COUNT
            ) or any(not isinstance(frame, Frame) for frame in captured_frames):
                raise CalibrationError(
                    'Camera runtime returned invalid consecutive metric frames',
                )
        else:
            captured_frames = tuple(
                self.snapshot(slot_id)
                for _frame_index in range(METRIC_CAPTURE_FRAME_COUNT)
            )

        frames: list[MetricTargetCorrespondences] = []
        previous_frame_counter: int | None = None
        for frame in captured_frames:
            if (
                not isinstance(frame.frame_counter, int)
                or isinstance(frame.frame_counter, bool)
                or frame.frame_counter < 0
                or (
                    previous_frame_counter is not None
                    and frame.frame_counter != previous_frame_counter + 1
                )
            ):
                raise CalibrationError(
                    'Metric capture did not receive three consecutive camera frames',
                )
            previous_frame_counter = frame.frame_counter
            frames.append(
                detect_and_assemble_metric_correspondences(
                    frame.data,
                    detector,
                    target=METRIC_TARGET,
                    camera_id=slot_id,
                    minimum_marker_count=(
                        self.configuration.metric_calibration_thresholds
                        .min_unique_target_fiducials
                    ),
                    minimum_spatial_coverage=(
                        self.configuration.metric_calibration_thresholds
                        .min_spatial_coverage
                    ),
                ),
            )
        return tuple(frames)

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
        with self._camera_management_lock:
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
        with self._camera_management_lock:
            return self.point_service.point_from_camera(logical_name, camera_point)

    def get_calibration_metrics(self, logical_name: str) -> CalibrationMetrics | None:
        status = self.get_camera_status(logical_name)
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError('Camera runtime returned no session camera')
        calibration = session_camera.calibration
        if not isinstance(calibration, PersistedCalibration):
            return None
        return calibration.metrics

    def get_calibration_records(self) -> dict[str, PersistedCalibration]:
        session_cameras = self._get_session_cameras()
        if session_cameras is None:
            raise CameraUnavailableError(
                'Camera runtime returned no session camera inventory',
            )
        return {
            camera.slot_id: camera.calibration
            for camera in session_cameras
            if isinstance(camera.calibration, PersistedCalibration)
        }

    def clear_overlay(self) -> None:
        self.point_service.clear_overlay()

    def _clear_area_before_calibration(self, logical_name: str) -> None:
        resolved_slot = self._resolve_camera_reference(logical_name)
        session_camera = self._get_session_camera(resolved_slot)
        if session_camera is None or not session_camera.area_enabled:
            return
        self._set_session_area_enabled(session_camera.slot_id, False)

    def _set_session_area_enabled(
        self,
        slot_id: str,
        area_enabled: bool,
    ) -> SessionCamera:
        set_area_enabled = getattr(self.camera_runtime, 'set_area_enabled', None)
        if not callable(set_area_enabled):
            raise SessionCameraError(
                'Camera runtime does not support session area state',
            )
        updated_camera = set_area_enabled(slot_id, area_enabled)
        if not isinstance(updated_camera, SessionCamera):
            raise SessionCameraError(
                'Camera runtime returned an invalid area session camera',
            )
        return updated_camera

    def _get_area_camera(
        self,
        slot_id: str,
    ) -> tuple[SessionCamera, CameraStatus]:
        camera = self._get_session_camera(slot_id)
        if camera is None:
            raise CameraSlotNotFoundError(
                f'Unknown session camera slot {slot_id!r}',
            )
        return camera, self.get_camera_status(camera.slot_id)

    def _get_camera_areas_locked(self) -> list[CameraArea]:
        cameras = self._get_session_cameras()
        if cameras is None:
            return []
        areas: list[CameraArea] = []
        for camera in cameras:
            area = self._build_camera_area(
                camera,
                self.get_camera_status(camera.slot_id),
            )
            if area.area_enabled and area.available_area is None:
                # An invalidating status must not leave an enabled flag that can
                # silently make an old area reappear after recovery.
                self._set_session_area_enabled(camera.slot_id, False)
                area = area._replace(area_enabled=False)
            areas.append(area)
        colour_index = 0
        for area_index, area in enumerate(areas):
            if not area.area_enabled or area.available_area is None:
                continue
            areas[area_index] = area._replace(
                area_colour=AREA_COLOURS[colour_index],
            )
            colour_index += 1
        return areas

    def _require_area_enablement(
        self,
        camera: SessionCamera,
        status: CameraStatus,
    ) -> None:
        if camera.state is not SessionCameraState.OPEN:
            raise CameraUnavailableError(
                camera.error_message
                or f'Camera {camera.slot_id!r} is not available',
            )
        if status.runtime_status is not RuntimeStatus.AVAILABLE:
            raise CameraUnavailableError(
                status.error_message
                or f'Camera {camera.slot_id!r} is not available',
            )
        calibration_status = self._get_calibration_status(status)
        if calibration_status is not CalibrationStatus.CALIBRATED:
            error = InvalidCalibrationStateError(
                f'Camera {camera.slot_id!r} calibration is '
                f'{calibration_status.value}',
            )
            error.code = f'CALIBRATION_{calibration_status.value}'
            raise error

    def _calculate_available_area(
        self,
        camera: SessionCamera,
        status: CameraStatus,
    ) -> Polygon:
        native_resolution = status.native_resolution
        calibration = camera.calibration
        camera_to_projector = getattr(calibration, 'camera_to_projector', None)
        if native_resolution is None or camera_to_projector is None:
            raise InvalidAvailableAreaError(
                f'Camera {camera.slot_id!r} has no usable calibrated area',
            )
        # Keep this footprint identical to the native-frame region accepted by
        # pointing; the calibration's tag-supported hull remains quality metadata.
        camera_frame = CoordinateBounds(
            0.0,
            0.0,
            float(native_resolution.width),
            float(native_resolution.height),
        )
        available_area = calculate_available_projector_area(
            camera_frame,
            native_resolution,
            camera_to_projector,
            self._projector_output_descriptor.projector_resolution,
        )
        if available_area is None:
            raise InvalidAvailableAreaError(
                f'Camera {camera.slot_id!r} has no usable diagnostic area',
            )
        return available_area

    def _build_camera_area(
        self,
        camera: SessionCamera,
        status: CameraStatus,
    ) -> CameraArea:
        area_colour = get_camera_area_colour(camera.slot_id)
        if not camera.area_enabled:
            return CameraArea(
                camera.slot_id,
                camera.display_name,
                False,
                None,
                area_colour,
            )
        try:
            self._require_area_enablement(camera, status)
            available_area = self._calculate_available_area(camera, status)
        except (
            CameraUnavailableError,
            InvalidCalibrationStateError,
            InvalidAvailableAreaError,
        ):
            available_area = None
        return CameraArea(
            camera.slot_id,
            camera.display_name,
            True,
            available_area,
            area_colour,
        )

    def _calibrate_camera(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> PersistedCalibration:
        with self._spatial_capture_operation_lock:
            return self._calibrate_camera_locked(logical_name, correspondences)

    def _calibrate_camera_locked(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None,
    ) -> PersistedCalibration:
        self._clear_area_before_calibration(logical_name)
        status = self._require_available_camera(logical_name)
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None or status.native_resolution is None:
            raise CameraUnavailableError(f'Camera {logical_name!r} has incomplete metadata')
        camera_slot_id = session_camera.slot_id
        camera_lifecycle_generation = session_camera.lifecycle_generation
        checked_correspondences = self._get_correspondences_for_operation(
            status,
            correspondences,
        )
        with self._camera_management_lock:
            current_status = self._require_available_camera(camera_slot_id)
            current_camera = self._get_session_camera(camera_slot_id)
            if current_camera is None or current_status.native_resolution is None:
                raise CameraUnavailableError(
                    f'Camera {logical_name!r} has incomplete metadata',
                )
            if current_camera.lifecycle_generation != camera_lifecycle_generation:
                raise CalibrationError(
                    'The selected camera changed during calibration',
                )
            result = calibrate_homography(
                checked_correspondences,
                self.calibration_pattern,
                self.configuration.calibration_thresholds,
                camera_resolution=current_status.native_resolution,
            )
            record = PersistedCalibration.from_result(
                result,
                current_status.native_resolution,
                self._projector_output_descriptor.projector_resolution,
                version=self.configuration.calibration_version,
                projector_output_descriptor=self._projector_output_descriptor,
                camera_id=current_camera.slot_id,
            )
            self._set_session_calibration(
                current_camera.slot_id,
                CalibrationStatus.UNVERIFIED,
                record,
            )
            return record

    def _verify_camera(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None = None,
    ) -> CalibrationStatus:
        with self._spatial_capture_operation_lock:
            with self._camera_management_lock:
                return self._verify_camera_locked(logical_name, correspondences)

    def _verify_camera_locked(
        self,
        logical_name: str,
        correspondences: CameraCorrespondences | Sequence[FiducialCorrespondence] | None,
    ) -> CalibrationStatus:
        self._clear_area_before_calibration(logical_name)
        status = self._require_available_camera(logical_name)
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None or status.native_resolution is None:
            raise CameraUnavailableError(f'Camera {logical_name!r} has incomplete metadata')
        calibration = session_camera.calibration
        if not isinstance(calibration, PersistedCalibration):
            return CalibrationStatus.UNCALIBRATED
        checked_correspondences = self._get_correspondences_for_operation(
            status,
            correspondences,
        )
        session_registry = CalibrationRegistry(
            {session_camera.slot_id: calibration},
            calibration_version=self.configuration.calibration_version,
            projector_resolution=self._projector_output_descriptor.projector_resolution,
            projector_output_descriptor=self._projector_output_descriptor,
        )
        calibration_status = session_registry.verify(
            session_camera.slot_id,
            checked_correspondences,
            camera_resolution=status.native_resolution,
            projector_resolution=self._projector_output_descriptor.projector_resolution,
            thresholds=self.configuration.calibration_thresholds,
            projector_output_descriptor=self._projector_output_descriptor,
            pattern=self.calibration_pattern,
        )
        self._set_session_calibration(
            session_camera.slot_id,
            calibration_status,
            calibration,
        )
        return calibration_status

    def _get_available_statuses(self) -> list[CameraStatus]:
        return [
            status
            for status in self.get_camera_statuses()
            if status.runtime_status is RuntimeStatus.AVAILABLE
        ]

    def _require_available_camera(self, logical_name: str) -> CameraStatus:
        status = self.get_camera_status(logical_name)
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is not None and session_camera.state is not SessionCameraState.OPEN:
            raise CameraUnavailableError(
                session_camera.error_message
                or f'Camera {logical_name!r} is not available',
            )
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
            # Allow the projector and camera exposure pipeline to settle after presentation.
            time.sleep(CALIBRATION_PATTERN_SETTLE_SECONDS)
            deadline = time.monotonic() + CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS
            best_correspondences: CameraCorrespondences | None = None
            last_error: CalibrationError | None = None
            while True:
                try:
                    candidate = self._get_correspondences(status, None)
                except CalibrationError as ex:
                    last_error = ex
                else:
                    if best_correspondences is None or (
                        len(candidate.unique_marker_ids),
                        len(candidate.correspondences),
                    ) > (
                        len(best_correspondences.unique_marker_ids),
                        len(best_correspondences.correspondences),
                    ):
                        best_correspondences = candidate
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.25)
            if best_correspondences is not None:
                return best_correspondences
            if last_error is not None:
                raise last_error
            raise CalibrationError('No calibration correspondences were detected')
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
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError(
                f'Camera {status.logical_name!r} has no session identity',
            )
        camera_id = session_camera.slot_id
        if correspondences is not None:
            if isinstance(correspondences, CameraCorrespondences):
                if correspondences.camera_id not in {None, camera_id}:
                    raise CalibrationError('Correspondences belong to another camera')
                return correspondences._replace(camera_id=camera_id)
            try:
                values = tuple(correspondences)
            except (TypeError, ValueError) as ex:
                raise CalibrationError('correspondences must be iterable') from ex
            return CameraCorrespondences(values, camera_id)

        frame = self.snapshot(status.logical_name)
        detector = self.detector
        if detector is None:
            detector = OpenCVArucoDetector()
        detected_markers = detect_fiducials(frame.data, detector)
        return assemble_correspondences(
            detected_markers,
            self.calibration_pattern,
            camera_id=camera_id,
        )

    def _get_calibration_status(self, status: CameraStatus) -> CalibrationStatus:
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError(
                f'Camera {status.logical_name!r} has no session state',
            )
        calibration = session_camera.calibration
        if (
            session_camera.calibration_status is CalibrationStatus.CALIBRATED
            and isinstance(calibration, PersistedCalibration)
            and (
                calibration.camera_id != session_camera.slot_id
                or calibration.version != self.configuration.calibration_version
                or calibration.camera_resolution != status.native_resolution
                or calibration.projector_output_descriptor
                != self._projector_output_descriptor
            )
        ):
            return CalibrationStatus.STALE
        return session_camera.calibration_status

    def _get_session_cameras(self) -> list[SessionCamera] | None:
        get_session_cameras = getattr(self.camera_runtime, 'get_session_cameras', None)
        if not callable(get_session_cameras):
            return None
        cameras = get_session_cameras()
        if not isinstance(cameras, list) or any(
            not isinstance(camera, SessionCamera)
            for camera in cameras
        ):
            raise SessionCameraError(
                'Camera runtime returned an invalid session camera list',
            )
        return sorted(cameras, key=_session_camera_sort_key)

    def _resolve_camera_reference(self, camera_reference: str) -> str:
        cameras = self._get_session_cameras()
        if cameras is None:
            return camera_reference
        for camera in cameras:
            if camera.slot_id == camera_reference:
                return camera.slot_id
        matching_cameras = [
            camera
            for camera in cameras
            if camera.display_name == camera_reference
        ]
        if len(matching_cameras) > 1:
            raise CameraUnavailableError(
                f'Camera name {camera_reference!r} is ambiguous',
            )
        if len(matching_cameras) == 1:
            return matching_cameras[0].slot_id
        return camera_reference

    def _get_session_camera(self, slot_id: str) -> SessionCamera | None:
        cameras = self._get_session_cameras()
        if cameras is None:
            return None
        for camera in cameras:
            if camera.slot_id == slot_id:
                return camera
        return None

    def _set_session_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: PersistedCalibration,
    ) -> None:
        set_calibration = getattr(self.camera_runtime, 'set_calibration', None)
        if not callable(set_calibration):
            raise SessionCameraError(
                'Camera runtime does not support session calibration',
            )
        updated_camera = set_calibration(
            slot_id,
            calibration_status,
            calibration,
        )
        if not isinstance(updated_camera, SessionCamera):
            raise SessionCameraError(
                'Camera runtime returned an invalid calibrated session camera',
            )


def _normalise_injected_metric_frames(
    correspondences: (
        MetricTargetCorrespondences
        | Sequence[MetricTargetCorrespondence]
        | Sequence[MetricTargetCorrespondences]
        | None
    ),
    camera_id: str,
) -> tuple[MetricTargetCorrespondences, ...] | None:
    if correspondences is None:
        return None
    if isinstance(correspondences, MetricTargetCorrespondences):
        frame = _replace_metric_camera_id(correspondences, camera_id)
        return (frame,) * METRIC_CAPTURE_FRAME_COUNT
    try:
        values = tuple(correspondences)
    except (TypeError, ValueError) as ex:
        raise CalibrationError('Metric correspondences must be iterable') from ex
    if all(isinstance(value, MetricTargetCorrespondence) for value in values):
        frame = MetricTargetCorrespondences(
            tuple(values),
            camera_id,
        )
        return (frame,) * METRIC_CAPTURE_FRAME_COUNT
    if all(isinstance(value, MetricTargetCorrespondences) for value in values):
        if len(values) != METRIC_CAPTURE_FRAME_COUNT:
            raise CalibrationError(
                f'Metric capture requires {METRIC_CAPTURE_FRAME_COUNT} frames',
            )
        return tuple(
            _replace_metric_camera_id(value, camera_id)
            for value in values
        )
    if all(
        isinstance(value, Sequence)
        and all(isinstance(item, MetricTargetCorrespondence) for item in value)
        for value in values
    ):
        if len(values) != METRIC_CAPTURE_FRAME_COUNT:
            raise CalibrationError(
                f'Metric capture requires {METRIC_CAPTURE_FRAME_COUNT} frames',
            )
        return tuple(
            MetricTargetCorrespondences(tuple(value), camera_id)
            for value in values
        )
    raise CalibrationError(
        'Metric correspondences must contain target correspondence values or frames',
    )


def _same_camera_calibration(first: object, second: object) -> bool:
    """Compare the camera geometry used by a capture, not object identity."""
    for field_name in (
        'camera_to_projector',
        'projector_output_descriptor',
        'camera_resolution',
        'version',
        'timestamp',
    ):
        first_value = getattr(first, field_name, None)
        second_value = getattr(second, field_name, None)
        if first_value != second_value:
            return False
    return True


def _replace_metric_camera_id(
    correspondences: MetricTargetCorrespondences,
    camera_id: str,
) -> MetricTargetCorrespondences:
    if correspondences.camera_id not in {None, camera_id}:
        raise CalibrationError('Metric correspondences belong to another camera')
    return correspondences._replace(camera_id=camera_id)


def _average_metric_correspondences(
    frames: Sequence[MetricTargetCorrespondences],
    maximum_corner_jitter_pixels: float,
    camera_id: str,
) -> MetricTargetCorrespondences:
    if len(frames) != METRIC_CAPTURE_FRAME_COUNT:
        raise CalibrationError(
            f'Metric capture requires {METRIC_CAPTURE_FRAME_COUNT} frames',
        )
    if not math.isfinite(maximum_corner_jitter_pixels) or maximum_corner_jitter_pixels <= 0:
        raise CalibrationError('Metric corner jitter tolerance is invalid')

    first_frame = validate_metric_correspondences(
        _replace_metric_camera_id(frames[0], camera_id),
        target=METRIC_TARGET,
    )
    first_by_corner = _metric_correspondences_by_corner(first_frame)
    first_marker_ids = first_frame.unique_marker_ids
    averaged: list[MetricTargetCorrespondence] = []
    frame_corner_maps = [first_by_corner]
    for frame in frames[1:]:
        checked_frame = validate_metric_correspondences(
            _replace_metric_camera_id(frame, camera_id),
            target=METRIC_TARGET,
        )
        frame_by_corner = _metric_correspondences_by_corner(checked_frame)
        if set(checked_frame.unique_marker_ids) != set(first_marker_ids):
            raise CalibrationError(
                'Metric capture frames do not contain the same target marker IDs',
            )
        if set(frame_by_corner) != set(first_by_corner):
            raise CalibrationError(
                'Metric capture frames do not contain the same target corners',
            )
        frame_corner_maps.append(frame_by_corner)
        for corner_key, first_correspondence in first_by_corner.items():
            correspondence = frame_by_corner[corner_key]
            if correspondence.surface_position != first_correspondence.surface_position:
                raise CalibrationError(
                    'Metric capture frames disagree about target corner positions',
                )
            try:
                corner_jitter = math.dist(
                    first_correspondence.camera_position,
                    correspondence.camera_position,
                )
            except (TypeError, ValueError) as ex:
                raise CalibrationError(
                    'Metric capture contains an invalid camera corner',
                ) from ex
            if not math.isfinite(corner_jitter) or corner_jitter > maximum_corner_jitter_pixels:
                raise CalibrationError(
                    'Metric capture detected movement beyond the stability tolerance',
                )

    for corner_key, first_correspondence in first_by_corner.items():
        frame_correspondences = [
            corner_map[corner_key]
            for corner_map in frame_corner_maps
        ]
        try:
            averaged_camera_position = Point2D(
                math.fsum(
                    correspondence.camera_position.x
                    for correspondence in frame_correspondences
                ) / METRIC_CAPTURE_FRAME_COUNT,
                math.fsum(
                    correspondence.camera_position.y
                    for correspondence in frame_correspondences
                ) / METRIC_CAPTURE_FRAME_COUNT,
            )
        except (AttributeError, TypeError, ValueError) as ex:
            raise CalibrationError(
                'Metric capture contains an invalid camera corner',
            ) from ex
        averaged.append(
            MetricTargetCorrespondence(
                first_correspondence.marker_id,
                first_correspondence.corner_index,
                first_correspondence.surface_position,
                averaged_camera_position,
            ),
        )
    return MetricTargetCorrespondences(tuple(averaged), camera_id)


def _metric_correspondences_by_corner(
    frame: MetricTargetCorrespondences,
) -> dict[tuple[int, int], MetricTargetCorrespondence]:
    values = frame.correspondences
    if any(not isinstance(value, MetricTargetCorrespondence) for value in values):
        raise CalibrationError('Metric capture contains malformed correspondences')
    result: dict[tuple[int, int], MetricTargetCorrespondence] = {}
    for correspondence in values:
        corner_key = (correspondence.marker_id, correspondence.corner_index)
        if corner_key in result:
            raise CalibrationError('Metric capture contains duplicate target corners')
        result[corner_key] = correspondence
    return result


def get_camera_area_colour(slot_id: str) -> tuple[int, int, int]:
    """Return the stable diagnostic colour assigned to a session slot."""
    prefix, separator, index = slot_id.rpartition('-')
    if prefix != 'camera' or separator == '' or not index.isdigit():
        return AREA_COLOURS[0]
    return AREA_COLOURS[int(index) % len(AREA_COLOURS)]


def _session_camera_sort_key(camera: SessionCamera) -> int:
    return int(camera.slot_id.rsplit('-', 1)[1])


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


__all__ = ['AREA_COLOURS', 'CameraArea', 'MultiVisionService', 'get_camera_area_colour']
