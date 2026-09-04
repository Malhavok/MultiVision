"""Service composition for MultiVision capabilities."""

from __future__ import annotations

import math
import pathlib
import statistics
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import (
    Callable,
    NamedTuple,
)

from multivision.benchmarking import BenchmarkMetrics, measure_timing
from multivision.calibration import (
    CalibrationMetrics,
    CalibrationResult,
    calibrate_homography,
)
from multivision.camera import CameraRuntime
from multivision.config import (
    CalibrationThresholds,
    Configuration,
    FiducialGroup,
    ProjectorOutputDescriptor,
    load_configuration,
)
from multivision.discovery import PlatformDeviceDiscovery
from multivision.errors import (
    CalibrationError,
    CameraSlotNotFoundError,
    CameraUnavailableError,
    FrameCaptureError,
    GeometryError,
    InvalidAvailableAreaError,
    InvalidCalibrationStateError,
    SessionCameraError,
)
from multivision.geometry import (
    CoordinateBounds,
    MatrixLike,
    Point2D,
    PointLike,
    Polygon,
    PreviewTransform,
    TagGeometry,
    build_protected_projector_regions,
    build_tag_geometry,
    calculate_available_projector_area,
    project_point,
    validate_homography,
)
from multivision.fiducials import (
    CachedTagDetectorFactory,
    CameraCorrespondences,
    FiducialCorrespondence,
    FiducialDetector,
    GroupTrackingError,
    MetricTargetCorrespondence,
    MetricTargetCorrespondences,
    OpenCVArucoDetector,
    PlanarTagObservation,
    TagDetectorFactory,
    assemble_correspondences,
    detect_and_assemble_metric_correspondences,
    detect_configured_fiducial_observations,
    detect_fiducials,
    detect_tag_observations,
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
from multivision.overlays import (
    AnyOverlayRequest,
    ArrowRequest,
    CircleRequest,
    FiducialAnchor,
    GridRequest,
    LineRequest,
    OverlayEntry,
    OverlayRegistry,
    PointReference,
    ProjectorCoverageGridRequest,
    ProjectorMaterialisation,
    RectRequest,
    RulerRequest,
    TextRequest,
    build_projector_coverage_grid_request,
    get_overlay_dependencies,
    get_overlay_point_references,
    is_dynamic_overlay_request,
    materialise_overlay,
    materialise_presentation,
    normalise_overlay_intensity,
)
from multivision.spatial import (
    CameraGeneration,
    SpatialState,
    SpatialTracker,
)
from multivision.pattern import (
    CalibrationPattern,
    PROJECTOR_CALIBRATION_MARKER_FAMILY,
    build_calibration_pattern,
    validate_tag_dictionary,
)
from multivision.persistence import (
    CalibrationRegistry,
    CalibrationStore,
    PersistedCalibration,
)
from multivision.service import PointOverlayService, RedCircleOverlay
from multivision.session import SessionCamera
from multivision.types import (
    CalibrationScope,
    CalibrationStage,
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    RuntimeStatus,
    Resolution,
    SessionCameraState,
    is_finite_real,
    is_valid_resolution,
)


CALIBRATION_PATTERN_WAIT_TIMEOUT_SECONDS = 5.0
# Bright projected patterns need time for camera auto-exposure and white balance to settle.
CALIBRATION_PATTERN_SETTLE_SECONDS = 5.0
CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS = 5.0
CALIBRATION_CAPTURE_CANDIDATE_FRAME_COUNT = 15
METRIC_CAPTURE_WAIT_TIMEOUT_SECONDS = 5.0
# The blank metric frame can trigger the same camera adaptation as the pattern.
METRIC_CAPTURE_SETTLE_SECONDS = 5.0
METRIC_CAPTURE_FRAME_COUNT = 3
METRIC_CAPTURE_CANDIDATE_FRAME_COUNT = 15
TRACKING_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
CALIBRATION_PATTERN_EDGE_MARGIN_PIXELS = 24.0
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


class FullCalibrationCameraResult(NamedTuple):
    """The independently verified outcome for one camera in a full run."""

    camera: str
    status: CalibrationStatus
    calibration_scope: CalibrationScope | None
    calibration: PersistedCalibration | None
    error: str | None = None

    def to_data(self) -> dict[str, object]:
        return {
            'camera': self.camera,
            'status': self.status.value,
            'calibration_scope': (
                None
                if self.calibration_scope is None
                else self.calibration_scope.value
            ),
            'calibration': (
                None
                if self.calibration is None
                else self.calibration.to_data()
            ),
            'error': self.error,
        }


class FullCalibrationResult(NamedTuple):
    """One-shot calibration report, including the session master quality gate."""

    master_camera: str | None
    master_gate_passed: bool
    cameras: Mapping[str, FullCalibrationCameraResult]

    def to_data(self) -> dict[str, object]:
        return {
            'master_camera': self.master_camera,
            'master_gate_passed': self.master_gate_passed,
            'cameras': {
                camera: result.to_data()
                for camera, result in self.cameras.items()
            },
        }


class ProjectionStatus(NamedTuple):
    """A structured explanation for unavailable projector geometry."""

    code: str
    message: str

    def to_data(self) -> dict[str, str]:
        return {'code': self.code, 'message': self.message}


class CameraRenderSnapshot(NamedTuple):
    """Immutable diagnostic and retained-frame input for one camera card."""

    slot_id: str
    logical_name: str
    status: CameraStatus
    camera_state: SessionCameraState
    lifecycle_generation: int
    frame: Frame | None = None
    calibration_metrics: CalibrationMetrics | None = None


class RenderSnapshot(NamedTuple):
    """One complete, immutable input for a projector render."""

    generation: int
    registry_generation: int
    spatial_generation: int
    spatial_state: SpatialState
    projector_output_descriptor: ProjectorOutputDescriptor
    overlays: tuple[OverlayEntry, ...]
    metric_generation: int = 0
    registry_entries: tuple[OverlayEntry, ...] | None = None
    global_overlay_intensity: float = 1.0
    protected_projector_regions: tuple[tuple[Point2D, ...], ...] = ()
    camera_snapshots: tuple[CameraRenderSnapshot, ...] = ()
    projector_areas: tuple[CameraArea, ...] = ()
    metric_state: MetricCalibrationStatus = MetricCalibrationStatus.UNCALIBRATED
    metric_ruler: MetricRulerOverlay | None = None
    point_overlay: RedCircleOverlay | None = None
    calibration_pattern_visible: bool = False
    metric_capture_active: bool = False
    calibration_pattern: CalibrationPattern | None = None

    @property
    def camera_statuses(self) -> tuple[CameraStatus, ...]:
        return tuple(camera.status for camera in self.camera_snapshots)

    @property
    def preview_cameras(self) -> tuple[CameraRenderSnapshot, ...]:
        return self.camera_snapshots

    @property
    def areas(self) -> tuple[CameraArea, ...]:
        return self.projector_areas

    @property
    def generic_overlays(self) -> tuple[OverlayEntry, ...]:
        return self.overlays

    @property
    def registry_snapshot(self) -> tuple[OverlayEntry, ...]:
        if self.registry_entries is not None:
            return self.registry_entries
        return self.overlays

    @property
    def resolved_overlays(self) -> tuple[OverlayEntry, ...]:
        return self.overlays

    @property
    def protected_regions(self) -> tuple[tuple[Point2D, ...], ...]:
        return self.protected_projector_regions

    @property
    def intensity(self) -> float:
        return self.global_overlay_intensity


class _CalibrationCapture(NamedTuple):
    """A stable correspondence burst and its measured capture noise."""

    correspondences: CameraCorrespondences
    noise: _CameraCaptureNoise | None


class _TrackingCycle(NamedTuple):
    """The short-lived, authority-tagged input for one tracking pass."""

    tracking_generation: int
    metric_generation: int
    projector_output_descriptor: ProjectorOutputDescriptor
    metric_calibration: object | None
    camera_states: Mapping[str, CameraGeneration]
    camera_transforms: Mapping[str, MatrixLike]
    frames_by_camera: Mapping[str, Frame]
    received_monotonic_seconds: float


class TagObservation(NamedTuple):
    """One detected tag with optional, independently validated projection."""

    marker_id: int
    camera: TagGeometry
    projector: TagGeometry | None = None
    projection_status: ProjectionStatus | None = None

    def to_data(self) -> dict[str, object]:
        return {
            'id': self.marker_id,
            'camera': _tag_geometry_to_data(self.camera),
            'projector': (
                None
                if self.projector is None
                else _tag_geometry_to_data(self.projector)
            ),
            'projection_status': (
                None
                if self.projection_status is None
                else self.projection_status.to_data()
            ),
        }


class TagInspectionResult(NamedTuple):
    """The read-only result of inspecting one retained camera frame."""

    camera: str
    camera_id: str | None
    dictionary: str
    frame_counter: int
    captured_at_seconds: float
    tags: tuple[TagObservation, ...]
    projection_status: ProjectionStatus | None = None

    def to_data(self) -> dict[str, object]:
        return {
            'camera': self.camera,
            'camera_id': self.camera_id,
            'dictionary': self.dictionary,
            'frame_counter': self.frame_counter,
            'captured_at_seconds': self.captured_at_seconds,
            'tags': [tag.to_data() for tag in self.tags],
            'projection_status': (
                None
                if self.projection_status is None
                else self.projection_status.to_data()
            ),
        }


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
        tag_detector_factory: TagDetectorFactory | None = None,
        calibration_pattern: CalibrationPattern | None = None,
        point_service: PointOverlayService | None = None,
        sleep_inhibitor: SleepInhibitor | None = None,
        tracking_clock: Callable[[], float] | None = None,
        spatial_tracker: SpatialTracker | None = None,
        clock: Callable[[], float] | None = None,
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
        self.overlay_registry = OverlayRegistry(
            self._projector_output_descriptor,
            max_batch_operations=configuration.max_batch_operations,
        )
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
            else _build_projector_calibration_pattern(
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
        self.tag_detector_factory = (
            tag_detector_factory
            if tag_detector_factory is not None
            else CachedTagDetectorFactory()
        )
        self.sleep_inhibitor = (
            sleep_inhibitor
            if sleep_inhibitor is not None
            else SystemSleepInhibitor()
        )
        self._lifecycle_lock = threading.RLock()
        self._camera_management_lock = threading.RLock()
        self._detector_lock = threading.RLock()
        self._calibration_capture_count = 0
        self._calibration_capture_lock = threading.RLock()
        self._last_camera_capture_noise: _CameraCaptureNoise | None = None
        self._calibration_pattern_presented = threading.Event()
        self._calibration_pattern_hold = False
        self._calibration_master_camera: str | None = None
        self._metric_capture_count = 0
        self._metric_capture_lock = threading.RLock()
        # Camera-pattern and metric-blank handshakes must never overlap.
        self._spatial_capture_operation_lock = threading.RLock()
        self._metric_capture_generation = 0
        self._metric_blank_presented = threading.Event()
        if tracking_clock is not None and clock is not None:
            raise ValueError('Specify only one tracking clock')
        effective_tracking_clock = tracking_clock or clock or time.monotonic
        if not callable(effective_tracking_clock):
            raise ValueError('tracking_clock must be callable')
        self._tracking_clock = effective_tracking_clock
        self._spatial_tracker = (
            spatial_tracker
            if spatial_tracker is not None
            else SpatialTracker(
                history_length=configuration.fiducial_history_length,
                grace_period_seconds=configuration.fiducial_grace_period_seconds,
                protection_margin_mm=configuration.fiducial_protection_margin_mm,
                update_deadband_mm=configuration.fiducial_update_deadband_mm,
                update_deadband_degrees=configuration.fiducial_update_deadband_degrees,
                clock=effective_tracking_clock,
            )
        )
        if not isinstance(self._spatial_tracker, SpatialTracker):
            raise ValueError('spatial_tracker must be SpatialTracker')
        self._tracking_control_lock = threading.RLock()
        self._tracking_generation = 0
        self._tracker_published_spatial_state: SpatialState | None = None
        self._camera_calibration_generations: dict[str, int] = {}
        self._camera_calibration_signatures: dict[str, object] = {}
        self._tracking_stop_event = threading.Event()
        self._tracking_thread: threading.Thread | None = None
        self._benchmark_metrics = BenchmarkMetrics()
        self._tracking_error: Exception | None = None
        self._spatial_state = SpatialState.empty()
        self._overlay_intensity = 1.0
        self._render_state_lock = threading.RLock()
        self._render_snapshot: RenderSnapshot | None = None
        self._render_generation = 0
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
                self.overlay_registry.invalidate_metric()
                self._invalidate_spatial_metric_state()
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
            if status is not MetricCalibrationStatus.CALIBRATED:
                self.overlay_registry.invalidate_metric()
                self._invalidate_spatial_metric_state()
            return status, self.metric_calibration_registry.record

    def clear_metric_calibration(self) -> None:
        """Clear the shared metric transform and its independent ruler state."""
        with self._camera_management_lock:
            self._metric_capture_generation += 1
            self.overlay_registry.invalidate_metric()
            self.metric_calibration_registry.clear()
            self._invalidate_spatial_metric_state()
            self._metric_ruler = None

    def create_projector_coverage_grid(
        self,
        request: ProjectorCoverageGridRequest,
    ) -> OverlayEntry:
        """Create a physical grid spanning the extrapolated projector footprint."""
        if not isinstance(request, ProjectorCoverageGridRequest):
            raise ValueError('request must be ProjectorCoverageGridRequest')
        with self._camera_management_lock:
            metric_calibration = self._require_overlay_metric_calibration()
            projector_resolution = self._projector_output_descriptor.projector_resolution
        grid_request = build_projector_coverage_grid_request(
            request,
            metric_calibration,
            projector_resolution,
        )
        return self.create_overlay(grid_request)

    def create_overlay(self, request: AnyOverlayRequest) -> OverlayEntry:
        """Create one overlay through the same transaction as a batch."""
        if not isinstance(request, (GridRequest, CircleRequest, RectRequest, TextRequest,
                                    LineRequest, RulerRequest, ArrowRequest)):
            raise ValueError('request must be an overlay request')
        entries = self.apply_overlay_batch(({'op': 'create', 'request': request},))
        with self._camera_management_lock:
            self._invalidate_unusable_overlay_dependencies()
        return entries[-1]

    def apply_overlay_batch(self, operations: Sequence[object]) -> list[OverlayEntry]:
        """Apply ordered overlay mutations with one bounded publication."""
        with measure_timing(
            self._benchmark_metrics,
            'registry_candidate_build_publication',
        ):
            canonical_operations = self._canonicalise_overlay_operations(operations)
            entries = self.overlay_registry.apply_batch(
                canonical_operations,
                materialise_request=self._materialise_overlay_request,
                max_operations=self.configuration.max_batch_operations,
            )
        self._benchmark_metrics.increment(
            'accepted_mutations',
            len(canonical_operations),
        )
        self._benchmark_metrics.increment('published_transactions')
        return entries

    apply_overlays_batch = apply_overlay_batch

    def update_overlay(
        self,
        selector: str | uuid.UUID,
        request: AnyOverlayRequest,
    ) -> OverlayEntry:
        """Replace one complete overlay specification without changing its order."""
        entries = self.apply_overlay_batch((
            {'op': 'update', 'selector': selector, 'request': request},
        ))
        return next(entry for entry in entries if entry.id == request.id)

    def list_overlays(self) -> list[OverlayEntry]:
        """Return retained generic overlays in deterministic insertion order."""
        with self._camera_management_lock:
            self._invalidate_unusable_overlay_dependencies()
            self._invalidate_stale_spatial_authorities_locked()
        self.get_spatial_state()
        return list(self.overlay_registry.resolve(self._materialise_overlay_request))

    @property
    def spatial_state(self) -> SpatialState:
        return self.get_spatial_state()

    def get_spatial_state(self) -> SpatialState:
        """Return the latest immutable spatial publication."""
        with self._tracking_control_lock:
            with self._render_state_lock:
                spatial_state = self._spatial_state
                tracker_state = self._tracker_published_spatial_state
            if tracker_state is spatial_state:
                refreshed_state = self._spatial_tracker.snapshot
                if refreshed_state is not spatial_state:
                    with self._render_state_lock:
                        if self._spatial_state is spatial_state:
                            self._spatial_state = refreshed_state
                            self._tracker_published_spatial_state = refreshed_state
                            spatial_state = refreshed_state
            return spatial_state

    def get_fiducial_groups(self) -> dict[str, object]:
        """Return a copy of the current session-local group definitions."""
        with self._camera_management_lock:
            return dict(self.configuration.fiducial_groups)

    def set_fiducial_group(
        self,
        group_name: str,
        dictionary: str,
        marker_size_mm: float,
    ) -> FiducialGroup:
        """Add or replace one tracking group without restarting the service."""
        group = FiducialGroup(dictionary, marker_size_mm)
        with self._camera_management_lock:
            groups = dict(self.configuration.fiducial_groups)
            groups[group_name] = group
            self.configuration = replace(self.configuration, fiducial_groups=groups)
            self._invalidate_spatial_state()
            return group

    def remove_fiducial_group(self, group_name: str) -> None:
        """Remove one session-local tracking group and its observations."""
        with self._camera_management_lock:
            if group_name not in self.configuration.fiducial_groups:
                raise KeyError(group_name)
            groups = dict(self.configuration.fiducial_groups)
            del groups[group_name]
            self.configuration = replace(self.configuration, fiducial_groups=groups)
            self._invalidate_spatial_state()

    def get_tracking_diagnostics(self) -> dict[str, object]:
        """Return the current tracking worker and authority diagnostics."""
        with self._camera_management_lock:
            cameras = self._get_session_cameras()
            camera_states: dict[str, CameraGeneration] = {}
            camera_transforms: dict[str, MatrixLike] = {}
            if cameras is not None:
                camera_states, camera_transforms = self._get_tracking_camera_inputs_locked(
                    cameras,
                    self._projector_output_descriptor,
                )
            metric_record = self.metric_calibration_registry.get_record()
            metric_usable = self.metric_calibration_registry.is_usable(
                self._projector_output_descriptor,
            )
        tracking_thread = self._tracking_thread
        return {
            'thread_alive': tracking_thread is not None and tracking_thread.is_alive(),
            'error': None if self._tracking_error is None else str(self._tracking_error),
            'metric_record_present': metric_record is not None,
            'metric_registry_usable': metric_usable,
            'camera_states': {
                slot: {
                    'lifecycle_generation': state.lifecycle_generation,
                    'calibration_generation': state.calibration_generation,
                    'is_available': state.is_available,
                    'is_open': state.is_open,
                    'is_calibrated': state.is_calibrated,
                }
                for slot, state in sorted(camera_states.items())
            },
            'camera_transform_slots': sorted(camera_transforms),
        }

    def get_benchmark_metrics(self) -> dict[str, object]:
        """Return process-local runtime counters for performance evidence."""
        return self._benchmark_metrics.snapshot()

    def record_benchmark_timing(
        self,
        component: str,
        elapsed_seconds: float,
        cpu_seconds: float,
        stalled: bool = False,
    ) -> None:
        """Accept display timings without coupling the renderer to benchmark code."""
        self._benchmark_metrics.record_timing(
            component,
            elapsed_seconds,
            cpu_seconds,
            stalled,
        )

    @property
    def spatial_tracker(self) -> SpatialTracker:
        return self._spatial_tracker

    @property
    def tracking_thread(self) -> threading.Thread | None:
        return self._tracking_thread

    @property
    def tracking_error(self) -> Exception | None:
        return self._tracking_error

    def run_tracking_cycle(self) -> SpatialState:
        """Run one retained-frame tracking cycle without touching camera handles."""
        return self._run_tracking_cycle()

    update_spatial_tracking = run_tracking_cycle
    track_once = run_tracking_cycle

    @property
    def overlay_intensity(self) -> float:
        with self._render_state_lock:
            return self._overlay_intensity

    @property
    def global_overlay_intensity(self) -> float:
        return self.overlay_intensity

    def get_overlay_intensity(self) -> float:
        return self.overlay_intensity

    def get_global_overlay_intensity(self) -> float:
        return self.overlay_intensity

    def set_overlay_intensity(self, intensity: object) -> float:
        """Set the session-local ordinary-overlay presentation intensity."""
        checked_intensity = normalise_overlay_intensity(intensity)
        with self._render_state_lock:
            self._overlay_intensity = checked_intensity
        return checked_intensity

    set_global_overlay_intensity = set_overlay_intensity

    def update_spatial_state(self, spatial_state: SpatialState) -> SpatialState:
        """Publish a complete spatial state for subsequent dynamic resolution."""
        if not isinstance(spatial_state, SpatialState):
            raise ValueError('spatial_state must be SpatialState')
        with self._tracking_control_lock:
            self._tracking_generation += 1
            with self._render_state_lock:
                self._spatial_state = spatial_state
                self._tracker_published_spatial_state = None
        return spatial_state

    set_spatial_state = update_spatial_state

    def _get_render_camera_statuses_locked(
        self,
        cameras: Sequence[SessionCamera] | None,
    ) -> list[CameraStatus]:
        """Read runtime status without falling back to a fresh camera frame."""
        get_statuses = getattr(self.camera_runtime, 'get_statuses', None)
        if callable(get_statuses):
            return self.get_camera_statuses()
        if cameras is None:
            return []
        get_status = getattr(self.camera_runtime, 'get_status', None)
        statuses: list[CameraStatus] = []
        for camera in cameras:
            if callable(get_status):
                status = get_status(camera.slot_id)
                if not isinstance(status, CameraStatus):
                    raise CameraUnavailableError(
                        f'Camera {camera.slot_id!r} returned an invalid status',
                    )
                _validate_camera_status(status, camera.slot_id)
                status = status._replace(
                    calibration_status=self._get_calibration_status(status),
                )
            else:
                status = CameraStatus(
                    camera.slot_id,
                    None if camera.device_info is None else camera.device_info.device_id,
                    {
                        SessionCameraState.OPEN: RuntimeStatus.AVAILABLE,
                        SessionCameraState.CLOSED: RuntimeStatus.STOPPED,
                        SessionCameraState.UNAVAILABLE: RuntimeStatus.UNAVAILABLE,
                    }[camera.state],
                    camera.calibration_status,
                    None if camera.device_info is None else camera.device_info.native_resolution,
                    0 if camera.frame_metadata is None else camera.frame_metadata.frame_counter,
                    camera.error_message,
                )
            statuses.append(status)
        return statuses

    def _capture_render_camera_inputs_locked(
        self,
    ) -> tuple[
        tuple[CameraRenderSnapshot, ...],
        tuple[CameraArea, ...],
        MetricCalibrationStatus,
        MetricRulerOverlay | None,
        RedCircleOverlay | None,
        bool,
        bool,
    ]:
        """Capture only retained diagnostic inputs for a complete render snapshot."""
        cameras = self._get_session_cameras()
        statuses = self._get_render_camera_statuses_locked(cameras)
        statuses_by_slot = {status.logical_name: status for status in statuses}
        latest_frames: Mapping[str, Frame] = {}
        snapshot_latest_frames = getattr(
            self.camera_runtime,
            'snapshot_latest_frames',
            None,
        )
        if callable(snapshot_latest_frames):
            candidate_frames = snapshot_latest_frames()
            if isinstance(candidate_frames, Mapping):
                latest_frames = {
                    slot_id: frame
                    for slot_id, frame in candidate_frames.items()
                    if isinstance(slot_id, str) and isinstance(frame, Frame)
                }

        camera_snapshots: list[CameraRenderSnapshot] = []
        if cameras is None or len(cameras) == 0:
            for status in statuses:
                camera_snapshots.append(
                    CameraRenderSnapshot(
                        status.logical_name,
                        status.logical_name,
                        status,
                        _get_camera_state_from_status(status),
                        0,
                        latest_frames.get(status.logical_name),
                    ),
                )
        else:
            for camera in cameras:
                status = statuses_by_slot.get(camera.slot_id)
                if status is None:
                    continue
                calibration_metrics = (
                    camera.calibration.metrics
                    if isinstance(camera.calibration, PersistedCalibration)
                    else None
                )
                camera_snapshots.append(
                    CameraRenderSnapshot(
                        camera.slot_id,
                        camera.display_name,
                        status,
                        camera.state,
                        camera.lifecycle_generation,
                        latest_frames.get(camera.slot_id),
                        calibration_metrics,
                    ),
                )

        get_metric_status = getattr(
            self.metric_calibration_registry,
            'get_status',
            None,
        )
        if callable(get_metric_status):
            metric_state = get_metric_status(self._projector_output_descriptor)
        else:
            metric_record = getattr(self.metric_calibration_registry, 'get_record', lambda: None)()
            metric_state = getattr(
                metric_record,
                'state',
                MetricCalibrationStatus.UNCALIBRATED,
            )
        if not isinstance(metric_state, MetricCalibrationStatus):
            metric_state = MetricCalibrationStatus.UNCALIBRATED
        metric_ruler = (
            self._metric_ruler
            if metric_state is MetricCalibrationStatus.CALIBRATED
            else None
        )
        with self._calibration_capture_lock:
            calibration_pattern_visible = (
                self._calibration_pattern_hold or self._calibration_capture_count > 0
            )
        with self._metric_capture_lock:
            metric_capture_active = self._metric_capture_count > 0
        return (
            tuple(camera_snapshots),
            tuple(self._get_camera_areas_locked())
            if cameras is not None and len(cameras) > 0
            else (),
            metric_state,
            metric_ruler,
            self.point_service.overlay,
            calibration_pattern_visible,
            metric_capture_active,
        )

    def get_render_snapshot(self) -> RenderSnapshot:
        """Resolve retained requests against one spatial generation."""
        with measure_timing(
            self._benchmark_metrics,
            'spatial_resolution_materialisation',
        ):
            return self._get_render_snapshot()

    def _get_render_snapshot(self) -> RenderSnapshot:
        """Resolve retained requests against one spatial generation."""
        with self._camera_management_lock:
            self._invalidate_unusable_overlay_dependencies()
            self._invalidate_stale_spatial_authorities_locked()
        for _attempt in range(2):
            with self._camera_management_lock:
                render_camera_inputs = self._capture_render_camera_inputs_locked()
                descriptor = self._projector_output_descriptor
                metric_generation = self._metric_capture_generation
            (
                camera_snapshots,
                projector_areas,
                metric_state,
                metric_ruler,
                point_overlay,
                pattern_visible,
                metric_capture_active,
            ) = render_camera_inputs
            spatial_state = self.get_spatial_state()
            registry_generation = self.overlay_registry.generation
            registry_entries = self.overlay_registry.snapshot()
            source_overlays = self.overlay_registry.resolve(
                lambda request: self._materialise_overlay_request(
                    request,
                    spatial_state,
                    descriptor,
                ),
            )
            protected_regions = _get_render_protection_regions(
                spatial_state,
                self.configuration.fiducial_protection_margin_mm,
            )
            with self._render_state_lock:
                global_overlay_intensity = self._overlay_intensity
            overlays = tuple(
                entry._replace(
                    materialised_primitives=materialise_presentation(
                        entry.materialised_primitives,
                        global_overlay_intensity,
                        protected_regions,
                    ),
                )
                for entry in source_overlays
            )
            snapshot: RenderSnapshot | None = None
            with self._camera_management_lock:
                self._invalidate_stale_spatial_authorities_locked()
                with self._render_state_lock:
                    if descriptor != self._projector_output_descriptor:
                        continue
                    if metric_generation != self._metric_capture_generation:
                        continue
                    if spatial_state is not self._spatial_state:
                        continue
                    if global_overlay_intensity != self._overlay_intensity:
                        continue

                    def publish_snapshot() -> None:
                        nonlocal snapshot
                        self._render_generation += 1
                        snapshot = RenderSnapshot(
                            self._render_generation,
                            registry_generation,
                            spatial_state.generation,
                            spatial_state,
                            descriptor,
                            overlays,
                            metric_generation,
                            registry_entries,
                            global_overlay_intensity,
                            protected_regions,
                            camera_snapshots,
                            projector_areas,
                            metric_state,
                            metric_ruler,
                            point_overlay,
                            pattern_visible,
                            metric_capture_active,
                            self.calibration_pattern,
                        )
                        self._render_snapshot = snapshot

                    if self.overlay_registry.publish_if_generation(
                        registry_generation,
                        publish_snapshot,
                    ):
                        assert snapshot is not None
                        return snapshot
        with self._camera_management_lock:
            current_descriptor = self._projector_output_descriptor
            current_metric_generation = self._metric_capture_generation
        with self._render_state_lock:
            current_spatial_state = self._spatial_state
            current_overlay_intensity = self._overlay_intensity
            previous_snapshot = self._render_snapshot
        current_registry_generation = self.overlay_registry.generation
        if previous_snapshot is not None and (
            previous_snapshot.registry_generation == current_registry_generation
            and previous_snapshot.spatial_state is current_spatial_state
            and previous_snapshot.projector_output_descriptor == current_descriptor
            and previous_snapshot.metric_generation == current_metric_generation
            and previous_snapshot.global_overlay_intensity == current_overlay_intensity
        ):
            return previous_snapshot
        raise RuntimeError('Authorities changed while assembling a render snapshot')

    capture_render_snapshot = get_render_snapshot
    capture_render_input = get_render_snapshot
    get_render_input = get_render_snapshot
    render_snapshot = get_render_snapshot

    def show_overlay(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_overlay_visibility(selector, True)

    def hide_overlay(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_overlay_visibility(selector, False)

    def _set_overlay_visibility(
        self,
        selector: str | uuid.UUID,
        visible: bool,
    ) -> OverlayEntry:
        with self._camera_management_lock:
            self._invalidate_unusable_overlay_dependencies()
            entry = self.overlay_registry.get(selector)
        if entry.visible == visible:
            return entry
        request = entry.request.model_copy(update={'visible': visible})
        entries = self.apply_overlay_batch(({
            'op': 'update',
            'selector': entry.id,
            'request': request,
        },))
        return next(updated_entry for updated_entry in entries if updated_entry.id == entry.id)

    def remove_overlay(self, selector: str | uuid.UUID) -> OverlayEntry:
        with self._camera_management_lock:
            return self.overlay_registry.remove(selector)

    def clear_overlays(self) -> None:
        with self._camera_management_lock:
            self.overlay_registry.clear()

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
            return self._calibration_pattern_hold or self._calibration_capture_count > 0

    def show_calibration_pattern(self) -> None:
        """Keep the calibration pattern projected for manual camera adjustment."""
        with self._calibration_capture_lock:
            self._calibration_pattern_hold = True

    def hide_calibration_pattern(self) -> None:
        """Stop the manual calibration-pattern display without changing calibration."""
        with self._calibration_capture_lock:
            self._calibration_pattern_hold = False

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
            self._tracking_stop_event = threading.Event()
            self._tracking_error = None
            self._tracking_thread = threading.Thread(
                target=self._tracking_loop,
                args=(self._tracking_stop_event,),
                daemon=True,
                name='multivision-fiducial-tracking',
            )
            self._is_running = True
            self._tracking_thread.start()

    def shutdown(self) -> None:
        """Stop capture and tracking workers and release every camera handle."""
        with self._lifecycle_lock:
            if self._has_stopped:
                return
            shutdown_error: Exception | None = None
            self._invalidate_spatial_state()
            tracking_thread = self._tracking_thread
            self._tracking_stop_event.set()
            if tracking_thread is not None:
                tracking_thread.join(TRACKING_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            if tracking_thread is not None and tracking_thread.is_alive():
                shutdown_error = RuntimeError(
                    'Fiducial tracking worker did not stop within '
                    f'{TRACKING_WORKER_SHUTDOWN_TIMEOUT_SECONDS} seconds',
                )
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

    def _tracking_loop(self, stop_event: threading.Event) -> None:
        interval_seconds = 1.0 / self.configuration.fiducial_tracking_rate_hz
        next_run_seconds = time.monotonic()
        while not stop_event.is_set():
            wait_seconds = max(0.0, next_run_seconds - time.monotonic())
            if stop_event.wait(wait_seconds):
                return
            next_run_seconds += interval_seconds
            try:
                self._run_tracking_cycle()
            except Exception as ex:  # noqa: BLE001 (Tracking must not stop capture.)
                self._tracking_error = ex
            if next_run_seconds < time.monotonic() - interval_seconds:
                next_run_seconds = time.monotonic()

    def _run_tracking_cycle(self) -> SpatialState:
        with measure_timing(self._benchmark_metrics, 'detection_tracking'):
            return self._run_tracking_cycle_unmeasured()

    def _run_tracking_cycle_unmeasured(self) -> SpatialState:
        cycle = self._get_tracking_cycle()
        if cycle is None:
            return self.spatial_state
        if not self._tracking_inputs_are_current(cycle):
            return self.spatial_state

        observations = []
        detector_failures: list[object] = []
        group_definitions = self.configuration.fiducial_groups
        for camera_slot in sorted(cycle.frames_by_camera):
            if cycle.frames_by_camera[camera_slot].frame_counter < 0:
                continue
            try:
                result = detect_configured_fiducial_observations(
                    cycle.frames_by_camera[camera_slot].data,
                    group_definitions,
                    self.tag_detector_factory,
                    cycle.camera_transforms.get(camera_slot),
                    metric_calibration=cycle.metric_calibration,
                    camera_slot=camera_slot,
                    camera_lifecycle_generation=cycle.camera_states[
                        camera_slot
                    ].lifecycle_generation,
                    camera_calibration_generation=cycle.camera_states[
                        camera_slot
                    ].calibration_generation,
                    frame_counter=cycle.frames_by_camera[camera_slot].frame_counter,
                    received_monotonic_seconds=cycle.received_monotonic_seconds,
                )
            except Exception as ex:  # noqa: BLE001 (One bad frame must not kill tracking.)
                detector_failures.extend(
                    GroupTrackingError(
                        group,
                        definition.dictionary,
                        'TRACKING_ERROR',
                        str(ex),
                    )
                    for group, definition in group_definitions.items()
                )
                continue
            observations.extend(result.observations)
            detector_failures.extend(result.errors)

        # Runtime-owned disconnects do not acquire the service lifecycle lock.
        # Recheck their authority after detection, before publication.
        if not self._tracking_inputs_are_current(cycle):
            self._invalidate_spatial_state_for_tracking_generation(
                cycle.tracking_generation,
            )
            return self.spatial_state

        with self._tracking_control_lock:
            if cycle.tracking_generation != self._tracking_generation:
                return self.spatial_state
            spatial_state = self._spatial_tracker.update(
                observations,
                detector_failures,
                metric_calibration=cycle.metric_calibration,
                camera_states=cycle.camera_states,
                frames_by_camera=cycle.frames_by_camera,
            )
            if cycle.tracking_generation != self._tracking_generation:
                return self.spatial_state
            with self._render_state_lock:
                self._spatial_state = spatial_state
                self._tracker_published_spatial_state = spatial_state
            return spatial_state

    def _get_tracking_cycle(self) -> _TrackingCycle | None:
        with self._camera_management_lock:
            cameras = self._get_session_cameras()
            if cameras is None:
                return None
            descriptor = self._projector_output_descriptor
            camera_states, camera_transforms = (
                self._get_tracking_camera_inputs_locked(cameras, descriptor)
            )
            metric_calibration = None
            if self.metric_calibration_registry.is_usable(descriptor):
                metric_calibration = self.metric_calibration_registry.get_record()

        with self._render_state_lock:
            metric_was_published = self._spatial_state.metric_calibration is not None
        if metric_calibration is None and metric_was_published:
            self._invalidate_spatial_metric_state()

        frames_by_camera = self._get_tracking_frames(
            tuple(
                camera_slot
                for camera_slot, state in camera_states.items()
                if state.is_available and state.is_open
            ),
        )
        if len(frames_by_camera) > 0:
            self._benchmark_metrics.increment(
                'tracking_frames',
                len(frames_by_camera),
            )
            self._benchmark_metrics.increment('tracking_cycles_with_frames')
        try:
            received_monotonic_seconds = self._tracking_clock()
        except Exception as ex:  # noqa: BLE001 (Injected clocks are a test boundary.)
            raise ValueError('tracking_clock failed') from ex
        if not is_finite_real(received_monotonic_seconds):
            raise ValueError('tracking_clock must return a finite number')
        with self._tracking_control_lock:
            self._tracking_generation += 1
            tracking_generation = self._tracking_generation
        return _TrackingCycle(
            tracking_generation,
            self._metric_capture_generation,
            descriptor,
            metric_calibration,
            camera_states,
            camera_transforms,
            frames_by_camera,
            float(received_monotonic_seconds),
        )

    def _get_tracking_camera_inputs_locked(
        self,
        cameras: Sequence[SessionCamera],
        descriptor: ProjectorOutputDescriptor,
    ) -> tuple[dict[str, CameraGeneration], dict[str, MatrixLike]]:
        camera_states: dict[str, CameraGeneration] = {}
        camera_transforms: dict[str, MatrixLike] = {}
        get_status = getattr(self.camera_runtime, 'get_status', None)
        for camera in cameras:
            status: CameraStatus | None = None
            runtime_is_available = camera.state is SessionCameraState.OPEN
            if callable(get_status):
                try:
                    candidate_status = get_status(camera.slot_id)
                except Exception:  # noqa: BLE001 (Runtime status is a hardware boundary.)
                    runtime_is_available = False
                else:
                    if isinstance(candidate_status, CameraStatus):
                        try:
                            _validate_camera_status(candidate_status, camera.slot_id)
                        except CameraUnavailableError:
                            runtime_is_available = False
                        else:
                            status = candidate_status
                            runtime_is_available = (
                                status.runtime_status is RuntimeStatus.AVAILABLE
                            )
                    else:
                        runtime_is_available = False
            is_open = camera.state is SessionCameraState.OPEN
            calibration = camera.calibration
            is_calibrated = (
                is_open
                and runtime_is_available
                and _is_tracking_calibration_usable(
                    camera,
                    status,
                    calibration,
                    descriptor,
                    self.configuration.calibration_version,
                )
            )
            calibration_generation = self._get_camera_calibration_generation(
                camera.slot_id,
                calibration,
            )
            camera_states[camera.slot_id] = CameraGeneration(
                camera.lifecycle_generation,
                calibration_generation,
                runtime_is_available,
                is_open,
                is_calibrated,
            )
            if not is_calibrated:
                continue
            try:
                camera_transforms[camera.slot_id] = validate_homography(
                    getattr(calibration, 'camera_to_projector'),
                )
            except (AttributeError, TypeError, ValueError):
                continue
        return camera_states, camera_transforms

    def _get_camera_calibration_generation(
        self,
        camera_slot: str,
        calibration: object | None,
    ) -> int:
        signature = _camera_calibration_signature(calibration)
        with self._tracking_control_lock:
            previous_signature = self._camera_calibration_signatures.get(camera_slot)
            if previous_signature is None and camera_slot not in self._camera_calibration_signatures:
                self._camera_calibration_signatures[camera_slot] = signature
                return self._camera_calibration_generations.get(camera_slot, 0)
            if previous_signature == signature:
                return self._camera_calibration_generations.get(camera_slot, 0)
            next_generation = self._camera_calibration_generations.get(camera_slot, 0) + 1
            self._camera_calibration_signatures[camera_slot] = signature
            self._camera_calibration_generations[camera_slot] = next_generation
            return next_generation

    def _get_tracking_frames(
        self,
        requested_camera_slots: Sequence[str],
    ) -> dict[str, Frame]:
        requested = set(requested_camera_slots)
        if len(requested) == 0:
            return {}
        snapshot_latest_frames = getattr(
            self.camera_runtime,
            'snapshot_latest_frames',
            None,
        )
        frames: Mapping[str, object] | None = None
        if callable(snapshot_latest_frames):
            try:
                candidate_frames = snapshot_latest_frames()
            except Exception:  # noqa: BLE001 (A missing retained frame is recoverable.)
                candidate_frames = None
            if isinstance(candidate_frames, Mapping):
                frames = candidate_frames
        if frames is not None:
            return {
                camera_slot: frame
                for camera_slot, frame in frames.items()
                if camera_slot in requested and isinstance(frame, Frame)
            }

        snapshot = getattr(self.camera_runtime, 'snapshot', None)
        if not callable(snapshot):
            return {}
        result: dict[str, Frame] = {}
        for camera_slot in sorted(requested):
            try:
                frame = snapshot(camera_slot)
            except Exception:  # noqa: BLE001 (A camera may lose its latest frame.)
                continue
            if isinstance(frame, Frame):
                result[camera_slot] = frame
        return result

    def _tracking_inputs_are_current(self, cycle: _TrackingCycle) -> bool:
        with self._camera_management_lock:
            if cycle.projector_output_descriptor != self._projector_output_descriptor:
                return False
            if cycle.metric_generation != self._metric_capture_generation:
                return False
            cameras = self._get_session_cameras()
            if cameras is None:
                return False
            current_states, _transforms = self._get_tracking_camera_inputs_locked(
                cameras,
                self._projector_output_descriptor,
            )
            if current_states != dict(cycle.camera_states):
                return False
            current_metric = None
            if self.metric_calibration_registry.is_usable(
                self._projector_output_descriptor,
            ):
                current_metric = self.metric_calibration_registry.get_record()
            if not (
                current_metric is cycle.metric_calibration
                or current_metric == cycle.metric_calibration
            ):
                return False
            return True

    def get_camera_status(self, logical_name: str) -> CameraStatus:
        with self._camera_management_lock:
            resolved_slot = self._resolve_camera_reference(logical_name)
            runtime_status = self.camera_runtime.get_status(resolved_slot)
            if not isinstance(runtime_status, CameraStatus):
                raise CameraUnavailableError(
                    f'Camera {logical_name!r} returned an invalid status',
                )
            _validate_camera_status(runtime_status, resolved_slot)
            if runtime_status.runtime_status is not RuntimeStatus.AVAILABLE:
                self.overlay_registry.invalidate_camera(resolved_slot)
                self._invalidate_spatial_state()
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
                if status.runtime_status is not RuntimeStatus.AVAILABLE:
                    self.overlay_registry.invalidate_camera(
                        self._resolve_camera_reference(status.logical_name),
                    )
                    self._invalidate_spatial_state()
                calibration_status = self._get_calibration_status(status)
                checked_statuses.append(status._replace(calibration_status=calibration_status))
            return checked_statuses

    def get_calibration_status_snapshot(
        self,
    ) -> tuple[CalibrationStage, MetricCalibrationStatus, list[CameraStatus]]:
        """Return one consistent aggregate, metric and camera status snapshot."""
        with self._camera_management_lock:
            metric_status = self.metric_calibration_registry.get_status(
                self._projector_output_descriptor,
            )
            camera_statuses = self.get_camera_statuses()
            if metric_status is MetricCalibrationStatus.CALIBRATED:
                stage = CalibrationStage.METRIC_CALIBRATED
            elif metric_status is MetricCalibrationStatus.STALE:
                stage = CalibrationStage.STALE
            elif any(
                status.calibration_status is CalibrationStatus.CALIBRATED
                for status in camera_statuses
            ):
                stage = CalibrationStage.CALIBRATED
            elif any(
                status.calibration_status is CalibrationStatus.STALE
                for status in camera_statuses
            ):
                stage = CalibrationStage.STALE
            elif any(
                status.calibration_status is CalibrationStatus.UNVERIFIED
                for status in camera_statuses
            ):
                stage = CalibrationStage.UNVERIFIED
            else:
                stage = CalibrationStage.UNCALIBRATED
            return stage, metric_status, camera_statuses

    def get_calibration_stage(self) -> CalibrationStage:
        """Return the highest complete calibration stage for the session."""
        stage, _metric_status, _camera_statuses = self.get_calibration_status_snapshot()
        return stage

    def snapshot(self, logical_name: str) -> Frame:
        """Return the latest frame retained by the persistent camera runtime."""
        frame = self.camera_runtime.snapshot(self._resolve_camera_reference(logical_name))
        if not isinstance(frame, Frame):
            raise FrameCaptureError(
                f'Camera {logical_name!r} returned an invalid frame',
            )
        return frame

    def inspect_tags(
        self,
        camera_reference: str,
        dictionary_name: str | None = None,
    ) -> TagInspectionResult:
        """Inspect every valid tag in one camera's retained latest frame."""
        if not isinstance(camera_reference, str) or len(camera_reference) == 0:
            raise ValueError('camera_reference must be a non-empty string')
        selected_dictionary = (
            self.configuration.tag_dictionary
            if dictionary_name is None
            else dictionary_name
        )
        selected_dictionary = validate_tag_dictionary(selected_dictionary)
        with self._camera_management_lock:
            resolved_slot = self._resolve_camera_reference(camera_reference)
            session_camera = self._get_session_camera(resolved_slot)
            if (
                session_camera is not None
                and session_camera.state is not SessionCameraState.OPEN
            ):
                raise CameraUnavailableError(
                    session_camera.error_message
                    or f'Camera {camera_reference!r} is not available',
                )
            get_status = getattr(self.camera_runtime, 'get_status', None)
            status: CameraStatus | None = None
            if callable(get_status):
                try:
                    status = get_status(resolved_slot)
                except CameraUnavailableError:
                    raise
                except (KeyError, ValueError) as ex:
                    raise CameraUnavailableError(
                        f'Camera {camera_reference!r} is not configured',
                    ) from ex
                if not isinstance(status, CameraStatus):
                    raise CameraUnavailableError(
                        f'Camera {camera_reference!r} returned an invalid status',
                    )
                _validate_camera_status(status, resolved_slot)
                if status.runtime_status is not RuntimeStatus.AVAILABLE:
                    raise CameraUnavailableError(
                        status.error_message
                        or f'Camera {camera_reference!r} is not available',
                    )
            frame = self.snapshot(resolved_slot)
            if (
                not isinstance(frame.frame_counter, int)
                or isinstance(frame.frame_counter, bool)
                or frame.frame_counter < 0
                or not is_finite_real(frame.captured_at_seconds)
            ):
                raise FrameCaptureError('Camera returned malformed frame metadata')
            captured_at_seconds = float(frame.captured_at_seconds)
            if session_camera is not None:
                camera_id = session_camera.slot_id
            elif status is not None:
                camera_id = status.device_id
            else:
                camera_id = None

        # Detection and projector geometry use the retained frame only – neither
        # operation should hold the service-wide camera lifecycle lock.
        detector = self.tag_detector_factory(selected_dictionary)
        camera_observations = detect_tag_observations(frame.data, detector)
        projection_status = self._get_tag_projection_status(resolved_slot)
        tags = tuple(
            self._build_tag_observation(
                resolved_slot,
                observation,
                projection_status,
            )
            for observation in camera_observations
        )
        return TagInspectionResult(
            camera_reference,
            camera_id,
            selected_dictionary,
            frame.frame_counter,
            captured_at_seconds,
            tags,
            projection_status,
        )

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
            # Reject an in-flight detector before the runtime releases its handle.
            self.overlay_registry.invalidate_camera(slot_id)
            self._invalidate_spatial_state()
            self.point_service.clear_overlay_for_camera(slot_id)
            camera = self.camera_runtime.close_camera(slot_id)
            if not isinstance(camera, SessionCamera):
                raise SessionCameraError(
                    'Camera runtime returned an invalid closed session camera',
                )
            if self._calibration_master_camera == camera.slot_id:
                self._calibration_master_camera = None
            return camera

    def open_camera(self, slot_id: str) -> SessionCamera:
        """Reopen one closed session slot with fresh spatial state."""
        with self._camera_management_lock:
            self.overlay_registry.invalidate_camera(slot_id)
            self._invalidate_spatial_state()
            self.point_service.clear_overlay_for_camera(slot_id)
            camera = self.camera_runtime.open_camera(slot_id)
            if not isinstance(camera, SessionCamera):
                raise SessionCameraError(
                    'Camera runtime returned an invalid opened session camera',
                )
            return camera

    def update_projector_descriptor(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | Resolution,
        output_identity: str = 'default',
    ) -> ProjectorOutputDescriptor:
        """Change the active output and invalidate all dependent geometry."""
        with self._spatial_capture_operation_lock:
            return self._update_projector_descriptor_locked(
                projector_output_descriptor,
                output_identity,
            )

    def _update_projector_descriptor_locked(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor | Resolution,
        output_identity: str = 'default',
    ) -> ProjectorOutputDescriptor:
        """Apply an output change while spatial capture is excluded."""
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
            # Validate the replacement pattern before mutating any output state.
            updated_calibration_pattern = _build_projector_calibration_pattern(
                checked_descriptor.projector_resolution,
            )
            # Complete the authority updates first so a rejected transition does
            # not leave the overlay registry describing a different output.
            self.camera_runtime.mark_calibrations_stale(checked_descriptor)
            self.calibration_registry.update_projector_descriptor(checked_descriptor)
            self.metric_calibration_registry.update_projector_descriptor(
                checked_descriptor,
            )
            self._metric_capture_generation += 1
            self.overlay_registry.invalidate_projector_output(checked_descriptor)
            self._invalidate_spatial_state()
            self._metric_ruler = None
            self.point_service.clear_overlay()
            self._projector_output_descriptor = checked_descriptor
            self.calibration_pattern = updated_calibration_pattern
            self._calibration_master_camera = None
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
                self.overlay_registry.invalidate_metric()
                self.metric_calibration_registry.clear()
                self._invalidate_spatial_metric_state()
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
                    self.configuration.metric_calibration_thresholds
                    .min_capture_marker_ratio,
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
                    record = self.metric_calibration_registry.register(
                        result,
                        self._projector_output_descriptor,
                        observation_camera_slot=camera.slot_id,
                        observation_camera_calibration=calibration,
                    )
                    self._metric_capture_generation += 1
                    # A detector that started before recalibration must not
                    # republish against the newly registered metric authority.
                    self._invalidate_spatial_metric_state()
                    return record
            finally:
                self._finish_metric_capture()

    def _canonicalise_overlay_operations(
        self,
        operations: Sequence[object],
    ) -> tuple[object, ...]:
        if not isinstance(operations, Sequence) or isinstance(
            operations,
            (str, bytes, bytearray),
        ):
            raise ValueError('operations must be an ordered sequence')
        canonical_operations: list[object] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                canonical_operations.append(operation)
                continue
            operation_copy = dict(operation)
            request_key = next(
                (
                    key
                    for key in ('request', 'specification', 'spec', 'overlay')
                    if key in operation_copy
                ),
                None,
            )
            request = operation_copy.get(request_key) if request_key is not None else None
            if isinstance(request, (GridRequest, CircleRequest, RectRequest, TextRequest,
                                    LineRequest, RulerRequest, ArrowRequest)):
                self._validate_overlay_fiducial_groups(request)
                operation_copy[request_key] = self._canonicalise_overlay_camera_references(
                    request,
                )
            canonical_operations.append(operation_copy)
        return tuple(canonical_operations)

    def _validate_overlay_fiducial_groups(
        self,
        request: AnyOverlayRequest,
    ) -> None:
        for reference in get_overlay_point_references(request):
            if (
                isinstance(reference, FiducialAnchor)
                and reference.group not in self.configuration.fiducial_groups
            ):
                raise ValueError(f'Unknown fiducial group {reference.group!r}')

    def _materialise_overlay_request(
        self,
        request: AnyOverlayRequest,
        spatial_state: SpatialState | None = None,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> ProjectorMaterialisation | None:
        """Take short authority snapshots, then build geometry without camera locks."""
        if not isinstance(request, (GridRequest, CircleRequest, RectRequest, TextRequest,
                                    LineRequest, RulerRequest, ArrowRequest)):
            raise ValueError('request must be an overlay request')
        with self._camera_management_lock:
            descriptor = projector_output_descriptor or self._projector_output_descriptor
            current_spatial_state = spatial_state
            if current_spatial_state is None:
                with self._render_state_lock:
                    current_spatial_state = self._spatial_state
            camera_dependencies, metric_dependency = get_overlay_dependencies(request)
            camera_authorities = (
                self._get_overlay_camera_authorities(camera_dependencies)
                if len(camera_dependencies) > 0
                else None
            )
            metric_calibration = None
            if metric_dependency:
                state_metric_calibration = getattr(
                    current_spatial_state,
                    'metric_calibration',
                    None,
                )
                if state_metric_calibration is not None:
                    metric_calibration = _get_current_spatial_metric_calibration(
                        current_spatial_state,
                        descriptor,
                    )
                elif self.metric_calibration_registry.is_usable(descriptor):
                    metric_calibration = self.metric_calibration_registry.get_record()
        if metric_dependency and metric_calibration is None:
            if is_dynamic_overlay_request(request):
                return None
            # Candidate construction must not invalidate already-published entries.
            # Authority transitions perform that invalidation at their own boundary.
            self._require_usable_metric_calibration()
            raise InvalidCalibrationStateError(
                'Metric calibration became unavailable while resolving the overlay',
            )
        try:
            return materialise_overlay(
                request,
                descriptor.projector_resolution,
                camera_authorities,
                metric_calibration,
                self.configuration.overlay_limits,
                current_spatial_state,
            )
        except ValueError as ex:
            if is_dynamic_overlay_request(request) and _is_unavailable_overlay_authority_error(ex):
                return None
            raise

    def _invalidate_unusable_overlay_dependencies(self) -> None:
        entries = self.overlay_registry.list()
        camera_dependencies = {
            camera_id
            for entry in entries
            for camera_id in entry.camera_dependencies
        }
        if len(camera_dependencies) > 0:
            camera_authorities = self._get_overlay_camera_authorities(
                tuple(sorted(camera_dependencies)),
            )
            for camera_id in sorted(camera_dependencies - camera_authorities.keys()):
                self.overlay_registry.invalidate_camera(camera_id)
                self._invalidate_spatial_state()
        metric_is_usable = self.metric_calibration_registry.is_usable(
            self._projector_output_descriptor,
        )
        if not metric_is_usable:
            self.overlay_registry.invalidate_metric()
            self._invalidate_spatial_metric_state()

    def _get_overlay_camera_authorities(
        self,
        requested_camera_references: Sequence[str] | None = None,
    ) -> dict[str, MatrixLike]:
        cameras = self._get_session_cameras()
        if cameras is None:
            return {}
        requested_references = (
            None
            if requested_camera_references is None
            else set(requested_camera_references)
        )
        authorities: dict[str, MatrixLike] = {}
        for camera in cameras:
            if requested_references is not None and not {
                camera.slot_id,
                camera.display_name,
            }.intersection(requested_references):
                continue
            if camera.state is not SessionCameraState.OPEN:
                continue
            try:
                status = self.camera_runtime.get_status(camera.slot_id)
            except CameraUnavailableError:
                continue
            if not isinstance(status, CameraStatus):
                continue
            if status.runtime_status is not RuntimeStatus.AVAILABLE:
                continue
            if camera.calibration_status is not CalibrationStatus.CALIBRATED:
                continue
            calibration = camera.calibration
            if calibration is None:
                continue
            if isinstance(calibration, PersistedCalibration) and (
                calibration.calibration_scope != CalibrationScope.GLOBAL.value
            ):
                continue
            if isinstance(calibration, PersistedCalibration) and (
                calibration.camera_id != camera.slot_id
                or calibration.version != self.configuration.calibration_version
                or calibration.camera_resolution != status.native_resolution
                or calibration.projector_output_descriptor
                != self._projector_output_descriptor
            ):
                continue
            if (
                getattr(calibration, 'projector_output_descriptor', None)
                not in (None, self._projector_output_descriptor)
            ):
                continue
            if (
                status.native_resolution is not None
                and getattr(calibration, 'camera_resolution', status.native_resolution)
                != status.native_resolution
            ):
                continue
            try:
                camera_to_projector = validate_homography(
                    getattr(calibration, 'camera_to_projector'),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            authorities[camera.slot_id] = camera_to_projector
        return authorities

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

    def _canonicalise_overlay_camera_references(
        self,
        request: AnyOverlayRequest,
    ) -> AnyOverlayRequest:
        updates: dict[str, PointReference] = {}
        for field_name, field_value in request:
            if (
                not isinstance(field_value, PointReference)
                or field_value.space != 'camera_px'
            ):
                continue
            assert field_value.camera is not None
            camera_slot_id = self._resolve_camera_reference(field_value.camera)
            if camera_slot_id != field_value.camera:
                updates[field_name] = field_value.model_copy(
                    update={'camera': camera_slot_id},
                )
        if len(updates) == 0:
            return request
        return request.model_copy(update=updates)

    def _invalidate_stale_spatial_authorities_locked(self) -> None:
        """Notice runtime-owned authority changes before resolving an overlay."""
        with self._render_state_lock:
            spatial_state = self._spatial_state
        if len(spatial_state.camera_generations) > 0:
            cameras = self._get_session_cameras()
            if cameras is not None:
                current_states, _transforms = self._get_tracking_camera_inputs_locked(
                    cameras,
                    self._projector_output_descriptor,
                )
                if current_states != dict(spatial_state.camera_generations):
                    self._invalidate_spatial_state()
                    return
        if spatial_state.metric_calibration is not None and not self.metric_calibration_registry.is_usable(
            self._projector_output_descriptor,
        ):
            self._invalidate_spatial_metric_state()

    def _invalidate_spatial_state_for_tracking_generation(
        self,
        tracking_generation: int,
    ) -> None:
        with self._tracking_control_lock:
            if tracking_generation != self._tracking_generation:
                return
            self._tracking_generation += 1
            self._spatial_tracker.reset(None)
            with self._render_state_lock:
                spatial_state = self._spatial_state
                self._spatial_state = SpatialState.empty()._replace(
                    generation=spatial_state.generation + 1,
                )
                self._tracker_published_spatial_state = None

    def _invalidate_spatial_state(self) -> None:
        """Invalidate spatial evidence and reject every in-flight tracking pass."""
        with self._tracking_control_lock:
            self._tracking_generation += 1
            self._spatial_tracker.reset(None)
            with self._render_state_lock:
                spatial_state = self._spatial_state
                self._spatial_state = SpatialState.empty()._replace(
                    generation=spatial_state.generation + 1,
                )
                self._tracker_published_spatial_state = None

    def _invalidate_spatial_metric_state(self) -> None:
        """Drop metric-dependent spatial evidence without retaining marker ghosts."""
        with self._tracking_control_lock:
            self._tracking_generation += 1
            self._spatial_tracker.reset(None)
            with self._render_state_lock:
                spatial_state = self._spatial_state
                self._spatial_state = SpatialState.empty()._replace(
                    generation=spatial_state.generation + 1,
                )
                self._tracker_published_spatial_state = None

    def _require_overlay_metric_calibration(self) -> MetricCalibrationRecord:
        try:
            return self._require_usable_metric_calibration()
        except InvalidCalibrationStateError:
            # A failed metric transition must not leave old materialised
            # projector geometry available to the display.
            self.overlay_registry.invalidate_metric()
            self._invalidate_spatial_metric_state()
            raise

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
            detector = OpenCVArucoDetector(METRIC_TARGET.marker_family)
        get_consecutive_frames = getattr(
            self.camera_runtime,
            'get_consecutive_frames',
            None,
        )
        if callable(get_consecutive_frames):
            captured_frames = get_consecutive_frames(
                slot_id,
                METRIC_CAPTURE_CANDIDATE_FRAME_COUNT,
                METRIC_CAPTURE_WAIT_TIMEOUT_SECONDS,
            )
            if not isinstance(captured_frames, Sequence) or len(captured_frames) < (
                METRIC_CAPTURE_FRAME_COUNT
            ) or any(not isinstance(frame, Frame) for frame in captured_frames):
                raise CalibrationError(
                    'Camera runtime returned invalid consecutive metric frames',
                )
            captured_frames = _select_stable_frame_window(
                captured_frames,
                self.configuration.metric_calibration_thresholds
                .max_capture_white_balance_delta,
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

    def full_calibrate(self) -> FullCalibrationResult:
        """Calibrate and freshly verify every available camera in one run."""
        with self._spatial_capture_operation_lock:
            statuses = self._get_available_statuses()
            if len(statuses) == 0:
                raise CameraUnavailableError('No configured camera is available for calibration')
            workflow_pattern = self.calibration_pattern
            workflow_descriptor = self._projector_output_descriptor
            workflow_version = self.configuration.calibration_version
            for status in statuses:
                self._clear_area_before_calibration(status.logical_name)

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
                captures = self._capture_calibration_bursts(statuses, allow_local=True)
                if (
                    self.calibration_pattern != workflow_pattern
                    or self._projector_output_descriptor != workflow_descriptor
                    or self.configuration.calibration_version != workflow_version
                ):
                    raise CalibrationError(
                        'Projector calibration authority changed during capture',
                    )
                candidates: dict[str, tuple[CameraStatus, int, PersistedCalibration, CalibrationScope, _CalibrationCapture]] = {}
                reports: dict[str, FullCalibrationCameraResult] = {}
                for status in statuses:
                    capture = captures.get(status.logical_name)
                    if not isinstance(capture, _CalibrationCapture):
                        reports[status.logical_name] = FullCalibrationCameraResult(
                            status.logical_name,
                            CalibrationStatus.UNCALIBRATED,
                            None,
                            None,
                            captures.get(status.logical_name, 'No calibration burst was captured'),
                        )
                        continue
                    try:
                        result = calibrate_homography(
                            capture.correspondences,
                            self.calibration_pattern,
                            self.configuration.calibration_thresholds,
                            camera_resolution=status.native_resolution,
                        )
                        scope = CalibrationScope.GLOBAL
                    except CalibrationError as global_error:
                        local_thresholds = replace(
                            self.configuration.calibration_thresholds,
                            min_unique_tags=1,
                            min_spatial_coverage=0.0,
                        )
                        try:
                            result = calibrate_homography(
                                capture.correspondences,
                                self.calibration_pattern,
                                local_thresholds,
                                camera_resolution=status.native_resolution,
                            )
                        except CalibrationError as local_error:
                            reports[status.logical_name] = FullCalibrationCameraResult(
                                status.logical_name,
                                CalibrationStatus.UNCALIBRATED,
                                None,
                                None,
                                f'{global_error}; local fallback failed: {local_error}',
                            )
                            continue
                        scope = (
                            CalibrationScope.LOCAL_LOW_CONFIDENCE
                            if result.metrics.unique_tag_count == 1
                            else CalibrationScope.LOCAL
                        )
                    result = _attach_camera_capture_noise(result, capture.noise)
                    session_camera = self._get_session_camera(status.logical_name)
                    if session_camera is None or status.native_resolution is None:
                        reports[status.logical_name] = FullCalibrationCameraResult(
                            status.logical_name,
                            CalibrationStatus.UNCALIBRATED,
                            scope,
                            None,
                            'Camera metadata changed during calibration',
                        )
                        continue
                    record = PersistedCalibration.from_result(
                        result,
                        status.native_resolution,
                        self._projector_output_descriptor.projector_resolution,
                        version=self.configuration.calibration_version,
                        projector_output_descriptor=self._projector_output_descriptor,
                        camera_id=session_camera.slot_id,
                        calibration_scope=scope.value,
                    )
                    candidates[status.logical_name] = (
                        status,
                        session_camera.lifecycle_generation,
                        record,
                        scope,
                        capture,
                    )
                    reports[status.logical_name] = FullCalibrationCameraResult(
                        status.logical_name,
                        CalibrationStatus.UNVERIFIED,
                        scope,
                        record,
                    )

                verification_statuses = tuple(
                    status
                    for status in statuses
                    if status.logical_name in candidates
                )
                expected = {
                    status.logical_name: candidates[status.logical_name][2]
                    for status in verification_statuses
                }
                verification_captures = self._capture_calibration_bursts(
                    verification_statuses,
                    expected_calibrations=expected,
                    allow_local=True,
                )
                accepted: dict[str, FullCalibrationCameraResult] = {}
                for logical_name in expected:
                    status, lifecycle_generation, record, scope, _capture = candidates[
                        logical_name
                    ]
                    verification_capture = verification_captures.get(logical_name)
                    if not isinstance(verification_capture, _CalibrationCapture):
                        reports[logical_name] = FullCalibrationCameraResult(
                            logical_name,
                            CalibrationStatus.STALE,
                            scope,
                            record,
                            str(verification_capture),
                        )
                        continue
                    verification_thresholds = self.configuration.calibration_thresholds
                    if scope is not CalibrationScope.GLOBAL:
                        verification_thresholds = replace(
                            verification_thresholds,
                            min_unique_tags=1,
                            min_spatial_coverage=0.0,
                        )
                    verification_registry = CalibrationRegistry(
                        {record.camera_id: record},
                        calibration_version=self.configuration.calibration_version,
                        projector_resolution=self._projector_output_descriptor.projector_resolution,
                        projector_output_descriptor=self._projector_output_descriptor,
                    )
                    verified_status = verification_registry.verify(
                        record.camera_id,
                        verification_capture.correspondences,
                        camera_resolution=status.native_resolution,
                        projector_resolution=self._projector_output_descriptor.projector_resolution,
                        thresholds=_derive_verification_thresholds(
                            record,
                            verification_thresholds,
                            verification_capture.noise,
                        ),
                        pattern=self.calibration_pattern,
                        projector_output_descriptor=self._projector_output_descriptor,
                    )
                    if verified_status is not CalibrationStatus.CALIBRATED:
                        reports[logical_name] = FullCalibrationCameraResult(
                            logical_name,
                            CalibrationStatus.STALE,
                            scope,
                            record,
                            'Fresh calibration verification did not pass',
                        )
                        continue
                    try:
                        with self._camera_management_lock:
                            current_camera = self._get_session_camera(logical_name)
                            if (
                                current_camera is None
                                or current_camera.lifecycle_generation != lifecycle_generation
                            ):
                                raise CalibrationError(
                                    'Camera changed during calibration verification',
                                )
                            if (
                                self.calibration_pattern != workflow_pattern
                                or self._projector_output_descriptor != workflow_descriptor
                                or self.configuration.calibration_version != workflow_version
                            ):
                                raise CalibrationError(
                                    'Projector calibration authority changed during publication',
                                )
                            self.calibration_store.save(record)
                            self._set_session_calibration(
                                current_camera.slot_id,
                                CalibrationStatus.CALIBRATED,
                                record,
                            )
                    except Exception as ex:  # noqa: BLE001 (report one-camera publication failures).
                        reports[logical_name] = FullCalibrationCameraResult(
                            logical_name,
                            CalibrationStatus.STALE,
                            scope,
                            record,
                            str(ex),
                        )
                        continue
                    accepted[logical_name] = FullCalibrationCameraResult(
                        logical_name,
                        CalibrationStatus.CALIBRATED,
                        scope,
                        record,
                    )
                    reports[logical_name] = accepted[logical_name]

                master_camera = None
                if accepted:
                    master_camera = max(
                        accepted,
                        key=lambda camera: _full_calibration_quality_score(
                            accepted[camera].calibration,
                        ),
                    )
                master_gate_passed = (
                    master_camera is not None
                    and _passes_full_calibration_master_gate(
                        accepted[master_camera].calibration,
                        workflow_pattern,
                        self.configuration.calibration_thresholds,
                    )
                )
                with self._camera_management_lock:
                    if master_camera is not None:
                        current_master = self._get_session_camera(master_camera)
                        try:
                            master_status = self.camera_runtime.get_status(master_camera)
                        except CameraUnavailableError:
                            master_status = None
                        if (
                            current_master is None
                            or current_master.state is not SessionCameraState.OPEN
                            or not isinstance(master_status, CameraStatus)
                            or master_status.runtime_status is not RuntimeStatus.AVAILABLE
                            or current_master.calibration_status is not CalibrationStatus.CALIBRATED
                        ):
                            master_camera = None
                    if master_camera is None:
                        master_gate_passed = False
                    self._calibration_master_camera = master_camera
                return FullCalibrationResult(master_camera, master_gate_passed, reports)
            finally:
                with self._calibration_capture_lock:
                    self._calibration_capture_count -= 1
                    if self._calibration_capture_count == 0:
                        self._calibration_pattern_presented.clear()

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

    @property
    def calibration_master_camera(self) -> str | None:
        """Return the latest usable full-calibration master for this session."""
        with self._camera_management_lock:
            master_camera = self._calibration_master_camera
            if master_camera is None:
                return None
            camera = self._get_session_camera(master_camera)
            try:
                status = self.camera_runtime.get_status(master_camera)
            except CameraUnavailableError:
                status = None
            if (
                camera is None
                or camera.state is not SessionCameraState.OPEN
                or not isinstance(status, CameraStatus)
                or status.runtime_status is not RuntimeStatus.AVAILABLE
                or camera.calibration_status is not CalibrationStatus.CALIBRATED
            ):
                self._calibration_master_camera = None
                return None
            return master_camera

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
        self.overlay_registry.invalidate_camera(resolved_slot)
        self._invalidate_spatial_state()
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
        # Global captures support the native frame; local captures must stay inside
        # the observed tag hull instead of silently extrapolating.
        camera_frame = CoordinateBounds(
            0.0,
            0.0,
            float(native_resolution.width),
            float(native_resolution.height),
        )
        calibrated_region = (
            calibration.valid_region
            if isinstance(calibration, PersistedCalibration)
            and calibration.calibration_scope != CalibrationScope.GLOBAL.value
            else camera_frame
        )
        available_area = calculate_available_projector_area(
            calibrated_region,
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
            if self._last_camera_capture_noise is not None:
                result = result._replace(
                    metrics=result.metrics._replace(
                        capture_median_sigma_pixels=(
                            self._last_camera_capture_noise.median_sigma_pixels
                        ),
                        capture_p95_sigma_pixels=(
                            self._last_camera_capture_noise.p95_sigma_pixels
                        ),
                        capture_max_sigma_pixels=(
                            self._last_camera_capture_noise.max_sigma_pixels
                        ),
                    ),
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
        try:
            checked_correspondences = self._get_correspondences_for_operation(
                status,
                correspondences,
                calibration,
            )
        except Exception:
            # A failed verification attempt must not leave a previously trusted
            # transform usable after its fresh evidence failed to arrive.
            self._set_session_calibration(
                session_camera.slot_id,
                CalibrationStatus.STALE,
                calibration,
            )
            raise
        session_registry = CalibrationRegistry(
            {session_camera.slot_id: calibration},
            calibration_version=self.configuration.calibration_version,
            projector_resolution=self._projector_output_descriptor.projector_resolution,
            projector_output_descriptor=self._projector_output_descriptor,
        )
        verification_thresholds = _derive_verification_thresholds(
            calibration,
            self.configuration.calibration_thresholds,
            self._last_camera_capture_noise,
        )
        calibration_status = session_registry.verify(
            session_camera.slot_id,
            checked_correspondences,
            camera_resolution=status.native_resolution,
            projector_resolution=self._projector_output_descriptor.projector_resolution,
            thresholds=verification_thresholds,
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
        expected_calibration: PersistedCalibration | None = None,
    ) -> CameraCorrespondences:
        self._last_camera_capture_noise = None
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
            capture = self._capture_camera_correspondences(
                status,
                expected_calibration=expected_calibration,
            )
            self._last_camera_capture_noise = capture.noise
            return capture.correspondences
        finally:
            with self._calibration_capture_lock:
                self._calibration_capture_count -= 1
                if self._calibration_capture_count == 0:
                    self._calibration_pattern_presented.clear()

    def _capture_camera_correspondences(
        self,
        status: CameraStatus,
        *,
        expected_calibration: PersistedCalibration | None = None,
        allow_local: bool = False,
    ) -> _CalibrationCapture:
        thresholds = self.configuration.calibration_thresholds
        get_consecutive_frames = getattr(
            self.camera_runtime,
            'get_consecutive_frames',
            None,
        )
        if callable(get_consecutive_frames) and CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS > 0:
            captured_frames = get_consecutive_frames(
                status.logical_name,
                CALIBRATION_CAPTURE_CANDIDATE_FRAME_COUNT,
                CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS,
            )
            if not isinstance(captured_frames, Sequence) or len(captured_frames) < 3:
                raise CalibrationError(
                    'Camera runtime returned invalid consecutive calibration frames',
                )
            if any(not isinstance(frame, Frame) for frame in captured_frames):
                raise CalibrationError(
                    'Camera runtime returned invalid consecutive calibration frames',
                )
            stable_frames = _select_stable_frame_window(
                captured_frames,
                thresholds.max_capture_white_balance_delta,
                window_size=CALIBRATION_CAPTURE_CANDIDATE_FRAME_COUNT,
            )
            detected_frames = tuple(
                self._get_correspondences_from_frame(status, frame)
                for frame in stable_frames
            )
            selected_frames = _select_camera_frame_window(
                detected_frames,
                self.calibration_pattern,
                thresholds,
                status.native_resolution,
                expected_calibration,
                minimum_common_marker_count=1 if allow_local else 2,
                minimum_common_corner_count=4 if allow_local else 8,
            )
            aggregated, noise = _aggregate_camera_correspondences(
                selected_frames,
                minimum_common_marker_count=1 if allow_local else 2,
                minimum_common_corner_count=4 if allow_local else 8,
            )
            _validate_camera_capture_noise(noise, thresholds)
            return _CalibrationCapture(aggregated, noise)

        deadline = time.monotonic() + CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS
        candidates: list[CameraCorrespondences] = []
        last_error: CalibrationError | None = None
        previous_frame_counter: int | None = None
        snapshot = getattr(self.camera_runtime, 'snapshot', None)
        while True:
            try:
                if callable(snapshot):
                    frame = snapshot(status.logical_name)
                    if not isinstance(frame, Frame):
                        raise CalibrationError('Camera runtime returned an invalid snapshot')
                    if (
                        previous_frame_counter is not None
                        and frame.frame_counter <= previous_frame_counter
                    ):
                        raise CalibrationError(
                            'Calibration capture did not receive fresh camera frames',
                        )
                    previous_frame_counter = frame.frame_counter
                    candidates.append(self._get_correspondences_from_frame(status, frame))
                else:
                    candidates.append(self._get_correspondences(status, None))
            except CalibrationError as ex:
                last_error = ex
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        if (
            callable(snapshot)
            and CALIBRATION_PATTERN_CAPTURE_TIMEOUT_SECONDS > 0
            and len(candidates) < 3
        ):
            raise CalibrationError(
                'Calibration capture did not receive three fresh camera frames',
            )
        if candidates:
            selected_candidates = candidates[-3:]
            aggregated, noise = _aggregate_camera_correspondences(
                selected_candidates,
                minimum_common_marker_count=1 if allow_local else 2,
                minimum_common_corner_count=4 if allow_local else 8,
            )
            _validate_camera_capture_noise(noise, thresholds)
            return _CalibrationCapture(aggregated, noise)
        if last_error is not None:
            raise last_error
        raise CalibrationError('No calibration correspondences were detected')

    def _capture_calibration_bursts(
        self,
        statuses: Sequence[CameraStatus],
        *,
        expected_calibrations: Mapping[str, PersistedCalibration] | None = None,
        allow_local: bool = False,
    ) -> dict[str, _CalibrationCapture | str]:
        expected = expected_calibrations or {}
        captures: dict[str, _CalibrationCapture | str] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(statuses))) as executor:
            futures = {
                status.logical_name: executor.submit(
                    self._capture_camera_correspondences,
                    status,
                    expected_calibration=expected.get(status.logical_name),
                    allow_local=allow_local,
                )
                for status in statuses
            }
            for logical_name, future in futures.items():
                try:
                    captures[logical_name] = future.result()
                except Exception as ex:  # noqa: BLE001 (one camera must not abort the run).
                    captures[logical_name] = str(ex)
        return captures

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

        return self._get_correspondences_from_frame(status, self.snapshot(status.logical_name))

    def _get_correspondences_from_frame(
        self,
        status: CameraStatus,
        frame: Frame,
    ) -> CameraCorrespondences:
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError(
                f'Camera {status.logical_name!r} has no session identity',
            )
        detector = self.detector
        if detector is None:
            detector = OpenCVArucoDetector(self.calibration_pattern.marker_family)
        with self._detector_lock:
            detected_markers = detect_fiducials(frame.data, detector)
        return assemble_correspondences(
            detected_markers,
            self.calibration_pattern,
            camera_id=session_camera.slot_id,
        )

    def _get_calibration_status(self, status: CameraStatus) -> CalibrationStatus:
        session_camera = self._get_session_camera(status.logical_name)
        if session_camera is None:
            raise CameraUnavailableError(
                f'Camera {status.logical_name!r} has no session state',
            )
        calibration = session_camera.calibration
        if session_camera.calibration_status is not CalibrationStatus.CALIBRATED:
            self.overlay_registry.invalidate_camera(session_camera.slot_id)
            return session_camera.calibration_status
        if (
            isinstance(calibration, PersistedCalibration)
            and (
                calibration.camera_id != session_camera.slot_id
                or calibration.version != self.configuration.calibration_version
                or calibration.camera_resolution != status.native_resolution
                or calibration.projector_output_descriptor
                != self._projector_output_descriptor
            )
        ):
            self.overlay_registry.invalidate_camera(session_camera.slot_id)
            return CalibrationStatus.STALE
        return session_camera.calibration_status

    def _get_tag_projection_status(
        self,
        camera_reference: str,
    ) -> ProjectionStatus | None:
        try:
            self.point_service.project_camera_points(camera_reference, ())
        except GeometryError as ex:
            return _projection_status_from_error(ex)
        except ValueError as ex:
            return ProjectionStatus('INVALID_HOMOGRAPHY', str(ex))
        return None

    def _build_tag_observation(
        self,
        camera_reference: str,
        observation: PlanarTagObservation,
        projection_status: ProjectionStatus | None,
    ) -> TagObservation:
        if projection_status is not None:
            return TagObservation(
                observation.marker_id,
                observation.camera,
                projection_status=projection_status,
            )
        try:
            projected_points = self.point_service.project_camera_points(
                camera_reference,
                observation.camera.corners + (observation.camera.centre,),
            )
            if len(projected_points) != 5:
                raise ValueError(
                    'Camera-to-projector authority returned an incomplete point set',
                )
            projector_geometry = build_tag_geometry(
                projected_points[:4],
                centre=projected_points[4],
            )
        except GeometryError as ex:
            return TagObservation(
                observation.marker_id,
                observation.camera,
                projection_status=_projection_status_from_error(ex),
            )
        except (TypeError, ValueError):
            return TagObservation(
                observation.marker_id,
                observation.camera,
                projection_status=ProjectionStatus(
                    'INVALID_HOMOGRAPHY',
                    'Camera-to-projector authority returned invalid tag geometry',
                ),
            )
        return TagObservation(
            observation.marker_id,
            observation.camera,
            projector_geometry,
        )

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
        self.overlay_registry.invalidate_camera(slot_id)
        self._invalidate_spatial_state()
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


class _CameraCaptureNoise(NamedTuple):
    median_sigma_pixels: float
    p95_sigma_pixels: float
    max_sigma_pixels: float


def _select_camera_frame_window(
    frames: Sequence[CameraCorrespondences],
    pattern: CalibrationPattern,
    thresholds: CalibrationThresholds,
    camera_resolution: Resolution | None,
    expected_calibration: PersistedCalibration | None = None,
    *,
    minimum_common_marker_count: int = 2,
    minimum_common_corner_count: int = 8,
) -> tuple[CameraCorrespondences, ...]:
    if camera_resolution is None:
        raise CalibrationError('Camera resolution is unavailable for calibration capture')
    if len(frames) <= 3:
        return tuple(frames)
    best_window: tuple[CameraCorrespondences, ...] | None = None
    best_score: tuple[float, ...] | None = None
    fallback_window: tuple[CameraCorrespondences, ...] | None = None
    fallback_score: tuple[int, float, float, int] | None = None
    for start_index in range(len(frames) - 2):
        window = tuple(frames[start_index:start_index + 3])
        common_marker_ids = set(window[0].unique_marker_ids)
        for frame in window[1:]:
            common_marker_ids.intersection_update(frame.unique_marker_ids)
        try:
            aggregated, window_noise = _aggregate_camera_correspondences(
                window,
                minimum_common_marker_count=minimum_common_marker_count,
                minimum_common_corner_count=minimum_common_corner_count,
            )
        except CalibrationError:
            continue
        noise_score = window_noise.p95_sigma_pixels if window_noise is not None else 0.0
        fallback_score_value = (
            len(common_marker_ids),
            -noise_score,
            -window_noise.max_sigma_pixels if window_noise is not None else 0.0,
            sum(len(frame.correspondences) for frame in window),
        )
        if fallback_score is None or fallback_score_value > fallback_score:
            fallback_window = window
            fallback_score = fallback_score_value
        try:
            _validate_camera_capture_noise(window_noise, thresholds)
            result = calibrate_homography(
                aggregated,
                pattern,
                thresholds,
                camera_resolution=camera_resolution,
            )
        except CalibrationError:
            continue
        metrics = result.metrics
        if expected_calibration is None:
            score = (
                len(common_marker_ids),
                metrics.inlier_ratio,
                metrics.spatial_coverage,
                -metrics.mean_reprojection_error,
                -noise_score,
                sum(len(frame.correspondences) for frame in window),
            )
        else:
            verification_thresholds = _derive_verification_thresholds(
                expected_calibration,
                thresholds,
                window_noise,
            )
            errors_by_marker: dict[int, list[float]] = {}
            for correspondence in aggregated.correspondences:
                predicted = project_point(
                    correspondence.projector_position,
                    expected_calibration.projector_to_camera,
                )
                errors_by_marker.setdefault(correspondence.marker_id, []).append(
                    math.dist(predicted, correspondence.camera_position),
                )
            accepted_errors = [
                statistics.median(marker_errors)
                for marker_errors in errors_by_marker.values()
                if statistics.median(marker_errors)
                <= verification_thresholds.max_reprojection_error
            ]
            accepted_marker_count = len(accepted_errors)
            mean_error = (
                sum(accepted_errors) / len(accepted_errors)
                if accepted_errors
                else float('inf')
            )
            minimum_accepted_marker_count = max(
                thresholds.min_unique_tags,
                math.ceil(0.5 * len(common_marker_ids)),
            )
            score = (
                int(
                    accepted_marker_count >= minimum_accepted_marker_count
                    and mean_error <= verification_thresholds.max_mean_reprojection_error
                ),
                accepted_marker_count,
                len(common_marker_ids),
                -mean_error,
                -max(accepted_errors) if accepted_errors else float('-inf'),
                -noise_score,
                sum(len(frame.correspondences) for frame in window),
            )
        if best_score is None or score > best_score:
            best_window = window
            best_score = score
    if best_window is not None:
        return best_window
    if fallback_window is not None:
        return fallback_window
    raise CalibrationError('No calibration frame window was selected')


def _aggregate_camera_correspondences(
    frames: Sequence[CameraCorrespondences],
    *,
    minimum_common_marker_count: int = 2,
    minimum_common_corner_count: int = 8,
) -> tuple[CameraCorrespondences, _CameraCaptureNoise | None]:
    if len(frames) == 0:
        raise CalibrationError('No calibration correspondences were detected')
    if len(frames) == 1:
        return frames[0], None

    marker_counts = tuple(len(frame.unique_marker_ids) for frame in frames)
    minimum_common_marker_count = max(
        minimum_common_marker_count,
        math.ceil(0.5 * statistics.median(marker_counts)),
    )
    common_marker_ids = set(frames[0].unique_marker_ids)
    frame_corner_maps = tuple(
        {
            (correspondence.marker_id, correspondence.corner_index): correspondence
            for correspondence in frame.correspondences
        }
        for frame in frames
    )
    for frame in frames[1:]:
        common_marker_ids.intersection_update(frame.unique_marker_ids)
    if len(common_marker_ids) < minimum_common_marker_count:
        raise CalibrationError(
            'Calibration frames do not contain enough common target tags',
        )

    common_corner_keys = tuple(
        sorted(
            corner_key
            for corner_key in frame_corner_maps[0]
            if corner_key[0] in common_marker_ids
            and all(corner_key in corner_map for corner_map in frame_corner_maps)
        ),
    )
    if len(common_corner_keys) < minimum_common_corner_count:
        raise CalibrationError(
            'Calibration frames do not contain enough common target corners',
        )

    noise_corner_maps = tuple(
        {
            (correspondence.marker_id, correspondence.corner_index): correspondence
            for correspondence in frame.correspondences
        }
        for frame in frames
    )
    averaged: list[FiducialCorrespondence] = []
    sigmas: list[float] = []
    first_map = frame_corner_maps[0]
    for corner_key in common_corner_keys:
        correspondences = [corner_map[corner_key] for corner_map in frame_corner_maps]
        first_correspondence = first_map[corner_key]
        if any(
            correspondence.projector_position != first_correspondence.projector_position
            for correspondence in correspondences
        ):
            raise CalibrationError(
                'Calibration frames disagree about projector corner positions',
            )
        median_position = Point2D(
            statistics.median(
                correspondence.camera_position.x
                for correspondence in correspondences
            ),
            statistics.median(
                correspondence.camera_position.y
                for correspondence in correspondences
            ),
        )
        noise_correspondences = tuple(
            corner_map[corner_key]
            for corner_map in noise_corner_maps
            if corner_key in corner_map
        )
        if len(noise_correspondences) >= 3:
            noise_median_position = Point2D(
                statistics.median(
                    correspondence.camera_position.x
                    for correspondence in noise_correspondences
                ),
                statistics.median(
                    correspondence.camera_position.y
                    for correspondence in noise_correspondences
                ),
            )
            deviations = tuple(
                math.dist(correspondence.camera_position, noise_median_position)
                for correspondence in noise_correspondences
            )
            sigma = 1.4826 * statistics.median(deviations)
            if not math.isfinite(sigma):
                raise CalibrationError('Calibration produced invalid camera noise metrics')
            sigmas.append(sigma)
        averaged.append(
            FiducialCorrespondence(
                first_correspondence.marker_id,
                first_correspondence.corner_index,
                first_correspondence.projector_position,
                median_position,
            ),
        )

    if not sigmas:
        return CameraCorrespondences(tuple(averaged), frames[0].camera_id), None
    sorted_sigmas = sorted(sigmas)
    p95_index = min(
        len(sorted_sigmas) - 1,
        max(0, math.ceil(0.95 * len(sorted_sigmas)) - 1),
    )
    noise = _CameraCaptureNoise(
        statistics.median(sorted_sigmas),
        sorted_sigmas[p95_index],
        max(sorted_sigmas),
    )
    return CameraCorrespondences(tuple(averaged), frames[0].camera_id), noise


def _validate_camera_capture_noise(
    noise: _CameraCaptureNoise | None,
    thresholds: CalibrationThresholds,
) -> None:
    if noise is not None and noise.p95_sigma_pixels > thresholds.max_capture_p95_sigma_pixels:
        raise CalibrationError(
            'Camera capture noise is too large for reliable calibration verification',
        )


def _derive_verification_thresholds(
    calibration: PersistedCalibration,
    thresholds: CalibrationThresholds,
    noise: _CameraCaptureNoise | None,
) -> CalibrationThresholds:
    if noise is None:
        return thresholds
    return replace(
        thresholds,
        max_mean_reprojection_error=(
            calibration.metrics.mean_reprojection_error
            + 3.0 * noise.median_sigma_pixels
        ),
        max_reprojection_error=(
            calibration.metrics.max_reprojection_error
            + 3.0 * noise.median_sigma_pixels
        ),
    )


def _select_stable_frame_window(
    frames: Sequence[Frame],
    maximum_white_balance_delta: float,
    *,
    window_size: int = METRIC_CAPTURE_FRAME_COUNT,
) -> tuple[Frame, ...]:
    if not math.isfinite(maximum_white_balance_delta) or maximum_white_balance_delta <= 0:
        raise CalibrationError('White-balance tolerance is invalid')
    if window_size <= 0 or len(frames) < window_size:
        raise CalibrationError('Capture did not contain enough consecutive frames')
    signatures = tuple(_white_balance_signature(frame.data) for frame in frames)
    for start_index in range(len(frames) - window_size + 1):
        window_signatures = signatures[start_index:start_index + window_size]
        if _white_balance_window_is_stable(
            window_signatures,
            maximum_white_balance_delta,
        ):
            return tuple(frames[start_index:start_index + window_size])
    raise CalibrationError(
        f'Capture did not find {window_size} consecutive white-balance-stable frames',
    )


def _white_balance_signature(frame_data: object) -> tuple[float, ...] | None:
    try:
        import numpy
        array = numpy.asarray(frame_data)
        if array.ndim < 3 or array.shape[2] < 3:
            return None
        channel_medians = numpy.median(
            array[..., :3].reshape(-1, 3),
            axis=0,
        )
        total = float(numpy.sum(channel_medians))
        if not math.isfinite(total) or total <= 0:
            return None
        signature = tuple(float(value) / total for value in channel_medians)
    except (ImportError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in signature):
        return None
    return signature


def _white_balance_window_is_stable(
    signatures: Sequence[tuple[float, ...] | None],
    maximum_delta: float,
) -> bool:
    if any(signature is None for signature in signatures):
        return True
    checked_signatures = tuple(
        signature for signature in signatures if signature is not None
    )
    return all(
        max(
            abs(first_channel - second_channel)
            for first_channel, second_channel in zip(first_signature, second_signature)
        ) <= maximum_delta
        for first_index, first_signature in enumerate(checked_signatures)
        for second_signature in checked_signatures[first_index + 1:]
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


def _full_calibration_quality_score(
    calibration: PersistedCalibration | None,
) -> tuple[float, ...]:
    if calibration is None:
        return (0.0, 0.0, 0.0, float('-inf'), float('-inf'))
    metrics = calibration.metrics
    return (
        float(metrics.unique_tag_count),
        metrics.spatial_coverage,
        metrics.inlier_ratio,
        -metrics.mean_reprojection_error,
        -metrics.max_reprojection_error,
        -(metrics.capture_p95_sigma_pixels or 0.0),
    )


def _attach_camera_capture_noise(
    result: CalibrationResult,
    noise: _CameraCaptureNoise | None,
) -> CalibrationResult:
    if noise is None:
        return result
    return result._replace(
        metrics=result.metrics._replace(
            capture_median_sigma_pixels=noise.median_sigma_pixels,
            capture_p95_sigma_pixels=noise.p95_sigma_pixels,
            capture_max_sigma_pixels=noise.max_sigma_pixels,
        ),
    )


def _passes_full_calibration_master_gate(
    calibration: PersistedCalibration | None,
    pattern: CalibrationPattern,
    thresholds: CalibrationThresholds,
) -> bool:
    if calibration is None or calibration.calibration_scope != CalibrationScope.GLOBAL.value:
        return False
    metrics = calibration.metrics
    minimum_tag_count = math.ceil(0.8 * len(pattern.markers))
    return (
        metrics.unique_tag_count >= minimum_tag_count
        and metrics.spatial_coverage >= thresholds.min_spatial_coverage
        and metrics.inlier_ratio >= thresholds.min_inlier_ratio
        and metrics.mean_reprojection_error <= thresholds.max_mean_reprojection_error
        and metrics.max_reprojection_error <= thresholds.max_reprojection_error
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


def _build_projector_calibration_pattern(
    projector_resolution: Resolution,
) -> CalibrationPattern:
    margin = CALIBRATION_PATTERN_EDGE_MARGIN_PIXELS
    return build_calibration_pattern(
        projector_resolution,
        usable_area=CoordinateBounds(
            margin,
            margin,
            projector_resolution.width - margin,
            projector_resolution.height - margin,
        ),
        marker_family=PROJECTOR_CALIBRATION_MARKER_FAMILY,
    )


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
    minimum_marker_ratio: float,
    camera_id: str,
) -> MetricTargetCorrespondences:
    if len(frames) != METRIC_CAPTURE_FRAME_COUNT:
        raise CalibrationError(
            f'Metric capture requires {METRIC_CAPTURE_FRAME_COUNT} frames',
        )
    if not math.isfinite(maximum_corner_jitter_pixels) or maximum_corner_jitter_pixels <= 0:
        raise CalibrationError('Metric corner jitter tolerance is invalid')
    if (
        not math.isfinite(minimum_marker_ratio)
        or not 0 < minimum_marker_ratio <= 1
    ):
        raise CalibrationError('Metric capture marker ratio is invalid')

    checked_frames = tuple(
        validate_metric_correspondences(
            _replace_metric_camera_id(frame, camera_id),
            target=METRIC_TARGET,
        )
        for frame in frames
    )
    frame_corner_maps = tuple(
        _metric_correspondences_by_corner(frame)
        for frame in checked_frames
    )
    common_marker_ids = set(checked_frames[0].unique_marker_ids)
    for frame in checked_frames[1:]:
        common_marker_ids.intersection_update(frame.unique_marker_ids)
    stable_marker_ratio = len(common_marker_ids) / len(METRIC_TARGET.markers)
    if stable_marker_ratio < minimum_marker_ratio:
        raise CalibrationError(
            'Metric capture does not contain enough stable target markers',
        )

    common_corner_keys = tuple(
        sorted(
            corner_key
            for corner_key in frame_corner_maps[0]
            if corner_key[0] in common_marker_ids
            and all(corner_key in corner_map for corner_map in frame_corner_maps)
        ),
    )
    if len(common_corner_keys) < 4:
        raise CalibrationError(
            'Metric capture does not contain enough stable target corners',
        )

    first_by_corner = frame_corner_maps[0]
    averaged: list[MetricTargetCorrespondence] = []
    for corner_key in common_corner_keys:
        first_correspondence = first_by_corner[corner_key]
        frame_correspondences = [
            corner_map[corner_key]
            for corner_map in frame_corner_maps
        ]
        try:
            median_camera_position = Point2D(
                statistics.median(
                    correspondence.camera_position.x
                    for correspondence in frame_correspondences
                ),
                statistics.median(
                    correspondence.camera_position.y
                    for correspondence in frame_correspondences
                ),
            )
            corner_deviations = tuple(
                math.dist(
                    correspondence.camera_position,
                    median_camera_position,
                )
                for correspondence in frame_correspondences
            )
        except (AttributeError, TypeError, ValueError) as ex:
            raise CalibrationError(
                'Metric capture contains an invalid camera corner',
            ) from ex
        if (
            not math.isfinite(statistics.median(corner_deviations))
            or statistics.median(corner_deviations) > maximum_corner_jitter_pixels
        ):
            raise CalibrationError(
                'Metric capture detected movement beyond the stability tolerance',
            )
        median_surface_position = first_correspondence.surface_position
        if any(
            correspondence.surface_position != median_surface_position
            for correspondence in frame_correspondences
        ):
            raise CalibrationError(
                'Metric capture frames disagree about target corner positions',
            )
        averaged_camera_position = median_camera_position
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


def _is_tracking_calibration_usable(
    camera: SessionCamera,
    status: CameraStatus | None,
    calibration: object | None,
    descriptor: ProjectorOutputDescriptor,
    calibration_version: int,
) -> bool:
    if camera.state is not SessionCameraState.OPEN:
        return False
    if status is not None and (
        status.runtime_status is not RuntimeStatus.AVAILABLE
        or status.calibration_status is not CalibrationStatus.CALIBRATED
    ):
        return False
    if camera.calibration_status is not CalibrationStatus.CALIBRATED or calibration is None:
        return False
    if (
        getattr(calibration, 'projector_output_descriptor', None)
        not in (None, descriptor)
    ):
        return False
    if status is not None and status.native_resolution is not None:
        if getattr(calibration, 'camera_resolution', status.native_resolution) != status.native_resolution:
            return False
    if isinstance(calibration, PersistedCalibration) and (
        calibration.camera_id != camera.slot_id
        or calibration.version != calibration_version
        or calibration.calibration_scope != CalibrationScope.GLOBAL.value
    ):
        return False
    try:
        validate_homography(getattr(calibration, 'camera_to_projector'))
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _camera_calibration_signature(calibration: object | None) -> object:
    if calibration is None:
        return None
    return (
        getattr(calibration, 'camera_to_projector', None),
        getattr(calibration, 'projector_output_descriptor', None),
        getattr(calibration, 'camera_resolution', None),
        getattr(calibration, 'version', None),
        getattr(calibration, 'timestamp', None),
    )


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


def _tag_geometry_to_data(geometry: TagGeometry) -> dict[str, object]:
    return {
        'corners': [[point.x, point.y] for point in geometry.corners],
        'centre': [geometry.centre.x, geometry.centre.y],
        'orientation_degrees': geometry.orientation_degrees,
        'area_px': geometry.area_px,
    }


def _projection_status_from_error(exception: Exception) -> ProjectionStatus:
    code = getattr(exception, 'code', 'CALIBRATION_INVALID')
    if not isinstance(code, str) or len(code) == 0:
        code = 'CALIBRATION_INVALID'
    return ProjectionStatus(code, str(exception))


def _get_render_protection_regions(
    spatial_state: SpatialState,
    protection_margin_mm: float,
) -> tuple[tuple[Point2D, ...], ...]:
    if len(spatial_state.protection_regions) > 0:
        return tuple(
            tuple(region)
            for region in spatial_state.protection_regions.values()
        )
    footprints = spatial_state.projector_footprints
    if len(footprints) == 0:
        footprints = {
            identity: observation.projector.corners
            for identity, observation in spatial_state.selected_observations.items()
            if observation.projector is not None
        }
    return build_protected_projector_regions(
        footprints,
        protection_margin_mm,
        spatial_state.metric_calibration,
    )


def _spatial_state_has_data(spatial_state: SpatialState) -> bool:
    return any(
        len(mapping) > 0
        for mapping in (
            spatial_state.selected_observations,
            spatial_state.last_seen_monotonic_seconds,
            spatial_state.stability_scores,
            spatial_state.projector_footprints,
            spatial_state.protection_regions,
            spatial_state.camera_generations,
        )
    ) or spatial_state.metric_calibration is not None or len(
        spatial_state.detector_failures
    ) > 0


def _get_current_spatial_metric_calibration(
    spatial_state: SpatialState,
    projector_output_descriptor: ProjectorOutputDescriptor,
) -> object | None:
    metric_calibration = getattr(spatial_state, 'metric_calibration', None)
    if metric_calibration is None:
        return None
    metric_descriptor = getattr(metric_calibration, 'projector_output_descriptor', None)
    if metric_descriptor is not None and metric_descriptor != projector_output_descriptor:
        return None
    metric_state = getattr(metric_calibration, 'state', None)
    metric_state_value = getattr(metric_state, 'value', metric_state)
    if metric_state is not None and metric_state_value != MetricCalibrationStatus.CALIBRATED.value:
        return None
    is_usable = getattr(metric_calibration, 'is_usable', None)
    if callable(is_usable) and is_usable() is not True:
        return None
    if is_usable is not None and not callable(is_usable) and is_usable is not True:
        return None
    return metric_calibration


def _is_unavailable_overlay_authority_error(exception: ValueError) -> bool:
    message = str(exception).lower()
    return any(
        phrase in message
        for phrase in (
            'unresolved',
            'no camera',
            'no usable',
            'no projector',
            'no surface',
            'metric calibration is not usable',
            'metric calibration has no',
            'a usable metric calibration is required',
            'arrow endpoints must be distinct',
            'cannot be converted',
            'projected safely',
            'horizon',
        )
    )


def _get_camera_state_from_status(status: CameraStatus) -> SessionCameraState:
    if status.runtime_status is RuntimeStatus.STOPPED:
        return SessionCameraState.CLOSED
    if status.runtime_status is RuntimeStatus.UNAVAILABLE:
        return SessionCameraState.UNAVAILABLE
    return SessionCameraState.OPEN


__all__ = [
    'AREA_COLOURS',
    'CameraArea',
    'CameraRenderSnapshot',
    'MultiVisionService',
    'ProjectionStatus',
    'RenderSnapshot',
    'TagInspectionResult',
    'TagObservation',
    'get_camera_area_colour',
]
