"""Local FastAPI boundary for the session-local MultiVision service."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import (
    FastAPI,
    Path,
    Query,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    UUID4,
    field_validator,
)

from multivision.application import (
    CameraArea,
    MultiVisionService,
    get_camera_area_colour,
)
from multivision.config import ProjectorOutputDescriptor
from multivision.errors import (
    CalibrationError,
    CameraSlotNotFoundError,
    CameraUnavailableError,
    ConfigurationError,
    FiducialDetectionError,
    FrameCaptureError,
    GeometryError,
    HardwareError,
    MultiVisionError,
    OverlayNotFoundError,
    SessionCameraError,
)
from multivision.fiducials import (
    FiducialCorrespondence,
    MetricTargetCorrespondence,
)
from multivision.geometry import (
    Point2D,
    Polygon,
)
from multivision.metric import (
    MetricCalibrationRecord,
    MetricCalibrationStatus,
    MetricRulerOverlay,
    MetricValidationRecord,
)
from multivision.metric_target import METRIC_TARGET
from multivision.overlays import (
    CircleRequest,
    GridRequest,
    LineRequest,
    OverlayEntry,
    ProjectorCoverageGridRequest,
    RectRequest,
    RulerRequest,
    TextRequest,
)
from multivision.persistence import PersistedCalibration
from multivision.pattern import validate_tag_dictionary
from multivision.service import RedCircleOverlay
from multivision.session import (
    FrameMetadata,
    SessionCamera,
)
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
    is_finite_real,
)


class PointPairRequest(BaseModel):
    """A finite coordinate pair in one of the calibration spaces."""

    model_config = ConfigDict(extra='forbid')

    x: float
    y: float

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_finite_coordinate(cls, value: Any) -> float:
        return _validate_coordinate(value)


class CorrespondenceRequest(BaseModel):
    """One detected marker corner supplied by a calibration client or test fake."""

    model_config = ConfigDict(extra='forbid')

    marker_id: StrictInt = Field(ge=0)
    corner_index: StrictInt = Field(ge=0, le=3)
    projector: PointPairRequest = Field(
        validation_alias=AliasChoices('projector', 'projector_position'),
    )
    camera: PointPairRequest = Field(
        validation_alias=AliasChoices('camera', 'camera_position'),
    )


class CalibrationRequest(BaseModel):
    """Optional camera and correspondences for calibration or verification."""

    model_config = ConfigDict(extra='forbid')

    camera: str | None = Field(default=None, min_length=1)
    correspondences: list[CorrespondenceRequest] | None = None


class MetricCorrespondenceRequest(BaseModel):
    """One injected metric-target corner for deterministic tests."""

    model_config = ConfigDict(extra='forbid')

    marker_id: StrictInt = Field(ge=0)
    corner_index: StrictInt = Field(ge=0, le=3)
    surface: PointPairRequest = Field(
        validation_alias=AliasChoices('surface', 'surface_position'),
    )
    camera: PointPairRequest = Field(
        validation_alias=AliasChoices('camera', 'camera_position'),
    )


class MetricCalibrationRequest(BaseModel):
    """The selected camera and optional test-only metric correspondence seam."""

    model_config = ConfigDict(extra='forbid')

    camera: str = Field(min_length=1)
    correspondences: list[MetricCorrespondenceRequest] | None = None


MetricUnitLiteral = Literal['mm', 'cm', 'in']


class MetricRulerRequest(BaseModel):
    """A physical ruler request in surface millimetres."""

    model_config = ConfigDict(extra='forbid')

    surface_start: PointPairRequest = Field(
        validation_alias=AliasChoices('from', 'surface_start'),
    )
    surface_end: PointPairRequest = Field(
        validation_alias=AliasChoices('to', 'surface_end'),
    )
    unit: MetricUnitLiteral = 'mm'
    observed_length: float | None = None
    observed_unit: MetricUnitLiteral = 'mm'

    @field_validator('observed_length', mode='before')
    @classmethod
    def validate_observed_length(cls, value: Any) -> float | None:
        if value is None:
            return None
        checked_length = _validate_coordinate(value)
        if checked_length <= 0:
            raise ValueError('observed_length must be positive')
        return checked_length


class CameraRenameRequest(BaseModel):
    """The new display name for a session camera slot."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)


class CameraAreaRequest(BaseModel):
    """The desired session-local diagnostic-area state."""

    model_config = ConfigDict(extra='forbid')

    enabled: StrictBool


class PointRequest(BaseModel):
    """A camera-native point request."""

    model_config = ConfigDict(extra='forbid')

    camera: str = Field(min_length=1)
    x: float
    y: float

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_finite_coordinate(cls, value: Any) -> float:
        return _validate_coordinate(value)


def _validate_tag_dictionary_query(value: str | None) -> str | None:
    if value is None:
        return None
    return validate_tag_dictionary(value)


TagDictionaryQuery = Annotated[
    str | None,
    Query(min_length=1),
    AfterValidator(_validate_tag_dictionary_query),
]


def create_app(
    service: MultiVisionService | None = None,
    *,
    manage_lifecycle: bool = True,
) -> FastAPI:
    """Build an app with optional ownership of the injected service lifecycle."""
    owned_service = service if service is not None else MultiVisionService()
    if not isinstance(manage_lifecycle, bool):
        raise TypeError('manage_lifecycle must be a bool')

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        if not manage_lifecycle:
            yield
            return

        lifecycle_error: BaseException | None = None
        try:
            owned_service.start()
            yield
        except BaseException as ex:  # noqa: BLE001 (Shutdown must also run after startup failure).
            lifecycle_error = ex
            raise
        finally:
            try:
                owned_service.shutdown()
            except BaseException:  # noqa: BLE001 (Preserve the original lifecycle failure).
                if lifecycle_error is None:
                    raise

    app = FastAPI(
        title='MultiVision',
        version='0.1.0',
        lifespan=lifespan,
    )
    app.state.multivision_service = owned_service

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            'REQUEST_VALIDATION_ERROR',
            'The request did not satisfy the endpoint schema',
            422,
            details=_validation_details(exception),
        )

    @app.exception_handler(MultiVisionError)
    async def handle_multivision_error(
        _request: Request,
        exception: MultiVisionError,
    ) -> JSONResponse:
        return _error_response(
            _error_code(exception),
            str(exception),
            _error_status(exception),
        )

    @app.exception_handler(ValueError)
    async def handle_boundary_error(
        _request: Request,
        exception: ValueError,
    ) -> JSONResponse:
        return _error_response('INVALID_REQUEST', str(exception), 422)

    app.add_exception_handler(TypeError, handle_boundary_error)

    @app.get('/health')
    def get_health() -> dict[str, Any]:
        statuses = owned_service.get_camera_statuses()
        available_count = sum(
            status.runtime_status is RuntimeStatus.AVAILABLE
            for status in statuses
        )
        return {
            'status': 'ok' if owned_service.is_running else 'starting',
            'service': 'running' if owned_service.is_running else 'stopped',
            'camera_count': len(statuses),
            'available_camera_count': available_count,
        }

    @app.get('/cameras')
    def get_cameras() -> list[dict[str, Any]]:
        session_cameras = owned_service.get_session_cameras()
        if len(session_cameras) > 0:
            areas = {
                area.slot_id: area
                for area in owned_service.get_camera_areas()
            }
            return [
                _session_camera_to_data(
                    owned_service,
                    camera,
                    areas[camera.slot_id],
                )
                for camera in session_cameras
            ]
        return [_camera_to_data(status) for status in owned_service.get_camera_statuses()]

    @app.post('/cameras/{slot_id}/rename')
    def rename_camera(
        slot_id: Annotated[str, Path(min_length=1)],
        request: CameraRenameRequest,
    ) -> dict[str, Any]:
        camera = owned_service.rename_camera(slot_id, request.name)
        return _session_camera_to_data(owned_service, camera)

    @app.post('/cameras/{slot_id}/area')
    def set_camera_area(
        slot_id: Annotated[str, Path(min_length=1)],
        request: CameraAreaRequest,
    ) -> dict[str, Any]:
        area = owned_service.set_area_enabled(slot_id, request.enabled)
        return _camera_area_to_data(owned_service, area)

    @app.post('/cameras/{slot_id}/close')
    def close_camera(
        slot_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        camera = owned_service.close_camera(slot_id)
        return _session_camera_to_data(owned_service, camera)

    @app.post('/cameras/{slot_id}/open')
    def open_camera(
        slot_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        camera = owned_service.open_camera(slot_id)
        return _session_camera_to_data(owned_service, camera)

    @app.get('/cameras/discovered')
    def get_discovered_cameras() -> list[dict[str, Any]]:
        return [
            _device_to_data(device)
            for device in owned_service.get_discovered_devices()
        ]

    @app.get('/cameras/{logical_name}/status')
    def get_camera_status(
        logical_name: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        camera = _find_session_camera(owned_service, logical_name)
        if camera is not None:
            return _session_camera_to_data(owned_service, camera)
        return _camera_to_data(owned_service.get_camera_status(logical_name))

    @app.get('/cameras/{camera}/tags')
    def get_camera_tags(
        camera: Annotated[str, Path(min_length=1)],
        dictionary: TagDictionaryQuery = None,
    ) -> dict[str, Any]:
        result = owned_service.inspect_tags(camera, dictionary)
        return _json_safe(result.to_data())

    @app.get('/cameras/{logical_name}/snapshot')
    def get_camera_snapshot(
        logical_name: Annotated[str, Path(min_length=1)],
    ) -> Response:
        frame = owned_service.snapshot(logical_name)
        return _frame_response(frame, logical_name)

    @app.post('/calibration/pattern')
    def show_calibration_pattern() -> dict[str, bool]:
        owned_service.show_calibration_pattern()
        return {'visible': True}

    @app.delete('/calibration/pattern')
    def hide_calibration_pattern() -> dict[str, bool]:
        owned_service.hide_calibration_pattern()
        return {'visible': False}

    @app.post('/calibration')
    def calibrate(request: CalibrationRequest | None = None) -> dict[str, Any]:
        request = CalibrationRequest() if request is None else request
        result = owned_service.calibrate(
            request.camera,
            _to_correspondences(request),
        )
        if isinstance(result, dict):
            return {
                'cameras': {
                    logical_name: _calibration_to_data(
                        record,
                        owned_service.get_camera_status(logical_name).calibration_status,
                    )
                    for logical_name, record in result.items()
                },
            }
        assert request.camera is not None
        return _calibration_to_data(
            result,
            owned_service.get_camera_status(request.camera).calibration_status,
        )

    @app.post('/calibration/verify')
    def verify_calibration(request: CalibrationRequest | None = None) -> dict[str, Any]:
        request = CalibrationRequest() if request is None else request
        result = owned_service.verify(
            request.camera,
            _to_correspondences(request),
        )
        if isinstance(result, dict):
            return {
                'cameras': {
                    logical_name: status.value
                    for logical_name, status in result.items()
                },
            }
        assert request.camera is not None
        return {
            'camera': request.camera,
            'status': result.value,
        }

    @app.get('/calibration/status')
    def get_calibration_status() -> dict[str, Any]:
        stage, metric_status, statuses = owned_service.get_calibration_status_snapshot()
        return {
            'calibration': stage.value,
            'metric_calibration': metric_status.value,
            'cameras': {
                status.logical_name: status.calibration_status.value
                for status in statuses
            },
            'calibrations': {
                camera_id: calibration.to_data()
                for camera_id, calibration in owned_service.get_calibration_records().items()
            },
        }

    @app.post('/metric/calibration')
    def calibrate_metric(request: MetricCalibrationRequest) -> dict[str, Any]:
        correspondences = _to_metric_correspondences(request)
        record = owned_service.calibrate_metric(request.camera, correspondences)
        return _json_safe(_metric_calibration_to_data(record))

    @app.get('/metric/calibration/status')
    def get_metric_calibration_status() -> dict[str, Any]:
        status, record = owned_service.get_metric_status_snapshot()
        return _json_safe(_metric_status_to_data(status, record, owned_service))

    @app.delete('/metric/calibration')
    def clear_metric_calibration() -> dict[str, Any]:
        owned_service.clear_metric_calibration()
        return {'cleared': True}

    @app.post('/metric/ruler')
    def set_metric_ruler(request: MetricRulerRequest) -> dict[str, Any]:
        surface_start = _point_from_request(request.surface_start)
        surface_end = _point_from_request(request.surface_end)
        ruler, validation = owned_service.set_metric_ruler_with_validation(
            surface_start,
            surface_end,
            request.unit,
            request.observed_length,
            request.observed_unit,
        )
        return _json_safe(_metric_ruler_to_data(ruler, validation, request))

    @app.delete('/metric/ruler')
    def clear_metric_ruler() -> dict[str, Any]:
        owned_service.clear_metric_ruler()
        return {'cleared': True}

    @app.post('/overlay/point')
    def point(request: PointRequest) -> dict[str, Any]:
        overlay = owned_service.point_from_camera(
            request.camera,
            (request.x, request.y),
        )
        return _overlay_to_data(overlay)

    @app.delete('/overlay')
    def clear_overlay() -> dict[str, Any]:
        owned_service.clear_overlay()
        return {'cleared': True}

    @app.post('/overlays/grid')
    def create_grid_overlay(request: GridRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.post('/overlays/grid/projector-footprint')
    def create_projector_coverage_grid_overlay(
        request: ProjectorCoverageGridRequest,
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(
            owned_service.create_projector_coverage_grid(request),
        )

    @app.post('/overlays/circle')
    def create_circle_overlay(request: CircleRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.post('/overlays/rect')
    def create_rect_overlay(request: RectRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.post('/overlays/text')
    def create_text_overlay(request: TextRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.post('/overlays/line')
    def create_line_overlay(request: LineRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.post('/overlays/ruler')
    def create_ruler_overlay(request: RulerRequest) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.create_overlay(request))

    @app.get('/overlays')
    def list_overlay_state() -> list[dict[str, Any]]:
        return [
            _overlay_entry_to_data(entry)
            for entry in owned_service.list_overlays()
        ]

    @app.post('/overlays/id/{overlay_id}/show')
    def show_overlay_by_id(
        overlay_id: Annotated[UUID4, Path()],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.show_overlay(overlay_id))

    @app.post('/overlays/id/{overlay_id}/hide')
    def hide_overlay_by_id(
        overlay_id: Annotated[UUID4, Path()],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.hide_overlay(overlay_id))

    @app.delete('/overlays/id/{overlay_id}')
    def remove_overlay_by_id(
        overlay_id: Annotated[UUID4, Path()],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.remove_overlay(overlay_id))

    @app.post('/overlays/name/{overlay_name}/show')
    def show_overlay_by_name(
        overlay_name: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.show_overlay(overlay_name))

    @app.post('/overlays/name/{overlay_name}/hide')
    def hide_overlay_by_name(
        overlay_name: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.hide_overlay(overlay_name))

    @app.delete('/overlays/name/{overlay_name}')
    def remove_overlay_by_name(
        overlay_name: Annotated[str, Path(min_length=1)],
    ) -> dict[str, Any]:
        return _overlay_entry_to_data(owned_service.remove_overlay(overlay_name))

    @app.delete('/overlays')
    def clear_overlays() -> dict[str, bool]:
        owned_service.clear_overlays()
        return {'cleared': True}

    return app


def _validate_coordinate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('coordinate must be a finite number')
    try:
        coordinate = float(value)
    except (OverflowError, TypeError, ValueError) as ex:
        raise ValueError('coordinate must be a finite number') from ex
    if not math.isfinite(coordinate):
        raise ValueError('coordinate must be finite')
    return coordinate


def _validation_details(exception: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {
            'location': [str(elem) for elem in error.get('loc', ())],
            'message': str(error.get('msg', 'Invalid value')),
            'type': str(error.get('type', 'value_error')),
        }
        for error in exception.errors()
    ]


def _to_correspondences(
    request: CalibrationRequest,
) -> tuple[FiducialCorrespondence, ...] | None:
    if request.correspondences is None:
        return None
    return tuple(
        _to_correspondence(correspondence)
        for correspondence in request.correspondences
    )


def _to_correspondence(request: CorrespondenceRequest) -> FiducialCorrespondence:
    return FiducialCorrespondence(
        request.marker_id,
        request.corner_index,
        _point_from_request(request.projector),
        _point_from_request(request.camera),
    )


def _to_metric_correspondences(
    request: MetricCalibrationRequest,
) -> tuple[MetricTargetCorrespondence, ...] | None:
    if request.correspondences is None:
        return None
    return tuple(
        MetricTargetCorrespondence(
            correspondence.marker_id,
            correspondence.corner_index,
            _point_from_request(correspondence.surface),
            _point_from_request(correspondence.camera),
        )
        for correspondence in request.correspondences
    )


def _point_from_request(request: PointPairRequest) -> Point2D:
    return Point2D(request.x, request.y)


def _metric_point_to_data(point: Point2D) -> list[float]:
    return [point.x, point.y]


def _metric_target_to_data() -> dict[str, Any]:
    return {
        'format': METRIC_TARGET.format_name,
        'format_name': METRIC_TARGET.format_name,
        'version': METRIC_TARGET.format_version,
        'format_version': METRIC_TARGET.format_version,
        'marker_family': METRIC_TARGET.marker_family,
        'marker_count': METRIC_TARGET.marker_count,
        'marker_ids': list(METRIC_TARGET.marker_ids),
        'page_size_mm': list(METRIC_TARGET.page_size_mm),
        'markers': [
            {
                'marker_id': marker.marker_id,
                'corners': [_metric_point_to_data(point) for point in marker.corners],
            }
            for marker in METRIC_TARGET.markers
        ],
        'orientation_cue': {
            'label': METRIC_TARGET.orientation_cue.label,
            'corners': [
                _metric_point_to_data(point)
                for point in METRIC_TARGET.orientation_cue.corners
            ],
            'text_position': _metric_point_to_data(
                METRIC_TARGET.orientation_cue.text_position,
            ),
        },
        'reference_segment': {
            'start': _metric_point_to_data(METRIC_TARGET.reference_segment.start),
            'end': _metric_point_to_data(METRIC_TARGET.reference_segment.end),
            'length_mm': METRIC_TARGET.reference_segment.length_mm,
            'label': METRIC_TARGET.reference_segment.label,
        },
    }


def _metric_calibration_to_data(
    record: MetricCalibrationRecord,
) -> dict[str, Any]:
    metrics = record.metrics
    return {
        'state': record.state.value,
        'projector_to_surface': (
            None
            if record.projector_to_surface is None
            else [list(row) for row in record.projector_to_surface]
        ),
        'surface_to_projector': (
            None
            if record.surface_to_projector is None
            else [list(row) for row in record.surface_to_projector]
        ),
        'projector_output_descriptor': _descriptor_to_data(
            record.projector_output_descriptor,
        ),
        'projector_resolution': _resolution_to_data(record.projector_resolution),
        'output_identity': record.output_identity,
        'target': _metric_target_to_data(),
        'target_format': record.target_format,
        'target_version': record.target_version,
        'marker_family': record.marker_family,
        'observation_camera_slot': record.observation_camera_slot,
        'observation_camera_id': record.observation_camera_id,
        'observation_camera_calibration_version': (
            record.observation_camera_calibration_version
        ),
        'observation_camera_calibration_timestamp': (
            record.observation_camera_calibration_timestamp
        ),
        'timestamp': record.timestamp,
        'metrics': None if metrics is None else {
            'unique_target_fiducial_count': metrics.unique_target_fiducial_count,
            'correspondence_corner_count': metrics.correspondence_corner_count,
            'ransac_inlier_count': metrics.ransac_inlier_count,
            'inlier_ratio': metrics.inlier_ratio,
            'mean_fit_error_mm': metrics.mean_fit_error_mm,
            'max_fit_error_mm': metrics.max_fit_error_mm,
            'target_page_spatial_coverage': metrics.target_page_spatial_coverage,
        },
        'fit_error_mm': record.fit_error_mm,
        'validation_records': [
            _metric_validation_to_data(validation)
            for validation in record.validation_records
        ],
        'latest_physical_validation_error_mm': (
            record.latest_physical_validation_error_mm
        ),
    }


def _descriptor_to_data(
    descriptor: ProjectorOutputDescriptor,
) -> dict[str, Any]:
    return {
        'projector_resolution': _resolution_to_data(descriptor.projector_resolution),
        'output_identity': descriptor.output_identity,
    }


def _metric_validation_to_data(
    validation: MetricValidationRecord,
) -> dict[str, Any]:
    return {
        'requested_length_mm': validation.requested_length_mm,
        'observed_length_mm': validation.observed_length_mm,
        'absolute_error_mm': validation.absolute_error_mm,
        'timestamp': validation.timestamp,
    }


def _metric_status_to_data(
    status: MetricCalibrationStatus,
    record: MetricCalibrationRecord | None,
    service: MultiVisionService,
) -> dict[str, Any]:
    if not isinstance(status, MetricCalibrationStatus):
        raise GeometryError('Metric service returned an invalid calibration state')
    descriptor = service.projector_output_descriptor
    state = status.value
    error_code = None if status is MetricCalibrationStatus.CALIBRATED else (
        'METRIC_STALE'
        if status is MetricCalibrationStatus.STALE
        else 'METRIC_UNAVAILABLE'
    )
    data: dict[str, Any] = {
        'state': state,
        'status': state,
        'applicable': status is MetricCalibrationStatus.CALIBRATED,
        'error_code': error_code,
        'projector_output_descriptor': _descriptor_to_data(descriptor),
        'projector_resolution': _resolution_to_data(descriptor.projector_resolution),
        'output_identity': descriptor.output_identity,
        'target': _metric_target_to_data(),
        'calibration': None,
    }
    if record is not None:
        calibration_data = _metric_calibration_to_data(record)
        data['calibration'] = calibration_data
        data.update(
            {
                key: value
                for key, value in calibration_data.items()
                if key not in {
                    'state',
                    'projector_output_descriptor',
                    'projector_resolution',
                    'output_identity',
                }
            },
        )
    return data


def _metric_ruler_to_data(
    ruler: MetricRulerOverlay,
    validation: MetricValidationRecord | None,
    request: MetricRulerRequest,
) -> dict[str, Any]:
    data = ruler.to_data()
    data.update(
        {
            'observed_length': request.observed_length,
            'observed_unit': (
                request.observed_unit if request.observed_length is not None else None
            ),
            'observed_length_mm': (
                None if validation is None else validation.observed_length_mm
            ),
            'absolute_error_mm': (
                None if validation is None else validation.absolute_error_mm
            ),
            'validation': (
                None if validation is None else _metric_validation_to_data(validation)
            ),
        },
    )
    return data


def _device_to_data(device: DeviceInfo) -> dict[str, Any]:
    return {
        'device_id': device.device_id,
        'name': device.name,
        'capture_index': device.capture_index,
        'backend_name': device.backend_name,
        'native_resolution': _resolution_to_data(device.native_resolution),
        'is_available': device.is_available,
        'is_stable_id': device.is_stable_id,
        'error_message': device.error_message,
    }


def _camera_to_data(status: CameraStatus) -> dict[str, Any]:
    lifecycle = _lifecycle_for_status(status)
    return {
        'camera': status.logical_name,
        'slot': status.logical_name,
        'name': status.logical_name,
        'device_id': status.device_id,
        'state': lifecycle,
        'lifecycle': lifecycle,
        'runtime_status': status.runtime_status.value,
        'calibration_status': status.calibration_status.value,
        'calibration': status.calibration_status.value,
        'native_resolution': _resolution_to_data(status.native_resolution),
        'frame_counter': status.frame_counter,
        'frame_metadata': None,
        'area_enabled': False,
        'area_colour': list(get_camera_area_colour(status.logical_name)),
        'available_area': None,
        'error_message': status.error_message,
    }


def _session_camera_to_data(
    service: MultiVisionService,
    camera: SessionCamera,
    area: CameraArea | None = None,
) -> dict[str, Any]:
    status = service.get_camera_status(camera.slot_id)
    area = service.get_camera_area(camera.slot_id) if area is None else area
    device_info = camera.device_info
    frame_metadata = camera.frame_metadata
    return {
        'camera': camera.display_name,
        'slot': camera.slot_id,
        'name': camera.display_name,
        'device_id': device_info.device_id if device_info is not None else None,
        'capture_index': camera.capture_index,
        'state': camera.state.value,
        'lifecycle': camera.state.value,
        'runtime_status': status.runtime_status.value,
        'calibration_status': status.calibration_status.value,
        'calibration': status.calibration_status.value,
        'native_resolution': _resolution_to_data(status.native_resolution),
        'frame_counter': status.frame_counter,
        'frame_metadata': _frame_metadata_to_data(frame_metadata),
        'area_enabled': area.area_enabled,
        'area_colour': list(area.area_colour),
        'available_area': _polygon_to_data(area.available_area),
        'error_message': status.error_message or camera.error_message,
    }


def _camera_area_to_data(
    service: MultiVisionService,
    area: CameraArea,
) -> dict[str, Any]:
    camera = _find_session_camera(service, area.slot_id)
    if camera is None:
        raise SessionCameraError(
            f'Camera slot {area.slot_id!r} is not in the session inventory',
        )
    return _session_camera_to_data(service, camera, area)


def _find_session_camera(
    service: MultiVisionService,
    camera_reference: str,
) -> SessionCamera | None:
    for camera in service.get_session_cameras():
        if camera.slot_id == camera_reference or camera.display_name == camera_reference:
            return camera
    return None


def _polygon_to_data(polygon: Polygon | None) -> list[list[float]] | None:
    if polygon is None:
        return None
    return [[point.x, point.y] for point in polygon]


def _lifecycle_for_status(status: CameraStatus) -> str:
    if status.runtime_status is RuntimeStatus.STOPPED:
        return 'CLOSED'
    if status.runtime_status in {RuntimeStatus.ERROR, RuntimeStatus.UNAVAILABLE}:
        return 'UNAVAILABLE'
    return 'OPEN'


def _frame_metadata_to_data(
    frame_metadata: FrameMetadata | None,
) -> dict[str, Any] | None:
    if frame_metadata is None:
        return None
    return {
        'frame_counter': frame_metadata.frame_counter,
        'captured_at_seconds': frame_metadata.captured_at_seconds,
        'native_resolution': _resolution_to_data(frame_metadata.native_resolution),
    }


def _resolution_to_data(resolution: Resolution | None) -> dict[str, int] | None:
    if resolution is None:
        return None
    return {'width': resolution.width, 'height': resolution.height}


def _calibration_to_data(
    calibration: PersistedCalibration,
    status: CalibrationStatus,
) -> dict[str, Any]:
    data = calibration.to_data()
    data.update({'status': status.value})
    return data


def _overlay_to_data(overlay: RedCircleOverlay) -> dict[str, Any]:
    return overlay.to_data()


def _overlay_entry_to_data(entry: OverlayEntry) -> dict[str, Any]:
    if not isinstance(entry, OverlayEntry):
        raise GeometryError('Overlay service returned an invalid overlay entry')
    request_data = entry.request.model_dump(mode='json')
    return {
        'id': str(entry.id),
        'name': entry.name,
        'kind': entry.kind,
        'visible': entry.visible,
        'request': request_data,
        'camera_dependencies': list(entry.camera_dependencies),
        'metric_dependency': entry.metric_dependency,
        'projector_output_descriptor': _descriptor_to_data(
            entry.projector_output_descriptor,
        ),
    }


def _frame_response(frame: Frame, logical_name: str) -> Response:
    if (
        not isinstance(frame.frame_counter, int)
        or isinstance(frame.frame_counter, bool)
        or frame.frame_counter < 0
        or not is_finite_real(frame.captured_at_seconds)
    ):
        raise FrameCaptureError('Camera returned malformed frame metadata')
    frame_data = frame.data
    if isinstance(frame_data, bytes):
        return Response(content=frame_data, media_type='application/octet-stream')
    if isinstance(frame_data, (bytearray, memoryview)):
        return Response(content=bytes(frame_data), media_type='application/octet-stream')

    image_response = _encoded_image_response(frame_data)
    if image_response is not None:
        return image_response

    return JSONResponse(
        {
            'camera': logical_name,
            'frame_counter': frame.frame_counter,
            'captured_at_seconds': frame.captured_at_seconds,
            'data': _json_safe(frame_data),
        },
    )


def _encoded_image_response(frame_data: Any) -> Response | None:
    if not hasattr(frame_data, 'shape'):
        return None
    try:
        import cv2
    except ImportError as ex:
        raise FrameCaptureError('OpenCV is required to encode camera frames') from ex

    try:
        success, encoded_image = cv2.imencode('.jpg', frame_data)
    except Exception as ex:  # noqa: BLE001 (OpenCV is an image encoding boundary).
        raise FrameCaptureError('Could not encode the camera frame') from ex
    if not success:
        raise FrameCaptureError('Could not encode the camera frame')
    try:
        encoded_data = encoded_image.tobytes()
    except Exception as ex:  # noqa: BLE001 (Encoded image data is an external value).
        raise FrameCaptureError('OpenCV returned malformed encoded image data') from ex
    return Response(content=encoded_data, media_type='image/jpeg')


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FrameCaptureError('Camera returned a non-finite frame value')
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(elem) for elem in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(elem) for key, elem in value.items()}
    return repr(value)


def _error_code(exception: MultiVisionError) -> str:
    explicit_code = getattr(exception, 'code', None)
    if isinstance(explicit_code, str) and len(explicit_code) > 0:
        return explicit_code
    error_codes: tuple[tuple[type[MultiVisionError], str], ...] = (
        (SessionCameraError, 'SESSION_CAMERA_ERROR'),
        (CameraUnavailableError, 'CAMERA_UNAVAILABLE'),
        (FrameCaptureError, 'FRAME_UNAVAILABLE'),
        (ConfigurationError, 'CONFIGURATION_ERROR'),
        (CalibrationError, 'CALIBRATION_ERROR'),
        (FiducialDetectionError, 'FIDUCIAL_DETECTION_ERROR'),
        (GeometryError, 'GEOMETRY_ERROR'),
        (HardwareError, 'HARDWARE_ERROR'),
    )
    for error_type, code in error_codes:
        if isinstance(exception, error_type):
            return code
    return 'MULTIVISION_ERROR'


def _error_status(exception: MultiVisionError) -> int:
    if isinstance(exception, (CameraSlotNotFoundError, OverlayNotFoundError)):
        return 404
    if isinstance(exception, SessionCameraError):
        return 409
    if isinstance(exception, (CameraUnavailableError, FrameCaptureError)):
        return 503
    if isinstance(exception, GeometryError):
        return 422
    if isinstance(exception, (CalibrationError, ConfigurationError)):
        return 422
    return 500


def _error_response(
    code: str,
    message: str,
    status_code: int,
    details: Any | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {'code': code, 'message': message}
    if details is not None:
        error['details'] = details
    return JSONResponse({'error': error}, status_code=status_code)


app = create_app()

__all__ = [
    'CalibrationRequest',
    'MetricCalibrationRequest',
    'MetricCorrespondenceRequest',
    'MetricRulerRequest',
    'CameraAreaRequest',
    'CameraRenameRequest',
    'CorrespondenceRequest',
    'PointPairRequest',
    'PointRequest',
    'app',
    'create_app',
]
