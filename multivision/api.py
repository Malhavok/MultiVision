"""Local FastAPI boundary for the session-local MultiVision service."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    Path,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
)

from multivision.application import (
    CameraArea,
    MultiVisionService,
    get_camera_area_colour,
)
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
    SessionCameraError,
)
from multivision.fiducials import FiducialCorrespondence
from multivision.geometry import (
    Point2D,
    Polygon,
)
from multivision.persistence import PersistedCalibration
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

    @app.get('/cameras/{logical_name}/snapshot')
    def get_camera_snapshot(
        logical_name: Annotated[str, Path(min_length=1)],
    ) -> Response:
        frame = owned_service.snapshot(logical_name)
        return _frame_response(frame, logical_name)

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
        statuses = owned_service.get_camera_statuses()
        return {
            'cameras': {
                status.logical_name: status.calibration_status.value
                for status in statuses
            },
            'calibrations': {
                camera_id: calibration.to_data()
                for camera_id, calibration in owned_service.get_calibration_records().items()
            },
        }

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


def _point_from_request(request: PointPairRequest) -> Point2D:
    return Point2D(request.x, request.y)


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
    if isinstance(exception, CameraSlotNotFoundError):
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
    'CameraAreaRequest',
    'CameraRenameRequest',
    'CorrespondenceRequest',
    'PointPairRequest',
    'PointRequest',
    'app',
    'create_app',
]
