"""Shared camera-point to projector-overlay operations."""

from __future__ import annotations

import math
import threading
from typing import (
    Any,
    NamedTuple,
    Protocol,
)

from multivision.config import ProjectorOutputDescriptor
from multivision.errors import (
    CameraUnavailableError,
    InvalidCalibrationStateError,
    InvalidHomographyError,
    PointOutsideCalibratedRegionError,
    PointOutsideProjectorError,
)
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    PointLike,
    PreviewTransform,
    coerce_point,
    is_finite_point,
    is_point_in_resolution,
    project_camera_to_projector,
)
from multivision.persistence import PersistedCalibration
from multivision.session import SessionCamera
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    Resolution,
    RuntimeStatus,
    SessionCameraState,
    is_finite_real,
    is_valid_resolution,
)


class CameraPointRuntime(Protocol):
    def get_status(self, logical_name: str) -> CameraStatus:
        ...


class CalibrationPointRegistry(Protocol):
    def get_status(
        self,
        camera_id: str,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
    ) -> CalibrationStatus:
        ...

    def get_status_error_code(
        self,
        camera_id: str,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
    ) -> str:
        ...

    def project_camera_to_projector(
        self,
        camera_id: str,
        point: PointLike,
        camera_resolution: Resolution,
        projector_resolution: Resolution,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> Point2D:
        ...


class _ResolvedCamera(NamedTuple):
    status: CameraStatus
    session_camera: SessionCamera | None
    calibration_id: str


class RedCircleOverlay(NamedTuple):
    """The one overlay primitive required by MVP0."""

    logical_name: str
    camera_id: str
    camera_point: Point2D
    projector_point: Point2D
    radius: int = 12
    colour: tuple[int, int, int] = (255, 0, 0)

    def to_data(self) -> dict[str, Any]:
        return {
            'camera': self.logical_name,
            'camera_id': self.camera_id,
            'camera_point': [self.camera_point.x, self.camera_point.y],
            'projector_point': [self.projector_point.x, self.projector_point.y],
            'radius': self.radius,
            'colour': list(self.colour),
        }


class PointOverlayService:
    """Own the shared point validation, projection and overlay state path."""

    def __init__(
        self,
        camera_runtime: CameraPointRuntime,
        calibration_registry: CalibrationPointRegistry,
        projector_resolution: Resolution,
        overlay_radius: int = 12,
        calibration_version: int = 1,
        projector_output_descriptor: ProjectorOutputDescriptor | None = None,
    ) -> None:
        if not is_valid_resolution(projector_resolution):
            raise ValueError('projector_resolution must be a positive Resolution')
        if (
            not isinstance(overlay_radius, int)
            or isinstance(overlay_radius, bool)
            or overlay_radius <= 0
        ):
            raise ValueError('overlay_radius must be a positive integer')
        if (
            not isinstance(calibration_version, int)
            or isinstance(calibration_version, bool)
            or calibration_version <= 0
        ):
            raise ValueError('calibration_version must be a positive integer')

        self.camera_runtime = camera_runtime
        self.calibration_registry = calibration_registry
        self.projector_resolution = projector_resolution
        self.projector_output_descriptor = projector_output_descriptor
        self.overlay_radius = overlay_radius
        self.calibration_version = calibration_version
        self._overlay: RedCircleOverlay | None = None
        self._lock = threading.RLock()

    @property
    def overlay(self) -> RedCircleOverlay | None:
        with self._lock:
            return self._overlay

    def point_from_preview(
        self,
        logical_name: str,
        preview_point: PointLike,
        preview_transform: PreviewTransform,
    ) -> RedCircleOverlay:
        """Convert one preview-local click through the native point path."""
        _validate_preview_transform(preview_transform)
        with self._lock:
            camera = self._get_available_camera(logical_name)
            if camera.status.native_resolution != preview_transform.camera_resolution:
                raise self._calibration_error(
                    'CAMERA_RESOLUTION_CHANGED',
                    f'Camera {logical_name!r} resolution does not match the preview',
                )
            camera_point = preview_transform.to_camera_native(preview_point)
            return self._point_from_camera(camera, camera_point)

    def point_from_camera(
        self,
        logical_name: str,
        camera_point: PointLike,
    ) -> RedCircleOverlay:
        """Validate and project a camera-native point, replacing the overlay on success."""
        checked_camera_point = coerce_point(camera_point)
        with self._lock:
            camera = self._get_available_camera(logical_name)
            return self._point_from_camera(camera, checked_camera_point)

    def clear_overlay(self) -> None:
        """Remove the current overlay without changing calibration state."""
        with self._lock:
            self._overlay = None

    def clear_overlay_for_camera(self, camera_id: str) -> None:
        """Remove an overlay only when it belongs to the changed camera."""
        if not isinstance(camera_id, str) or len(camera_id) == 0:
            raise ValueError('camera_id must be a non-empty string')
        with self._lock:
            overlay = self._overlay
            if overlay is None or overlay.camera_id != camera_id:
                return
            self._overlay = None

    def rename_overlay_camera(self, camera_id: str, logical_name: str) -> None:
        """Keep an owned overlay associated with a renamed session camera."""
        if not isinstance(camera_id, str) or len(camera_id) == 0:
            raise ValueError('camera_id must be a non-empty string')
        if not isinstance(logical_name, str) or len(logical_name) == 0:
            raise ValueError('logical_name must be a non-empty string')
        with self._lock:
            overlay = self._overlay
            if overlay is None:
                return
            if overlay.camera_id != camera_id:
                return
            self._overlay = overlay._replace(logical_name=logical_name)

    def _get_available_camera(self, camera_reference: str) -> _ResolvedCamera:
        if not isinstance(camera_reference, str) or len(camera_reference) == 0:
            raise ValueError('camera_reference must be a non-empty string')
        session_camera = self._find_session_camera(camera_reference)
        resolved_reference = (
            session_camera.slot_id
            if session_camera is not None
            else camera_reference
        )
        try:
            status = self.camera_runtime.get_status(resolved_reference)
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
        if status.logical_name != resolved_reference:
            raise CameraUnavailableError(
                f'Camera {camera_reference!r} returned a status for another camera',
            )
        if session_camera is not None and session_camera.state is not SessionCameraState.OPEN:
            raise CameraUnavailableError(
                session_camera.error_message
                or f'Camera {camera_reference!r} is not available',
            )
        if status.runtime_status is not RuntimeStatus.AVAILABLE:
            raise CameraUnavailableError(
                status.error_message
                or f'Camera {camera_reference!r} is not available',
            )
        if not is_valid_resolution(status.native_resolution):
            raise CameraUnavailableError(
                f'Camera {camera_reference!r} has no native resolution',
            )
        calibration_id = (
            session_camera.slot_id
            if session_camera is not None
            else status.device_id
        )
        if not isinstance(calibration_id, str) or len(calibration_id) == 0:
            raise CameraUnavailableError(
                f'Camera {camera_reference!r} has no calibration identity',
            )
        return _ResolvedCamera(status, session_camera, calibration_id)

    def _find_session_camera(self, camera_reference: str) -> SessionCamera | None:
        get_session_cameras = getattr(self.camera_runtime, 'get_session_cameras', None)
        if not callable(get_session_cameras):
            return None
        session_cameras = get_session_cameras()
        if not isinstance(session_cameras, list) or any(
            not isinstance(camera, SessionCamera)
            for camera in session_cameras
        ):
            raise CameraUnavailableError('Camera runtime returned an invalid session camera list')
        for camera in session_cameras:
            if camera.slot_id == camera_reference:
                return camera
        matches = [
            camera
            for camera in session_cameras
            if camera.display_name == camera_reference
        ]
        if len(matches) > 1:
            raise CameraUnavailableError(
                f'Camera name {camera_reference!r} is ambiguous',
            )
        return matches[0] if len(matches) == 1 else None

    def _point_from_camera(
        self,
        camera: _ResolvedCamera,
        camera_point: Point2D,
    ) -> RedCircleOverlay:
        status = camera.status
        assert status.native_resolution is not None
        if not is_point_in_resolution(camera_point, status.native_resolution):
            raise PointOutsideCalibratedRegionError(
                f'Camera point {camera_point!r} is outside the native camera bounds',
            )
        session_calibration_error_code = self._get_session_calibration_error_code(camera)
        if session_calibration_error_code is not None:
            raise self._calibration_error(
                session_calibration_error_code,
                f'Camera {status.logical_name!r} calibration is not applicable',
            )
        calibration_status = self._get_calibration_status(camera)
        if not isinstance(calibration_status, CalibrationStatus):
            raise self._calibration_error(
                'CALIBRATION_INVALID',
                f'Camera {status.logical_name!r} returned an invalid calibration status',
            )
        if calibration_status is not CalibrationStatus.CALIBRATED:
            get_status_error_code = getattr(
                self.calibration_registry,
                'get_status_error_code',
                None,
            )
            error_code = _calibration_error_code(calibration_status)
            if camera.session_camera is None and callable(get_status_error_code):
                error_code = get_status_error_code(
                    camera.calibration_id,
                    camera_resolution=status.native_resolution,
                    projector_resolution=self.projector_resolution,
                )
            raise self._calibration_error(
                error_code,
                f'Camera {status.logical_name!r} calibration is {calibration_status.value}',
            )
        projector_point = self._project_camera_point(camera, camera_point)
        if not isinstance(projector_point, Point2D) or not is_finite_point(projector_point):
            raise InvalidHomographyError(
                'Camera-to-projector homography produced a non-finite point',
            )
        if not is_point_in_resolution(projector_point, self.projector_resolution):
            raise PointOutsideProjectorError(
                f'Projected point {projector_point!r} is outside projector bounds',
            )

        overlay = RedCircleOverlay(
            camera.session_camera.display_name
            if camera.session_camera is not None
            else status.logical_name,
            camera.calibration_id,
            camera_point,
            projector_point,
            self.overlay_radius,
        )
        with self._lock:
            self._overlay = overlay
        return overlay

    def _get_session_calibration_error_code(
        self,
        camera: _ResolvedCamera,
    ) -> str | None:
        session_camera = camera.session_camera
        if session_camera is None:
            return None
        calibration = session_camera.calibration
        if calibration is None:
            return None
        if not isinstance(calibration, PersistedCalibration):
            return None
        if calibration.camera_id != session_camera.slot_id:
            return 'CALIBRATION_INVALID'
        if calibration.version != self.calibration_version:
            return 'CALIBRATION_STALE'
        if calibration.camera_resolution != camera.status.native_resolution:
            return 'CAMERA_RESOLUTION_CHANGED'
        if calibration.projector_resolution != self.projector_resolution:
            return 'PROJECTOR_RESOLUTION_CHANGED'
        if (
            self.projector_output_descriptor is not None
            and calibration.projector_output_descriptor != self.projector_output_descriptor
        ):
            return 'PROJECTOR_OUTPUT_CHANGED'
        return None

    def _get_calibration_status(self, camera: _ResolvedCamera) -> CalibrationStatus:
        session_camera = camera.session_camera
        if session_camera is not None:
            return session_camera.calibration_status
        if self.projector_output_descriptor is None:
            return self.calibration_registry.get_status(
                camera.calibration_id,
                camera_resolution=camera.status.native_resolution,
                projector_resolution=self.projector_resolution,
            )
        return self.calibration_registry.get_status(
            camera.calibration_id,
            camera_resolution=camera.status.native_resolution,
            projector_resolution=self.projector_resolution,
            projector_output_descriptor=self.projector_output_descriptor,
        )

    def _project_camera_point(
        self,
        camera: _ResolvedCamera,
        camera_point: Point2D,
    ) -> Point2D:
        session_camera = camera.session_camera
        if session_camera is None:
            if self.projector_output_descriptor is None:
                return self.calibration_registry.project_camera_to_projector(
                    camera.calibration_id,
                    camera_point,
                    camera_resolution=camera.status.native_resolution,
                    projector_resolution=self.projector_resolution,
                )
            return self.calibration_registry.project_camera_to_projector(
                camera.calibration_id,
                camera_point,
                camera_resolution=camera.status.native_resolution,
                projector_resolution=self.projector_resolution,
                projector_output_descriptor=self.projector_output_descriptor,
            )
        if session_camera.calibration is None:
            raise InvalidHomographyError(
                'Session camera has no current calibration transform',
            )

        calibration = session_camera.calibration
        camera_to_projector = getattr(calibration, 'camera_to_projector', None)
        if camera_to_projector is None:
            raise InvalidHomographyError(
                'Session camera calibration has no camera-to-projector transform',
            )
        camera_bounds = CoordinateBounds(
            0.0,
            0.0,
            float(camera.status.native_resolution.width),
            float(camera.status.native_resolution.height),
        )
        return project_camera_to_projector(
            camera_point,
            camera_to_projector,
            calibrated_region=camera_bounds,
            projector_resolution=self.projector_resolution,
        )

    @staticmethod
    def _calibration_error(
        code: str,
        message: str,
    ) -> InvalidCalibrationStateError:
        error = InvalidCalibrationStateError(message)
        error.code = code
        return error


def _calibration_error_code(status: CalibrationStatus) -> str:
    return {
        CalibrationStatus.UNCALIBRATED: 'CALIBRATION_UNCALIBRATED',
        CalibrationStatus.UNVERIFIED: 'CALIBRATION_UNVERIFIED',
        CalibrationStatus.STALE: 'CALIBRATION_STALE',
        CalibrationStatus.CALIBRATED: 'CALIBRATED',
    }.get(status, 'CALIBRATION_INVALID')


def _validate_preview_transform(preview_transform: PreviewTransform) -> None:
    if not isinstance(preview_transform, PreviewTransform):
        raise TypeError('preview_transform must be PreviewTransform')
    if not is_valid_resolution(preview_transform.preview_size):
        raise ValueError('preview_transform has an invalid preview resolution')
    if not is_valid_resolution(preview_transform.camera_resolution):
        raise ValueError('preview_transform has an invalid camera resolution')
    if not is_finite_real(preview_transform.scale) or preview_transform.scale <= 0:
        raise ValueError('preview_transform has an invalid scale')
    bounds = preview_transform.content_bounds
    if not isinstance(bounds, CoordinateBounds):
        raise ValueError('preview_transform has invalid content bounds')
    if not all(is_finite_real(value) for value in bounds):
        raise ValueError('preview_transform has invalid content bounds')
    coordinate_tolerance = 1e-9
    if not (
        -coordinate_tolerance <= bounds.left < bounds.right
        <= preview_transform.preview_size.width + coordinate_tolerance
        and -coordinate_tolerance <= bounds.top < bounds.bottom
        <= preview_transform.preview_size.height + coordinate_tolerance
    ):
        raise ValueError('preview_transform content bounds exceed the preview')
    if not (
        math.isclose(
            bounds.right - bounds.left,
            preview_transform.camera_resolution.width * preview_transform.scale,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            bounds.bottom - bounds.top,
            preview_transform.camera_resolution.height * preview_transform.scale,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError('preview_transform scale does not match its content bounds')


__all__ = [
    'CalibrationPointRegistry',
    'CameraPointRuntime',
    'PointOverlayService',
    'RedCircleOverlay',
]
