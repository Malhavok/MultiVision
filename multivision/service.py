"""Shared camera-point to projector-overlay operations."""

from __future__ import annotations

import math
import threading
from typing import (
    Any,
    NamedTuple,
    Protocol,
)

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
)
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    Resolution,
    RuntimeStatus,
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
    ) -> Point2D:
        ...


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
    ) -> None:
        if not is_valid_resolution(projector_resolution):
            raise ValueError('projector_resolution must be a positive Resolution')
        if (
            not isinstance(overlay_radius, int)
            or isinstance(overlay_radius, bool)
            or overlay_radius <= 0
        ):
            raise ValueError('overlay_radius must be a positive integer')

        self.camera_runtime = camera_runtime
        self.calibration_registry = calibration_registry
        self.projector_resolution = projector_resolution
        self.overlay_radius = overlay_radius
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
            status = self._get_available_status(logical_name)
            if status.native_resolution != preview_transform.camera_resolution:
                raise self._calibration_error(
                    'CAMERA_RESOLUTION_CHANGED',
                    f'Camera {logical_name!r} resolution does not match the preview',
                )
            camera_point = preview_transform.to_camera_native(preview_point)
            return self._point_from_camera_status(status, camera_point)

    def point_from_camera(
        self,
        logical_name: str,
        camera_point: PointLike,
    ) -> RedCircleOverlay:
        """Validate and project a camera-native point, replacing the overlay on success."""
        checked_camera_point = coerce_point(camera_point)
        with self._lock:
            status = self._get_available_status(logical_name)
            return self._point_from_camera_status(status, checked_camera_point)

    def clear_overlay(self) -> None:
        """Remove the current overlay without changing calibration state."""
        with self._lock:
            self._overlay = None

    def _get_available_status(self, logical_name: str) -> CameraStatus:
        if not isinstance(logical_name, str) or len(logical_name) == 0:
            raise ValueError('logical_name must be a non-empty string')
        try:
            status = self.camera_runtime.get_status(logical_name)
        except CameraUnavailableError:
            raise
        except (KeyError, ValueError) as ex:
            raise CameraUnavailableError(
                f'Camera {logical_name!r} is not configured',
            ) from ex
        if not isinstance(status, CameraStatus):
            raise CameraUnavailableError(
                f'Camera {logical_name!r} returned an invalid status',
            )
        if status.logical_name != logical_name:
            raise CameraUnavailableError(
                f'Camera {logical_name!r} returned a status for another camera',
            )
        if status.runtime_status is not RuntimeStatus.AVAILABLE:
            raise CameraUnavailableError(
                status.error_message
                or f'Camera {logical_name!r} is not available',
            )
        if (
            not isinstance(status.device_id, str)
            or len(status.device_id) == 0
            or not is_valid_resolution(status.native_resolution)
        ):
            raise CameraUnavailableError(
                f'Camera {logical_name!r} has no stable device or native resolution',
            )
        return status

    def _point_from_camera_status(
        self,
        status: CameraStatus,
        camera_point: Point2D,
    ) -> RedCircleOverlay:
        assert status.device_id is not None
        assert status.native_resolution is not None
        if not is_point_in_resolution(camera_point, status.native_resolution):
            raise PointOutsideCalibratedRegionError(
                f'Camera point {camera_point!r} is outside the native camera bounds',
            )
        calibration_status = self.calibration_registry.get_status(
            status.device_id,
            camera_resolution=status.native_resolution,
            projector_resolution=self.projector_resolution,
        )
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
            error_code = (
                get_status_error_code(
                    status.device_id,
                    camera_resolution=status.native_resolution,
                    projector_resolution=self.projector_resolution,
                )
                if callable(get_status_error_code)
                else _calibration_error_code(calibration_status)
            )
            raise self._calibration_error(
                error_code,
                f'Camera {status.logical_name!r} calibration is {calibration_status.value}',
            )
        projector_point = self.calibration_registry.project_camera_to_projector(
            status.device_id,
            camera_point,
            camera_resolution=status.native_resolution,
            projector_resolution=self.projector_resolution,
        )
        if not isinstance(projector_point, Point2D) or not is_finite_point(projector_point):
            raise InvalidHomographyError(
                'Camera-to-projector homography produced a non-finite point',
            )
        if not is_point_in_resolution(projector_point, self.projector_resolution):
            raise PointOutsideProjectorError(
                f'Projected point {projector_point!r} is outside projector bounds',
            )

        overlay = RedCircleOverlay(
            status.logical_name,
            status.device_id,
            camera_point,
            projector_point,
            self.overlay_radius,
        )
        with self._lock:
            self._overlay = overlay
        return overlay

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
