class MultiVisionError(Exception):
    """Base class for expected MultiVision failures."""


class ConfigurationError(MultiVisionError):
    """Raised when the central configuration cannot be read or validated."""


class GeometryError(MultiVisionError, ValueError):
    """Base class for invalid coordinate and homography operations."""


class PointOutsidePreviewError(GeometryError):
    """Raised when a preview point falls in letterbox padding."""

    code = 'POINT_OUTSIDE_PREVIEW'


class PointOutsideCalibratedRegionError(GeometryError):
    """Raised when a camera point is outside calibrated spatial support."""

    code = 'POINT_OUTSIDE_CALIBRATED_REGION'


class PointOutsideProjectorError(GeometryError):
    """Raised when a projected point falls outside the projector output."""

    code = 'POINT_OUTSIDE_PROJECTOR_BOUNDS'


class InvalidCalibrationStateError(GeometryError):
    """Raised when spatial geometry is requested before verification."""

    code = 'CALIBRATION_INVALID'


class InvalidHomographyError(GeometryError):
    """Raised when a homography cannot safely project points."""

    code = 'INVALID_HOMOGRAPHY'


class FiducialDetectionError(MultiVisionError):
    """Raised when the fiducial detector cannot be initialised or run."""


class CalibrationError(MultiVisionError, ValueError):
    """Raised when correspondences cannot produce a trustworthy calibration."""


class HardwareError(MultiVisionError):
    """Base class for failures at a hardware boundary."""


class CameraOpenError(HardwareError):
    """Raised when a camera cannot be opened."""


class CameraUnavailableError(HardwareError):
    """Raised when a configured camera is not available."""

    code = 'CAMERA_UNAVAILABLE'


class FrameCaptureError(HardwareError):
    """Raised when an open camera cannot provide a usable frame."""
