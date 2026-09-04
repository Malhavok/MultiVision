"""Coordinate-aware overlay requests and source-space geometry builders."""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    UUID4,
    field_validator,
    model_validator,
)

from multivision.errors import (
    ConfigurationError,
    OverlayNotFoundError,
)
from multivision.geometry import (
    CoordinateBounds,
    MatrixLike,
    Point2D,
    TagGeometry,
    calculate_polygon_area,
    camera_to_projector as project_camera_point,
    coerce_point,
    is_point_in_region,
    project_point,
    project_tag_geometry,
    validate_homography,
    validate_planar_corners,
)
from multivision.types import (
    Resolution,
    is_finite_real,
    is_valid_resolution,
)


if TYPE_CHECKING:
    from multivision.config import ProjectorOutputDescriptor


class PointReferenceSpace(str, Enum):
    CAMERA_PX = 'camera_px'
    PROJECTOR_PX = 'projector_px'
    SURFACE_MM = 'surface_mm'


class GeometrySpace(str, Enum):
    PROJECTOR_PX = 'projector_px'
    SURFACE_MM = 'surface_mm'


OverlayUnit: TypeAlias = Literal['px', 'mm', 'cm', 'in']
PhysicalUnit: TypeAlias = Literal['mm', 'cm', 'in']
PointSpace: TypeAlias = Literal['camera_px', 'projector_px', 'surface_mm']

PHYSICAL_UNITS: frozenset[PhysicalUnit] = frozenset(('mm', 'cm', 'in'))

ANGLE_CONVENTION = (
    'zero points along positive x; positive angles are counter-clockwise '
    'with x right and y up'
)

_COLOUR_PATTERN = re.compile(r'^#[0-9a-fA-F]{6}$')


def _get_unit_to_mm() -> Mapping[str, float]:
    # Import lazily because configuration imports OverlayConfiguration from here.
    from multivision.metric import UNIT_TO_MM

    return UNIT_TO_MM


@dataclass(frozen=True)
class OverlayConfiguration:
    """Positive limits that bound overlay work before it reaches the registry."""

    max_overlay_vertices: int = 10_000
    max_overlay_segments: int = 5_000
    max_overlay_ticks: int = 200
    max_overlay_label_characters: int = 256

    def __post_init__(self) -> None:
        for field_name in (
            'max_overlay_vertices',
            'max_overlay_segments',
            'max_overlay_ticks',
            'max_overlay_label_characters',
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigurationError(f'{field_name} must be a positive integer')

    @classmethod
    def from_data(
        cls: type['OverlayConfiguration'],
        data: Any,
    ) -> 'OverlayConfiguration':
        if not isinstance(data, Mapping):
            raise ValueError('overlay_limits must be an object')
        allowed_fields = {
            'max_overlay_vertices',
            'max_overlay_segments',
            'max_overlay_ticks',
            'max_overlay_label_characters',
        }
        unknown_fields = set(data) - allowed_fields
        if len(unknown_fields) > 0:
            raise ValueError(f'Unknown overlay limit fields: {sorted(unknown_fields)!r}')
        defaults = cls()
        return cls(
            max_overlay_vertices=data.get(
                'max_overlay_vertices',
                defaults.max_overlay_vertices,
            ),
            max_overlay_segments=data.get(
                'max_overlay_segments',
                defaults.max_overlay_segments,
            ),
            max_overlay_ticks=data.get('max_overlay_ticks', defaults.max_overlay_ticks),
            max_overlay_label_characters=data.get(
                'max_overlay_label_characters',
                defaults.max_overlay_label_characters,
            ),
        )

    def to_data(self) -> dict[str, int]:
        return {
            'max_overlay_vertices': self.max_overlay_vertices,
            'max_overlay_segments': self.max_overlay_segments,
            'max_overlay_ticks': self.max_overlay_ticks,
            'max_overlay_label_characters': self.max_overlay_label_characters,
        }


class PointReference(BaseModel):
    """An immutable point whose space and units cannot be inferred implicitly."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    space: PointSpace
    x: float
    y: float
    camera: str | None = None
    unit: OverlayUnit | None = None

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_coordinate(
        cls: type['PointReference'],
        value: Any,
    ) -> float:
        if not is_finite_real(value):
            raise ValueError('Point coordinates must be finite numbers')
        return float(value)

    @model_validator(mode='before')
    @classmethod
    def normalise_point_units(
        cls: type['PointReference'],
        value: Any,
    ) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        space = data.get('space')
        unit = data.get('unit')
        if unit is None:
            unit = 'mm' if space == PointReferenceSpace.SURFACE_MM.value else 'px'
            data['unit'] = unit
        if (
            space == PointReferenceSpace.SURFACE_MM.value
            and isinstance(unit, str)
            and unit in PHYSICAL_UNITS
        ):
            multiplier = _get_unit_to_mm()[unit]
            for coordinate_name in ('x', 'y'):
                coordinate = data.get(coordinate_name)
                if is_finite_real(coordinate):
                    data[coordinate_name] = float(coordinate) * multiplier
            data['unit'] = 'mm'
        return data

    @model_validator(mode='after')
    def validate_point_is_finite(self) -> 'PointReference':
        if not is_finite_real(self.x) or not is_finite_real(self.y):
            raise ValueError('Point coordinates must remain finite after unit conversion')
        return self

    @model_validator(mode='after')
    def validate_space_identity(self) -> 'PointReference':
        if self.space == PointReferenceSpace.CAMERA_PX.value:
            if not isinstance(self.camera, str) or len(self.camera.strip()) == 0:
                raise ValueError('camera_px points require a camera identity')
            if self.unit != 'px':
                raise ValueError('camera_px points require px units')
            return self
        if self.camera is not None:
            raise ValueError('Only camera_px points may carry a camera identity')
        if self.space == PointReferenceSpace.PROJECTOR_PX.value:
            if self.unit != 'px':
                raise ValueError('projector_px points require px units')
            return self
        if self.unit != 'mm':
            raise ValueError('surface_mm points require mm, cm or in units')
        return self


class SurfaceAnchor(BaseModel):
    """An immutable point in the calibrated surface coordinate system."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    type: Literal['surface']
    x: float
    y: float
    unit: PhysicalUnit

    @model_validator(mode='before')
    @classmethod
    def normalise_units(cls: type['SurfaceAnchor'], value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        unit = data.get('unit')
        if isinstance(unit, str) and unit in PHYSICAL_UNITS and unit != 'mm':
            multiplier = _get_unit_to_mm()[unit]
            for coordinate_name in ('x', 'y'):
                coordinate = data.get(coordinate_name)
                if is_finite_real(coordinate):
                    data[coordinate_name] = float(coordinate) * multiplier
            data['unit'] = 'mm'
        return data

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_coordinate(cls: type['SurfaceAnchor'], value: Any) -> float:
        if not is_finite_real(value):
            raise ValueError('Surface anchor coordinates must be finite numbers')
        return float(value)


class ProjectorAnchor(BaseModel):
    """An immutable point in projector-native pixels."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    type: Literal['projector']
    x: float
    y: float
    unit: Literal['px']

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_coordinate(cls: type['ProjectorAnchor'], value: Any) -> float:
        if not is_finite_real(value):
            raise ValueError('Projector anchor coordinates must be finite numbers')
        return float(value)


class LocalOffset(BaseModel):
    """A finite metric offset expressed in the marker-local surface frame."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    x: float
    y: float
    unit: PhysicalUnit

    @model_validator(mode='before')
    @classmethod
    def normalise_units(cls: type['LocalOffset'], value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        unit = data.get('unit')
        if isinstance(unit, str) and unit in PHYSICAL_UNITS and unit != 'mm':
            multiplier = _get_unit_to_mm()[unit]
            for coordinate_name in ('x', 'y'):
                coordinate = data.get(coordinate_name)
                if is_finite_real(coordinate):
                    data[coordinate_name] = float(coordinate) * multiplier
            data['unit'] = 'mm'
        return data

    @field_validator('x', 'y', mode='before')
    @classmethod
    def validate_coordinate(cls: type['LocalOffset'], value: Any) -> float:
        if not is_finite_real(value):
            raise ValueError('Local offsets must be finite numbers')
        return float(value)


class FiducialAnchor(BaseModel):
    """An immutable namespace-qualified marker anchor."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    type: Literal['fiducial']
    group: str
    id: int
    local_offset: LocalOffset | None = None
    follow_rotation: bool = False

    @field_validator('group')
    @classmethod
    def validate_group(cls: type['FiducialAnchor'], value: str) -> str:
        if len(value.strip()) == 0:
            raise ValueError('Fiducial anchor groups must be non-empty')
        return value

    @field_validator('id', mode='before')
    @classmethod
    def validate_marker_id(cls: type['FiducialAnchor'], value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('Fiducial anchor ids must be non-negative integers')
        return value

    @model_validator(mode='after')
    def validate_rotation_flag(self) -> 'FiducialAnchor':
        if not isinstance(self.follow_rotation, bool):
            raise ValueError('follow_rotation must be boolean')
        return self


Anchor: TypeAlias = Annotated[
    SurfaceAnchor | ProjectorAnchor | FiducialAnchor,
    Field(discriminator='type'),
]
AnchorReference: TypeAlias = Anchor | PointReference


class Quantity(BaseModel):
    """An immutable finite quantity normalised to millimetres or pixels."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    value: float
    unit: OverlayUnit

    @model_validator(mode='before')
    @classmethod
    def normalise_physical_unit(
        cls: type['Quantity'],
        value: Any,
    ) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        unit = data.get('unit')
        quantity_value = data.get('value')
        if (
            isinstance(unit, str)
            and unit in PHYSICAL_UNITS
            and unit != 'mm'
            and is_finite_real(quantity_value)
        ):
            data['value'] = float(quantity_value) * _get_unit_to_mm()[unit]
            data['unit'] = 'mm'
        return data

    @field_validator('value', mode='before')
    @classmethod
    def validate_value(
        cls: type['Quantity'],
        value: Any,
    ) -> float:
        if not is_finite_real(value):
            raise ValueError('Quantity values must be finite numbers')
        return float(value)

    @model_validator(mode='after')
    def validate_normalised_value(self) -> 'Quantity':
        if self.unit != 'px' and not math.isfinite(
            self.value * _get_unit_to_mm()[self.unit],
        ):
            raise ValueError('Quantity must remain finite in millimetres')
        return self

    def to_mm(self) -> float:
        if self.unit == 'px':
            raise ValueError('Pixel quantities do not have a physical millimetre value')
        value_mm = self.value
        if not math.isfinite(value_mm):
            raise ValueError('Quantity is not finite in millimetres')
        return value_mm

    def validate_for_space(self, space: GeometrySpace | str) -> 'Quantity':
        space_value = space.value if isinstance(space, GeometrySpace) else space
        if space_value == GeometrySpace.PROJECTOR_PX.value and self.unit != 'px':
            raise ValueError('projector_px geometry requires px quantities')
        if space_value == GeometrySpace.SURFACE_MM.value and self.unit == 'px':
            raise ValueError('surface_mm geometry requires mm, cm or in quantities')
        if space_value not in {
            GeometrySpace.PROJECTOR_PX.value,
            GeometrySpace.SURFACE_MM.value,
        }:
            raise ValueError(f'Unsupported geometry space: {space!r}')
        return self

    def require_positive(self) -> 'Quantity':
        if self.value <= 0:
            raise ValueError('Overlay dimensions must be positive')
        return self


MIN_OVERLAY_INTENSITY = 0.0
MAX_OVERLAY_INTENSITY = 1.0


def normalise_overlay_intensity(value: Any) -> float:
    """Validate one presentation intensity in the inclusive public range."""
    try:
        if not is_finite_real(value):
            raise ValueError('overlay intensity must be finite')
        checked_value = float(value)
    except (TypeError, ValueError, OverflowError) as ex:
        raise ValueError('overlay intensity must be finite') from ex
    if not MIN_OVERLAY_INTENSITY <= checked_value <= MAX_OVERLAY_INTENSITY:
        raise ValueError('overlay intensity must be between 0.0 and 1.0')
    return checked_value


def apply_overlay_intensity_to_colour(
    colour: tuple[int, int, int],
    intensity: float,
) -> tuple[int, int, int]:
    """Apply intensity to colour only, retaining geometry and line width."""
    checked_intensity = normalise_overlay_intensity(intensity)
    return tuple(
        min(255, max(0, round(channel * checked_intensity)))
        for channel in colour
    )


class OverlayStyle(BaseModel):
    """The small immutable style vocabulary shared by generic overlays."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    colour: tuple[int, int, int] = (255, 255, 255)
    fill: bool = False
    line_width_px: int = 1
    intensity: float = 1.0

    @field_validator('colour', mode='before')
    @classmethod
    def normalise_colour(
        cls: type['OverlayStyle'],
        value: Any,
    ) -> tuple[int, int, int]:
        if not isinstance(value, str) or _COLOUR_PATTERN.fullmatch(value) is None:
            raise ValueError('colour must be a six-digit HEX value')
        return tuple(int(value[idx:idx + 2], 16) for idx in (1, 3, 5))

    @field_validator('line_width_px')
    @classmethod
    def validate_line_width(
        cls: type['OverlayStyle'],
        value: int,
    ) -> int:
        if value <= 0:
            raise ValueError('line_width_px must be positive')
        return value

    @field_validator('intensity', mode='before')
    @classmethod
    def validate_intensity(
        cls: type['OverlayStyle'],
        value: Any,
    ) -> float:
        return normalise_overlay_intensity(value)


class OverlayExtent(BaseModel):
    """A finite width and height used by an explicitly bounded grid."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    width: Quantity
    height: Quantity


def _validate_angle(value: Any) -> float:
    if not is_finite_real(value):
        raise ValueError('angle_deg must be finite')
    return float(value)


MIN_OVERLAY_LABEL_SCALE = 0.1
MAX_OVERLAY_LABEL_SCALE = 32.0


def _validate_scale(value: Any) -> float:
    if (
        not is_finite_real(value)
        or not MIN_OVERLAY_LABEL_SCALE <= float(value) <= MAX_OVERLAY_LABEL_SCALE
    ):
        raise ValueError(
            f'scale must be finite and between {MIN_OVERLAY_LABEL_SCALE} '
            f'and {MAX_OVERLAY_LABEL_SCALE}',
        )
    return float(value)


def _validate_overlay_label(value: str | None) -> str | None:
    if value is not None and len(value) > OverlayConfiguration().max_overlay_label_characters:
        raise ValueError('label exceeds the configured character limit')
    return value


def _validate_overlay_uuid(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError as ex:
            raise ValueError('Overlay id must be a valid UUID4') from ex
    return value


def _validate_overlay_name(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value.strip()) == 0 or '/' in value:
        raise ValueError('Overlay names must be non-empty and cannot contain /')
    try:
        uuid.UUID(value)
    except ValueError:
        return value
    raise ValueError('Overlay names cannot use UUID syntax')


class OverlayRequest(BaseModel):
    """Common immutable identity and visibility fields for an overlay request."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str | None = None
    visible: bool = True
    style: OverlayStyle = Field(default_factory=OverlayStyle)

    @field_validator('id', mode='before')
    @classmethod
    def validate_id(
        cls: type['OverlayRequest'],
        value: Any,
    ) -> Any:
        return _validate_overlay_uuid(value)

    @field_validator('name')
    @classmethod
    def validate_name(
        cls: type['OverlayRequest'],
        value: str | None,
    ) -> str | None:
        return _validate_overlay_name(value)


class GridRequest(OverlayRequest):
    kind: Literal['grid'] = 'grid'
    origin: AnchorReference
    geometry_space: Literal['projector_px', 'surface_mm']
    spacing: Quantity
    extent: OverlayExtent
    angle_deg: float = 0.0

    @field_validator('angle_deg', mode='before')
    @classmethod
    def validate_angle(
        cls: type['GridRequest'],
        value: Any,
    ) -> float:
        return _validate_angle(value)

    @model_validator(mode='after')
    def validate_grid(self) -> 'GridRequest':
        self.spacing.validate_for_space(self.geometry_space).require_positive()
        self.extent.width.validate_for_space(self.geometry_space).require_positive()
        self.extent.height.validate_for_space(self.geometry_space).require_positive()
        if self.style.fill:
            raise ValueError('Grid styles cannot request fill')
        return self


class ProjectorCoverageGridRequest(OverlayRequest):
    """A physical grid whose finite extent is derived from the projector output."""

    spacing: Quantity
    angle_deg: float = 0.0

    @field_validator('angle_deg', mode='before')
    @classmethod
    def validate_angle(
        cls: type['ProjectorCoverageGridRequest'],
        value: Any,
    ) -> float:
        return _validate_angle(value)

    @model_validator(mode='after')
    def validate_grid(self) -> 'ProjectorCoverageGridRequest':
        self.spacing.validate_for_space(GeometrySpace.SURFACE_MM).require_positive()
        if self.style.fill:
            raise ValueError('Grid styles cannot request fill')
        return self


class CircleRequest(OverlayRequest):
    kind: Literal['circle'] = 'circle'
    centre: AnchorReference
    geometry_space: Literal['projector_px', 'surface_mm']
    radius: Quantity

    @model_validator(mode='after')
    def validate_circle(self) -> 'CircleRequest':
        self.radius.validate_for_space(self.geometry_space).require_positive()
        return self


class RectRequest(OverlayRequest):
    kind: Literal['rect'] = 'rect'
    centre: AnchorReference
    geometry_space: Literal['projector_px', 'surface_mm']
    width: Quantity
    height: Quantity
    angle_deg: float = 0.0
    label: str | None = None
    label_angle_deg: float = 0.0
    label_scale: float = 1.0

    @field_validator('angle_deg', 'label_angle_deg', mode='before')
    @classmethod
    def validate_angle(
        cls: type['RectRequest'],
        value: Any,
    ) -> float:
        return _validate_angle(value)

    @field_validator('label')
    @classmethod
    def validate_label(
        cls: type['RectRequest'],
        value: str | None,
    ) -> str | None:
        return _validate_overlay_label(value)

    @field_validator('label_scale', mode='before')
    @classmethod
    def validate_label_scale(
        cls: type['RectRequest'],
        value: Any,
    ) -> float:
        return _validate_scale(value)

    @model_validator(mode='after')
    def validate_rect(self) -> 'RectRequest':
        self.width.validate_for_space(self.geometry_space).require_positive()
        self.height.validate_for_space(self.geometry_space).require_positive()
        return self


class TextRequest(OverlayRequest):
    kind: Literal['text'] = 'text'
    position: AnchorReference
    text: str
    angle_deg: float = 0.0
    scale: float = 1.0

    @field_validator('text')
    @classmethod
    def validate_text(
        cls: type['TextRequest'],
        value: str,
    ) -> str:
        if len(value) == 0:
            raise ValueError('text must not be empty')
        checked_text = _validate_overlay_label(value)
        assert checked_text is not None
        return checked_text

    @field_validator('angle_deg', mode='before')
    @classmethod
    def validate_angle(
        cls: type['TextRequest'],
        value: Any,
    ) -> float:
        return _validate_angle(value)

    @field_validator('scale', mode='before')
    @classmethod
    def validate_text_scale(
        cls: type['TextRequest'],
        value: Any,
    ) -> float:
        return _validate_scale(value)


class LineRequest(OverlayRequest):
    kind: Literal['line'] = 'line'
    start: AnchorReference
    end: AnchorReference
    label: str | None = None

    @field_validator('label')
    @classmethod
    def validate_label(
        cls: type['LineRequest'],
        value: str | None,
    ) -> str | None:
        return _validate_overlay_label(value)

    @model_validator(mode='after')
    def validate_line(self) -> 'LineRequest':
        if self.style.fill:
            raise ValueError('Line styles cannot request fill')
        return self


class RulerRequest(OverlayRequest):
    kind: Literal['ruler'] = 'ruler'
    start: AnchorReference
    end: AnchorReference
    measurement_space: Literal['projector_px', 'surface_mm']
    unit: OverlayUnit
    label: str | None = None

    @field_validator('label')
    @classmethod
    def validate_label(
        cls: type['RulerRequest'],
        value: str | None,
    ) -> str | None:
        return _validate_overlay_label(value)

    @model_validator(mode='after')
    def validate_ruler(self) -> 'RulerRequest':
        if self.measurement_space == GeometrySpace.PROJECTOR_PX.value:
            if self.unit != 'px':
                raise ValueError('projector_px rulers require px units')
        elif self.unit == 'px':
            raise ValueError('surface_mm rulers require mm, cm or in units')
        if self.style.fill:
            raise ValueError('Ruler styles cannot request fill')
        return self


MAX_ARROW_HEAD_DIMENSION = 10_000.0


class ArrowRequest(OverlayRequest):
    """A declarative arrow whose endpoints may resolve independently."""

    kind: Literal['arrow'] = 'arrow'
    start: AnchorReference
    end: AnchorReference
    geometry_space: Literal['projector_px', 'surface_mm']
    head_length: Quantity
    head_width: Quantity
    label: str | None = None

    @field_validator('label')
    @classmethod
    def validate_label(cls: type['ArrowRequest'], value: str | None) -> str | None:
        return _validate_overlay_label(value)

    @model_validator(mode='after')
    def validate_arrow(self) -> 'ArrowRequest':
        self.head_length.validate_for_space(self.geometry_space).require_positive()
        self.head_width.validate_for_space(self.geometry_space).require_positive()
        for quantity, field_name in (
            (self.head_length, 'head_length'),
            (self.head_width, 'head_width'),
        ):
            if quantity.value > MAX_ARROW_HEAD_DIMENSION:
                raise ValueError(f'{field_name} exceeds the bounded arrow dimension')
        return self


AnyOverlayRequest: TypeAlias = (
    GridRequest
    | CircleRequest
    | RectRequest
    | TextRequest
    | LineRequest
    | RulerRequest
    | ArrowRequest
)
_OVERLAY_REQUEST_TYPES = (
    GridRequest,
    CircleRequest,
    RectRequest,
    TextRequest,
    LineRequest,
    RulerRequest,
    ArrowRequest,
)


class SourceSegment(NamedTuple):
    """One finite segment in a declared source coordinate space."""

    start: Point2D
    end: Point2D


class CircleGeometry(NamedTuple):
    """A circle retained in its requested geometry space until materialisation."""

    centre: Point2D
    radius: float
    geometry_space: str
    style: OverlayStyle


class RectGeometry(NamedTuple):
    """The four source-space corners of a rotated rectangle."""

    centre: Point2D
    corners: tuple[Point2D, ...]
    width: float
    height: float
    angle_deg: float
    geometry_space: str
    style: OverlayStyle


class TextGeometry(NamedTuple):
    """A floating projector-native text anchor."""

    position: Point2D
    text: str
    angle_deg: float
    scale: float
    style: OverlayStyle


class GridGeometry(NamedTuple):
    """Finite square-grid segments in the requested source space."""

    origin: Point2D
    spacing: float
    width: float
    height: float
    angle_deg: float
    geometry_space: str
    vertical_segments: tuple[SourceSegment, ...]
    horizontal_segments: tuple[SourceSegment, ...]
    style: OverlayStyle

    @property
    def segments(self) -> tuple[SourceSegment, ...]:
        return self.vertical_segments + self.horizontal_segments


class LineGeometry(NamedTuple):
    """A literal projector-native segment."""

    start: Point2D
    end: Point2D
    label: str | None
    style: OverlayStyle


class RulerGeometry(NamedTuple):
    """A finite ruler line and distance in its requested measurement space."""

    start: Point2D
    end: Point2D
    measurement_space: str
    unit: str
    length: float
    length_mm: float | None
    label: str | None
    style: OverlayStyle


class ArrowGeometry(NamedTuple):
    """An arrow retained in its declared source coordinate space."""

    start: Point2D
    end: Point2D
    head: tuple[Point2D, Point2D, Point2D]
    head_length: float
    head_width: float
    geometry_space: str
    label: str | None
    style: OverlayStyle


class ProjectorSegment(NamedTuple):
    """One raster-safe projector-native line segment."""

    start: Point2D
    end: Point2D
    style: OverlayStyle


class ProjectorPolygon(NamedTuple):
    """One raster-safe, projector-native filled polygon."""

    points: tuple[Point2D, ...]
    style: OverlayStyle


class ProjectorLabel(NamedTuple):
    """One decorative label positioned in projector-native coordinates."""

    position: Point2D
    text: str
    style: OverlayStyle
    angle_deg: float = 0.0
    scale: float = 1.0


class ProjectorMaterialisation(NamedTuple):
    """Immutable projector primitives produced from one overlay request."""

    segments: tuple[ProjectorSegment, ...] = ()
    polygons: tuple[ProjectorPolygon, ...] = ()
    labels: tuple[ProjectorLabel, ...] = ()


def materialise_presentation(
    materialisation: ProjectorMaterialisation,
    global_intensity: float = 1.0,
    protected_regions: Sequence[Sequence[PointLike]] = (),
) -> ProjectorMaterialisation:
    """Return a presentation-only copy with intensity and protection applied.

    The supplied projector primitives are immutable source geometry.  Protected
    regions are applied to the copy, so registry requests and resolved geometry
    remain suitable for later frames and marker recovery.
    """
    if not isinstance(materialisation, ProjectorMaterialisation):
        raise ValueError('materialisation must be ProjectorMaterialisation')
    checked_global_intensity = normalise_overlay_intensity(global_intensity)
    checked_regions = _normalise_protected_regions(protected_regions)
    segments: list[ProjectorSegment] = []
    for segment in materialisation.segments:
        presentation_style = _presentation_style(segment.style, checked_global_intensity)
        segments.extend(
            _subtract_segment_from_protected_regions(
                segment.start,
                segment.end,
                presentation_style,
                checked_regions,
            ),
        )
    polygons: list[ProjectorPolygon] = []
    for polygon in materialisation.polygons:
        if _polygon_intersects_protected_regions(polygon.points, checked_regions):
            continue
        polygons.append(
            ProjectorPolygon(
                polygon.points,
                _presentation_style(polygon.style, checked_global_intensity),
            ),
        )
    labels = tuple(
        ProjectorLabel(
            label.position,
            label.text,
            _presentation_style(label.style, checked_global_intensity),
            label.angle_deg,
            label.scale,
        )
        for label in materialisation.labels
        if not _point_in_protected_regions(label.position, checked_regions)
    )
    return ProjectorMaterialisation(tuple(segments), tuple(polygons), labels)


# Longer aliases make the presentation boundary discoverable without adding a
# second implementation authority.
materialise_overlay_presentation = materialise_presentation
apply_overlay_presentation = materialise_presentation


def _presentation_style(style: OverlayStyle, global_intensity: float) -> OverlayStyle:
    if not isinstance(style, OverlayStyle):
        raise ValueError('projector primitive styles must be OverlayStyle values')
    effective_intensity = normalise_overlay_intensity(global_intensity * style.intensity)
    return style.model_copy(
        update={
            'colour': apply_overlay_intensity_to_colour(style.colour, effective_intensity),
            'intensity': 1.0,
        },
    )


def _normalise_protected_regions(
    protected_regions: Sequence[Sequence[PointLike]],
) -> tuple[tuple[Point2D, ...], ...]:
    if isinstance(protected_regions, Mapping):
        protected_regions = tuple(protected_regions.values())
    if isinstance(protected_regions, (str, bytes, bytearray)):
        raise ValueError('protected_regions must be an ordered collection')
    try:
        region_values = tuple(protected_regions)
    except TypeError as ex:
        raise ValueError('protected_regions must be an ordered collection') from ex

    regions: list[tuple[Point2D, ...]] = []
    for region in region_values:
        if isinstance(region, CoordinateBounds):
            points = (
                Point2D(region.left, region.top),
                Point2D(region.right, region.top),
                Point2D(region.right, region.bottom),
                Point2D(region.left, region.bottom),
            )
        else:
            if isinstance(region, (Mapping, Set, str, bytes, bytearray)):
                raise ValueError('protected regions must contain polygons')
            try:
                points = tuple(
                    Point2D(float(point[0]), float(point[1]))
                    for point in region
                )
            except (IndexError, TypeError, ValueError, OverflowError) as ex:
                raise ValueError('protected regions must contain polygons') from ex
        if len(points) < 3 or any(
            not is_finite_real(value)
            for point in points
            for value in point
        ):
            raise ValueError('protected regions must contain finite polygons')
        if calculate_polygon_area(points) <= 0:
            raise ValueError('protected regions must contain non-degenerate polygons')
        regions.append(points)
    return tuple(regions)


def _subtract_segment_from_protected_regions(
    start: Point2D,
    end: Point2D,
    style: OverlayStyle,
    protected_regions: Sequence[Sequence[Point2D]],
) -> tuple[ProjectorSegment, ...]:
    difference = Point2D(end.x - start.x, end.y - start.y)
    if difference == Point2D(0.0, 0.0):
        return ()
    parameters = [0.0, 1.0]
    for region in protected_regions:
        for idx, region_start in enumerate(region):
            region_end = region[(idx + 1) % len(region)]
            intersection = _segment_intersection_parameter(
                start,
                difference,
                region_start,
                Point2D(region_end.x - region_start.x, region_end.y - region_start.y),
            )
            if intersection is not None:
                parameters.extend(intersection)
    ordered_parameters = _unique_sorted_parameters(parameters)
    output: list[ProjectorSegment] = []
    for idx in range(len(ordered_parameters) - 1):
        start_parameter = ordered_parameters[idx]
        end_parameter = ordered_parameters[idx + 1]
        if end_parameter - start_parameter <= 1e-12:
            continue
        midpoint_parameter = (start_parameter + end_parameter) / 2.0
        midpoint = Point2D(
            start.x + difference.x * midpoint_parameter,
            start.y + difference.y * midpoint_parameter,
        )
        if _point_in_protected_regions(midpoint, protected_regions):
            continue
        output.append(
            ProjectorSegment(
                Point2D(start.x + difference.x * start_parameter, start.y + difference.y * start_parameter),
                Point2D(start.x + difference.x * end_parameter, start.y + difference.y * end_parameter),
                style,
            ),
        )
    return tuple(output)


def _segment_intersection_parameter(
    start: Point2D,
    difference: Point2D,
    edge_start: Point2D,
    edge_difference: Point2D,
) -> tuple[float, ...] | None:
    denominator = difference.x * edge_difference.y - difference.y * edge_difference.x
    offset = Point2D(edge_start.x - start.x, edge_start.y - start.y)
    if abs(denominator) <= 1e-12:
        if abs(offset.x * difference.y - offset.y * difference.x) > 1e-9:
            return None
        length_squared = difference.x * difference.x + difference.y * difference.y
        if length_squared == 0:
            return None
        first = (offset.x * difference.x + offset.y * difference.y) / length_squared
        second_offset = Point2D(edge_start.x + edge_difference.x - start.x, edge_start.y + edge_difference.y - start.y)
        second = (second_offset.x * difference.x + second_offset.y * difference.y) / length_squared
        return tuple(value for value in (first, second) if 0.0 <= value <= 1.0)
    segment_parameter = (
        offset.x * edge_difference.y - offset.y * edge_difference.x
    ) / denominator
    edge_parameter = (
        offset.x * difference.y - offset.y * difference.x
    ) / denominator
    if 0.0 <= segment_parameter <= 1.0 and 0.0 <= edge_parameter <= 1.0:
        return (segment_parameter,)
    return None


def _unique_sorted_parameters(parameters: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(min(1.0, max(0.0, value)) for value in parameters)
    unique: list[float] = []
    for value in ordered:
        if len(unique) == 0 or abs(value - unique[-1]) > 1e-9:
            unique.append(value)
    return tuple(unique)


def _point_in_protected_regions(
    point: Point2D,
    protected_regions: Sequence[Sequence[Point2D]],
) -> bool:
    return any(is_point_in_region(point, region) for region in protected_regions)


def is_polygon_intersecting_protected_regions(
    polygon: Sequence[PointLike],
    protected_regions: Sequence[Sequence[PointLike]],
) -> bool:
    """Return whether a polygon would draw over a protected projector region."""
    checked_polygon = tuple(coerce_point(point) for point in polygon)
    checked_regions = _normalise_protected_regions(protected_regions)
    if len(checked_polygon) < 3:
        raise ValueError('polygon must contain at least three finite points')
    return _polygon_intersects_protected_regions(checked_polygon, checked_regions)


def is_circle_intersecting_protected_regions(
    centre: PointLike,
    radius: float,
    protected_regions: Sequence[Sequence[PointLike]],
) -> bool:
    """Return whether a filled projector circle would touch protected output."""
    checked_centre = coerce_point(centre)
    try:
        checked_radius = float(radius)
    except (TypeError, ValueError, OverflowError) as ex:
        raise ValueError('circle radius must be finite and non-negative') from ex
    if not math.isfinite(checked_radius) or checked_radius < 0:
        raise ValueError('circle radius must be finite and non-negative')
    checked_regions = _normalise_protected_regions(protected_regions)
    for region in checked_regions:
        if is_point_in_region(checked_centre, region):
            return True
        if any(
            math.hypot(
                point.x - checked_centre.x,
                point.y - checked_centre.y,
            ) <= checked_radius
            for point in region
        ):
            return True
        for idx, segment_start in enumerate(region):
            segment_end = region[(idx + 1) % len(region)]
            if _distance_to_segment(
                checked_centre,
                segment_start,
                segment_end,
            ) <= checked_radius:
                return True
    return False


def _distance_to_segment(
    point: Point2D,
    segment_start: Point2D,
    segment_end: Point2D,
) -> float:
    difference_x = segment_end.x - segment_start.x
    difference_y = segment_end.y - segment_start.y
    length_squared = difference_x * difference_x + difference_y * difference_y
    if length_squared == 0:
        return math.hypot(
            point.x - segment_start.x,
            point.y - segment_start.y,
        )
    fraction = (
        (point.x - segment_start.x) * difference_x
        + (point.y - segment_start.y) * difference_y
    ) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    nearest_point = Point2D(
        segment_start.x + fraction * difference_x,
        segment_start.y + fraction * difference_y,
    )
    return math.hypot(
        point.x - nearest_point.x,
        point.y - nearest_point.y,
    )


def _polygon_intersects_protected_regions(
    polygon: Sequence[Point2D],
    protected_regions: Sequence[Sequence[Point2D]],
) -> bool:
    for region in protected_regions:
        if any(is_point_in_region(point, region) for point in polygon):
            return True
        if any(is_point_in_region(point, polygon) for point in region):
            return True
        for idx, point in enumerate(polygon):
            next_point = polygon[(idx + 1) % len(polygon)]
            for region_idx, region_point in enumerate(region):
                region_next = region[(region_idx + 1) % len(region)]
                if _segment_intersection_parameter(
                    point,
                    Point2D(next_point.x - point.x, next_point.y - point.y),
                    region_point,
                    Point2D(region_next.x - region_point.x, region_next.y - region_point.y),
                ) is not None:
                    return True
    return False


class OverlayEntry(NamedTuple):
    """One immutable declarative overlay and its last resolved presentation."""

    id: uuid.UUID
    name: str | None
    kind: str
    visible: bool
    request: AnyOverlayRequest
    materialised_primitives: ProjectorMaterialisation
    camera_dependencies: tuple[str, ...]
    metric_dependency: bool
    projector_output_descriptor: ProjectorOutputDescriptor
    insertion_sequence: int
    is_dynamic: bool = False
    is_resolved: bool = True
    unresolved_reason: str | None = None
    fiducial_dependencies: tuple[tuple[str, int], ...] = ()

    @property
    def specification(self) -> AnyOverlayRequest:
        """Return the immutable request retained by the registry."""
        return self.request

    @property
    def resolved(self) -> bool:
        return self.is_resolved

    @property
    def unresolved(self) -> bool:
        return not self.is_resolved

    @property
    def dynamic(self) -> bool:
        return self.is_dynamic


class OverlayRegistry:
    """Publish complete overlay candidates with a single atomic snapshot swap."""

    def __init__(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
        *,
        id_factory: Callable[[], uuid.UUID | str] | None = None,
        max_operations: int = 100,
        max_batch_operations: int | None = None,
    ) -> None:
        _validate_projector_output_descriptor(projector_output_descriptor)
        if id_factory is not None and not callable(id_factory):
            raise ValueError('id_factory must be callable')
        if max_batch_operations is not None:
            max_operations = max_batch_operations
        _validate_batch_limit(max_operations)
        self._max_operations = max_operations
        self._projector_output_descriptor = projector_output_descriptor
        self._id_factory = id_factory
        self._entries: dict[uuid.UUID, OverlayEntry] = {}
        self._names: dict[str, uuid.UUID] = {}
        self._next_insertion_sequence = 0
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def snapshot(self) -> tuple[OverlayEntry, ...]:
        """Return one immutable registry view without exposing mutable indexes."""
        with self._lock:
            return tuple(self._entries.values())

    def create(
        self,
        request: AnyOverlayRequest,
        materialised_primitives: ProjectorMaterialisation,
    ) -> OverlayEntry:
        """Compatibility wrapper for one atomic create operation."""
        if not isinstance(materialised_primitives, ProjectorMaterialisation):
            raise ValueError('materialised_primitives must be ProjectorMaterialisation')
        entries = self.apply_batch(
            ({'op': 'create', 'request': request},),
            materialise_request=lambda _request: materialised_primitives,
        )
        return entries[-1]

    def create_specification(
        self,
        request: AnyOverlayRequest,
        materialise_request: Callable[[AnyOverlayRequest], ProjectorMaterialisation | None],
    ) -> OverlayEntry:
        """Create a request while allowing an unavailable dynamic authority."""
        entries = self.apply_batch(
            ({'op': 'create', 'request': request},),
            materialise_request=materialise_request,
        )
        return entries[-1]

    def update(
        self,
        selector: str | uuid.UUID,
        request: AnyOverlayRequest,
        materialise_request: Callable[[AnyOverlayRequest], ProjectorMaterialisation | None]
        | None = None,
    ) -> OverlayEntry:
        """Replace one complete request while preserving its identity and order."""
        entries = self.apply_batch(
            ({'op': 'update', 'selector': selector, 'request': request},),
            materialise_request=materialise_request,
        )
        return next(entry for entry in entries if entry.id == request.id)

    def apply_batch(
        self,
        operations: Sequence[object],
        *,
        materialise_request: Callable[[AnyOverlayRequest], ProjectorMaterialisation | None]
        | None = None,
        max_operations: int | None = None,
    ) -> list[OverlayEntry]:
        """Validate and publish an ordered batch, or publish nothing at all.

        Candidate construction deliberately happens outside ``_lock``.  The final
        generation check makes a candidate based on an older registry impossible
        to publish after a concurrent mutation.
        """
        if not isinstance(operations, Sequence) or isinstance(
            operations,
            (str, bytes, bytearray),
        ):
            raise ValueError('operations must be an ordered sequence')
        operation_values = tuple(operations)
        if max_operations is not None:
            _validate_batch_limit(max_operations)
        operation_limit = (
            self._max_operations
            if max_operations is None
            else min(self._max_operations, max_operations)
        )
        if len(operation_values) > operation_limit:
            raise ValueError('overlay batch exceeds the configured operation limit')
        if len(operation_values) == 0:
            raise ValueError('overlay batch must contain at least one operation')

        with self._lock:
            starting_generation = self._generation
            starting_entries = tuple(self._entries.values())
            starting_next_sequence = self._next_insertion_sequence
            descriptor = self._projector_output_descriptor
        candidate_entries, next_sequence = _apply_overlay_operations(
            starting_entries,
            starting_next_sequence,
            descriptor,
            operation_values,
            materialise_request,
            self._id_factory,
        )
        candidate_by_id = {entry.id: entry for entry in candidate_entries}
        candidate_names = {
            entry.name: entry.id
            for entry in candidate_entries
            if entry.name is not None
        }
        with self._lock:
            if self._generation != starting_generation:
                raise RuntimeError('Overlay registry changed while building the batch')
            self._entries = candidate_by_id
            self._names = candidate_names
            self._next_insertion_sequence = next_sequence
            self._generation += 1
            return list(candidate_entries)

    # These names keep the service boundary readable and make the transaction
    # seam convenient for callers that describe the operation as a mutation.
    apply_operations = apply_batch
    batch = apply_batch

    def publish_if_generation(
        self,
        generation: int,
        publish: Callable[[], None],
    ) -> bool:
        """Run one publication callback while the registry generation is stable."""
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise ValueError('generation must be an integer')
        if not callable(publish):
            raise ValueError('publish must be callable')
        with self._lock:
            if self._generation != generation:
                return False
            publish()
            return True

    def resolve(
        self,
        spatial_state_or_materialise_request: object,
        materialise_request: Callable[
            [AnyOverlayRequest, object],
            ProjectorMaterialisation | None,
        ]
        | None = None,
    ) -> tuple[OverlayEntry, ...]:
        """Resolve one registry snapshot without publishing its materialisation.

        The one-argument form is retained for the existing service seam.  The
        two-argument form makes the spatial snapshot explicit for callers that
        own resolution, while keeping all work outside the registry lock.
        """
        if materialise_request is None:
            if not callable(spatial_state_or_materialise_request):
                raise ValueError('resolve requires a materialisation callback')
            materialise = spatial_state_or_materialise_request
        else:
            if not callable(materialise_request):
                raise ValueError('materialise_request must be callable')
            spatial_state = spatial_state_or_materialise_request

            def materialise(request: AnyOverlayRequest) -> ProjectorMaterialisation | None:
                return materialise_request(request, spatial_state)

        entries = self.snapshot()
        resolved_entries: list[OverlayEntry] = []
        for entry in entries:
            if not entry.is_dynamic:
                resolved_entries.append(entry)
                continue
            materialisation = materialise(entry.request)
            if materialisation is None:
                resolved_entries.append(
                    entry._replace(
                        materialised_primitives=ProjectorMaterialisation(),
                        is_resolved=False,
                        unresolved_reason='spatial authority unavailable',
                    ),
                )
                continue
            _validate_projector_materialisation(materialisation)
            resolved_entries.append(
                entry._replace(
                    materialised_primitives=materialisation,
                    is_resolved=True,
                    unresolved_reason=None,
                ),
            )
        return tuple(resolved_entries)

    def list(self) -> list[OverlayEntry]:
        return list(self.snapshot())

    def get(self, selector: str | uuid.UUID) -> OverlayEntry:
        with self._lock:
            return self._entries[self._resolve_selector(selector)]

    def show(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_visibility(selector, True)

    def hide(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_visibility(selector, False)

    def remove(self, selector: str | uuid.UUID) -> OverlayEntry:
        with self._lock:
            removed_entry = self._entries[self._resolve_selector(selector)]
        self.apply_batch(({'op': 'remove', 'selector': selector},))
        return removed_entry

    def clear(self) -> None:
        with self._lock:
            if len(self._entries) == 0:
                return
            self._entries = {}
            self._names = {}
            self._generation += 1

    def invalidate_camera(self, camera_id: str) -> None:
        if not isinstance(camera_id, str) or len(camera_id) == 0:
            raise ValueError('camera_id must be a non-empty string')
        self._invalidate_matching(lambda entry: camera_id in entry.camera_dependencies)

    def invalidate_metric(self) -> None:
        self._invalidate_matching(lambda entry: entry.metric_dependency)

    def invalidate_projector_output(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
    ) -> None:
        _validate_projector_output_descriptor(projector_output_descriptor)
        with self._lock:
            if projector_output_descriptor == self._projector_output_descriptor:
                return
            self._projector_output_descriptor = projector_output_descriptor
            updated_entries = tuple(
                entry._replace(
                    materialised_primitives=ProjectorMaterialisation(),
                    projector_output_descriptor=projector_output_descriptor,
                    is_resolved=False,
                    unresolved_reason='projector output changed',
                )
                if entry.is_dynamic
                else entry._replace(
                    projector_output_descriptor=projector_output_descriptor,
                )
                for entry in self._entries.values()
                if entry.is_dynamic
                or (
                    len(entry.camera_dependencies) == 0
                    and not entry.metric_dependency
                )
            )
            self._entries = {entry.id: entry for entry in updated_entries}
            self._names = {
                entry.name: entry.id
                for entry in updated_entries
                if entry.name is not None
            }
            self._generation += 1

    def _set_visibility(
        self,
        selector: str | uuid.UUID,
        visible: bool,
    ) -> OverlayEntry:
        with self._lock:
            overlay_id = self._resolve_selector(selector)
            entry = self._entries[overlay_id]
        if entry.visible == visible:
            return entry
        updated_request = entry.request.model_copy(update={'visible': visible})
        updated_entries = self.apply_batch(
            ({'op': 'update', 'selector': overlay_id, 'request': updated_request},),
            materialise_request=(
                lambda _request: (
                    entry.materialised_primitives
                    if entry.is_resolved
                    else None
                )
            ),
        )
        return next(item for item in updated_entries if item.id == overlay_id)

    def _make_overlay_id(self, request_id: uuid.UUID) -> uuid.UUID:
        overlay_id = request_id if self._id_factory is None else self._id_factory()
        if isinstance(overlay_id, str):
            try:
                overlay_id = uuid.UUID(overlay_id)
            except ValueError as ex:
                raise ValueError('id_factory must return a UUID4') from ex
        if not isinstance(overlay_id, uuid.UUID) or overlay_id.version != 4:
            raise ValueError('Overlay ids must be UUID4 values')
        return overlay_id

    def _resolve_selector(self, selector: str | uuid.UUID) -> uuid.UUID:
        if isinstance(selector, uuid.UUID):
            overlay_id = selector
        elif isinstance(selector, str):
            try:
                overlay_id = uuid.UUID(selector)
            except ValueError:
                overlay_id = self._names.get(selector)
                if overlay_id is None:
                    raise OverlayNotFoundError(f'Unknown overlay {selector!r}') from None
        else:
            raise ValueError('overlay selector must be an id or name')
        if overlay_id not in self._entries:
            raise OverlayNotFoundError(f'Unknown overlay {selector!r}')
        return overlay_id

    def _invalidate_matching(self, predicate: Callable[[OverlayEntry], bool]) -> None:
        while True:
            with self._lock:
                starting_generation = self._generation
                entries = tuple(self._entries.values())
            retained_entries = tuple(
                entry._replace(
                    materialised_primitives=ProjectorMaterialisation(),
                    is_resolved=False,
                    unresolved_reason='spatial authority unavailable',
                )
                if predicate(entry) and entry.is_dynamic
                else entry
                for entry in entries
                if not predicate(entry) or entry.is_dynamic
            )
            if retained_entries == entries:
                return
            with self._lock:
                if (
                    self._generation != starting_generation
                    or tuple(self._entries.values()) != entries
                ):
                    continue
                self._entries = {entry.id: entry for entry in retained_entries}
                self._names = {
                    entry.name: entry.id
                    for entry in retained_entries
                    if entry.name is not None
                }
                self._generation += 1
                return


def _validate_batch_limit(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError('max_operations must be a positive integer')


def _apply_overlay_operations(
    starting_entries: tuple[OverlayEntry, ...],
    next_sequence: int,
    descriptor: ProjectorOutputDescriptor,
    operations: tuple[object, ...],
    materialise_request: Callable[[AnyOverlayRequest], ProjectorMaterialisation | None]
    | None,
    id_factory: Callable[[], uuid.UUID | str] | None,
) -> tuple[list[OverlayEntry], int]:
    candidate = list(starting_entries)
    for operation in operations:
        operation_name, selector, request = _normalise_overlay_operation(operation)
        if operation_name == 'remove':
            entry_index = _find_candidate_index(candidate, selector)
            candidate.pop(entry_index)
            continue
        if not isinstance(request, _OVERLAY_REQUEST_TYPES):
            raise ValueError('request must be an overlay request')
        if operation_name == 'create':
            overlay_id = request.id if id_factory is None else _make_batch_overlay_id(id_factory)
            if overlay_id in {entry.id for entry in candidate}:
                raise ValueError(f'Overlay id {overlay_id} is already in use')
            if any(entry.name == request.name and request.name is not None for entry in candidate):
                raise ValueError(f'Overlay name {request.name!r} is already in use')
            if overlay_id != request.id:
                request = request.model_copy(update={'id': overlay_id})
            materialisation = _materialise_candidate_request(request, materialise_request)
            candidate.append(
                _make_overlay_entry(
                    request,
                    materialisation,
                    descriptor,
                    next_sequence,
                ),
            )
            next_sequence += 1
            continue
        if operation_name != 'update':
            raise ValueError(f'Unknown overlay operation {operation_name!r}')
        entry_index = _find_candidate_index(candidate, selector)
        existing = candidate[entry_index]
        if request.id != existing.id:
            raise ValueError('Overlay update request id must match its selector')
        if any(
            idx != entry_index and entry.name == request.name and request.name is not None
            for idx, entry in enumerate(candidate)
        ):
            raise ValueError(f'Overlay name {request.name!r} is already in use')
        if request == existing.request and not existing.is_dynamic:
            materialisation = existing.materialised_primitives
        elif request == existing.request and materialise_request is None:
            materialisation = (
                existing.materialised_primitives
                if existing.is_resolved
                else None
            )
        else:
            materialisation = _materialise_candidate_request(
                request,
                materialise_request,
            )
        candidate[entry_index] = _make_overlay_entry(
            request,
            materialisation,
            descriptor,
            existing.insertion_sequence,
        )
    return candidate, next_sequence


def _normalise_overlay_operation(
    operation: object,
) -> tuple[str, str | uuid.UUID | None, AnyOverlayRequest | None]:
    if isinstance(operation, Mapping):
        operation_name = operation.get(
            'op',
            operation.get('operation', operation.get('action', operation.get('type'))),
        )
        selector = operation.get(
            'selector',
            operation.get('target', operation.get('overlay_id', operation.get('id'))),
        )
        request = operation.get(
            'request',
            operation.get('specification', operation.get('spec')),
        )
    else:
        operation_name = getattr(
            operation,
            'op',
            getattr(operation, 'operation', getattr(operation, 'action', None)),
        )
        selector = getattr(
            operation,
            'selector',
            getattr(operation, 'target', getattr(operation, 'overlay_id', None)),
        )
        request = getattr(
            operation,
            'request',
            getattr(operation, 'specification', getattr(operation, 'spec', None)),
        )
    if not isinstance(operation_name, str):
        raise ValueError('overlay operation must specify create, update or remove')
    operation_name = operation_name.lower()
    if operation_name not in {'create', 'update', 'remove'}:
        raise ValueError(f'Unknown overlay operation {operation_name!r}')
    if operation_name == 'remove':
        if not isinstance(selector, (str, uuid.UUID)):
            raise ValueError('remove operations require an overlay selector')
        return operation_name, selector, None
    return operation_name, selector, request


def _find_candidate_index(
    candidate: Sequence[OverlayEntry],
    selector: str | uuid.UUID | None,
) -> int:
    if not isinstance(selector, (str, uuid.UUID)):
        raise ValueError('overlay selector must be an id or name')
    if isinstance(selector, uuid.UUID):
        for idx, entry in enumerate(candidate):
            if entry.id == selector:
                return idx
    else:
        try:
            overlay_id = uuid.UUID(selector)
        except ValueError:
            overlay_id = None
        for idx, entry in enumerate(candidate):
            if (overlay_id is not None and entry.id == overlay_id) or entry.name == selector:
                return idx
    raise OverlayNotFoundError(f'Unknown overlay {selector!r}')


def _make_batch_overlay_id(factory: Callable[[], uuid.UUID | str]) -> uuid.UUID:
    value = factory()
    try:
        checked_value = uuid.UUID(value) if isinstance(value, str) else value
    except ValueError as ex:
        raise ValueError('id_factory must return a UUID4') from ex
    if not isinstance(checked_value, uuid.UUID) or checked_value.version != 4:
        raise ValueError('Overlay ids must be UUID4 values')
    return checked_value


def _materialise_candidate_request(
    request: AnyOverlayRequest,
    materialise_request: Callable[[AnyOverlayRequest], ProjectorMaterialisation | None] | None,
) -> ProjectorMaterialisation | None:
    if materialise_request is None:
        raise ValueError('materialise_request is required for a batch')
    materialisation = materialise_request(request)
    if materialisation is not None:
        _validate_projector_materialisation(materialisation)
    elif not is_dynamic_overlay_request(request):
        raise ValueError('static overlay materialisation is unavailable')
    return materialisation


def _make_overlay_entry(
    request: AnyOverlayRequest,
    materialisation: ProjectorMaterialisation | None,
    descriptor: ProjectorOutputDescriptor,
    insertion_sequence: int,
) -> OverlayEntry:
    camera_dependencies, metric_dependency = get_overlay_dependencies(request)
    fiducial_dependencies = get_overlay_fiducial_dependencies(request)
    dynamic = is_dynamic_overlay_request(request)
    return OverlayEntry(
        request.id,
        request.name,
        request.kind,
        request.visible,
        request,
        materialisation or ProjectorMaterialisation(),
        camera_dependencies,
        metric_dependency,
        descriptor,
        insertion_sequence,
        dynamic,
        materialisation is not None,
        None if materialisation is not None else 'spatial authority unavailable',
        fiducial_dependencies,
    )


def _validate_projector_materialisation(
    materialisation: ProjectorMaterialisation,
) -> None:
    for primitives in (
        materialisation.segments,
        materialisation.polygons,
        materialisation.labels,
    ):
        if not isinstance(primitives, tuple):
            raise ValueError('materialised primitives must be immutable tuples')

    for segment in materialisation.segments:
        if not isinstance(segment, ProjectorSegment):
            raise ValueError('materialised segments must be ProjectorSegment values')
        _validate_finite_geometry_point(segment.start)
        _validate_finite_geometry_point(segment.end)
        if not isinstance(segment.style, OverlayStyle):
            raise ValueError('materialised segment styles must be OverlayStyle values')

    for polygon in materialisation.polygons:
        if not isinstance(polygon, ProjectorPolygon):
            raise ValueError('materialised polygons must be ProjectorPolygon values')
        if not isinstance(polygon.points, tuple) or len(polygon.points) < 3:
            raise ValueError('materialised polygon points must be immutable tuples')
        for point in polygon.points:
            _validate_finite_geometry_point(point)
        if not isinstance(polygon.style, OverlayStyle):
            raise ValueError('materialised polygon styles must be OverlayStyle values')

    for label in materialisation.labels:
        if not isinstance(label, ProjectorLabel):
            raise ValueError('materialised labels must be ProjectorLabel values')
        _validate_finite_geometry_point(label.position)
        if not isinstance(label.text, str):
            raise ValueError('materialised label text must be a string')
        if (
            not is_finite_real(label.angle_deg)
            or not is_finite_real(label.scale)
            or not MIN_OVERLAY_LABEL_SCALE <= label.scale <= MAX_OVERLAY_LABEL_SCALE
        ):
            raise ValueError('materialised label rotation and scale are invalid')
        if not isinstance(label.style, OverlayStyle):
            raise ValueError('materialised label styles must be OverlayStyle values')


def get_overlay_point_references(
    request: AnyOverlayRequest,
) -> tuple[AnchorReference, ...]:
    if isinstance(request, (GridRequest,)):
        return (request.origin,)
    if isinstance(request, (CircleRequest, RectRequest)):
        return (request.centre,)
    if isinstance(request, TextRequest):
        return (request.position,)
    if isinstance(request, (LineRequest, RulerRequest, ArrowRequest)):
        return (request.start, request.end)
    raise ValueError('request must be an overlay request')


def get_overlay_fiducial_dependencies(
    request: AnyOverlayRequest,
) -> tuple[tuple[str, int], ...]:
    """Return marker identities referenced by a declarative request."""
    dependencies: list[tuple[str, int]] = []
    for reference in get_overlay_point_references(request):
        if isinstance(reference, FiducialAnchor):
            identity = (reference.group, reference.id)
            if identity not in dependencies:
                dependencies.append(identity)
    return tuple(dependencies)


def is_dynamic_overlay_request(request: AnyOverlayRequest) -> bool:
    """Return whether resolving this request depends on changing spatial state."""
    return len(get_overlay_fiducial_dependencies(request)) > 0


def get_overlay_dependencies(
    request: AnyOverlayRequest,
) -> tuple[tuple[str, ...], bool]:
    point_references = get_overlay_point_references(request)
    metric_dependency = False
    camera_dependencies: list[str] = []
    for point_reference in point_references:
        if isinstance(point_reference, PointReference):
            metric_dependency = metric_dependency or (
                point_reference.space == PointReferenceSpace.SURFACE_MM.value
            )
            camera_id = point_reference.camera
            if (
                point_reference.space == PointReferenceSpace.CAMERA_PX.value
                and camera_id is not None
                and camera_id not in camera_dependencies
            ):
                camera_dependencies.append(camera_id)
            continue
        if isinstance(point_reference, SurfaceAnchor):
            metric_dependency = True
            continue
        if isinstance(point_reference, FiducialAnchor):
            metric_dependency = metric_dependency or (
                point_reference.local_offset is not None
                or point_reference.follow_rotation
            )

    if isinstance(request, (GridRequest, CircleRequest, RectRequest, ArrowRequest)):
        metric_dependency = metric_dependency or (
            request.geometry_space == GeometrySpace.SURFACE_MM.value
        )
    elif isinstance(request, RulerRequest):
        metric_dependency = metric_dependency or (
            request.measurement_space == GeometrySpace.SURFACE_MM.value
        )
    return tuple(camera_dependencies), metric_dependency


def _validate_projector_output_descriptor(value: object) -> None:
    # Import lazily because configuration imports the overlay limits above.
    from multivision.config import ProjectorOutputDescriptor

    if not isinstance(value, ProjectorOutputDescriptor):
        raise ValueError('projector_output_descriptor is invalid')
    if not is_valid_resolution(value.projector_resolution):
        raise ValueError('projector_output_descriptor is invalid')


class AnchorResolution(NamedTuple):
    """One resolved point and optional marker orientation in source space."""

    position: Point2D
    orientation_degrees: float = 0.0


ResolvedAnchor = AnchorResolution


def resolve_anchor(
    anchor: AnchorReference,
    geometry_space: GeometrySpace | str,
    spatial_state: object | None = None,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> AnchorResolution:
    """Resolve one fixed or namespaced marker anchor without changing it."""
    # Accepting the state before the space keeps this helper convenient for
    # callers that naturally start with the changing authority.
    if not isinstance(geometry_space, (GeometrySpace, str)) and isinstance(
        spatial_state,
        (GeometrySpace, str),
    ):
        geometry_space, spatial_state = spatial_state, geometry_space
    target_space = _normalise_geometry_space(geometry_space)
    authority = metric_calibration
    if authority is None:
        authority = getattr(spatial_state, 'metric_calibration', None)
    if isinstance(anchor, PointReference):
        return AnchorResolution(
            resolve_point_reference(
                anchor,
                target_space,
                camera_to_projector,
                authority,
            ),
        )
    if not isinstance(anchor, (SurfaceAnchor, ProjectorAnchor, FiducialAnchor)):
        raise ValueError('anchor must be a valid anchor or point reference')

    if isinstance(anchor, SurfaceAnchor):
        position = Point2D(anchor.x, anchor.y)
        if target_space == GeometrySpace.PROJECTOR_PX.value:
            position = _project_surface_point_to_projector(
                position,
                authority,
            )
        return AnchorResolution(position)

    if isinstance(anchor, ProjectorAnchor):
        position = Point2D(anchor.x, anchor.y)
        if target_space == GeometrySpace.SURFACE_MM.value:
            position = _project_projector_point_to_surface(
                position,
                authority,
            )
        return AnchorResolution(position)

    observation = _find_spatial_observation(spatial_state, anchor.group, anchor.id)
    if observation is None:
        raise ValueError(
            f'Fiducial anchor {anchor.group!r}/{anchor.id} is unresolved',
        )
    requires_metric = anchor.local_offset is not None or anchor.follow_rotation
    if requires_metric and authority is None:
        raise ValueError('Marker-local anchors require usable metric calibration')
    if requires_metric:
        _require_metric_authority(authority)

    surface_geometry = getattr(observation, 'surface', None)
    projector_geometry = getattr(observation, 'projector', None)
    if surface_geometry is not None:
        _validate_fiducial_geometry(surface_geometry)
    if projector_geometry is not None:
        _validate_fiducial_geometry(projector_geometry)

    if anchor.local_offset is not None and surface_geometry is None:
        if authority is None or projector_geometry is None:
            raise ValueError('Marker-local offsets require surface geometry')
        surface_geometry = _project_marker_geometry(
            projector_geometry,
            authority,
            'projector_to_surface',
        )

    if target_space == GeometrySpace.SURFACE_MM.value:
        if surface_geometry is None:
            if authority is None or projector_geometry is None:
                raise ValueError('Fiducial has no usable surface geometry')
            surface_geometry = _project_marker_geometry(
                projector_geometry,
                authority,
                'projector_to_surface',
            )
        marker_position = surface_geometry.centre
        marker_orientation = surface_geometry.orientation_degrees
    else:
        if projector_geometry is None:
            if authority is None or surface_geometry is None:
                raise ValueError('Fiducial has no usable projector geometry')
            projector_geometry = _project_marker_geometry(
                surface_geometry,
                authority,
                'surface_to_projector',
            )
        marker_position = projector_geometry.centre
        marker_orientation = projector_geometry.orientation_degrees

    if anchor.local_offset is not None:
        assert surface_geometry is not None
        offset = _rotate_marker_local_offset(
            anchor.local_offset.x,
            anchor.local_offset.y,
            surface_geometry.orientation_degrees,
        )
        surface_position = Point2D(
            surface_geometry.centre.x + offset.x,
            surface_geometry.centre.y + offset.y,
        )
        marker_position = (
            surface_position
            if target_space == GeometrySpace.SURFACE_MM.value
            else _project_surface_point_to_projector(surface_position, authority)
        )
        if target_space == GeometrySpace.SURFACE_MM.value:
            marker_orientation = surface_geometry.orientation_degrees

    if not anchor.follow_rotation:
        marker_orientation = 0.0
    _validate_finite_geometry_point(marker_position)
    if not is_finite_real(marker_orientation):
        raise ValueError('Fiducial orientation must be finite')
    return AnchorResolution(marker_position, marker_orientation)


def _find_spatial_observation(
    spatial_state: object | None,
    group: str,
    marker_id: int,
) -> object | None:
    if spatial_state is None:
        return None
    get_observation = getattr(spatial_state, 'get_observation', None)
    if callable(get_observation):
        return get_observation(group, marker_id)
    observations = getattr(spatial_state, 'selected_observations', None)
    if observations is None:
        observations = getattr(spatial_state, 'observations', spatial_state)
    if isinstance(observations, Mapping):
        return observations.get((group, marker_id))
    return None


def _validate_fiducial_geometry(geometry: object) -> None:
    if not isinstance(geometry, TagGeometry):
        raise ValueError('Fiducial geometry must be TagGeometry')
    try:
        validate_planar_corners(geometry.corners)
    except (TypeError, ValueError, OverflowError) as ex:
        raise ValueError('Fiducial geometry must be a valid quadrilateral') from ex
    _validate_finite_geometry_point(geometry.centre)
    if (
        not is_finite_real(geometry.orientation_degrees)
        or not is_finite_real(geometry.area_px)
        or geometry.area_px <= 0
    ):
        raise ValueError('Fiducial geometry must be finite and usable')


def _require_metric_authority(metric_calibration: object) -> None:
    for direction in ('projector_to_surface', 'surface_to_projector'):
        try:
            _get_metric_matrix(metric_calibration, direction)
        except ValueError:
            continue
        return
    raise ValueError('Metric calibration is not usable')


def _project_marker_geometry(
    geometry: TagGeometry,
    metric_calibration: object,
    direction: str,
) -> TagGeometry:
    matrix = _get_metric_matrix(metric_calibration, direction)
    try:
        return project_tag_geometry(geometry, matrix)
    except (TypeError, ValueError) as ex:
        raise ValueError('Fiducial geometry cannot be projected safely') from ex


def _rotate_marker_local_offset(
    x_offset: float,
    y_offset: float,
    orientation_degrees: float,
) -> Point2D:
    angle_radians = math.radians(orientation_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    result = Point2D(
        cosine * x_offset - sine * y_offset,
        sine * x_offset + cosine * y_offset,
    )
    _validate_finite_geometry_point(result)
    return result


def resolve_point_reference(
    point: PointReference,
    geometry_space: GeometrySpace | str,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> Point2D:
    """Resolve one point through the existing camera and metric authorities."""
    if not isinstance(point, PointReference):
        raise ValueError('point must be a PointReference')
    target_space = _normalise_geometry_space(geometry_space)
    point_position = Point2D(point.x, point.y)

    if point.space == PointReferenceSpace.PROJECTOR_PX.value:
        projector_position = point_position
    elif point.space == PointReferenceSpace.CAMERA_PX.value:
        transform = _get_camera_to_projector_transform(
            camera_to_projector,
            point.camera,
        )
        projector_position = project_camera_point(point_position, transform)
    else:
        projector_position = None

    if target_space == GeometrySpace.PROJECTOR_PX.value:
        if projector_position is not None:
            return projector_position
        return _project_surface_point_to_projector(
            point_position,
            metric_calibration,
        )

    if point.space == PointReferenceSpace.SURFACE_MM.value:
        return point_position
    if projector_position is None:
        raise ValueError('Point cannot be converted to surface_mm')
    return _project_projector_point_to_surface(
        projector_position,
        metric_calibration,
    )


def _resolve_overlay_reference(
    reference: AnchorReference,
    geometry_space: GeometrySpace | str,
    spatial_state: object | None,
    camera_to_projector: object | None,
    metric_calibration: object | None,
) -> AnchorResolution:
    if isinstance(reference, PointReference):
        return AnchorResolution(
            resolve_point_reference(
                reference,
                geometry_space,
                camera_to_projector,
                metric_calibration,
            ),
        )
    return resolve_anchor(
        reference,
        geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    )


def build_circle(
    request: CircleRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> CircleGeometry:
    """Build a finite circle after converting its anchor to its source space."""
    if not isinstance(request, CircleRequest):
        raise ValueError('request must be a CircleRequest')
    centre = _resolve_overlay_reference(
        request.centre,
        request.geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    radius = request.radius.value
    return CircleGeometry(centre, radius, request.geometry_space, request.style)


def build_rotated_rect(
    request: RectRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> RectGeometry:
    """Build four rotated corners in the requested source space."""
    if not isinstance(request, RectRequest):
        raise ValueError('request must be a RectRequest')
    resolved_centre = _resolve_overlay_reference(
        request.centre,
        request.geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    )
    centre = resolved_centre.position
    angle_deg = request.angle_deg + resolved_centre.orientation_degrees
    width = request.width.value
    height = request.height.value
    half_width = width / 2.0
    half_height = height / 2.0
    corners = tuple(
        _rotate_source_offset(
            centre,
            x_offset,
            y_offset,
            angle_deg,
        )
        for x_offset, y_offset in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    )
    for corner in corners:
        _validate_finite_geometry_point(corner)
    return RectGeometry(
        centre,
        corners,
        width,
        height,
        angle_deg,
        request.geometry_space,
        request.style,
    )


def build_text(
    request: TextRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> TextGeometry:
    """Resolve one floating text anchor to projector-native coordinates."""
    if not isinstance(request, TextRequest):
        raise ValueError('request must be a TextRequest')
    resolved_position = _resolve_overlay_reference(
        request.position,
        GeometrySpace.PROJECTOR_PX,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    )
    return TextGeometry(
        resolved_position.position,
        request.text,
        request.angle_deg + resolved_position.orientation_degrees,
        request.scale,
        request.style,
    )


def build_projector_coverage_grid_request(
    request: ProjectorCoverageGridRequest,
    metric_calibration: object,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
) -> GridRequest:
    """Build a finite surface grid covering the complete projector output."""
    if not isinstance(request, ProjectorCoverageGridRequest):
        raise ValueError('request must be ProjectorCoverageGridRequest')
    from multivision.metric import calculate_projector_surface_grid_layout

    layout = calculate_projector_surface_grid_layout(
        metric_calibration,
        projector_resolution,
        request.spacing.value,
    )
    return GridRequest(
        id=request.id,
        name=request.name,
        visible=request.visible,
        style=request.style,
        origin={
            'space': GeometrySpace.SURFACE_MM.value,
            'x': layout.origin.x,
            'y': layout.origin.y,
            'unit': 'mm',
        },
        geometry_space=GeometrySpace.SURFACE_MM.value,
        spacing=request.spacing,
        extent={
            'width': {
                'value': layout.width,
                'unit': 'mm',
            },
            'height': {
                'value': layout.height,
                'unit': 'mm',
            },
        },
        angle_deg=layout.angle_deg + request.angle_deg,
    )


def build_grid(
    request: GridRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> GridGeometry:
    """Build a finite square grid with exact source-space spacing."""
    if not isinstance(request, GridRequest):
        raise ValueError('request must be a GridRequest')
    resolved_origin = _resolve_overlay_reference(
        request.origin,
        request.geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    )
    origin = resolved_origin.position
    angle_deg = request.angle_deg + resolved_origin.orientation_degrees
    spacing = request.spacing.value
    width = request.extent.width.value
    height = request.extent.height.value
    vertical_count = _calculate_grid_line_count(width, spacing)
    horizontal_count = _calculate_grid_line_count(height, spacing)
    segment_count = vertical_count + horizontal_count
    limits = (
        OverlayConfiguration()
        if overlay_configuration is None
        else overlay_configuration
    )
    if not isinstance(limits, OverlayConfiguration):
        raise ValueError('overlay_configuration must be OverlayConfiguration')
    if segment_count > limits.max_overlay_segments:
        raise ValueError('Grid exceeds the configured segment budget')
    if 2 * segment_count > limits.max_overlay_vertices:
        raise ValueError('Grid exceeds the configured vertex budget')

    vertical_segments = tuple(
        SourceSegment(
            _rotate_source_offset(origin, x_idx * spacing, 0.0, angle_deg),
            _rotate_source_offset(origin, x_idx * spacing, height, angle_deg),
        )
        for x_idx in range(vertical_count)
    )
    horizontal_segments = tuple(
        SourceSegment(
            _rotate_source_offset(origin, 0.0, y_idx * spacing, angle_deg),
            _rotate_source_offset(origin, width, y_idx * spacing, angle_deg),
        )
        for y_idx in range(horizontal_count)
    )
    for segment in vertical_segments + horizontal_segments:
        _validate_finite_geometry_point(segment.start)
        _validate_finite_geometry_point(segment.end)
    return GridGeometry(
        origin,
        spacing,
        width,
        height,
        angle_deg,
        request.geometry_space,
        vertical_segments,
        horizontal_segments,
        request.style,
    )


def build_line(
    request: LineRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> LineGeometry:
    """Build a literal projector-native line from independently resolved endpoints."""
    if not isinstance(request, LineRequest):
        raise ValueError('request must be a LineRequest')
    start = _resolve_overlay_reference(
        request.start,
        GeometrySpace.PROJECTOR_PX,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    end = _resolve_overlay_reference(
        request.end,
        GeometrySpace.PROJECTOR_PX,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    return LineGeometry(start, end, request.label, request.style)


def build_arrow(
    request: ArrowRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> ArrowGeometry:
    """Build one deterministic shaft and triangle head in source space."""
    if not isinstance(request, ArrowRequest):
        raise ValueError('request must be an ArrowRequest')
    start = _resolve_overlay_reference(
        request.start,
        request.geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    end = _resolve_overlay_reference(
        request.end,
        request.geometry_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    difference_x = end.x - start.x
    difference_y = end.y - start.y
    length = math.hypot(difference_x, difference_y)
    if not math.isfinite(length) or length == 0:
        raise ValueError('Arrow endpoints must be distinct and finite')
    direction_x = difference_x / length
    direction_y = difference_y / length
    head_length = request.head_length.value
    head_width = request.head_width.value
    base = Point2D(
        end.x - direction_x * head_length,
        end.y - direction_y * head_length,
    )
    perpendicular = Point2D(-direction_y, direction_x)
    head = (
        end,
        Point2D(
            base.x + perpendicular.x * head_width / 2.0,
            base.y + perpendicular.y * head_width / 2.0,
        ),
        Point2D(
            base.x - perpendicular.x * head_width / 2.0,
            base.y - perpendicular.y * head_width / 2.0,
        ),
    )
    _validate_finite_geometry_point(start)
    _validate_finite_geometry_point(end)
    for point in head:
        _validate_finite_geometry_point(point)
    return ArrowGeometry(
        start,
        end,
        head,
        head_length,
        head_width,
        request.geometry_space,
        request.label,
        request.style,
    )


def build_ruler(
    request: RulerRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    spatial_state: object | None = None,
) -> RulerGeometry:
    """Build a ruler in projector pixels or canonical surface millimetres."""
    if not isinstance(request, RulerRequest):
        raise ValueError('request must be a RulerRequest')
    start = _resolve_overlay_reference(
        request.start,
        request.measurement_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    end = _resolve_overlay_reference(
        request.end,
        request.measurement_space,
        spatial_state,
        camera_to_projector,
        metric_calibration,
    ).position
    if request.measurement_space == GeometrySpace.SURFACE_MM.value:
        from multivision.metric import calculate_surface_distance_mm

        length_mm = calculate_surface_distance_mm(start, end)
        output_length = length_mm / _get_unit_to_mm()[request.unit]
    else:
        length_mm = None
        output_length = math.hypot(end.x - start.x, end.y - start.y)
    if not math.isfinite(output_length) or output_length <= 0:
        raise ValueError('Ruler endpoints must be distinct and finite')
    return RulerGeometry(
        start,
        end,
        request.measurement_space,
        request.unit,
        output_length,
        length_mm,
        request.label,
        request.style,
    )


DEFAULT_CIRCLE_SAMPLE_COUNT = 64


def materialise_circle(
    request_or_geometry: CircleRequest | CircleGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Sample and clip one circle in projector-native coordinates."""
    geometry = (
        build_circle(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, CircleRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, CircleGeometry):
        raise ValueError('request_or_geometry must be CircleRequest or CircleGeometry')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    sample_count = DEFAULT_CIRCLE_SAMPLE_COUNT
    if 2 * sample_count > limits.max_overlay_vertices:
        raise ValueError('Circle exceeds the configured vertex budget')
    if sample_count > limits.max_overlay_segments:
        raise ValueError('Circle exceeds the configured segment budget')

    source_points = tuple(
        Point2D(
            geometry.centre.x
            + geometry.radius * math.cos(2 * math.pi * idx / sample_count),
            geometry.centre.y
            - geometry.radius * math.sin(2 * math.pi * idx / sample_count),
        )
        for idx in range(sample_count)
    )
    _validate_circle_horizon(geometry, metric_calibration)
    projector_points = _project_source_points(
        source_points,
        geometry.geometry_space,
        metric_calibration,
    )
    if geometry.style.fill:
        polygon = _clip_projector_polygon(projector_points, bounds, geometry.style)
        return ProjectorMaterialisation(polygons=() if polygon is None else (polygon,))
    segments = _materialise_closed_edges(projector_points, bounds, geometry.style)
    return ProjectorMaterialisation(segments=segments)


def materialise_rect(
    request_or_geometry: RectRequest | RectGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Project and clip all four edges of a rotated rectangle."""
    geometry = (
        build_rotated_rect(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, RectRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, RectGeometry):
        raise ValueError('request_or_geometry must be RectRequest or RectGeometry')
    label = (
        request_or_geometry.label
        if isinstance(request_or_geometry, RectRequest)
        else None
    )
    label_angle_deg = (
        request_or_geometry.label_angle_deg
        if isinstance(request_or_geometry, RectRequest)
        else 0.0
    )
    label_scale = (
        request_or_geometry.label_scale
        if isinstance(request_or_geometry, RectRequest)
        else 1.0
    )
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    if 4 > limits.max_overlay_segments or 8 > limits.max_overlay_vertices:
        raise ValueError('Rectangle exceeds the configured primitive budget')
    projector_points = _project_source_points(
        geometry.corners,
        geometry.geometry_space,
        metric_calibration,
    )
    if geometry.style.fill:
        polygon = _clip_projector_polygon(projector_points, bounds, geometry.style)
        materialisation = ProjectorMaterialisation(
            polygons=() if polygon is None else (polygon,),
        )
    else:
        materialisation = ProjectorMaterialisation(
            segments=_materialise_closed_edges(projector_points, bounds, geometry.style),
        )
    projector_centre = _project_source_points(
        (geometry.centre,),
        geometry.geometry_space,
        metric_calibration,
    )[0]
    labels = _materialise_optional_label_at_position(
        label,
        projector_centre,
        materialisation,
        bounds,
        geometry.style,
        label_angle_deg,
        label_scale,
        limits,
    )
    return materialisation._replace(labels=labels)


def materialise_text(
    request_or_geometry: TextRequest | TextGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Resolve one floating text label in projector-native coordinates."""
    geometry = (
        build_text(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, TextRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, TextGeometry):
        raise ValueError('request_or_geometry must be TextRequest or TextGeometry')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    _validate_finite_geometry_point(geometry.position)
    if not isinstance(geometry.text, str) or len(geometry.text) == 0:
        raise ValueError('text must be a non-empty string')
    if (
        not is_finite_real(geometry.angle_deg)
        or not is_finite_real(geometry.scale)
        or not MIN_OVERLAY_LABEL_SCALE <= geometry.scale <= MAX_OVERLAY_LABEL_SCALE
    ):
        raise ValueError('text rotation and scale are invalid')
    if len(geometry.text) > limits.max_overlay_label_characters:
        raise ValueError('text exceeds the configured character limit')
    label = ProjectorLabel(
        _round_projector_point(geometry.position, bounds),
        geometry.text,
        geometry.style,
        geometry.angle_deg,
        geometry.scale,
    )
    return ProjectorMaterialisation(labels=(label,))


def materialise_grid(
    request_or_geometry: GridRequest | GridGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Project and independently clip a finite grid's source-space segments."""
    geometry = (
        build_grid(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            overlay_configuration,
            spatial_state,
        )
        if isinstance(request_or_geometry, GridRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, GridGeometry):
        raise ValueError('request_or_geometry must be GridRequest or GridGeometry')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    source_segments = geometry.segments
    if len(source_segments) > limits.max_overlay_segments:
        raise ValueError('Grid exceeds the configured segment budget')
    if 2 * len(source_segments) > limits.max_overlay_vertices:
        raise ValueError('Grid exceeds the configured vertex budget')
    segments: list[ProjectorSegment] = []
    for source_segment in source_segments:
        segments.extend(
            _materialise_source_segment(
                source_segment,
                geometry.geometry_space,
                geometry.style,
                bounds,
                metric_calibration,
            ),
        )
    return ProjectorMaterialisation(segments=tuple(segments))


def materialise_line(
    request_or_geometry: LineRequest | LineGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Resolve, clip and optionally label a literal line segment."""
    geometry = (
        build_line(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, LineRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, LineGeometry):
        raise ValueError('request_or_geometry must be LineRequest or LineGeometry')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    if 1 > limits.max_overlay_segments or 2 > limits.max_overlay_vertices:
        raise ValueError('Line exceeds the configured primitive budget')
    segment = _clip_projector_segment(geometry.start, geometry.end, bounds, geometry.style)
    segments = () if segment is None else (segment,)
    labels = _materialise_optional_label(
        geometry.label,
        geometry.start,
        geometry.end,
        segments,
        bounds,
        geometry.style,
        limits,
    )
    return ProjectorMaterialisation(segments=segments, labels=labels)


def materialise_arrow(
    request_or_geometry: ArrowRequest | ArrowGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Project and clip one arrow shaft and its deterministic triangle head."""
    geometry = (
        build_arrow(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, ArrowRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, ArrowGeometry):
        raise ValueError('request_or_geometry must be ArrowRequest or ArrowGeometry')
    _validate_finite_geometry_point(geometry.start)
    _validate_finite_geometry_point(geometry.end)
    if len(geometry.head) != 3:
        raise ValueError('Arrow heads must contain exactly three points')
    for point in geometry.head:
        _validate_finite_geometry_point(point)
    for dimension, field_name in (
        (geometry.head_length, 'head_length'),
        (geometry.head_width, 'head_width'),
    ):
        if (
            not is_finite_real(dimension)
            or dimension <= 0
            or dimension > MAX_ARROW_HEAD_DIMENSION
        ):
            raise ValueError(f'{field_name} must be positive and bounded')
    source_length = math.hypot(
        geometry.end.x - geometry.start.x,
        geometry.end.y - geometry.start.y,
    )
    if not math.isfinite(source_length) or source_length == 0:
        raise ValueError('Arrow endpoints must be distinct and finite')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    if 1 > limits.max_overlay_segments or 5 > limits.max_overlay_vertices:
        raise ValueError('Arrow exceeds the configured primitive budget')
    projector_start, projector_end = _project_source_points(
        (geometry.start, geometry.end),
        geometry.geometry_space,
        metric_calibration,
    )
    projector_head = _project_source_points(
        geometry.head,
        geometry.geometry_space,
        metric_calibration,
    )
    shaft = _clip_projector_segment(
        projector_start,
        projector_end,
        bounds,
        geometry.style,
    )
    polygon = _clip_projector_polygon(projector_head, bounds, geometry.style)
    segments = () if shaft is None else (shaft,)
    polygons = () if polygon is None else (polygon,)
    labels = _materialise_optional_label(
        geometry.label,
        projector_start,
        projector_end,
        segments,
        bounds,
        geometry.style,
        limits,
    )
    return ProjectorMaterialisation(segments, polygons, labels)


def materialise_ruler(
    request_or_geometry: RulerRequest | RulerGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Build physical or projector-pixel ticks, then clip every segment."""
    geometry = (
        build_ruler(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            spatial_state,
        )
        if isinstance(request_or_geometry, RulerRequest)
        else request_or_geometry
    )
    if not isinstance(geometry, RulerGeometry):
        raise ValueError('request_or_geometry must be RulerRequest or RulerGeometry')
    bounds, limits = _normalise_materialisation_arguments(
        projector_resolution,
        overlay_configuration,
    )
    projector_start, projector_end = _project_ruler_endpoints(
        geometry,
        metric_calibration,
    )
    main_segment = _clip_projector_segment(
        projector_start,
        projector_end,
        bounds,
        geometry.style,
    )
    tick_spacing, tick_count = _calculate_ruler_tick_layout(
        geometry,
        limits.max_overlay_ticks,
    )
    segment_count = 1 + tick_count
    if segment_count > limits.max_overlay_segments:
        raise ValueError('Ruler exceeds the configured segment budget')
    if 2 * segment_count > limits.max_overlay_vertices:
        raise ValueError('Ruler exceeds the configured vertex budget')
    source_ticks = _build_ruler_ticks(geometry, tick_spacing, tick_count)
    segments = [] if main_segment is None else [main_segment]
    for source_tick in source_ticks:
        segments.extend(
            _materialise_source_segment(
                source_tick,
                geometry.measurement_space,
                geometry.style,
                bounds,
                metric_calibration,
            ),
        )
    label = geometry.label
    if label is None:
        label = f'{geometry.length:.1f} {geometry.unit}'
    labels = _materialise_optional_label(
        label,
        projector_start,
        projector_end,
        tuple(segments),
        bounds,
        geometry.style,
        limits,
    )
    return ProjectorMaterialisation(segments=tuple(segments), labels=labels)


def materialise_overlay(
    request: AnyOverlayRequest,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
    spatial_state: object | None = None,
) -> ProjectorMaterialisation:
    """Materialise one validated request through the appropriate pure builder."""
    if not isinstance(request, _OVERLAY_REQUEST_TYPES):
        raise ValueError('request must be an overlay request')
    materialise = {
        'grid': materialise_grid,
        'circle': materialise_circle,
        'rect': materialise_rect,
        'text': materialise_text,
        'line': materialise_line,
        'ruler': materialise_ruler,
        'arrow': materialise_arrow,
    }[request.kind]
    return materialise(
        request,
        projector_resolution,
        camera_to_projector,
        metric_calibration,
        overlay_configuration,
        spatial_state,
    )


def _normalise_materialisation_arguments(
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    overlay_configuration: OverlayConfiguration | None,
) -> tuple[CoordinateBounds, OverlayConfiguration]:
    if isinstance(projector_resolution, CoordinateBounds):
        if (
            projector_resolution.left != 0
            or projector_resolution.top != 0
            or not float(projector_resolution.right).is_integer()
            or not float(projector_resolution.bottom).is_integer()
        ):
            raise ValueError('projector bounds must be an origin-based integer rectangle')
        projector_resolution = Resolution(
            int(projector_resolution.right),
            int(projector_resolution.bottom),
        )
    if not isinstance(projector_resolution, Resolution):
        try:
            width, height = projector_resolution
        except (TypeError, ValueError):
            raise ValueError('projector_resolution must contain width and height') from None
        projector_resolution = Resolution(width, height)
    if not is_valid_resolution(projector_resolution):
        raise ValueError('projector_resolution must be a positive resolution')
    if overlay_configuration is not None and not isinstance(
        overlay_configuration,
        OverlayConfiguration,
    ):
        raise ValueError('overlay_configuration must be OverlayConfiguration')
    limits = overlay_configuration or OverlayConfiguration()
    return CoordinateBounds(0, 0, projector_resolution.width, projector_resolution.height), limits


def _project_source_points(
    source_points: Sequence[Point2D],
    geometry_space: str,
    metric_calibration: object | None,
) -> tuple[Point2D, ...]:
    if geometry_space == GeometrySpace.PROJECTOR_PX.value:
        projected_points = tuple(source_points)
    else:
        matrix = _get_metric_matrix(metric_calibration, 'surface_to_projector')
        denominators = tuple(
            matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2]
            for point in source_points
        )
        if any(
            not math.isfinite(denominator) or abs(denominator) <= 1e-12
            for denominator in denominators
        ) or len({denominator > 0 for denominator in denominators}) != 1:
            raise ValueError('Source geometry crosses the projective horizon')
        try:
            projected_points = tuple(project_point(point, matrix) for point in source_points)
        except (OverflowError, ValueError) as ex:
            raise ValueError('Source geometry cannot be projected safely') from ex
    for point in projected_points:
        _validate_finite_geometry_point(point)
    return projected_points


def _validate_circle_horizon(
    geometry: CircleGeometry,
    metric_calibration: object | None,
) -> None:
    if geometry.geometry_space == GeometrySpace.PROJECTOR_PX.value:
        return
    matrix = _get_metric_matrix(metric_calibration, 'surface_to_projector')
    denominator_at_centre = (
        matrix[2][0] * geometry.centre.x
        + matrix[2][1] * geometry.centre.y
        + matrix[2][2]
    )
    denominator_radius = geometry.radius * math.hypot(matrix[2][0], matrix[2][1])
    if (
        not math.isfinite(denominator_at_centre)
        or not math.isfinite(denominator_radius)
        or abs(denominator_at_centre) <= denominator_radius
    ):
        raise ValueError('Circle crosses the projective horizon')


def _materialise_source_segment(
    source_segment: SourceSegment,
    geometry_space: str,
    style: OverlayStyle,
    bounds: CoordinateBounds,
    metric_calibration: object | None,
) -> tuple[ProjectorSegment, ...]:
    projector_start, projector_end = _project_source_points(
        (source_segment.start, source_segment.end),
        geometry_space,
        metric_calibration,
    )
    segment = _clip_projector_segment(projector_start, projector_end, bounds, style)
    return () if segment is None else (segment,)


def _project_ruler_endpoints(
    geometry: RulerGeometry,
    metric_calibration: object | None,
) -> tuple[Point2D, Point2D]:
    return _project_source_points(
        (geometry.start, geometry.end),
        geometry.measurement_space,
        metric_calibration,
    )


def _calculate_ruler_tick_layout(
    geometry: RulerGeometry,
    maximum_tick_count: int,
) -> tuple[float, int]:
    from multivision.metric import calculate_ruler_tick_layout

    length = (
        geometry.length_mm
        if geometry.measurement_space == GeometrySpace.SURFACE_MM.value
        else geometry.length
    )
    return calculate_ruler_tick_layout(length, maximum_tick_count)


def _build_ruler_ticks(
    geometry: RulerGeometry,
    spacing: float,
    tick_count: int,
) -> tuple[SourceSegment, ...]:
    length = (
        geometry.length_mm
        if geometry.measurement_space == GeometrySpace.SURFACE_MM.value
        else geometry.length
    )
    if length is None or not math.isfinite(length):
        raise ValueError('Ruler length must be finite')
    if tick_count == 0:
        return ()
    difference_x = geometry.end.x - geometry.start.x
    difference_y = geometry.end.y - geometry.start.y
    unit_x = difference_x / length
    unit_y = difference_y / length
    normal_x = -unit_y
    normal_y = unit_x
    ticks: list[SourceSegment] = []
    for tick_index in range(1, tick_count + 1):
        distance = tick_index * spacing
        centre = Point2D(
            geometry.start.x + unit_x * distance,
            geometry.start.y + unit_y * distance,
        )
        half_length = 5.0 if distance % 10.0 == 0 else 3.0
        ticks.append(
            SourceSegment(
                Point2D(
                    centre.x - normal_x * half_length,
                    centre.y - normal_y * half_length,
                ),
                Point2D(
                    centre.x + normal_x * half_length,
                    centre.y + normal_y * half_length,
                ),
            ),
        )
    return tuple(ticks)


def _materialise_closed_edges(
    points: Sequence[Point2D],
    bounds: CoordinateBounds,
    style: OverlayStyle,
) -> tuple[ProjectorSegment, ...]:
    segments: list[ProjectorSegment] = []
    for idx, point in enumerate(points):
        segment = _clip_projector_segment(
            point,
            points[(idx + 1) % len(points)],
            bounds,
            style,
        )
        if segment is not None:
            segments.append(segment)
    return tuple(segments)


def _clip_projector_segment(
    start: Point2D,
    end: Point2D,
    bounds: CoordinateBounds,
    style: OverlayStyle,
) -> ProjectorSegment | None:
    difference_x = end.x - start.x
    difference_y = end.y - start.y
    if not math.isfinite(difference_x) or not math.isfinite(difference_y):
        raise ValueError('Projector segment is not finite')
    parameters = [0.0, 1.0]
    for position, difference, lower, upper in (
        (start.x, difference_x, bounds.left, bounds.right),
        (start.y, difference_y, bounds.top, bounds.bottom),
    ):
        if difference == 0:
            if position < lower or position >= upper:
                return None
            continue
        lower_parameter = (lower - position) / difference
        upper_parameter = (upper - position) / difference
        if lower_parameter > upper_parameter:
            lower_parameter, upper_parameter = upper_parameter, lower_parameter
        parameters[0] = max(parameters[0], lower_parameter)
        parameters[1] = min(parameters[1], upper_parameter)
        if parameters[0] > parameters[1]:
            return None
    clipped_start = Point2D(
        start.x + parameters[0] * difference_x,
        start.y + parameters[0] * difference_y,
    )
    clipped_end = Point2D(
        start.x + parameters[1] * difference_x,
        start.y + parameters[1] * difference_y,
    )
    rounded_start = _round_projector_point(clipped_start, bounds)
    rounded_end = _round_projector_point(clipped_end, bounds)
    if rounded_start == rounded_end:
        return None
    return ProjectorSegment(rounded_start, rounded_end, style)


def _clip_projector_polygon(
    points: Sequence[Point2D],
    bounds: CoordinateBounds,
    style: OverlayStyle,
) -> ProjectorPolygon | None:
    clipped_points = list(points)
    for axis, boundary, keeps_greater in (
        ('x', bounds.left, True),
        ('x', bounds.right, False),
        ('y', bounds.top, True),
        ('y', bounds.bottom, False),
    ):
        if len(clipped_points) == 0:
            return None
        output: list[Point2D] = []
        previous_point = clipped_points[-1]
        previous_value = previous_point.x if axis == 'x' else previous_point.y
        previous_inside = (
            previous_value >= boundary
            if keeps_greater
            else previous_value <= boundary
        )
        for current_point in clipped_points:
            current_value = current_point.x if axis == 'x' else current_point.y
            current_inside = (
                current_value >= boundary
                if keeps_greater
                else current_value <= boundary
            )
            if current_inside != previous_inside:
                fraction = (boundary - previous_value) / (current_value - previous_value)
                output.append(
                    Point2D(
                        previous_point.x + fraction * (current_point.x - previous_point.x),
                        previous_point.y + fraction * (current_point.y - previous_point.y),
                    ),
                )
            if current_inside:
                output.append(current_point)
            previous_point = current_point
            previous_value = current_value
            previous_inside = current_inside
        clipped_points = output
    rounded_points = tuple(_round_projector_point(point, bounds) for point in clipped_points)
    distinct_points = tuple(
        point
        for idx, point in enumerate(rounded_points)
        if idx == 0 or point != rounded_points[idx - 1]
    )
    if len(distinct_points) > 1 and distinct_points[0] == distinct_points[-1]:
        distinct_points = distinct_points[:-1]
    if len(distinct_points) < 3 or _polygon_area(distinct_points) <= 0:
        return None
    return ProjectorPolygon(distinct_points, style)


def _polygon_area(points: Sequence[Point2D]) -> float:
    return abs(
        sum(
            point.x * points[(idx + 1) % len(points)].y
            - points[(idx + 1) % len(points)].x * point.y
            for idx, point in enumerate(points)
        )
        / 2.0
    )


def _round_projector_point(point: Point2D, bounds: CoordinateBounds) -> Point2D:
    return Point2D(
        float(min(max(round(point.x), int(bounds.left)), int(bounds.right) - 1)),
        float(min(max(round(point.y), int(bounds.top)), int(bounds.bottom) - 1)),
    )


def _materialise_optional_label(
    label: str | None,
    start: Point2D,
    end: Point2D,
    visible_segments: Sequence[ProjectorSegment],
    bounds: CoordinateBounds,
    style: OverlayStyle,
    limits: OverlayConfiguration,
) -> tuple[ProjectorLabel, ...]:
    if label is None:
        return ()
    midpoint = Point2D((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
    return _materialise_optional_label_at_position(
        label,
        midpoint,
        ProjectorMaterialisation(segments=tuple(visible_segments)),
        bounds,
        style,
        0.0,
        1.0,
        limits,
    )


def _materialise_optional_label_at_position(
    label: str | None,
    position: Point2D,
    materialisation: ProjectorMaterialisation,
    bounds: CoordinateBounds,
    style: OverlayStyle,
    angle_deg: float,
    scale: float,
    limits: OverlayConfiguration,
) -> tuple[ProjectorLabel, ...]:
    if label is None:
        return ()
    if len(label) > limits.max_overlay_label_characters:
        raise ValueError('label exceeds the configured character limit')
    if (
        len(materialisation.segments) == 0
        and len(materialisation.polygons) == 0
    ):
        return ()
    return (
        ProjectorLabel(
            _round_projector_point(position, bounds),
            label,
            style,
            angle_deg,
            scale,
        ),
    )


def _calculate_grid_line_count(extent: float, spacing: float) -> int:
    try:
        ratio = extent / spacing
    except OverflowError as ex:
        raise ValueError('Grid extent and spacing produce a non-finite line count') from ex
    if not math.isfinite(ratio):
        raise ValueError('Grid extent and spacing produce a non-finite line count')
    return math.floor(ratio) + 1


def _normalise_geometry_space(space: GeometrySpace | str) -> str:
    space_value = space.value if isinstance(space, GeometrySpace) else space
    if space_value not in {
        GeometrySpace.PROJECTOR_PX.value,
        GeometrySpace.SURFACE_MM.value,
    }:
        raise ValueError(f'Unsupported geometry space: {space!r}')
    return space_value


def _get_camera_to_projector_transform(
    authority: object | None,
    camera_id: str | None,
) -> MatrixLike:
    if not isinstance(camera_id, str) or len(camera_id) == 0:
        raise ValueError('camera_px points require a camera identity')
    if authority is None:
        raise ValueError(f'No camera-to-projector transform for {camera_id!r}')

    if isinstance(authority, Mapping):
        transform = authority.get(camera_id)
    else:
        get_status = getattr(authority, 'get_status', None)
        if callable(get_status):
            status = get_status(camera_id)
            status_value = getattr(status, 'value', status)
            if status_value != 'CALIBRATED':
                raise ValueError(f'Camera {camera_id!r} calibration is not usable')
        get_record = getattr(authority, 'get_record', None)
        transform = get_record(camera_id) if callable(get_record) else authority

    if transform is None:
        raise ValueError(f'No camera-to-projector transform for {camera_id!r}')
    record_camera_id = getattr(transform, 'camera_id', None)
    if record_camera_id is not None and record_camera_id != camera_id:
        raise ValueError(f'Camera transform does not belong to {camera_id!r}')
    matrix = getattr(transform, 'camera_to_projector', transform)
    return validate_homography(matrix)


def _get_metric_matrix(metric_calibration: object | None, direction: str) -> MatrixLike:
    if metric_calibration is None:
        raise ValueError('A usable metric calibration is required')
    is_usable = getattr(metric_calibration, 'is_usable', None)
    if callable(is_usable):
        if is_usable() is not True:
            raise ValueError('Metric calibration is not usable')
    elif is_usable is not None and is_usable is not True:
        raise ValueError('Metric calibration is not usable')

    get_record = getattr(metric_calibration, 'get_record', None)
    record = get_record() if callable(get_record) else metric_calibration
    if record is None:
        raise ValueError('A usable metric calibration is required')
    state = getattr(record, 'state', None)
    state_value = getattr(state, 'value', state)
    if state is not None and state_value != 'CALIBRATED':
        raise ValueError('Metric calibration is not usable')
    matrix = getattr(record, direction, None)
    if matrix is None:
        raise ValueError('Metric calibration has no complete transform pair')
    return validate_homography(matrix)


def _project_surface_point_to_projector(
    point: Point2D,
    metric_calibration: object | None,
) -> Point2D:
    return project_point(
        point,
        _get_metric_matrix(metric_calibration, 'surface_to_projector'),
    )


def _project_projector_point_to_surface(
    point: Point2D,
    metric_calibration: object | None,
) -> Point2D:
    return project_point(
        point,
        _get_metric_matrix(metric_calibration, 'projector_to_surface'),
    )


def _rotate_source_offset(
    centre: Point2D,
    x_offset: float,
    y_offset: float,
    angle_deg: float,
) -> Point2D:
    angle_radians = math.radians(angle_deg)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    # Source coordinates grow downwards, so this preserves the public CCW angle.
    return Point2D(
        centre.x + cosine * x_offset + sine * y_offset,
        centre.y - sine * x_offset + cosine * y_offset,
    )


def _validate_finite_geometry_point(point: Point2D) -> None:
    if (
        not isinstance(point, Point2D)
        or not all(is_finite_real(value) for value in point)
    ):
        raise ValueError('Geometry points must be finite Point2D values')


__all__ = [
    'ANGLE_CONVENTION',
    'AnyOverlayRequest',
    'Anchor',
    'AnchorReference',
    'AnchorResolution',
    'ArrowGeometry',
    'ArrowRequest',
    'MAX_ARROW_HEAD_DIMENSION',
    'MAX_OVERLAY_INTENSITY',
    'MAX_OVERLAY_LABEL_SCALE',
    'MIN_OVERLAY_INTENSITY',
    'MIN_OVERLAY_LABEL_SCALE',
    'CircleRequest',
    'GeometrySpace',
    'GridRequest',
    'LineRequest',
    'OverlayConfiguration',
    'OverlayExtent',
    'OverlayRequest',
    'OverlayStyle',
    'OverlayUnit',
    'PhysicalUnit',
    'PointReference',
    'PointReferenceSpace',
    'ProjectorAnchor',
    'PointSpace',
    'Quantity',
    'RectRequest',
    'RulerRequest',
    'ResolvedAnchor',
    'SurfaceAnchor',
    'TextRequest',
    'PHYSICAL_UNITS',
    'CircleGeometry',
    'FiducialAnchor',
    'GridGeometry',
    'ProjectorCoverageGridRequest',
    'LineGeometry',
    'RectGeometry',
    'RulerGeometry',
    'TextGeometry',
    'SourceSegment',
    'ProjectorLabel',
    'ProjectorMaterialisation',
    'apply_overlay_intensity_to_colour',
    'ProjectorPolygon',
    'OverlayEntry',
    'OverlayRegistry',
    'ProjectorSegment',
    'get_overlay_dependencies',
    'get_overlay_fiducial_dependencies',
    'is_circle_intersecting_protected_regions',
    'is_polygon_intersecting_protected_regions',
    'get_overlay_point_references',
    'is_dynamic_overlay_request',
    'DEFAULT_CIRCLE_SAMPLE_COUNT',
    'build_circle',
    'build_grid',
    'build_projector_coverage_grid_request',
    'build_line',
    'build_rotated_rect',
    'build_ruler',
    'build_text',
    'build_arrow',
    'materialise_arrow',
    'materialise_circle',
    'materialise_grid',
    'materialise_line',
    'materialise_overlay',
    'materialise_overlay_presentation',
    'materialise_presentation',
    'normalise_overlay_intensity',
    'materialise_rect',
    'materialise_ruler',
    'materialise_text',
    'resolve_anchor',
    'resolve_point_reference',
    'LocalOffset',
]
