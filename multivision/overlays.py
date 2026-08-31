"""Coordinate-aware overlay requests and source-space geometry builders."""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
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
    camera_to_projector as project_camera_point,
    project_point,
    validate_homography,
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


class OverlayStyle(BaseModel):
    """The small immutable style vocabulary shared by generic overlays."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)

    colour: tuple[int, int, int] = (255, 255, 255)
    fill: bool = False
    line_width_px: int = 1

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
    origin: PointReference
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
    centre: PointReference
    geometry_space: Literal['projector_px', 'surface_mm']
    radius: Quantity

    @model_validator(mode='after')
    def validate_circle(self) -> 'CircleRequest':
        self.radius.validate_for_space(self.geometry_space).require_positive()
        return self


class RectRequest(OverlayRequest):
    kind: Literal['rect'] = 'rect'
    centre: PointReference
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
    position: PointReference
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
    start: PointReference
    end: PointReference
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
    start: PointReference
    end: PointReference
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


AnyOverlayRequest: TypeAlias = (
    GridRequest
    | CircleRequest
    | RectRequest
    | TextRequest
    | LineRequest
    | RulerRequest
)
_OVERLAY_REQUEST_TYPES = (
    GridRequest,
    CircleRequest,
    RectRequest,
    TextRequest,
    LineRequest,
    RulerRequest,
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


class OverlayEntry(NamedTuple):
    """One immutable generic overlay and the authorities it depends on."""

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


class OverlayRegistry:
    """Keep session-local generic overlays in deterministic insertion order."""

    def __init__(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
        *,
        id_factory: Callable[[], uuid.UUID | str] | None = None,
    ) -> None:
        _validate_projector_output_descriptor(projector_output_descriptor)
        if id_factory is not None and not callable(id_factory):
            raise ValueError('id_factory must be callable')
        self._projector_output_descriptor = projector_output_descriptor
        self._id_factory = id_factory
        self._entries: dict[uuid.UUID, OverlayEntry] = {}
        self._names: dict[str, uuid.UUID] = {}
        self._next_insertion_sequence = 0
        self._lock = threading.RLock()

    def create(
        self,
        request: AnyOverlayRequest,
        materialised_primitives: ProjectorMaterialisation,
    ) -> OverlayEntry:
        """Add one request only after its complete materialisation has succeeded."""
        if not isinstance(request, _OVERLAY_REQUEST_TYPES):
            raise ValueError('request must be an overlay request')
        if not isinstance(materialised_primitives, ProjectorMaterialisation):
            raise ValueError('materialised_primitives must be ProjectorMaterialisation')
        _validate_projector_materialisation(materialised_primitives)
        camera_dependencies, metric_dependency = get_overlay_dependencies(request)
        with self._lock:
            overlay_id = self._make_overlay_id(request.id)
            if overlay_id in self._entries:
                raise ValueError(f'Overlay id {overlay_id} is already in use')
            if request.name is not None and request.name in self._names:
                raise ValueError(f'Overlay name {request.name!r} is already in use')
            if overlay_id != request.id:
                request = request.model_copy(update={'id': overlay_id})
            entry = OverlayEntry(
                overlay_id,
                request.name,
                request.kind,
                request.visible,
                request,
                materialised_primitives,
                camera_dependencies,
                metric_dependency,
                self._projector_output_descriptor,
                self._next_insertion_sequence,
            )
            self._entries[overlay_id] = entry
            if entry.name is not None:
                self._names[entry.name] = overlay_id
            self._next_insertion_sequence += 1
            return entry

    def list(self) -> list[OverlayEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, selector: str | uuid.UUID) -> OverlayEntry:
        with self._lock:
            return self._entries[self._resolve_selector(selector)]

    def show(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_visibility(selector, True)

    def hide(self, selector: str | uuid.UUID) -> OverlayEntry:
        return self._set_visibility(selector, False)

    def remove(self, selector: str | uuid.UUID) -> OverlayEntry:
        with self._lock:
            overlay_id = self._resolve_selector(selector)
            entry = self._entries.pop(overlay_id)
            if entry.name is not None:
                del self._names[entry.name]
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._names.clear()

    def invalidate_camera(self, camera_id: str) -> None:
        if not isinstance(camera_id, str) or len(camera_id) == 0:
            raise ValueError('camera_id must be a non-empty string')
        with self._lock:
            self._remove_matching(
                lambda entry: camera_id in entry.camera_dependencies,
            )

    def invalidate_metric(self) -> None:
        with self._lock:
            self._remove_matching(lambda entry: entry.metric_dependency)

    def invalidate_projector_output(
        self,
        projector_output_descriptor: ProjectorOutputDescriptor,
    ) -> None:
        _validate_projector_output_descriptor(projector_output_descriptor)
        with self._lock:
            if projector_output_descriptor == self._projector_output_descriptor:
                return
            self._projector_output_descriptor = projector_output_descriptor
            self._remove_matching(lambda _entry: True)

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
            updated_entry = entry._replace(visible=visible)
            self._entries[overlay_id] = updated_entry
            return updated_entry

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
                    raise OverlayNotFoundError(
                        f'Unknown overlay {selector!r}',
                    ) from None
        else:
            raise ValueError('overlay selector must be an id or name')
        if overlay_id not in self._entries:
            raise OverlayNotFoundError(
                f'Unknown overlay {selector!r}',
            )
        return overlay_id

    def _remove_matching(self, predicate: Callable[[OverlayEntry], bool]) -> None:
        removed_ids = [
            overlay_id
            for overlay_id, entry in self._entries.items()
            if predicate(entry)
        ]
        for overlay_id in removed_ids:
            entry = self._entries.pop(overlay_id)
            if entry.name is not None:
                del self._names[entry.name]


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
) -> tuple[PointReference, ...]:
    if isinstance(request, GridRequest):
        return (request.origin,)
    if isinstance(request, (CircleRequest, RectRequest)):
        return (request.centre,)
    if isinstance(request, TextRequest):
        return (request.position,)
    if isinstance(request, (LineRequest, RulerRequest)):
        return (request.start, request.end)
    raise ValueError('request must be an overlay request')


def get_overlay_dependencies(
    request: AnyOverlayRequest,
) -> tuple[tuple[str, ...], bool]:
    point_references = get_overlay_point_references(request)
    metric_dependency = any(
        point_reference.space == PointReferenceSpace.SURFACE_MM.value
        for point_reference in point_references
    )
    if isinstance(request, (GridRequest, CircleRequest, RectRequest)):
        metric_dependency = metric_dependency or (
            request.geometry_space == GeometrySpace.SURFACE_MM.value
        )
    elif isinstance(request, RulerRequest):
        metric_dependency = metric_dependency or (
            request.measurement_space == GeometrySpace.SURFACE_MM.value
        )

    camera_dependencies: list[str] = []
    for point_reference in point_references:
        camera_id = point_reference.camera
        if (
            point_reference.space == PointReferenceSpace.CAMERA_PX.value
            and camera_id is not None
            and camera_id not in camera_dependencies
        ):
            camera_dependencies.append(camera_id)
    return tuple(camera_dependencies), metric_dependency


def _validate_projector_output_descriptor(value: object) -> None:
    # Import lazily because configuration imports the overlay limits above.
    from multivision.config import ProjectorOutputDescriptor

    if not isinstance(value, ProjectorOutputDescriptor):
        raise ValueError('projector_output_descriptor is invalid')
    if not is_valid_resolution(value.projector_resolution):
        raise ValueError('projector_output_descriptor is invalid')


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


def build_circle(
    request: CircleRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> CircleGeometry:
    """Build a finite circle after converting its anchor to its source space."""
    if not isinstance(request, CircleRequest):
        raise ValueError('request must be a CircleRequest')
    centre = resolve_point_reference(
        request.centre,
        request.geometry_space,
        camera_to_projector,
        metric_calibration,
    )
    radius = request.radius.value
    return CircleGeometry(centre, radius, request.geometry_space, request.style)


def build_rotated_rect(
    request: RectRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> RectGeometry:
    """Build four rotated corners in the requested source space."""
    if not isinstance(request, RectRequest):
        raise ValueError('request must be a RectRequest')
    centre = resolve_point_reference(
        request.centre,
        request.geometry_space,
        camera_to_projector,
        metric_calibration,
    )
    width = request.width.value
    height = request.height.value
    half_width = width / 2.0
    half_height = height / 2.0
    corners = tuple(
        _rotate_source_offset(
            centre,
            x_offset,
            y_offset,
            request.angle_deg,
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
        request.angle_deg,
        request.geometry_space,
        request.style,
    )


def build_text(
    request: TextRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> TextGeometry:
    """Resolve one floating text anchor to projector-native coordinates."""
    if not isinstance(request, TextRequest):
        raise ValueError('request must be a TextRequest')
    position = resolve_point_reference(
        request.position,
        GeometrySpace.PROJECTOR_PX,
        camera_to_projector,
        metric_calibration,
    )
    return TextGeometry(
        position,
        request.text,
        request.angle_deg,
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
) -> GridGeometry:
    """Build a finite square grid with exact source-space spacing."""
    if not isinstance(request, GridRequest):
        raise ValueError('request must be a GridRequest')
    origin = resolve_point_reference(
        request.origin,
        request.geometry_space,
        camera_to_projector,
        metric_calibration,
    )
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
            _rotate_source_offset(origin, x_idx * spacing, 0.0, request.angle_deg),
            _rotate_source_offset(origin, x_idx * spacing, height, request.angle_deg),
        )
        for x_idx in range(vertical_count)
    )
    horizontal_segments = tuple(
        SourceSegment(
            _rotate_source_offset(origin, 0.0, y_idx * spacing, request.angle_deg),
            _rotate_source_offset(origin, width, y_idx * spacing, request.angle_deg),
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
        request.angle_deg,
        request.geometry_space,
        vertical_segments,
        horizontal_segments,
        request.style,
    )


def build_line(
    request: LineRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> LineGeometry:
    """Build a literal projector-native line from independently resolved endpoints."""
    if not isinstance(request, LineRequest):
        raise ValueError('request must be a LineRequest')
    start = resolve_point_reference(
        request.start,
        GeometrySpace.PROJECTOR_PX,
        camera_to_projector,
        metric_calibration,
    )
    end = resolve_point_reference(
        request.end,
        GeometrySpace.PROJECTOR_PX,
        camera_to_projector,
        metric_calibration,
    )
    return LineGeometry(start, end, request.label, request.style)


def build_ruler(
    request: RulerRequest,
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
) -> RulerGeometry:
    """Build a ruler in projector pixels or canonical surface millimetres."""
    if not isinstance(request, RulerRequest):
        raise ValueError('request must be a RulerRequest')
    start = resolve_point_reference(
        request.start,
        request.measurement_space,
        camera_to_projector,
        metric_calibration,
    )
    end = resolve_point_reference(
        request.end,
        request.measurement_space,
        camera_to_projector,
        metric_calibration,
    )
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
) -> ProjectorMaterialisation:
    """Sample and clip one circle in projector-native coordinates."""
    geometry = (
        build_circle(request_or_geometry, camera_to_projector, metric_calibration)
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
) -> ProjectorMaterialisation:
    """Project and clip all four edges of a rotated rectangle."""
    geometry = (
        build_rotated_rect(request_or_geometry, camera_to_projector, metric_calibration)
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
) -> ProjectorMaterialisation:
    """Resolve one floating text label in projector-native coordinates."""
    geometry = (
        build_text(request_or_geometry, camera_to_projector, metric_calibration)
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
) -> ProjectorMaterialisation:
    """Project and independently clip a finite grid's source-space segments."""
    geometry = (
        build_grid(
            request_or_geometry,
            camera_to_projector,
            metric_calibration,
            overlay_configuration,
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
) -> ProjectorMaterialisation:
    """Resolve, clip and optionally label a literal line segment."""
    geometry = (
        build_line(request_or_geometry, camera_to_projector, metric_calibration)
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


def materialise_ruler(
    request_or_geometry: RulerRequest | RulerGeometry,
    projector_resolution: Resolution | CoordinateBounds | Sequence[int],
    camera_to_projector: object | None = None,
    metric_calibration: object | None = None,
    overlay_configuration: OverlayConfiguration | None = None,
) -> ProjectorMaterialisation:
    """Build physical or projector-pixel ticks, then clip every segment."""
    geometry = (
        build_ruler(request_or_geometry, camera_to_projector, metric_calibration)
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
    }[request.kind]
    return materialise(
        request,
        projector_resolution,
        camera_to_projector,
        metric_calibration,
        overlay_configuration,
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
    if callable(is_usable) and is_usable() is not True:
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
    'MAX_OVERLAY_LABEL_SCALE',
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
    'PointSpace',
    'Quantity',
    'RectRequest',
    'RulerRequest',
    'TextRequest',
    'PHYSICAL_UNITS',
    'CircleGeometry',
    'GridGeometry',
    'ProjectorCoverageGridRequest',
    'LineGeometry',
    'RectGeometry',
    'RulerGeometry',
    'TextGeometry',
    'SourceSegment',
    'ProjectorLabel',
    'ProjectorMaterialisation',
    'ProjectorPolygon',
    'OverlayEntry',
    'OverlayRegistry',
    'ProjectorSegment',
    'get_overlay_dependencies',
    'get_overlay_point_references',
    'DEFAULT_CIRCLE_SAMPLE_COUNT',
    'build_circle',
    'build_grid',
    'build_projector_coverage_grid_request',
    'build_line',
    'build_rotated_rect',
    'build_ruler',
    'build_text',
    'materialise_circle',
    'materialise_grid',
    'materialise_line',
    'materialise_overlay',
    'materialise_rect',
    'materialise_ruler',
    'materialise_text',
    'resolve_point_reference',
]
