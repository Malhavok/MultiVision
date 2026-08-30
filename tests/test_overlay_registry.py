import uuid
from types import SimpleNamespace

import pytest

from multivision.config import ProjectorOutputDescriptor
from multivision.geometry import Point2D
from multivision.overlays import (
    CircleRequest,
    LineRequest,
    OverlayRegistry,
    ProjectorMaterialisation,
    ProjectorSegment,
)
from multivision.types import Resolution


def _descriptor() -> ProjectorOutputDescriptor:
    return ProjectorOutputDescriptor(Resolution(100, 80), 'projector-a')


def _line_request(
    *,
    name: str | None = None,
    overlay_id: uuid.UUID | None = None,
) -> LineRequest:
    data: dict[str, object] = {
        'start': {'space': 'projector_px', 'x': 1, 'y': 1},
        'end': {'space': 'projector_px', 'x': 10, 'y': 10},
    }
    if name is not None:
        data['name'] = name
    if overlay_id is not None:
        data['id'] = overlay_id
    return LineRequest(**data)


def _materialisation() -> ProjectorMaterialisation:
    return ProjectorMaterialisation(
        segments=(
            ProjectorSegment(
                Point2D(1, 1),
                Point2D(10, 10),
                _line_request().style,
            ),
        ),
    )


def test_registry_requires_the_projector_descriptor_type() -> None:
    with pytest.raises(ValueError, match='descriptor'):
        OverlayRegistry(SimpleNamespace(
            projector_resolution=Resolution(100, 80),
            output_identity='projector-a',
        ))  # type: ignore[arg-type]


def test_registry_preserves_identity_order_and_visibility() -> None:
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    registry = OverlayRegistry(_descriptor())

    first = registry.create(
        _line_request(name='first', overlay_id=first_id),
        _materialisation(),
    )
    second = registry.create(
        _line_request(name='second', overlay_id=second_id),
        _materialisation(),
    )

    assert [entry.id for entry in registry.list()] == [first.id, second.id]
    assert registry.hide('first') == registry.hide(first.id)
    assert registry.get('first').visible is False
    assert registry.show(str(first.id)).visible is True
    assert registry.get('first').insertion_sequence < registry.get('second').insertion_sequence


def test_registry_rejects_mutable_materialised_primitives() -> None:
    registry = OverlayRegistry(_descriptor())
    request = _line_request()
    segments = list(_materialisation().segments)

    with pytest.raises(ValueError, match='immutable'):
        registry.create(
            request,
            ProjectorMaterialisation(segments=segments),
        )
    assert registry.list() == []


def test_registry_rejects_implicit_replacement() -> None:
    registry = OverlayRegistry(_descriptor())
    first_id = uuid.uuid4()
    registry.create(_line_request(name='same', overlay_id=first_id), _materialisation())

    with pytest.raises(ValueError, match='name'):
        registry.create(_line_request(name='same'), _materialisation())
    with pytest.raises(ValueError, match='id'):
        registry.create(_line_request(overlay_id=first_id), _materialisation())
    assert len(registry.list()) == 1


def test_registry_invalidates_only_declared_dependencies() -> None:
    registry = OverlayRegistry(_descriptor())
    camera_request = LineRequest(
        start={'space': 'camera_px', 'camera': 'camera-0', 'x': 1, 'y': 1},
        end={'space': 'projector_px', 'x': 10, 'y': 10},
    )
    metric_request = CircleRequest(
        centre={'space': 'projector_px', 'x': 20, 'y': 20},
        geometry_space='surface_mm',
        radius={'value': 10, 'unit': 'mm'},
    )
    projector_request = _line_request()
    registry.create(camera_request, _materialisation())
    registry.create(metric_request, ProjectorMaterialisation())
    registry.create(projector_request, _materialisation())

    registry.invalidate_camera('camera-0')
    assert len(registry.list()) == 2
    registry.invalidate_metric()
    assert len(registry.list()) == 1
    registry.invalidate_projector_output(
        ProjectorOutputDescriptor(Resolution(120, 80), 'projector-b'),
    )
    assert registry.list() == []


def test_registry_does_not_invalidate_without_an_output_transition() -> None:
    registry = OverlayRegistry(_descriptor())
    entry = registry.create(_line_request(name='persistent'), _materialisation())

    registry.invalidate_projector_output(_descriptor())

    assert registry.list() == [entry]


def test_registry_remove_and_clear_are_independent_of_legacy_state() -> None:
    registry = OverlayRegistry(_descriptor())
    entry = registry.create(_line_request(name='temporary'), _materialisation())

    assert registry.remove(entry.id) == entry
    assert registry.list() == []
    registry.create(_line_request(name='temporary'), _materialisation())
    registry.clear()
    assert registry.list() == []
