import threading
import uuid
from types import SimpleNamespace

import pytest

from multivision.config import ProjectorOutputDescriptor
from multivision.geometry import Point2D
from multivision.overlays import (
    CircleRequest,
    FiducialAnchor,
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


def _dynamic_line_request(
    *,
    name: str | None = None,
    overlay_id: uuid.UUID | None = None,
) -> LineRequest:
    data: dict[str, object] = {
        'start': FiducialAnchor(type='fiducial', group='cards', id=7),
        'end': {'type': 'projector', 'x': 10, 'y': 10, 'unit': 'px'},
    }
    if name is not None:
        data['name'] = name
    if overlay_id is not None:
        data['id'] = overlay_id
    return LineRequest(**data)


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
    remaining_entries = registry.list()
    assert len(remaining_entries) == 1, f'{remaining_entries=}'
    assert remaining_entries[0].request == projector_request
    assert remaining_entries[0].projector_output_descriptor == ProjectorOutputDescriptor(
        Resolution(120, 80),
        'projector-b',
    )


def test_registry_output_invalidation_discards_static_dependent_entries() -> None:
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
    registry.create(metric_request, _materialisation())
    registry.create(projector_request, _materialisation())

    registry.invalidate_projector_output(
        ProjectorOutputDescriptor(Resolution(120, 80), 'projector-b'),
    )

    entries = registry.list()
    assert [entry.request for entry in entries] == [projector_request]
    assert entries[0].projector_output_descriptor == ProjectorOutputDescriptor(
        Resolution(120, 80),
        'projector-b',
    )


def test_registry_output_invalidation_keeps_static_and_unresolves_dynamic_entries() -> None:
    registry = OverlayRegistry(_descriptor())
    static_entry = registry.create(_line_request(name='static'), _materialisation())
    dynamic_entry = registry.create_specification(
        _dynamic_line_request(name='dynamic'),
        lambda _request: None,
    )

    next_descriptor = ProjectorOutputDescriptor(Resolution(120, 80), 'projector-b')
    registry.invalidate_projector_output(next_descriptor)

    entries = registry.list()
    assert [entry.id for entry in entries] == [static_entry.id, dynamic_entry.id]
    assert entries[0].is_resolved, f'{entries=}'
    assert entries[0].projector_output_descriptor == next_descriptor
    assert entries[1].unresolved, f'{entries=}'
    assert entries[1].unresolved_reason == 'projector output changed'


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


def test_registry_batch_preserves_order_and_applies_operations_in_order() -> None:
    registry = OverlayRegistry(_descriptor())
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    first = _line_request(name='first', overlay_id=first_id)
    second = _line_request(name='second', overlay_id=second_id)
    replacement = _line_request(name='renamed', overlay_id=first_id)

    entries = registry.apply_batch(
        (
            {'op': 'create', 'request': first},
            {'op': 'create', 'request': second},
            {'op': 'update', 'selector': first_id, 'request': replacement},
            {'op': 'remove', 'selector': second_id},
        ),
        materialise_request=lambda _request: _materialisation(),
    )

    assert [entry.id for entry in entries] == [first_id]
    assert entries[0].name == 'renamed'
    assert entries[0].insertion_sequence == 0
    assert registry.generation == 1


def test_registry_batch_failure_has_no_partial_publication() -> None:
    registry = OverlayRegistry(_descriptor())
    original = registry.create(_line_request(name='original'), _materialisation())
    original_snapshot = registry.snapshot()
    original_generation = registry.generation

    with pytest.raises(ValueError, match='name'):
        registry.apply_batch(
            (
                {
                    'op': 'create',
                    'request': _line_request(name='temporary'),
                },
                {
                    'op': 'create',
                    'request': _line_request(name='original'),
                },
            ),
            materialise_request=lambda _request: _materialisation(),
        )

    assert registry.snapshot() == original_snapshot
    assert registry.generation == original_generation
    assert registry.get(original.id) == original


def test_registry_dynamic_specification_is_retained_until_resolution() -> None:
    registry = OverlayRegistry(_descriptor())
    request = _dynamic_line_request(name='marker-line')
    entry = registry.create_specification(request, lambda _request: None)

    assert entry.specification == request
    assert entry.unresolved
    assert registry.list()[0].specification == request
    materialised = registry.resolve(
        SimpleNamespace(generation=1),
        lambda _request, _state: _materialisation(),
    )[0]

    assert materialised.resolved
    assert registry.list()[0].unresolved
    assert registry.list()[0].specification == request


def test_registry_batch_limit_is_checked_before_candidate_work() -> None:
    registry = OverlayRegistry(_descriptor(), max_batch_operations=2)
    requests = tuple(_line_request(name=f'line-{idx}') for idx in range(3))
    callback_called = False

    def materialise(_request: object) -> ProjectorMaterialisation:
        nonlocal callback_called
        callback_called = True
        return _materialisation()

    with pytest.raises(ValueError, match='operation limit'):
        registry.apply_batch(
            tuple({'op': 'create', 'request': request} for request in requests),
            materialise_request=materialise,
        )

    assert not callback_called
    assert registry.list() == []


def test_registry_per_call_limit_cannot_bypass_configured_limit() -> None:
    registry = OverlayRegistry(_descriptor(), max_batch_operations=1)
    requests = (
        {'op': 'create', 'request': _line_request(name='first')},
        {'op': 'create', 'request': _line_request(name='second')},
    )

    with pytest.raises(ValueError, match='operation limit'):
        registry.apply_batch(
            requests,
            materialise_request=lambda _request: _materialisation(),
            max_operations=100,
        )

    assert registry.list() == []


def test_registry_readers_observe_only_old_or_new_batch_snapshots() -> None:
    registry = OverlayRegistry(_descriptor())
    old_entry = registry.create(_line_request(name='old'), _materialisation())
    started = threading.Event()
    release = threading.Event()
    published: list[OverlayEntry] = []

    def materialise(_request: object) -> ProjectorMaterialisation:
        started.set()
        assert release.wait(2)
        return _materialisation()

    def apply_batch() -> None:
        published.extend(registry.apply_batch(
            ({'op': 'create', 'request': _line_request(name='new')},),
            materialise_request=materialise,
        ))

    worker = threading.Thread(target=apply_batch)
    worker.start()
    assert started.wait(2)
    assert [entry.name for entry in registry.snapshot()] == [old_entry.name]
    release.set()
    worker.join(2)

    assert not worker.is_alive()
    assert [entry.name for entry in registry.snapshot()] == ['old', 'new']
    assert [entry.name for entry in published] == ['old', 'new']
