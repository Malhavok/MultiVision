import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from multivision.config import Configuration
from multivision.errors import ConfigurationError
from multivision.geometry import (
    CoordinateBounds,
    Point2D,
    invert_homography,
    project_point,
)
from multivision.overlays import (
    CircleRequest,
    GridRequest,
    LineRequest,
    ProjectorCoverageGridRequest,
    OverlayConfiguration,
    OverlayStyle,
    PointReference,
    ProjectorLabel,
    ProjectorMaterialisation,
    Quantity,
    RectRequest,
    RulerRequest,
    TextRequest,
    build_circle,
    build_grid,
    build_line,
    build_projector_coverage_grid_request,
    build_rotated_rect,
    build_ruler,
    materialise_circle,
    materialise_grid,
    materialise_line,
    materialise_rect,
    materialise_ruler,
    materialise_text,
    resolve_point_reference,
)
from multivision.types import Resolution


def test_rectangle_and_floating_text_materialise_rotated_scaled_labels() -> None:
    bounds = CoordinateBounds(0, 0, 200, 200)
    rectangle = RectRequest(
        centre={'space': 'projector_px', 'x': 100, 'y': 80},
        geometry_space='projector_px',
        width={'value': 40, 'unit': 'px'},
        height={'value': 20, 'unit': 'px'},
        label='card-1',
        label_angle_deg=15,
        label_scale=1.5,
    )
    rectangle_materialisation = materialise_rect(rectangle, bounds)
    assert rectangle_materialisation.labels[0] == ProjectorLabel(
        Point2D(100, 80),
        'card-1',
        rectangle.style,
        15.0,
        1.5,
    ), f'{rectangle_materialisation=}'

    text = TextRequest(
        position={'space': 'projector_px', 'x': 25, 'y': 30},
        text='floating',
        angle_deg=-20,
        scale=2,
    )
    text_materialisation = materialise_text(text, bounds)
    assert text_materialisation.labels == (
        ProjectorLabel(Point2D(25, 30), 'floating', text.style, -20.0, 2.0),
    ), f'{text_materialisation=}'


    configuration = Configuration.from_data(
        {
            'overlay_limits': {
                'max_overlay_vertices': 12,
                'max_overlay_segments': 8,
                'max_overlay_ticks': 4,
                'max_overlay_label_characters': 16,
            },
        },
    )

    expected_limits = OverlayConfiguration(
        max_overlay_vertices=12,
        max_overlay_segments=8,
        max_overlay_ticks=4,
        max_overlay_label_characters=16,
    )
    assert configuration.overlay_limits == expected_limits, f'{configuration.overlay_limits=}'
    round_tripped_configuration = Configuration.from_data(configuration.to_data())
    assert round_tripped_configuration == configuration, f'{round_tripped_configuration=}'


@pytest.mark.parametrize('invalid_value', [True, 0, -1, 1.5])
def test_overlay_configuration_rejects_invalid_limits(invalid_value: object) -> None:
    with pytest.raises(ConfigurationError):
        OverlayConfiguration(max_overlay_vertices=invalid_value)  # type: ignore[arg-type]


def test_point_reference_normalises_physical_units() -> None:
    point = PointReference(space='surface_mm', x=2, y=1.5, unit='cm')

    assert (point.x, point.y, point.unit) == (20.0, 15.0, 'mm'), f'{point=}'


def test_point_reference_requires_explicit_camera_identity() -> None:
    with pytest.raises(ValidationError):
        PointReference(space='camera_px', x=1, y=2)

    with pytest.raises(ValidationError):
        PointReference(
            space='projector_px',
            x=1,
            y=2,
            camera='camera-1',
        )


def test_quantities_and_geometry_spaces_must_agree() -> None:
    quantity = Quantity(value=2, unit='in')
    assert (quantity.value, quantity.unit, quantity.to_mm()) == (50.8, 'mm', 50.8), f'{quantity=}'

    grid = GridRequest(
        origin={'space': 'surface_mm', 'x': 1, 'y': 2},
        geometry_space='surface_mm',
        spacing={'value': 1, 'unit': 'cm'},
        extent={
            'width': {'value': 10, 'unit': 'cm'},
            'height': {'value': 5, 'unit': 'cm'},
        },
        angle_deg=15,
    )
    assert grid.spacing.value == 10.0, f'{grid=}'

    rectangle = RectRequest(
        centre={'space': 'projector_px', 'x': 10, 'y': 20},
        geometry_space='projector_px',
        width={'value': 30, 'unit': 'px'},
        height={'value': 15, 'unit': 'px'},
        angle_deg=-15,
    )
    assert rectangle.angle_deg == -15.0, f'{rectangle=}'

    with pytest.raises(ValidationError):
        CircleRequest(
            centre={'space': 'projector_px', 'x': 0, 'y': 0},
            geometry_space='projector_px',
            radius={'value': 1, 'unit': 'cm'},
        )


def test_styles_and_labels_are_strict() -> None:
    style = OverlayStyle(colour='#12aBef', line_width_px=2)
    assert style.colour == (18, 171, 239), f'{style=}'

    with pytest.raises(ValidationError):
        LineRequest(
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 1, 'y': 1},
            style={'fill': True},
        )

    with pytest.raises(ValidationError):
        LineRequest(
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 1, 'y': 1},
            label='x' * 257,
        )


def test_requests_reject_unknown_fields_and_invalid_uuid_versions() -> None:
    with pytest.raises(ValidationError):
        RulerRequest(
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 1, 'y': 1},
            measurement_space='projector_px',
            unit='px',
            unexpected=True,
        )

    with pytest.raises(ValidationError):
        LineRequest(
            id=uuid.uuid1(),
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 1, 'y': 1},
        )

    with pytest.raises(ValidationError):
        LineRequest(
            name=' ',
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 1, 'y': 1},
        )


def test_resolves_mixed_points_through_existing_homographies() -> None:
    camera_to_projector = (
        (1.0, 0.0, 100.0),
        (0.0, 1.0, 50.0),
        (0.0, 0.0, 1.0),
    )
    projector_to_surface = (
        (2.0, 0.1, 10.0),
        (0.2, 3.0, 20.0),
        (0.001, 0.002, 1.0),
    )
    surface_to_projector = invert_homography(projector_to_surface)
    metric_calibration = SimpleNamespace(
        state='CALIBRATED',
        projector_to_surface=projector_to_surface,
        surface_to_projector=surface_to_projector,
    )
    camera_point = PointReference(
        space='camera_px',
        camera='camera-1',
        x=10,
        y=20,
    )
    projector_point = project_point(
        Point2D(camera_point.x, camera_point.y),
        camera_to_projector,
    )
    expected_surface_point = project_point(projector_point, projector_to_surface)

    actual_surface_point = resolve_point_reference(
        camera_point,
        'surface_mm',
        {'camera-1': camera_to_projector},
        metric_calibration,
    )
    actual_projector_point = resolve_point_reference(
        PointReference(
            space='surface_mm',
            x=expected_surface_point.x,
            y=expected_surface_point.y,
        ),
        'projector_px',
        metric_calibration=metric_calibration,
    )

    assert actual_surface_point == expected_surface_point, f'{actual_surface_point=}'
    assert actual_projector_point.x == pytest.approx(
        projector_point.x,
    ), f'{actual_projector_point=}'
    assert actual_projector_point.y == pytest.approx(
        projector_point.y,
    ), f'{actual_projector_point=}'


def test_builds_source_shapes_without_pixel_scale_approximations() -> None:
    projector_to_surface = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.001, 0.0, 1.0),
    )
    metric_calibration = SimpleNamespace(
        state='CALIBRATED',
        projector_to_surface=projector_to_surface,
        surface_to_projector=invert_homography(projector_to_surface),
    )
    camera_to_projector = {
        'camera-1': ((1.0, 0.0, 10.0), (0.0, 1.0, 20.0), (0.0, 0.0, 1.0)),
    }

    circle = build_circle(
        CircleRequest(
            centre={'space': 'camera_px', 'camera': 'camera-1', 'x': 5, 'y': 7},
            geometry_space='surface_mm',
            radius={'value': 1, 'unit': 'in'},
        ),
        camera_to_projector,
        metric_calibration,
    )
    assert circle.centre == project_point(
        (15, 27),
        metric_calibration.projector_to_surface,
    )
    assert circle.radius == 25.4

    rectangle = build_rotated_rect(
        RectRequest(
            centre={'space': 'surface_mm', 'x': 50, 'y': 60},
            geometry_space='surface_mm',
            width={'value': 20, 'unit': 'mm'},
            height={'value': 10, 'unit': 'mm'},
            angle_deg=90,
        ),
    )
    assert rectangle.corners == (
        rectangle.centre._replace(x=45, y=70),
        rectangle.centre._replace(x=45, y=50),
        rectangle.centre._replace(x=55, y=50),
        rectangle.centre._replace(x=55, y=70),
    ), f'{rectangle.corners=}'

    grid = build_grid(
        GridRequest(
            origin={'space': 'projector_px', 'x': 100, 'y': 200},
            geometry_space='projector_px',
            spacing={'value': 10, 'unit': 'px'},
            extent={
                'width': {'value': 20, 'unit': 'px'},
                'height': {'value': 10, 'unit': 'px'},
            },
        ),
    )
    assert len(grid.segments) == 5, f'{grid.segments=}'
    assert grid.vertical_segments[1].start == (110, 200), f'{grid.vertical_segments=}'
    assert grid.horizontal_segments[1].end == (120, 210), f'{grid.horizontal_segments=}'

    line = build_line(
        LineRequest(
            start={'space': 'projector_px', 'x': 1, 'y': 2},
            end={'space': 'surface_mm', 'x': 3, 'y': 4},
        ),
        metric_calibration=metric_calibration,
    )
    assert line.start == (1, 2), f'{line=}'
    assert line.end == project_point(
        (3, 4),
        metric_calibration.surface_to_projector,
    ), f'{line=}'

    ruler = build_ruler(
        RulerRequest(
            start={'space': 'projector_px', 'x': 0, 'y': 0},
            end={'space': 'projector_px', 'x': 3, 'y': 4},
            measurement_space='projector_px',
            unit='px',
        ),
    )
    assert ruler.length == 5.0, f'{ruler=}'
    assert ruler.length_mm is None, f'{ruler=}'


def test_projector_coverage_grid_derives_a_finite_surface_extent() -> None:
    request = ProjectorCoverageGridRequest(
        name='coverage',
        spacing={'value': 35, 'unit': 'mm'},
    )
    metric_calibration = SimpleNamespace(
        projector_to_surface=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        surface_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    grid_request = build_projector_coverage_grid_request(
        request,
        metric_calibration,
        Resolution(100, 80),
    )

    assert grid_request.origin.x == -20.0, f'{grid_request=}'
    assert grid_request.origin.y == -30.0, f'{grid_request=}'
    assert grid_request.extent.width.value == 140.0, f'{grid_request=}'
    assert grid_request.extent.height.value == 140.0, f'{grid_request=}'
    assert grid_request.spacing.value == 35.0, f'{grid_request=}'


def test_grid_generation_honours_segment_and_vertex_budgets() -> None:
    request = GridRequest(
        origin={'space': 'projector_px', 'x': 0, 'y': 0},
        geometry_space='projector_px',
        spacing={'value': 10, 'unit': 'px'},
        extent={
            'width': {'value': 20, 'unit': 'px'},
            'height': {'value': 10, 'unit': 'px'},
        },
    )

    with pytest.raises(ValueError, match='segment budget'):
        build_grid(
            request,
            overlay_configuration=OverlayConfiguration(max_overlay_segments=4),
        )
    with pytest.raises(ValueError, match='vertex budget'):
        build_grid(
            request,
            overlay_configuration=OverlayConfiguration(max_overlay_vertices=9),
        )


def test_materialises_circles_before_projective_transformation() -> None:
    surface_to_projector = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.001, 0.0, 1.0),
    )
    metric_calibration = SimpleNamespace(
        state='CALIBRATED',
        projector_to_surface=invert_homography(surface_to_projector),
        surface_to_projector=surface_to_projector,
    )
    request = CircleRequest(
        centre={'space': 'surface_mm', 'x': 100, 'y': 100},
        geometry_space='surface_mm',
        radius={'value': 20, 'unit': 'mm'},
    )

    materialisation = materialise_circle(
        request,
        Resolution(500, 500),
        metric_calibration=metric_calibration,
    )

    assert isinstance(materialisation, ProjectorMaterialisation), f'{materialisation=}'
    assert len(materialisation.segments) == 64, f'{len(materialisation.segments)=}'
    expected_first_point = project_point((120, 100), surface_to_projector)
    assert materialisation.segments[0].start == Point2D(
        round(expected_first_point.x),
        round(expected_first_point.y),
    ), f'{materialisation.segments[0]=}'


def test_clipping_keeps_outline_edges_independent_and_offscreen_geometry_empty() -> None:
    crossing_outline = RectRequest(
        centre={'space': 'projector_px', 'x': 2, 'y': 40},
        geometry_space='projector_px',
        width={'value': 20, 'unit': 'px'},
        height={'value': 10, 'unit': 'px'},
    )
    offscreen_line = LineRequest(
        start={'space': 'projector_px', 'x': -20, 'y': 5},
        end={'space': 'projector_px', 'x': -10, 'y': 15},
    )

    crossing_materialisation = materialise_rect(
        crossing_outline,
        Resolution(100, 80),
    )
    offscreen_materialisation = materialise_line(
        offscreen_line,
        Resolution(100, 80),
    )

    assert len(crossing_materialisation.segments) == 3, (
        f'{crossing_materialisation=}'
    )
    assert all(
        not (
            segment.start.x == segment.end.x == 0
            and segment.start.y != segment.end.y
        )
        for segment in crossing_materialisation.segments
    ), f'{crossing_materialisation=}'
    assert offscreen_materialisation == ProjectorMaterialisation(), (
        f'{offscreen_materialisation=}'
    )


def test_clips_filled_shapes_but_does_not_close_outline_at_projector_boundary() -> None:
    filled_rect = RectRequest(
        centre={'space': 'projector_px', 'x': 2, 'y': 2},
        geometry_space='projector_px',
        width={'value': 10, 'unit': 'px'},
        height={'value': 10, 'unit': 'px'},
        style={'fill': True},
    )
    enclosing_outline = RectRequest(
        centre={'space': 'projector_px', 'x': 50, 'y': 40},
        geometry_space='projector_px',
        width={'value': 200, 'unit': 'px'},
        height={'value': 200, 'unit': 'px'},
    )

    filled_materialisation = materialise_rect(filled_rect, Resolution(100, 80))
    outline_materialisation = materialise_rect(enclosing_outline, Resolution(100, 80))

    assert len(filled_materialisation.polygons) == 1, f'{filled_materialisation=}'
    assert all(
        0 <= coordinate < limit
        for point in filled_materialisation.polygons[0].points
        for coordinate, limit in ((point.x, 100), (point.y, 80))
    ), f'{filled_materialisation=}'
    assert outline_materialisation == ProjectorMaterialisation(), f'{outline_materialisation=}'


def test_materialises_grid_and_mixed_line_endpoints() -> None:
    grid = GridRequest(
        origin={'space': 'projector_px', 'x': 10, 'y': 10},
        geometry_space='projector_px',
        spacing={'value': 10, 'unit': 'px'},
        extent={
            'width': {'value': 20, 'unit': 'px'},
            'height': {'value': 20, 'unit': 'px'},
        },
    )
    grid_materialisation = materialise_grid(grid, Resolution(40, 40))
    assert len(grid_materialisation.segments) == 6, f'{grid_materialisation=}'
    assert all(
        0 <= coordinate < limit
        for segment in grid_materialisation.segments
        for point in (segment.start, segment.end)
        for coordinate, limit in ((point.x, 40), (point.y, 40))
    ), f'{grid_materialisation=}'

    metric_calibration = SimpleNamespace(
        state='CALIBRATED',
        projector_to_surface=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        surface_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    line = LineRequest(
        start={'space': 'projector_px', 'x': -10, 'y': 5},
        end={'space': 'surface_mm', 'x': 20, 'y': 5},
    )
    line_materialisation = materialise_line(
        line,
        Resolution(40, 40),
        metric_calibration=metric_calibration,
    )
    assert line_materialisation.segments[0].start == Point2D(0, 5), (
        f'{line_materialisation=}'
    )
    assert line_materialisation.segments[0].end == Point2D(20, 5), (
        f'{line_materialisation=}'
    )


def test_materialisation_honours_fixed_shape_budgets() -> None:
    circle = CircleRequest(
        centre={'space': 'projector_px', 'x': 10, 'y': 10},
        geometry_space='projector_px',
        radius={'value': 5, 'unit': 'px'},
    )
    with pytest.raises(ValueError, match='vertex budget'):
        materialise_circle(
            circle,
            Resolution(40, 40),
            overlay_configuration=OverlayConfiguration(max_overlay_vertices=127),
        )

    line = LineRequest(
        start={'space': 'projector_px', 'x': 0, 'y': 0},
        end={'space': 'projector_px', 'x': 10, 'y': 10},
    )
    with pytest.raises(ValueError, match='primitive budget'):
        materialise_line(
            line,
            Resolution(40, 40),
            overlay_configuration=OverlayConfiguration(max_overlay_vertices=1),
        )

    rectangle = RectRequest(
        centre={'space': 'projector_px', 'x': 10, 'y': 10},
        geometry_space='projector_px',
        width={'value': 10, 'unit': 'px'},
        height={'value': 10, 'unit': 'px'},
    )
    with pytest.raises(ValueError, match='primitive budget'):
        materialise_rect(
            rectangle,
            Resolution(40, 40),
            overlay_configuration=OverlayConfiguration(max_overlay_segments=3),
        )


def test_materialisation_rejects_source_geometry_crossing_projective_horizon() -> None:
    surface_to_projector = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
    )
    metric_calibration = SimpleNamespace(
        state='CALIBRATED',
        projector_to_surface=invert_homography(surface_to_projector),
        surface_to_projector=surface_to_projector,
    )
    rectangle = RectRequest(
        centre={'space': 'surface_mm', 'x': 0, 'y': 0},
        geometry_space='surface_mm',
        width={'value': 4, 'unit': 'mm'},
        height={'value': 2, 'unit': 'mm'},
    )
    grid = GridRequest(
        origin={'space': 'surface_mm', 'x': -2, 'y': 0},
        geometry_space='surface_mm',
        spacing={'value': 2, 'unit': 'mm'},
        extent={
            'width': {'value': 4, 'unit': 'mm'},
            'height': {'value': 2, 'unit': 'mm'},
        },
    )
    ruler = RulerRequest(
        start={'space': 'surface_mm', 'x': -2, 'y': 0},
        end={'space': 'surface_mm', 'x': 2, 'y': 0},
        measurement_space='surface_mm',
        unit='mm',
    )

    for materialise, request in (
        (materialise_rect, rectangle),
        (materialise_grid, grid),
        (materialise_ruler, ruler),
    ):
        with pytest.raises(ValueError, match='projective horizon'):
            materialise(
                request,
                Resolution(100, 100),
                metric_calibration=metric_calibration,
            )


def test_materialises_bounded_projector_ruler_ticks() -> None:
    request = RulerRequest(
        start={'space': 'projector_px', 'x': 10, 'y': 10},
        end={'space': 'projector_px', 'x': 90, 'y': 10},
        measurement_space='projector_px',
        unit='px',
    )

    materialisation = materialise_ruler(
        request,
        Resolution(100, 80),
        overlay_configuration=OverlayConfiguration(max_overlay_ticks=3),
    )

    assert len(materialisation.segments) == 4, f'{materialisation=}'
    assert materialisation.labels[0].text == '80.0 px', f'{materialisation.labels=}'


def test_ruler_budget_is_checked_before_tick_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = RulerRequest(
        start={'space': 'projector_px', 'x': 10, 'y': 10},
        end={'space': 'projector_px', 'x': 90, 'y': 10},
        measurement_space='projector_px',
        unit='px',
    )

    def fail_if_ticks_are_built(*_args: object, **_kwargs: object) -> None:
        raise AssertionError('ticks were allocated before the budget was checked')

    monkeypatch.setattr(
        'multivision.overlays._build_ruler_ticks',
        fail_if_ticks_are_built,
    )
    with pytest.raises(ValueError, match='segment budget'):
        materialise_ruler(
            request,
            Resolution(100, 80),
            overlay_configuration=OverlayConfiguration(max_overlay_segments=1),
        )


def test_rejects_unusable_calibration_authorities() -> None:
    class CameraAuthority:
        def get_status(self, _camera_id: str) -> str:
            return 'STALE'

        def get_record(self, _camera_id: str) -> object:
            return ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    with pytest.raises(ValueError):
        resolve_point_reference(
            PointReference(space='camera_px', camera='camera-1', x=1, y=2),
            'projector_px',
            CameraAuthority(),
        )

    class MetricAuthority:
        def is_usable(self) -> bool:
            return False

        projector_to_surface = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    with pytest.raises(ValueError):
        resolve_point_reference(
            PointReference(space='surface_mm', x=1, y=2),
            'projector_px',
            metric_calibration=MetricAuthority(),
        )
