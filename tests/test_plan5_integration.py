import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.config import Configuration, ProjectorOutputDescriptor
from multivision.display import ProjectorRenderer
from multivision.geometry import Point2D, project_point
from multivision.metric import (
    MetricCalibrationMetrics,
    MetricCalibrationResult,
    MetricCalibrationStatus,
    MetricHomographyPair,
)
from multivision.metric_target import METRIC_TARGET
from multivision.overlays import (
    CircleRequest,
    GridRequest,
    LineRequest,
    OverlayConfiguration,
    OverlayStyle,
    ProjectorMaterialisation,
    ProjectorPolygon,
    ProjectorSegment,
    RectRequest,
    RulerRequest,
    materialise_circle,
    materialise_grid,
    materialise_rect,
    materialise_ruler,
)
from multivision.service import RedCircleOverlay
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
)


PROJECTOR_RESOLUTION = Resolution(400, 300)
SURFACE_TO_PROJECTOR = (
    (2.4, 0.35, 55.0),
    (0.2, 2.1, 42.0),
    (0.0011, 0.0008, 1.0),
)
CAMERA_TO_PROJECTOR = (
    (1.0, 0.08, 12.0),
    (0.03, 0.98, 9.0),
    (0.0002, 0.0001, 1.0),
)


class LegacyPointService:
    def __init__(self) -> None:
        self.overlay = RedCircleOverlay(
            'camera-0',
            'camera-0',
            Point2D(10, 20),
            Point2D(22, 30),
        )
        self.projector_resolution = PROJECTOR_RESOLUTION
        self.projector_output_descriptor = None

    def point_from_camera(self, _camera: str, _point: object) -> RedCircleOverlay:
        return self.overlay

    def clear_overlay(self) -> None:
        self.overlay = None

    def clear_overlay_for_camera(self, _camera: str) -> None:
        self.overlay = None

    def rename_overlay_camera(self, _camera: str, _name: str) -> None:
        return None


class OverlayRuntime:
    def __init__(self) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    'camera-device-0',
                    'Camera 0',
                    0,
                    PROJECTOR_RESOLUTION,
                ),
            ],
        )
        self.frame_counter = 0

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        runtime_status = (
            RuntimeStatus.AVAILABLE
            if camera.state.value == 'OPEN'
            else RuntimeStatus.UNAVAILABLE
        )
        return CameraStatus(
            slot_id,
            None,
            runtime_status,
            camera.calibration_status,
            PROJECTOR_RESOLUTION,
        )

    def get_statuses(self) -> list[CameraStatus]:
        return [self.get_status(camera.slot_id) for camera in self.registry.get_cameras()]

    def set_calibration(
        self,
        slot_id: str,
        calibration_status: CalibrationStatus,
        calibration: object,
    ) -> object:
        return self.registry.set_calibration(slot_id, calibration_status, calibration)

    def mark_calibrations_stale(self, _descriptor: object) -> list[object]:
        return []

    def close_camera(self, slot_id: str) -> object:
        return self.registry.close(slot_id)

    def open_camera(self, slot_id: str) -> object:
        return self.registry.open(slot_id)

    def rename_camera(self, slot_id: str, name: str) -> object:
        return self.registry.rename(slot_id, name)

    def snapshot(self, slot_id: str) -> Frame:
        self.frame_counter += 1
        return Frame(f'frame-{slot_id}', self.frame_counter, 0.0)


def _camera_calibration() -> SimpleNamespace:
    return SimpleNamespace(
        camera_to_projector=CAMERA_TO_PROJECTOR,
        projector_output_descriptor=Configuration(
            projector_resolution=PROJECTOR_RESOLUTION,
        ).projector_output_descriptor,
        camera_resolution=PROJECTOR_RESOLUTION,
        version=1,
        timestamp=1.0,
    )


def _metric_result() -> MetricCalibrationResult:
    return MetricCalibrationResult(
        MetricHomographyPair.from_surface_to_projector(SURFACE_TO_PROJECTOR),
        MetricCalibrationMetrics(4, 16, 16, 1.0, 0.0, 0.0, 1.0),
        PROJECTOR_RESOLUTION,
        METRIC_TARGET.format_name,
        METRIC_TARGET.format_version,
        METRIC_TARGET.marker_family,
    )


def _service(
    *,
    overlay_limits: OverlayConfiguration | None = None,
) -> tuple[MultiVisionService, OverlayRuntime]:
    runtime = OverlayRuntime()
    configuration = Configuration(
        projector_resolution=PROJECTOR_RESOLUTION,
        overlay_limits=overlay_limits or OverlayConfiguration(),
    )
    service = MultiVisionService(
        configuration,
        camera_runtime=runtime,  # type: ignore[arg-type]
        point_service=LegacyPointService(),  # type: ignore[arg-type]
    )
    runtime.registry.set_calibration(
        'camera-0',
        CalibrationStatus.CALIBRATED,
        _camera_calibration(),
    )
    service.metric_registry.register(
        _metric_result(),
        configuration.projector_output_descriptor,
        observation_camera_slot='camera-0',
    )
    return service, runtime


def test_plan5_projector_only_service_does_not_require_calibration() -> None:
    service = MultiVisionService(
        Configuration(projector_resolution=PROJECTOR_RESOLUTION),
    )

    entry = service.create_overlay(
        LineRequest(
            start={'space': 'projector_px', 'x': 10, 'y': 10},
            end={'space': 'projector_px', 'x': 100, 'y': 80},
        ),
    )

    assert entry.camera_dependencies == (), f'{entry=}'
    assert not entry.metric_dependency, f'{entry=}'
    assert service.list_overlays() == [entry]


def test_plan5_pure_geometry_uses_skewed_projective_authorities() -> None:
    metric_calibration = SimpleNamespace(
        state=MetricCalibrationStatus.CALIBRATED,
        projector_to_surface=MetricHomographyPair.from_surface_to_projector(
            SURFACE_TO_PROJECTOR,
        ).projector_to_surface,
        surface_to_projector=SURFACE_TO_PROJECTOR,
    )
    circle_request = CircleRequest(
        centre={'space': 'surface_mm', 'x': 90, 'y': 65},
        geometry_space='surface_mm',
        radius={'value': 20, 'unit': 'mm'},
        style={'colour': '#123456', 'line_width_px': 3},
    )
    circle = materialise_circle(circle_request, PROJECTOR_RESOLUTION, metric_calibration=metric_calibration)

    assert len(circle.segments) == 64, f'{len(circle.segments)=}'
    expected_first = project_point(Point2D(110, 65), SURFACE_TO_PROJECTOR)
    assert circle.segments[0].start == Point2D(round(expected_first.x), round(expected_first.y))
    segment_lengths = {
        round(
            ((segment.end.x - segment.start.x) ** 2 + (segment.end.y - segment.start.y) ** 2) ** 0.5,
            3,
        )
        for segment in circle.segments
    }
    assert len(segment_lengths) > 1, f'{segment_lengths=}'

    rectangle = materialise_rect(
        RectRequest(
            centre={'space': 'surface_mm', 'x': 90, 'y': 65},
            geometry_space='surface_mm',
            width={'value': 30, 'unit': 'mm'},
            height={'value': 20, 'unit': 'mm'},
            angle_deg=27,
        ),
        PROJECTOR_RESOLUTION,
        metric_calibration=metric_calibration,
    )
    grid = materialise_grid(
        GridRequest(
            origin={'space': 'surface_mm', 'x': 60, 'y': 45},
            geometry_space='surface_mm',
            spacing={'value': 10, 'unit': 'mm'},
            extent={
                'width': {'value': 30, 'unit': 'mm'},
                'height': {'value': 20, 'unit': 'mm'},
            },
            angle_deg=-18,
        ),
        PROJECTOR_RESOLUTION,
        metric_calibration=metric_calibration,
    )
    ruler = materialise_ruler(
        RulerRequest(
            start={'space': 'surface_mm', 'x': 60, 'y': 60},
            end={'space': 'surface_mm', 'x': 160, 'y': 60},
            measurement_space='surface_mm',
            unit='cm',
        ),
        PROJECTOR_RESOLUTION,
        metric_calibration=metric_calibration,
    )

    assert len(rectangle.segments) == 4, f'{rectangle=}'
    assert len(grid.segments) == 7, f'{grid=}'
    assert ruler.labels[0].text == '10.0 cm', f'{ruler.labels=}'
    assert all(
        point.x == round(point.x) and point.y == round(point.y)
        for materialisation in (circle, rectangle, grid, ruler)
        for segment in materialisation.segments
        for point in (segment.start, segment.end)
    )


def test_plan5_camera_anchors_build_all_physical_shapes_in_surface_space() -> None:
    metric_calibration = SimpleNamespace(
        state=MetricCalibrationStatus.CALIBRATED,
        projector_to_surface=MetricHomographyPair.from_surface_to_projector(
            SURFACE_TO_PROJECTOR,
        ).projector_to_surface,
        surface_to_projector=SURFACE_TO_PROJECTOR,
    )
    camera_point = project_point(Point2D(40, 30), CAMERA_TO_PROJECTOR)
    surface_centre = project_point(
        camera_point,
        metric_calibration.projector_to_surface,
    )
    camera_anchor = {
        'space': 'camera_px',
        'camera': 'camera-0',
        'x': 40,
        'y': 30,
    }

    circle = materialise_circle(
        CircleRequest(
            centre=camera_anchor,
            geometry_space='surface_mm',
            radius={'value': 10, 'unit': 'mm'},
        ),
        PROJECTOR_RESOLUTION,
        CAMERA_TO_PROJECTOR,
        metric_calibration,
    )
    rectangle = materialise_rect(
        RectRequest(
            centre=camera_anchor,
            geometry_space='surface_mm',
            width={'value': 20, 'unit': 'mm'},
            height={'value': 10, 'unit': 'mm'},
        ),
        PROJECTOR_RESOLUTION,
        CAMERA_TO_PROJECTOR,
        metric_calibration,
    )
    grid = materialise_grid(
        GridRequest(
            origin=camera_anchor,
            geometry_space='surface_mm',
            spacing={'value': 10, 'unit': 'mm'},
            extent={
                'width': {'value': 20, 'unit': 'mm'},
                'height': {'value': 20, 'unit': 'mm'},
            },
        ),
        PROJECTOR_RESOLUTION,
        CAMERA_TO_PROJECTOR,
        metric_calibration,
    )

    expected_circle_start = project_point(
        Point2D(surface_centre.x + 10, surface_centre.y),
        SURFACE_TO_PROJECTOR,
    )
    expected_rectangle_corner = project_point(
        Point2D(surface_centre.x - 10, surface_centre.y - 5),
        SURFACE_TO_PROJECTOR,
    )
    expected_grid_origin = project_point(surface_centre, SURFACE_TO_PROJECTOR)
    assert circle.segments[0].start == Point2D(
        round(expected_circle_start.x),
        round(expected_circle_start.y),
    )
    assert rectangle.segments[0].start == Point2D(
        round(expected_rectangle_corner.x),
        round(expected_rectangle_corner.y),
    )
    assert grid.segments[0].start == Point2D(
        round(expected_grid_origin.x),
        round(expected_grid_origin.y),
    )


def test_plan5_service_is_atomic_and_invalidates_only_required_dependencies() -> None:
    service, runtime = _service(
        overlay_limits=OverlayConfiguration(max_overlay_segments=100),
    )
    projector_line = service.create_overlay(
        LineRequest(
            name='projector-only',
            start={'space': 'projector_px', 'x': 10, 'y': 10},
            end={'space': 'projector_px', 'x': 120, 'y': 80},
        ),
    )
    camera_line = service.create_overlay(
        LineRequest(
            name='camera-line',
            start={'space': 'camera_px', 'camera': 'camera-0', 'x': 20, 'y': 20},
            end={'space': 'projector_px', 'x': 120, 'y': 80},
        ),
    )
    physical_circle = service.create_overlay(
        CircleRequest(
            name='physical-circle',
            centre={'space': 'camera_px', 'camera': 'camera-0', 'x': 30, 'y': 25},
            geometry_space='surface_mm',
            radius={'value': 12, 'unit': 'mm'},
        ),
    )

    assert [entry.name for entry in service.list_overlays()] == [
        'projector-only',
        'camera-line',
        'physical-circle',
    ]
    assert camera_line.camera_dependencies == ('camera-0',)
    assert physical_circle.camera_dependencies == ('camera-0',)
    assert physical_circle.metric_dependency
    assert service.hide_overlay(projector_line.id).visible is False
    assert service.hide_overlay('projector-only').visible is False
    assert service.show_overlay('projector-only').visible is True
    with pytest.raises(ValueError, match='segment budget'):
        service.create_overlay(
            GridRequest(
                origin={'space': 'projector_px', 'x': 0, 'y': 0},
                geometry_space='projector_px',
                spacing={'value': 10, 'unit': 'px'},
                extent={
                    'width': {'value': 1000, 'unit': 'px'},
                    'height': {'value': 1000, 'unit': 'px'},
                },
            ),
        )
    assert [entry.id for entry in service.list_overlays()] == [
        projector_line.id,
        camera_line.id,
        physical_circle.id,
    ]

    service.clear_metric_calibration()
    assert [entry.name for entry in service.list_overlays()] == [
        'projector-only',
        'camera-line',
    ]
    runtime.registry.mark_unavailable('camera-0', 'disconnected')
    assert [entry.name for entry in service.list_overlays()] == ['projector-only']

    service.update_projector_descriptor(
        ProjectorOutputDescriptor(Resolution(300, 200), 'projector-replaced'),
    )
    assert service.list_overlays() == []


def test_plan5_api_is_strict_json_safe_and_keeps_legacy_point_boundary() -> None:
    service, _runtime = _service()
    with TestClient(create_app(service, manage_lifecycle=False)) as client:
        responses = [
            client.post(
                f'/overlays/{kind}',
                json=spec,
            )
            for kind, spec in (
                (
                    'grid',
                    {
                        'origin': {'space': 'projector_px', 'x': 1, 'y': 2},
                        'geometry_space': 'projector_px',
                        'spacing': {'value': 10, 'unit': 'px'},
                        'extent': {
                            'width': {'value': 30, 'unit': 'px'},
                            'height': {'value': 20, 'unit': 'px'},
                        },
                    },
                ),
                (
                    'circle',
                    {
                        'centre': {'space': 'projector_px', 'x': 40, 'y': 40},
                        'geometry_space': 'projector_px',
                        'radius': {'value': 10, 'unit': 'px'},
                        'style': {'colour': '#abcdef', 'line_width_px': 2},
                    },
                ),
                (
                    'rect',
                    {
                        'centre': {'space': 'projector_px', 'x': 50, 'y': 40},
                        'geometry_space': 'projector_px',
                        'width': {'value': 20, 'unit': 'px'},
                        'height': {'value': 10, 'unit': 'px'},
                        'angle_deg': 15,
                    },
                ),
                (
                    'line',
                    {
                        'start': {'space': 'projector_px', 'x': 5, 'y': 5},
                        'end': {'space': 'projector_px', 'x': 90, 'y': 70},
                        'label': 'safe',
                    },
                ),
                (
                    'ruler',
                    {
                        'start': {'space': 'projector_px', 'x': 5, 'y': 5},
                        'end': {'space': 'projector_px', 'x': 90, 'y': 5},
                        'measurement_space': 'projector_px',
                        'unit': 'px',
                    },
                ),
            )
        ]
        listing = client.get('/overlays')
        malformed = client.post(
            '/overlays/line',
            json={
                'start': {'space': 'projector_px', 'x': 0, 'y': 0},
                'end': {'space': 'projector_px', 'x': 1, 'y': 1},
                'unexpected': True,
            },
        )
        camera_point = client.post(
            '/overlay/point',
            json={'camera': 'camera-0', 'x': 10, 'y': 20},
        )

    assert all(response.status_code == 200 for response in responses), [
        response.text for response in responses
    ]
    assert listing.status_code == 200
    assert all('materialised_primitives' not in entry for entry in listing.json())
    json.dumps(listing.json(), allow_nan=False)
    assert malformed.status_code == 422, malformed.text
    assert camera_point.status_code == 200, camera_point.text
    assert service.overlay is not None


class RecordingSurface:
    def __init__(self) -> None:
        self.size = (100, 80)

    def fill(self, _colour: tuple[int, int, int]) -> None:
        pass


class RecordingPygame:
    def __init__(self) -> None:
        self.draw_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.rendered_text: list[str] = []
        self.draw = SimpleNamespace(
            polygon=lambda *arguments: self.draw_calls.append(('polygon', arguments)),
            line=lambda *arguments: self.draw_calls.append(('line', arguments)),
        )
        self.font = SimpleNamespace(
            Font=lambda _name, _size: SimpleNamespace(
                render=lambda text, _antialias, _colour: self._render(text),
            ),
        )

    def _render(self, text: str) -> RecordingSurface:
        self.rendered_text.append(text)
        return RecordingSurface()


def test_plan5_display_consumes_immutable_projector_primitives_in_layer_order() -> None:
    pygame_module = RecordingPygame()
    renderer = ProjectorRenderer(pygame_module)
    surface = RecordingSurface()
    style = OverlayStyle(colour='#102030', line_width_px=3)
    entries = [
        SimpleNamespace(
            kind='line',
            visible=True,
            insertion_sequence=0,
            materialised_primitives=ProjectorMaterialisation(
                segments=(ProjectorSegment(Point2D(2, 3), Point2D(20, 30), style),),
                labels=(),
            ),
        ),
        SimpleNamespace(
            kind='rect',
            visible=True,
            insertion_sequence=1,
            materialised_primitives=ProjectorMaterialisation(
                polygons=(
                    ProjectorPolygon(
                        (Point2D(10, 10), Point2D(20, 10), Point2D(20, 20)),
                        style,
                    ),
                ),
            ),
        ),
        SimpleNamespace(
            kind='circle',
            visible=False,
            insertion_sequence=2,
            materialised_primitives=ProjectorMaterialisation(
                segments=(ProjectorSegment(Point2D(1, 1), Point2D(5, 5), style),),
            ),
        ),
    ]

    renderer.render_generic_overlays(surface, entries)

    assert [name for name, _arguments in pygame_module.draw_calls] == ['polygon', 'line']
    assert pygame_module.draw_calls[0][1][1] == style.colour
    assert pygame_module.draw_calls[1][1][-1] == style.line_width_px
    assert pygame_module.rendered_text == []

    first_draw_calls = list(pygame_module.draw_calls)
    pygame_module.draw_calls.clear()
    renderer.render_generic_overlays(surface, entries)
    assert pygame_module.draw_calls == first_draw_calls, (
        f'{pygame_module.draw_calls=}, {first_draw_calls=}'
    )
