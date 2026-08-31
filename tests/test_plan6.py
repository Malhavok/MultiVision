import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from multivision.api import create_app
from multivision.application import MultiVisionService
from multivision.cli import MultiVisionClient, ServiceResponse, main
from multivision.config import Configuration
from multivision.errors import (
    FrameCaptureError,
    PointOutsideCalibratedRegionError,
)
from multivision.fiducials import DetectedMarker
from multivision.geometry import (
    Point2D,
    build_tag_geometry,
    project_tag_geometry,
)
from multivision.pattern import DICT_5X5_1000
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Frame,
    Resolution,
    RuntimeStatus,
)


CAMERA_RESOLUTION = Resolution(100, 100)
IDENTITY_HOMOGRAPHY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _corners(x_pos: float, y_pos: float, size: float = 10.0) -> tuple[Point2D, ...]:
    return (
        Point2D(x_pos, y_pos),
        Point2D(x_pos + size, y_pos),
        Point2D(x_pos + size, y_pos + size),
        Point2D(x_pos, y_pos + size),
    )


class InspectionRuntime:
    def __init__(
        self,
        calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED,
        calibration: object | None = None,
        frame: Frame | None = None,
    ) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [DeviceInfo('device-0', 'Camera 0', capture_index=0)],
        )
        if calibration_status is not CalibrationStatus.UNCALIBRATED:
            self.registry.set_calibration(
                'camera-0',
                calibration_status,
                calibration if calibration is not None else SimpleNamespace(),
            )
        self.frame = frame if frame is not None else Frame(object(), 42, 123.5)
        self.snapshot_count = 0

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        return CameraStatus(
            slot_id,
            'device-0',
            RuntimeStatus.AVAILABLE,
            camera.calibration_status,
            CAMERA_RESOLUTION,
        )

    def snapshot(self, slot_id: str) -> Frame:
        assert slot_id == 'camera-0', f'{slot_id=}'
        self.snapshot_count += 1
        return self.frame

    def get_statuses(self) -> list[CameraStatus]:
        return [self.get_status('camera-0')]


class TagDetector:
    def __init__(self, detections: tuple[DetectedMarker, ...]) -> None:
        self.detections = detections
        self.frames: list[object] = []

    def detect(self, frame: object) -> tuple[DetectedMarker, ...]:
        self.frames.append(frame)
        return self.detections


class Plan6ServiceTest(unittest.TestCase):
    def _service(
        self,
        detections: tuple[DetectedMarker, ...],
        *,
        calibration_status: CalibrationStatus = CalibrationStatus.CALIBRATED,
        homography: tuple[tuple[float, float, float], ...] = IDENTITY_HOMOGRAPHY,
        calibrated_region: tuple[Point2D, ...] = (
            Point2D(0, 0),
            Point2D(100, 0),
            Point2D(100, 100),
            Point2D(0, 100),
        ),
        projector_resolution: Resolution = CAMERA_RESOLUTION,
    ) -> tuple[MultiVisionService, InspectionRuntime, TagDetector]:
        calibration = SimpleNamespace(
            camera_to_projector=homography,
            valid_region=calibrated_region,
        )
        runtime = InspectionRuntime(calibration_status, calibration)
        detector = TagDetector(detections)
        service = MultiVisionService(
            Configuration(projector_resolution=projector_resolution),
            camera_runtime=runtime,  # type: ignore[arg-type]
            tag_detector_factory=lambda _dictionary: detector,
        )
        return service, runtime, detector

    def test_inspection_uses_one_owned_latest_frame_session_reference_and_no_overlay_mutation(self) -> None:
        detections = (DetectedMarker(23, _corners(10, 10)),)
        service, runtime, detector = self._service(detections)
        runtime.registry.rename('camera-0', 'overhead')
        overlay = service.point_from_camera('overhead', (5, 5))
        calibration_before = runtime.registry.get('camera-0').calibration
        calibration_records_before = service.get_calibration_records()

        result = service.inspect_tags('overhead', DICT_5X5_1000)

        assert runtime.snapshot_count == 1, f'{runtime.snapshot_count=}'
        assert result.camera == 'overhead'
        assert result.camera_id == 'camera-0'
        assert result.frame_counter == 42
        assert result.captured_at_seconds == 123.5
        assert detector.frames == [runtime.frame.data], f'{detector.frames=}'
        assert service.overlay == overlay, f'{service.overlay=}'
        assert runtime.registry.get('camera-0').calibration == calibration_before
        assert service.get_calibration_records() == calibration_records_before
        assert result.tags[0].projector is not None

    def test_service_attaches_projective_projector_observation(self) -> None:
        homography = ((1, 0.2, 0), (0.2, 1, 0), (0.01, 0.005, 1))
        service, _runtime, _detector = self._service(
            (DetectedMarker(23, _corners(10, 10)),),
            homography=homography,
        )

        result = service.inspect_tags('camera-0')
        expected = project_tag_geometry(build_tag_geometry(_corners(10, 10)), homography)

        assert result.tags[0].projector == expected, f'{result=}, {expected=}'
        assert result.tags[0].projector.orientation_degrees != (
            result.tags[0].camera.orientation_degrees
        )

    def test_camera_wide_projection_status_is_shared_and_empty_frames_have_no_tag_status(self) -> None:
        detections = (DetectedMarker(1, _corners(10, 10)), DetectedMarker(2, _corners(30, 30)))
        for calibration_status, expected_code in [
            (CalibrationStatus.UNCALIBRATED, 'CALIBRATION_UNCALIBRATED'),
            (CalibrationStatus.STALE, 'CALIBRATION_STALE'),
        ]:
            with self.subTest(calibration_status=calibration_status):
                service, _runtime, _detector = self._service(
                    detections,
                    calibration_status=calibration_status,
                )
                result = service.inspect_tags('camera-0')
                assert result.projection_status is not None
                assert result.projection_status.code == expected_code
                assert all(
                    tag.projector is None
                    and tag.projection_status == result.projection_status
                    for tag in result.tags
                ), f'{result=}'

        service, _runtime, _detector = self._service(
            (),
            calibration_status=CalibrationStatus.UNCALIBRATED,
        )
        result = service.inspect_tags('camera-0')
        assert result.tags == ()
        assert result.projection_status is not None
        assert result.projection_status.code == 'CALIBRATION_UNCALIBRATED'

    def test_invalid_camera_wide_homography_is_repeated_for_each_raw_tag(self) -> None:
        service, _runtime, _detector = self._service(
            (DetectedMarker(1, _corners(10, 10)), DetectedMarker(2, _corners(30, 30))),
            homography=((1, 0, 0), (0, 0, 0), (0, 0, 1)),
        )

        result = service.inspect_tags('camera-0')

        assert result.projection_status is not None
        assert result.projection_status.code == 'INVALID_HOMOGRAPHY'
        assert all(
            tag.projector is None
            and tag.projection_status == result.projection_status
            for tag in result.tags
        ), f'{result=}'

    def test_per_tag_projection_status_order_is_region_then_homography_then_bounds(self) -> None:
        cases = [
            (
                'POINT_OUTSIDE_CALIBRATED_REGION',
                ((0, 0), (50, 0), (50, 50), (0, 50)),
                IDENTITY_HOMOGRAPHY,
                CAMERA_RESOLUTION,
            ),
            (
                'INVALID_HOMOGRAPHY',
                ((0, 0), (100, 0), (100, 100), (0, 100)),
                ((1, 0, 0), (0, 1, 0), (-0.02, 0, 1)),
                CAMERA_RESOLUTION,
            ),
            (
                'POINT_OUTSIDE_PROJECTOR_BOUNDS',
                ((0, 0), (100, 0), (100, 100), (0, 100)),
                ((1, 0, 150), (0, 1, 0), (0, 0, 1)),
                Resolution(200, 200),
            ),
        ]
        for expected_code, region, homography, projector_resolution in cases:
            with self.subTest(expected_code=expected_code):
                service, _runtime, _detector = self._service(
                    (DetectedMarker(4, _corners(40, 40, 20)),),
                    calibrated_region=region,
                    homography=homography,
                    projector_resolution=projector_resolution,
                )
                result = service.inspect_tags('camera-0')
                assert result.projection_status is None, f'{result=}'
                assert result.tags[0].projector is None, f'{result=}'
                assert result.tags[0].projection_status is not None
                assert result.tags[0].projection_status.code == expected_code

    def test_legacy_point_projection_uses_the_persisted_support_region(self) -> None:
        service, _runtime, _detector = self._service(
            (),
            calibrated_region=((0, 0), (50, 0), (50, 50), (0, 50)),
        )

        projected = service.point_from_camera('camera-0', (10, 10))
        assert projected.projector_point == Point2D(10, 10)
        with self.assertRaises(PointOutsideCalibratedRegionError):
            service.point_from_camera('camera-0', (75, 75))

    def test_snapshot_compatibility_and_calibration_detector_remain_separate(self) -> None:
        calibration_detector = TagDetector((DetectedMarker(0, _corners(10, 10)),))
        tag_detector = TagDetector(())
        runtime = InspectionRuntime()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            detector=calibration_detector,
            tag_detector_factory=lambda _dictionary: tag_detector,
        )

        frame = service.snapshot('camera-0')
        correspondences = service._get_correspondences_from_frame(
            runtime.get_status('camera-0'),
            frame,
        )
        service.inspect_tags('camera-0')

        assert frame is runtime.frame
        assert len(correspondences.correspondences) == 4
        assert calibration_detector.frames == [frame.data], f'{calibration_detector.frames=}'
        assert tag_detector.frames == [frame.data], f'{tag_detector.frames=}'

    def test_missing_latest_frame_is_an_explicit_capture_failure(self) -> None:
        class MissingFrameRuntime(InspectionRuntime):
            def snapshot(self, _slot_id: str) -> Frame:
                raise FrameCaptureError('no latest frame')

        runtime = MissingFrameRuntime()
        service = MultiVisionService(
            Configuration(),
            camera_runtime=runtime,  # type: ignore[arg-type]
            tag_detector_factory=lambda _dictionary: TagDetector(()),
        )

        with self.assertRaises(FrameCaptureError):
            service.inspect_tags('camera-0')


class Plan6ApiCliTest(unittest.TestCase):
    def test_api_returns_repeated_json_safe_observations_and_delegates_without_camera_work(self) -> None:
        class FakeResult:
            def to_data(self) -> dict[str, Any]:
                return {
                    'camera': 'overhead',
                    'camera_id': 'camera-0',
                    'dictionary': DICT_5X5_1000,
                    'frame_counter': 7,
                    'captured_at_seconds': 123.5,
                    'tags': [
                        {
                            'id': 4,
                            'camera': {
                                'corners': ((1.0, 2.0),) * 4,
                                'centre': (1.0, 2.0),
                                'orientation_degrees': 0.0,
                                'area_px': 0.0,
                            },
                            'projector': None,
                            'projection_status': None,
                        },
                        {
                            'id': 4,
                            'camera': {
                                'corners': ((3.0, 4.0),) * 4,
                                'centre': (3.0, 4.0),
                                'orientation_degrees': 90.0,
                                'area_px': 0.0,
                            },
                            'projector': None,
                            'projection_status': {
                                'code': 'POINT_OUTSIDE_CALIBRATED_REGION',
                                'message': 'outside',
                            },
                        },
                    ],
                    'projection_status': None,
                }

        class FakeService:
            def __init__(self) -> None:
                self.requests: list[tuple[str, str | None]] = []

            def inspect_tags(self, camera: str, dictionary: str | None) -> FakeResult:
                self.requests.append((camera, dictionary))
                return FakeResult()

        service = FakeService()
        with TestClient(create_app(service, manage_lifecycle=False)) as client:
            response = client.get(
                '/cameras/overhead/tags',
                params={'dictionary': DICT_5X5_1000},
            )

        data = response.json()
        assert response.status_code == 200, f'{response.text=}'
        assert [tag['id'] for tag in data['tags']] == [4, 4], f'{data=}'
        json.dumps(data, allow_nan=False)
        assert service.requests == [('overhead', DICT_5X5_1000)], f'{service.requests=}'

    def test_cli_delegates_and_reports_transport_failures_without_camera_imports(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"tags": []}')

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                ['tags', 'list', '--camera', 'overhead'],
                MultiVisionClient('http://service.test', request_sender=request_sender),
            )
        assert result == 0
        assert requests[0][0:3] == ('GET', 'http://service.test/cameras/overhead/tags', None)
        assert json.loads(output.getvalue()) == {'tags': []}

        error_output = io.StringIO()
        failing_client = MultiVisionClient(
            'http://service.test',
            request_sender=lambda *_arguments: (_ for _ in ()).throw(OSError('offline')),
        )
        with redirect_stderr(error_output):
            result = main(['tags', 'list', '--camera', 'overhead'], failing_client)
        assert result == 1
        assert 'Could not contact MultiVision service' in error_output.getvalue()
        cli_module = sys.modules['multivision.cli']
        assert 'cv2' not in cli_module.__dict__
        assert 'MultiVisionService' not in cli_module.__dict__
