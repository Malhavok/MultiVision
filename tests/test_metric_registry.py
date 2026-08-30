import math
import unittest
from types import SimpleNamespace

from multivision.application import MultiVisionService
from multivision.config import Configuration, ProjectorOutputDescriptor
from multivision.metric import (
    MetricCalibrationMetrics,
    MetricCalibrationRegistry,
    MetricCalibrationResult,
    MetricCalibrationStatus,
    MetricHomographyPair,
)
from multivision.metric_target import METRIC_TARGET
from multivision.session import SessionCameraRegistry
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    DeviceInfo,
    Resolution,
    RuntimeStatus,
)


class MetricCalibrationRegistryTest(unittest.TestCase):
    def test_registry_has_one_session_local_record_and_reachable_stale_state(self) -> None:
        descriptor = ProjectorOutputDescriptor(Resolution(800, 600), 'projector-a')
        registry = MetricCalibrationRegistry(descriptor)
        result = _metric_result(descriptor.projector_resolution)

        assert registry.get_status() is MetricCalibrationStatus.UNCALIBRATED
        with self.assertRaises(ValueError):
            registry.register(
                result,
                ProjectorOutputDescriptor(descriptor.projector_resolution, 'projector-b'),
                'camera-0',
            )
        assert registry.get_status() is MetricCalibrationStatus.UNCALIBRATED
        first_record = registry.register(result, descriptor, 'camera-0')
        second_record = registry.register(result, descriptor, 'camera-1')

        assert registry.get_record() is second_record, f'{registry.get_record()=}'
        assert first_record is not second_record
        assert second_record.observation_camera_slot == 'camera-1'
        assert not hasattr(second_record, 'mm_per_pixel')

        registry.update_projector_descriptor(
            ProjectorOutputDescriptor(Resolution(1024, 600), 'projector-a'),
        )
        assert registry.get_status() is MetricCalibrationStatus.STALE
        assert registry.is_usable() is False
        registry.clear()
        assert registry.get_status() is MetricCalibrationStatus.UNCALIBRATED

    def test_registry_rejects_malformed_calibration_records(self) -> None:
        descriptor = ProjectorOutputDescriptor(Resolution(800, 600), 'projector-a')
        result = _metric_result(descriptor.projector_resolution)
        malformed_homography = result._replace(
            homography=MetricHomographyPair(
                result.homography.projector_to_surface,
                ((1, 0, 10), (0, 1, 0), (0, 0, 1)),
            ),
        )
        malformed_metrics = result._replace(
            metrics=result.metrics._replace(mean_fit_error_mm=math.nan),
        )
        malformed_counts = result._replace(
            metrics=result.metrics._replace(correspondence_corner_count=4),
        )
        registry = MetricCalibrationRegistry(descriptor)

        for malformed_result in (
            malformed_homography,
            malformed_metrics,
            malformed_counts,
        ):
            with self.subTest(malformed_result=malformed_result):
                with self.assertRaises(ValueError):
                    registry.register(malformed_result, descriptor, 'camera-0')
                assert registry.get_status() is MetricCalibrationStatus.UNCALIBRATED

    def test_validation_records_preserve_transform_and_reject_malformed_measurements(self) -> None:
        descriptor = ProjectorOutputDescriptor(Resolution(800, 600), 'projector-a')
        registry = MetricCalibrationRegistry(descriptor)
        registry.register(_metric_result(descriptor.projector_resolution), descriptor, 'camera-0')
        before = registry.get_record()
        assert before is not None

        validation = registry.add_validation_record(10, 1, 'cm', 'in', timestamp=4.0)
        after = registry.get_record()
        assert after is not None
        assert validation.requested_length_mm == 100.0, f'{validation=}'
        assert validation.observed_length_mm == 25.4, f'{validation=}'
        assert validation.absolute_error_mm == 74.6, f'{validation=}'
        assert after.projector_to_surface == before.projector_to_surface
        assert after.surface_to_projector == before.surface_to_projector
        assert after.validation_records == (validation,)
        assert after.latest_physical_validation_error_mm == validation.absolute_error_mm

        for requested_length, observed_length, requested_unit, observed_unit in (
            (0, 10, 'mm', 'mm'),
            (10, 0, 'mm', 'mm'),
            (math.nan, 10, 'mm', 'mm'),
            (10, math.inf, 'mm', 'mm'),
            (10, 10, 'metres', 'mm'),
            (10, 10, 'mm', 'metres'),
        ):
            with self.subTest(
                requested_length=requested_length,
                observed_length=observed_length,
                requested_unit=requested_unit,
                observed_unit=observed_unit,
            ):
                with self.assertRaises(ValueError):
                    registry.add_validation_record(
                        requested_length,
                        observed_length,
                        requested_unit,
                        observed_unit,
                    )
        assert registry.get_record() == after

        registry.update_projector_descriptor(
            ProjectorOutputDescriptor(descriptor.projector_resolution, 'projector-b'),
        )
        with self.assertRaises(ValueError):
            registry.add_validation_record(10, 10)
        assert registry.get_record() == after._replace(
            state=MetricCalibrationStatus.STALE,
        )

    def test_service_clear_is_idempotent_and_removes_metric_ruler(self) -> None:
        runtime = _SessionRuntime()
        descriptor = ProjectorOutputDescriptor(Resolution(800, 600), 'projector-a')
        service = MultiVisionService(
            Configuration(
                projector_resolution=descriptor.projector_resolution,
                projector_output_identity=descriptor.output_identity,
            ),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )
        service.metric_registry.register(
            _metric_result(descriptor.projector_resolution),
            descriptor,
            'camera-0',
        )
        service._metric_ruler = object()

        service.clear_metric_calibration()
        service.clear_metric_calibration()

        assert service.metric_calibration is None
        assert service.metric_ruler is None
        assert service.get_metric_status() is MetricCalibrationStatus.UNCALIBRATED

    def test_service_stales_all_camera_geometry_metric_record_and_ruler_together(self) -> None:
        runtime = _SessionRuntime()
        old_descriptor = ProjectorOutputDescriptor(Resolution(800, 600), 'projector-a')
        for slot_id in ('camera-0', 'camera-1'):
            runtime.registry.set_calibration(
                slot_id,
                CalibrationStatus.CALIBRATED,
                SimpleNamespace(projector_output_descriptor=old_descriptor),
            )
        service = MultiVisionService(
            Configuration(
                projector_resolution=old_descriptor.projector_resolution,
                projector_output_identity=old_descriptor.output_identity,
            ),
            camera_runtime=runtime,  # type: ignore[arg-type]
        )
        service.metric_registry.register(
            _metric_result(old_descriptor.projector_resolution),
            old_descriptor,
            'camera-0',
        )
        service._metric_ruler = object()

        new_descriptor = ProjectorOutputDescriptor(Resolution(1024, 600), 'projector-b')
        service.update_projector_descriptor(new_descriptor)

        assert service.projector_output_descriptor == new_descriptor
        assert service.configuration.projector_output_descriptor == new_descriptor
        assert service.metric_registry.get_status() is MetricCalibrationStatus.STALE
        assert service.metric_ruler is None
        assert all(
            camera.calibration_status is CalibrationStatus.STALE
            for camera in runtime.registry.get_cameras()
        ), f'{runtime.registry.get_cameras()=}'

        restarted_service = MultiVisionService(
            Configuration(
                projector_resolution=new_descriptor.projector_resolution,
                projector_output_identity=new_descriptor.output_identity,
            ),
            camera_runtime=_SessionRuntime(),  # type: ignore[arg-type]
        )
        assert (
            restarted_service.metric_registry.get_status()
            is MetricCalibrationStatus.UNCALIBRATED
        )


def _metric_result(projector_resolution: Resolution) -> MetricCalibrationResult:
    identity = MetricHomographyPair.from_projector_to_surface(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    return MetricCalibrationResult(
        identity,
        MetricCalibrationMetrics(20, 80, 80, 1.0, 0.0, 0.0, 0.8),
        projector_resolution,
        METRIC_TARGET.format_name,
        METRIC_TARGET.format_version,
        METRIC_TARGET.marker_family,
        'camera-0',
    )


class _SessionRuntime:
    def __init__(self) -> None:
        self.registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo('device-0', 'Camera 0', capture_index=0),
                DeviceInfo('device-1', 'Camera 1', capture_index=1),
            ],
        )

    def get_session_cameras(self) -> list[object]:
        return self.registry.get_cameras()

    def get_status(self, slot_id: str) -> CameraStatus:
        camera = self.registry.get(slot_id)
        return CameraStatus(
            slot_id,
            camera.device_info.device_id if camera.device_info is not None else None,
            RuntimeStatus.AVAILABLE,
            camera.calibration_status,
            Resolution(800, 600),
        )

    def mark_calibrations_stale(
        self,
        descriptor: ProjectorOutputDescriptor,
    ) -> list[object]:
        return self.registry.mark_calibrations_stale(descriptor)

    def set_calibration(
        self,
        slot_id: str,
        status: CalibrationStatus,
        calibration: object,
    ) -> object:
        return self.registry.set_calibration(slot_id, status, calibration)


if __name__ == '__main__':
    unittest.main()
