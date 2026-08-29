import unittest

from multivision.errors import (
    ActiveCameraLimitError,
    CameraSlotNotFoundError,
    CameraStateError,
    DuplicateCameraNameError,
    SessionCameraError,
)
from multivision.session import (
    FrameMetadata,
    SessionCamera,
    SessionCameraRegistry,
)
from multivision.types import (
    CalibrationStatus,
    DeviceInfo,
    Resolution,
    SessionCameraState,
)


class SessionCameraRegistryTest(unittest.TestCase):
    def test_slots_have_default_names_and_deterministic_capture_order(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([7, 2, 4])

        cameras = registry.get_cameras()
        assert [camera.slot_id for camera in cameras] == [
            'camera-0',
            'camera-1',
            'camera-2',
        ], f'{cameras=}'
        assert [camera.display_name for camera in cameras] == [
            'camera-0',
            'camera-1',
            'camera-2',
        ], f'{cameras=}'
        assert [camera.capture_index for camera in cameras] == [2, 4, 7], f'{cameras=}'
        assert all(
            camera.device_info is not None
            and not camera.device_info.is_stable_id
            for camera in cameras
        ), f'{cameras=}'
        assert all(
            camera.state is SessionCameraState.OPEN
            for camera in cameras
        ), f'{cameras=}'

    def test_inventory_reads_do_not_expose_mutable_registry_state(self) -> None:
        metadata = {'source': {'name': 'startup'}}
        calibration = {'matrix': [[1, 0], [0, 1]]}
        registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo(
                    'device',
                    'Camera',
                    capture_index=0,
                    metadata=metadata,
                ),
            ],
        )
        metadata['source']['name'] = 'mutated externally'
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, calibration)
        calibration['matrix'][0][0] = 99

        camera = registry.get('camera-0')
        camera.display_name = 'mutated'
        camera.state = SessionCameraState.CLOSED
        assert camera.device_info is not None
        assert camera.device_info.metadata is not None
        camera.device_info.metadata['source'] = 'mutated'

        current_camera = registry.get('camera-0')
        assert current_camera.display_name == 'camera-0', f'{current_camera=}'
        assert current_camera.state is SessionCameraState.OPEN, f'{current_camera=}'
        assert current_camera.device_info is not None
        assert current_camera.device_info.metadata == {
            'source': {'name': 'startup'},
        }, f'{current_camera=}'
        assert current_camera.calibration == {'matrix': [[1, 0], [0, 1]]}, f'{current_camera=}'
        current_camera.calibration['matrix'][0][0] = 88
        assert registry.get('camera-0').calibration == {'matrix': [[1, 0], [0, 1]]}

    def test_rename_is_unique_and_preserves_camera_state(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0, 1])
        registry.set_frame_metadata(
            'camera-0',
            FrameMetadata(12, 123.5, Resolution(640, 480)),
        )
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
        renamed_camera = registry.rename('camera-0', 'overhead')

        assert renamed_camera.slot_id == 'camera-0', f'{renamed_camera=}'
        with self.assertRaises(AttributeError):
            renamed_camera.slot_id = 'camera-1'
        assert renamed_camera.display_name == 'overhead', f'{renamed_camera=}'
        assert renamed_camera.capture_index == 0, f'{renamed_camera=}'
        assert renamed_camera.frame_metadata == FrameMetadata(
            12,
            123.5,
            Resolution(640, 480),
        ), f'{renamed_camera=}'
        assert renamed_camera.calibration_status is CalibrationStatus.CALIBRATED
        assert renamed_camera.calibration == 'transform'
        with self.assertRaises(DuplicateCameraNameError):
            registry.rename('camera-1', 'overhead')
        assert registry.get('camera-1').display_name == 'camera-1'

    def test_area_enablement_defaults_disabled_and_requires_calibration(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0])
        camera = registry.get('camera-0')

        assert camera.area_enabled is False, f'{camera=}'
        with self.assertRaises(CameraStateError):
            registry.set_area_enabled('camera-0', True)
        assert registry.get('camera-0').area_enabled is False

        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
        enabled_camera = registry.set_area_enabled('camera-0', True)
        assert enabled_camera.area_enabled is True, f'{enabled_camera=}'
        assert registry.set_area_enabled('camera-0', True).area_enabled is True
        assert registry.set_area_enabled('camera-0', False).area_enabled is False
        with self.assertRaises(SessionCameraError):
            registry.set_area_enabled('camera-0', 1)  # type: ignore[arg-type]

    def test_area_enablement_preserves_slot_identity_and_rename_state(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0, 1])
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
        registry.set_area_enabled('camera-0', True)
        renamed_camera = registry.rename('camera-0', 'overhead')

        assert renamed_camera.slot_id == 'camera-0', f'{renamed_camera=}'
        with self.assertRaises(AttributeError):
            renamed_camera.slot_id = 'camera-1'
        assert renamed_camera.display_name == 'overhead', f'{renamed_camera=}'
        assert renamed_camera.area_enabled is True, f'{renamed_camera=}'
        assert renamed_camera.calibration == 'transform', f'{renamed_camera=}'
        assert registry.get('camera-1').area_enabled is False

    def test_area_enablement_is_independent_and_recalibration_invalidates_one_slot(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0, 1])
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform-0')
        registry.set_calibration('camera-1', CalibrationStatus.CALIBRATED, 'transform-1')
        registry.set_area_enabled('camera-0', True)
        registry.set_area_enabled('camera-1', True)

        registry.set_area_enabled('camera-0', False)
        assert registry.get('camera-0').area_enabled is False
        assert registry.get('camera-1').area_enabled is True
        assert registry.get('camera-1').calibration == 'transform-1'

        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform-0-new')
        assert registry.get('camera-0').area_enabled is False
        assert registry.get('camera-0').calibration == 'transform-0-new'
        assert registry.get('camera-1').area_enabled is True
        assert registry.get('camera-1').calibration == 'transform-1'

    def test_area_enablement_is_session_local_and_calibration_owns_geometry(self) -> None:
        calibration = {'valid_region': [(0, 0), (10, 0), (10, 10)]}
        registry = SessionCameraRegistry.from_capture_indexes([2, 0, 1])
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, calibration)
        registry.set_area_enabled('camera-0', True)

        enabled_camera = registry.get('camera-0')
        assert enabled_camera.area_enabled is True, f'{enabled_camera=}'
        assert enabled_camera.calibration == calibration, f'{enabled_camera=}'
        assert not hasattr(enabled_camera, 'available_area'), f'{enabled_camera=}'
        assert [camera.slot_id for camera in registry.get_cameras()] == [
            'camera-0',
            'camera-1',
            'camera-2',
        ], f'{registry.get_cameras()=}'

        new_session_registry = SessionCameraRegistry.from_capture_indexes([2, 0, 1])
        assert new_session_registry.get('camera-0').area_enabled is False

    def test_close_and_reopen_clear_frame_calibration_and_area(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0])
        registry.set_frame_metadata('camera-0', FrameMetadata(3, 4.0))
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
        registry.set_area_enabled('camera-0', True)
        closed_camera = registry.close('camera-0')

        assert closed_camera.state is SessionCameraState.CLOSED
        assert closed_camera.frame_metadata is None
        assert closed_camera.calibration_status is CalibrationStatus.UNCALIBRATED
        assert closed_camera.calibration is None
        assert closed_camera.area_enabled is False

        reopened_camera = registry.open('camera-0')
        assert reopened_camera.state is SessionCameraState.OPEN
        assert reopened_camera.calibration_status is CalibrationStatus.UNCALIBRATED
        assert reopened_camera.calibration is None
        assert reopened_camera.area_enabled is False

    def test_disconnect_invalidates_camera_state_and_spatial_ownership(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0])
        registry.set_frame_metadata('camera-0', FrameMetadata(3, 4.0))
        registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'transform')
        registry.set_area_enabled('camera-0', True)
        unavailable_camera = registry.mark_unavailable(
            'camera-0',
            'Camera camera-0 became unavailable',
        )

        assert unavailable_camera.state is SessionCameraState.UNAVAILABLE
        assert unavailable_camera.frame_metadata is None
        assert unavailable_camera.calibration_status is CalibrationStatus.UNCALIBRATED
        assert unavailable_camera.calibration is None
        assert unavailable_camera.area_enabled is False
        with self.assertRaises(CameraStateError):
            registry.open('camera-0')

    def test_unavailable_camera_cannot_be_reopened(self) -> None:
        registry = SessionCameraRegistry.from_devices(
            [DeviceInfo('device', 'Camera', capture_index=0, is_available=False)],
        )

        camera = registry.get('camera-0')
        assert camera.state is SessionCameraState.UNAVAILABLE
        with self.assertRaises(CameraStateError):
            registry.open('camera-0')

    def test_unavailable_slots_do_not_consume_active_capacity(self) -> None:
        registry = SessionCameraRegistry.from_devices(
            [
                DeviceInfo('missing', 'Missing', capture_index=0, is_available=False),
                DeviceInfo('camera-1', 'Camera 1', capture_index=1),
                DeviceInfo('camera-2', 'Camera 2', capture_index=2),
                DeviceInfo('camera-3', 'Camera 3', capture_index=3),
                DeviceInfo('camera-4', 'Camera 4', capture_index=4),
                DeviceInfo('camera-5', 'Camera 5', capture_index=5),
            ],
        )

        assert registry.active_count == 4
        assert registry.get('camera-4').state is SessionCameraState.OPEN
        assert registry.get('camera-5').state is SessionCameraState.CLOSED

    def test_invalid_state_and_unknown_slot_are_rejected(self) -> None:
        with self.assertRaises(SessionCameraError):
            FrameMetadata(frame_counter=-1)
        with self.assertRaises(SessionCameraError):
            SessionCameraRegistry.from_capture_indexes([0, 0])
        with self.assertRaises(CameraSlotNotFoundError):
            SessionCameraRegistry().get('camera-0')

    def test_inventory_sizes_obey_the_four_camera_active_limit(self) -> None:
        for device_count, expected_active_count in ((0, 0), (1, 1), (4, 4), (5, 4)):
            with self.subTest(device_count=device_count):
                registry = SessionCameraRegistry.from_capture_indexes(
                    range(device_count),
                )
                cameras = registry.get_cameras()

                assert registry.active_count == expected_active_count, f'{cameras=}'
                assert sum(
                    camera.state is SessionCameraState.OPEN
                    for camera in cameras
                ) == expected_active_count, f'{cameras=}'

    def test_more_than_four_slots_start_with_only_four_open(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes(range(5))

        assert registry.active_count == 4
        assert registry.get('camera-4').state is SessionCameraState.CLOSED
        with self.assertRaises(ActiveCameraLimitError):
            registry.open('camera-4')

        registry.close('camera-0')
        assert registry.open('camera-4').state is SessionCameraState.OPEN
        assert registry.active_count == 4

    def test_malformed_session_state_is_rejected(self) -> None:
        invalid_states = [
            lambda: SessionCamera('camera-x', 'camera-x', 0),
            lambda: SessionCamera('camera-0', '', 0),
            lambda: SessionCamera('camera-0', 'camera-0', True),
            lambda: SessionCamera(
                'camera-0',
                'camera-0',
                0,
                state='OPEN',  # type: ignore[arg-type]
            ),
            lambda: SessionCamera(
                'camera-0',
                'camera-0',
                0,
                frame_metadata=object(),  # type: ignore[arg-type]
            ),
            lambda: SessionCamera(
                'camera-0',
                'camera-0',
                0,
                calibration_status='CALIBRATED',  # type: ignore[arg-type]
            ),
            lambda: SessionCamera(
                'camera-0',
                'camera-0',
                0,
                state=SessionCameraState.CLOSED,
                frame_metadata=FrameMetadata(1),
            ),
            lambda: SessionCamera(
                'camera-0',
                'camera-0',
                0,
                calibration_status=CalibrationStatus.CALIBRATED,
            ),
        ]

        for make_invalid_state in invalid_states:
            with self.subTest(make_invalid_state=make_invalid_state):
                with self.assertRaises(SessionCameraError):
                    make_invalid_state()

        malformed_device = DeviceInfo(
            'device',
            'Camera',
            capture_index=0,
            metadata=[],  # type: ignore[arg-type]
        )
        with self.assertRaises(SessionCameraError):
            SessionCameraRegistry.from_devices([malformed_device])

    def test_calibration_state_cannot_become_inconsistent(self) -> None:
        registry = SessionCameraRegistry.from_capture_indexes([0])

        with self.assertRaises(SessionCameraError):
            registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, None)
        assert registry.get('camera-0').calibration_status is CalibrationStatus.UNCALIBRATED

        registry.close('camera-0')
        with self.assertRaises(SessionCameraError):
            registry.set_calibration('camera-0', CalibrationStatus.CALIBRATED, 'stale')
        assert registry.get('camera-0').calibration_status is CalibrationStatus.UNCALIBRATED


if __name__ == '__main__':
    unittest.main()
