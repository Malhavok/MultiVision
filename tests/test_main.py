import unittest
from unittest.mock import patch

from multivision.config import Configuration
from multivision.main import main
from multivision.types import (
    CalibrationStatus,
    CameraStatus,
    Resolution,
    RuntimeStatus,
)


LIFECYCLE_EVENTS: list[str] = []


class FakeCameraRuntime:
    def __init__(self) -> None:
        self.started = False
        self.shutdown_called = False
        self.statuses: list[CameraStatus] = []

    def start(self) -> None:
        self.started = True
        LIFECYCLE_EVENTS.append('camera.start')

    def shutdown(self) -> None:
        self.shutdown_called = True
        LIFECYCLE_EVENTS.append('camera.shutdown')

    def get_statuses(self) -> list[CameraStatus]:
        return self.statuses


class FakeService:
    def __init__(
        self,
        camera_runtime: FakeCameraRuntime,
        configuration: Configuration,
    ) -> None:
        self.camera_runtime = camera_runtime
        self.configuration = configuration
        self.calibration_pattern = object()

    def get_camera_statuses(self) -> list[CameraStatus]:
        return self.camera_runtime.get_statuses()

    def snapshot(self, logical_name: str) -> object:
        raise AssertionError(f'Unexpected snapshot request for {logical_name!r}')

    def get_calibration_metrics(self, logical_name: str) -> None:
        return None

    def point_from_preview(self, *arguments: object) -> object:
        raise AssertionError(f'Unexpected point request: {arguments!r}')

    @property
    def overlay(self) -> None:
        return None

    def start(self) -> None:
        self.camera_runtime.start()

    def shutdown(self) -> None:
        self.camera_runtime.shutdown()


class FakeDisplayRuntime:
    instances: list['FakeDisplayRuntime'] = []

    def __init__(
        self,
        service: FakeService,
        configuration: object,
        **dependencies: object,
    ) -> None:
        self.service = service
        self.configuration = configuration
        self.dependencies = dependencies
        self.ran = False
        self.statuses_during_run: list[object] = []
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def run(self) -> None:
        self.ran = True
        LIFECYCLE_EVENTS.append('display.run')
        get_statuses = getattr(self.service, 'get_camera_statuses', None)
        self.statuses_during_run = get_statuses() if callable(get_statuses) else []

    def shutdown(self) -> None:
        self.shutdown_called = True
        LIFECYCLE_EVENTS.append('display.shutdown')


class FakeApiServerRuntime:
    instances: list['FakeApiServerRuntime'] = []

    def __init__(self, service: object) -> None:
        self.service = service
        self.started = False
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True
        LIFECYCLE_EVENTS.append('api.start')

    def shutdown(self) -> None:
        self.shutdown_called = True
        LIFECYCLE_EVENTS.append('api.shutdown')


class MainTest(unittest.TestCase):
    def test_main_marks_an_unavailable_camera_without_blocking_the_ui(self) -> None:
        FakeDisplayRuntime.instances.clear()
        FakeApiServerRuntime.instances.clear()
        configuration = Configuration(
        )
        camera_runtime = FakeCameraRuntime()
        camera_runtime.statuses = [
            CameraStatus(
                logical_name='overhead',
                device_id='missing-device',
                runtime_status=RuntimeStatus.UNAVAILABLE,
                calibration_status=CalibrationStatus.UNCALIBRATED,
            ),
        ]
        service = FakeService(camera_runtime, configuration)

        with (
            patch('multivision.main.MultiVisionService', return_value=service),
            patch('multivision.main.PygameDisplayRuntime', FakeDisplayRuntime),
            patch('multivision.main.ApiServerRuntime', FakeApiServerRuntime),
        ):
            main()

        display_runtime = FakeDisplayRuntime.instances[0]
        assert len(display_runtime.statuses_during_run) == 1
        assert display_runtime.statuses_during_run[0].runtime_status.value == 'UNAVAILABLE'
        assert display_runtime.ran, f'{display_runtime=}'
        assert display_runtime.shutdown_called, f'{display_runtime=}'
        assert FakeApiServerRuntime.instances[0].shutdown_called

    def test_main_owns_runtime_lifecycle(self) -> None:
        FakeDisplayRuntime.instances.clear()
        FakeApiServerRuntime.instances.clear()
        LIFECYCLE_EVENTS.clear()
        configuration = Configuration(
            projector_resolution=Resolution(1600, 900),
        )

        camera_runtime = FakeCameraRuntime()
        service = FakeService(camera_runtime, configuration)

        with (
            patch('multivision.main.MultiVisionService', return_value=service),
            patch('multivision.main.PygameDisplayRuntime', FakeDisplayRuntime),
            patch('multivision.main.ApiServerRuntime', FakeApiServerRuntime),
        ):
            main()

        display_runtime = FakeDisplayRuntime.instances[0]
        api_runtime = FakeApiServerRuntime.instances[0]
        assert camera_runtime.started, f'{camera_runtime=}'
        assert camera_runtime.shutdown_called, f'{camera_runtime=}'
        assert display_runtime.ran, f'{display_runtime=}'
        assert display_runtime.shutdown_called, f'{display_runtime=}'
        assert api_runtime.started, f'{api_runtime=}'
        assert api_runtime.shutdown_called, f'{api_runtime=}'
        assert LIFECYCLE_EVENTS == [
            'camera.start',
            'api.start',
            'display.run',
            'display.shutdown',
            'api.shutdown',
            'camera.shutdown',
        ], f'{LIFECYCLE_EVENTS=}'
        assert display_runtime.service is service
        assert display_runtime.configuration.projector_resolution == Resolution(1600, 900)
        assert set(display_runtime.dependencies) == {'calibration_pattern'}


if __name__ == '__main__':
    unittest.main()
