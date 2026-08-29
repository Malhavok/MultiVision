import os
import subprocess
import unittest
from unittest.mock import patch

from multivision.application import MultiVisionService
from multivision.hardware import SystemSleepInhibitor


class FakeSleepProcess:
    def __init__(self) -> None:
        self.is_running = True
        self.terminate_count = 0
        self.kill_count = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> None:
        return None if self.is_running else 0

    def terminate(self) -> None:
        self.terminate_count += 1
        self.is_running = False

    def kill(self) -> None:
        self.kill_count += 1
        self.is_running = False

    def wait(self, timeout: float) -> None:
        self.wait_timeouts.append(timeout)


class FakeCameraRuntime:
    def __init__(self) -> None:
        self.start_count = 0
        self.shutdown_count = 0

    def start(self) -> None:
        self.start_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeSleepInhibitor:
    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1


class SystemSleepInhibitorTest(unittest.TestCase):
    def test_mac_start_and_stop_manage_one_caffeinate_process(self) -> None:
        process = FakeSleepProcess()
        with (
            patch('multivision.hardware.sys.platform', 'darwin'),
            patch('multivision.hardware.shutil.which', return_value='/usr/bin/caffeinate'),
            patch('multivision.hardware.subprocess.Popen', return_value=process) as popen,
        ):
            inhibitor = SystemSleepInhibitor()
            inhibitor.start()
            inhibitor.start()
            inhibitor.stop()

        popen.assert_called_once_with(
            ['/usr/bin/caffeinate', '-dimsu', '-w', str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert process.terminate_count == 1, f'{process.terminate_count=}'
        assert process.wait_timeouts == [2.0], f'{process.wait_timeouts=}'

    def test_non_mac_start_is_a_no_op(self) -> None:
        with patch('multivision.hardware.sys.platform', 'linux'):
            inhibitor = SystemSleepInhibitor()
            inhibitor.start()
            inhibitor.stop()

    def test_service_starts_and_stops_sleep_inhibition_with_camera_runtime(self) -> None:
        camera_runtime = FakeCameraRuntime()
        sleep_inhibitor = FakeSleepInhibitor()
        service = MultiVisionService(
            camera_runtime=camera_runtime,  # type: ignore[arg-type]
            sleep_inhibitor=sleep_inhibitor,
        )

        service.start()
        service.shutdown()

        assert camera_runtime.start_count == 1, f'{camera_runtime.start_count=}'
        assert camera_runtime.shutdown_count == 1, f'{camera_runtime.shutdown_count=}'
        assert sleep_inhibitor.start_count == 1, f'{sleep_inhibitor.start_count=}'
        assert sleep_inhibitor.stop_count == 1, f'{sleep_inhibitor.stop_count=}'


if __name__ == '__main__':
    unittest.main()
