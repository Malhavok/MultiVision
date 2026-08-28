import threading
import time
import unittest

from fastapi.testclient import TestClient

from multivision.application import MultiVisionService
from multivision.config import Configuration
from multivision.server import ApiServerRuntime


class FakeApiServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False
        self.stopped = threading.Event()

    def run(self) -> None:
        self.started = True
        while not self.should_exit:
            self.stopped.wait(0.001)
        self.stopped.set()


class ServerTest(unittest.TestCase):
    def test_start_timeout_does_not_wait_forever_for_a_stuck_server(self) -> None:
        class NeverStartingServer:
            def __init__(self) -> None:
                self.started = False
                self.should_exit = False
                self.stop_requested = threading.Event()

            def run(self) -> None:
                self.stop_requested.wait(10)

        service = MultiVisionService(Configuration())
        fake_server = NeverStartingServer()
        api_runtime = ApiServerRuntime(
            service,
            startup_timeout_seconds=0.01,
            server_factory=lambda _application, _host, _port: fake_server,
        )

        started_at = time.monotonic()
        with self.assertRaises(TimeoutError):
            api_runtime.start()
        duration_seconds = time.monotonic() - started_at
        assert duration_seconds < 1, f'{duration_seconds=}'
        assert fake_server.should_exit

        fake_server.stop_requested.set()
        assert api_runtime._thread is not None
        api_runtime._thread.join(1)
        assert not api_runtime._thread.is_alive()

    def test_api_worker_starts_and_stops_without_owning_service_lifecycle(self) -> None:
        service = MultiVisionService(Configuration())
        fake_server = FakeApiServer()
        factory_arguments: list[tuple[object, str, int]] = []

        def make_server(application: object, host: str, port: int) -> FakeApiServer:
            factory_arguments.append((application, host, port))
            return fake_server

        api_runtime = ApiServerRuntime(
            service,
            host='localhost',
            port=8123,
            startup_timeout_seconds=1,
            server_factory=make_server,
        )
        with TestClient(api_runtime.application) as client:
            response = client.get('/health')
        assert response.status_code == 200, response.text
        assert response.json()['service'] == 'stopped'
        assert not service.is_running, f'{service.is_running=}'

        api_runtime.start()
        api_runtime.shutdown()

        assert len(factory_arguments) == 1, f'{factory_arguments=}'
        assert factory_arguments[0][1:] == ('localhost', 8123)
        assert fake_server.should_exit
        assert fake_server.stopped.is_set()
        assert not service.is_running, f'{service.is_running=}'
        assert not api_runtime.is_running, f'{api_runtime.is_running=}'


if __name__ == '__main__':
    unittest.main()
