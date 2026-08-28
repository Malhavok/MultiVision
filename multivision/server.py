"""Run the local FastAPI server without taking ownership of the application."""

from __future__ import annotations

import math
import threading
import time
from typing import (
    Any,
    Protocol,
)

from multivision.api import create_app
from multivision.application import MultiVisionService


DEFAULT_API_HOST = '127.0.0.1'
DEFAULT_API_PORT = 8000
DEFAULT_API_STARTUP_TIMEOUT_SECONDS = 10.0


class ApiServer(Protocol):
    started: bool
    should_exit: bool

    def run(self) -> None:
        ...


class ApiServerFactory(Protocol):
    def __call__(self, application: Any, host: str, port: int) -> ApiServer:
        ...


class ApiServerRuntime:
    """Own the API worker while the process lifecycle remains in ``main``."""

    def __init__(
        self,
        service: MultiVisionService,
        host: str = DEFAULT_API_HOST,
        port: int = DEFAULT_API_PORT,
        startup_timeout_seconds: float = DEFAULT_API_STARTUP_TIMEOUT_SECONDS,
        server_factory: ApiServerFactory | None = None,
    ) -> None:
        if not isinstance(service, MultiVisionService):
            raise TypeError('service must be a MultiVisionService')
        if not isinstance(host, str) or len(host) == 0:
            raise ValueError('host must be a non-empty string')
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise ValueError('port must be between 1 and 65535')
        if (
            not isinstance(startup_timeout_seconds, (int, float))
            or isinstance(startup_timeout_seconds, bool)
            or not math.isfinite(startup_timeout_seconds)
            or startup_timeout_seconds <= 0
        ):
            raise ValueError('startup_timeout_seconds must be positive')
        if server_factory is not None and not callable(server_factory):
            raise TypeError('server_factory must be callable')

        self.service = service
        self.host = host
        self.port = port
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.application = create_app(service, manage_lifecycle=False)
        self._server_factory = (
            server_factory if server_factory is not None else _make_uvicorn_server
        )
        self._server: ApiServer | None = None
        self._thread: threading.Thread | None = None
        self._thread_finished = threading.Event()
        self._server_error: BaseException | None = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        if self._is_running:
            return
        if self._thread is not None:
            raise RuntimeError('The API server has already stopped')

        self._server = self._server_factory(self.application, self.host, self.port)
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name='multivision-api',
        )
        self._thread.start()
        deadline = time.monotonic() + self.startup_timeout_seconds
        while True:
            if self._server_error is not None:
                error = self._server_error
                self._request_stop()
                self._thread.join(self.startup_timeout_seconds)
                self._server_error = None
                if self._thread.is_alive():
                    raise RuntimeError(
                        'The MultiVision API server failed to start and did not stop '
                        'during cleanup',
                    ) from error
                raise RuntimeError('The MultiVision API server failed to start') from error
            if self._server.started:
                self._is_running = True
                return
            if not self._thread.is_alive():
                raise RuntimeError('The MultiVision API server stopped during startup')
            if time.monotonic() >= deadline:
                self._request_stop()
                self._thread.join(self.startup_timeout_seconds)
                if self._thread.is_alive():
                    raise TimeoutError(
                        'The MultiVision API server did not start or stop in time',
                    )
                raise TimeoutError('The MultiVision API server did not start in time')
            self._thread_finished.wait(0.01)

    def shutdown(self) -> None:
        if self._server is None or self._thread is None:
            return
        self._request_stop()
        self._thread.join(self.startup_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError('The MultiVision API server did not stop in time')
        self._is_running = False
        if self._server_error is not None:
            error = self._server_error
            self._server_error = None
            raise RuntimeError('The MultiVision API server failed') from error

    def _run_server(self) -> None:
        assert self._server is not None
        try:
            self._server.run()
        except BaseException as ex:  # noqa: BLE001 (The main thread must own cleanup.)
            self._server_error = ex
        finally:
            self._thread_finished.set()

    def _request_stop(self) -> None:
        assert self._server is not None
        self._server.should_exit = True


def _make_uvicorn_server(application: Any, host: str, port: int) -> ApiServer:
    try:
        import uvicorn
    except ImportError as ex:
        raise RuntimeError(
            'Uvicorn is required to run the MultiVision API server',
        ) from ex

    configuration = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level='info',
    )
    return uvicorn.Server(configuration)


__all__ = [
    'ApiServer',
    'ApiServerFactory',
    'ApiServerRuntime',
    'DEFAULT_API_HOST',
    'DEFAULT_API_PORT',
]
