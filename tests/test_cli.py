import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from multivision.cli import (
    MultiVisionClient,
    ServiceResponse,
    main,
)


class CliTest(unittest.TestCase):
    def test_commands_send_capability_requests_to_the_service(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"ok": true}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        commands = [
            ['status'],
            ['cameras', 'list'],
            ['cameras', 'rename', 'camera-0', 'overhead'],
            ['cameras', 'close', 'camera-1'],
            ['cameras', 'open', 'camera-1'],
            ['cameras', 'area', 'enable', 'camera-0'],
            ['cameras', 'area', 'disable', 'camera-0'],
            ['calibrate', '--camera', 'overhead'],
            ['calibration', 'verify', '--camera', 'overhead'],
            ['calibration', 'status'],
            ['calibration', 'pattern', 'show'],
            ['calibration', 'pattern', 'hide'],
            ['point', '--camera', 'overhead', '--x', '12', '--y', '34'],
            ['overlay', 'clear'],
        ]

        for command in commands:
            with self.subTest(command=command):
                assert main(command, client) == 0

        assert [(method, url, payload) for method, url, payload, _ in requests] == [
            ('GET', 'http://service.test/health', None),
            ('GET', 'http://service.test/cameras', None),
            (
                'POST',
                'http://service.test/cameras/camera-0/rename',
                {'name': 'overhead'},
            ),
            ('POST', 'http://service.test/cameras/camera-1/close', None),
            ('POST', 'http://service.test/cameras/camera-1/open', None),
            (
                'POST',
                'http://service.test/cameras/camera-0/area',
                {'enabled': True},
            ),
            (
                'POST',
                'http://service.test/cameras/camera-0/area',
                {'enabled': False},
            ),
            ('POST', 'http://service.test/calibration', {'camera': 'overhead'}),
            ('POST', 'http://service.test/calibration/verify', {'camera': 'overhead'}),
            ('GET', 'http://service.test/calibration/status', None),
            ('POST', 'http://service.test/calibration/pattern', None),
            ('DELETE', 'http://service.test/calibration/pattern', None),
            ('POST', 'http://service.test/overlay/point', {
                'camera': 'overhead',
                'x': 12.0,
                'y': 34.0,
            }),
            ('DELETE', 'http://service.test/overlay', None),
        ]

    def test_metric_commands_delegate_to_http_with_expected_payloads(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"ok": true}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        commands = [
            ['metric', 'calibrate', '--camera', 'camera-0'],
            ['metric', 'status'],
            ['metric', 'clear'],
            [
                'metric', 'ruler', '--from-mm', '100,100', '--to-mm', '300,100',
                '--unit', 'cm', '--observed-length', '20', '--observed-unit', 'cm',
            ],
            ['metric', 'ruler', 'clear'],
        ]

        for command in commands:
            with self.subTest(command=command):
                assert main(command, client) == 0

        assert [(method, url, payload) for method, url, payload, _ in requests] == [
            ('POST', 'http://service.test/metric/calibration', {'camera': 'camera-0'}),
            ('GET', 'http://service.test/metric/calibration/status', None),
            ('DELETE', 'http://service.test/metric/calibration', None),
            (
                'POST',
                'http://service.test/metric/ruler',
                {
                    'from': {'x': 100.0, 'y': 100.0},
                    'to': {'x': 300.0, 'y': 100.0},
                    'unit': 'cm',
                    'observed_length': 20.0,
                    'observed_unit': 'cm',
                },
            ),
            ('DELETE', 'http://service.test/metric/ruler', None),
        ]

    def test_metric_target_generation_writes_locally_without_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(500, 'application/json', b'{"error": {}}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = pathlib.Path(temporary_directory) / 'metric-target.svg'
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    ['metric', 'target', 'generate', '--output', str(output_path)],
                    client,
                )

            assert result == 0
            assert output_path.read_bytes().startswith(b'<?xml'), f'{output_path=}'
            assert 'Wrote metric target' in output.getvalue(), f'{output.getvalue()=}'

        assert requests == [], f'{requests=}'

    def test_metric_ruler_rejects_invalid_values_before_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"ok": true}')

        client = MultiVisionClient(request_sender=request_sender)
        invalid_commands = [
            ['metric', 'ruler', '--from-mm', 'nan,1', '--to-mm', '2,3'],
            [
                'metric', 'ruler', '--from-mm', '1,1', '--to-mm', '2,3',
                '--observed-length', '0',
            ],
        ]
        for command in invalid_commands:
            with self.subTest(command=command), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(command, client)

        error_output = io.StringIO()
        with redirect_stderr(error_output):
            assert main(
                ['metric', 'ruler', '--from-mm', '1,1', '--to-mm', '2,3',
                 '--observed-unit', 'cm'],
                client,
            ) == 1
        assert '--observed-unit requires --observed-length' in error_output.getvalue()

        for command in [
            [
                'metric', 'ruler', '--from-mm', '1,1', '--to-mm', '2,3',
                '--observed-unit', 'mm',
            ],
            ['metric', 'ruler', 'clear', '--unit', 'cm'],
            ['metric', 'ruler', 'clear', '--observed-unit', 'mm'],
        ]:
            with self.subTest(command=command), redirect_stderr(io.StringIO()):
                assert main(command, client) == 1
        assert requests == [], f'{requests=}'

    def test_area_command_prints_service_schema_and_preserves_structured_failures(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            if payload == {'enabled': True}:
                return ServiceResponse(
                    200,
                    'application/json',
                    json.dumps({
                        'slot': 'camera-0',
                        'name': 'overhead',
                        'area_enabled': True,
                        'available_area': [[1.0, 2.0], [3.0, 4.0]],
                    }).encode('utf-8'),
                )
            return ServiceResponse(
                422,
                'application/json',
                json.dumps({
                    'error': {
                        'code': 'AVAILABLE_AREA_INVALID',
                        'message': 'area is degenerate',
                    },
                }).encode('utf-8'),
            )

        client = MultiVisionClient(
            'http://service.test',
            request_sender=request_sender,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(['cameras', 'area', 'enable', 'camera-0'], client)
        assert result == 0
        assert json.loads(output.getvalue())['area_enabled'] is True

        error_output = io.StringIO()
        with redirect_stderr(error_output):
            result = main(['cameras', 'area', 'disable', 'camera-0'], client)
        assert result == 1
        assert 'AVAILABLE_AREA_INVALID' in error_output.getvalue()
        assert [
            (url, payload)
            for _method, url, payload, _timeout_seconds in requests
        ] == [
            (
                'http://service.test/cameras/camera-0/area',
                {'enabled': True},
            ),
            (
                'http://service.test/cameras/camera-0/area',
                {'enabled': False},
            ),
        ], f'{requests=}'

    def test_snapshot_can_write_the_service_response_without_opening_a_camera(self) -> None:
        output = io.StringIO()
        client = MultiVisionClient(
            request_sender=lambda method, url, payload, timeout: ServiceResponse(
                200,
                'image/jpeg',
                b'fake-image',
            ),
        )
        with redirect_stdout(output):
            result = main(['snapshot', 'overhead'], client)

        assert result == 0
        assert 'Received 10 bytes' in output.getvalue(), f'{output.getvalue()=}'

    def test_snapshot_output_is_written_without_camera_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = pathlib.Path(temporary_directory) / 'snapshot.jpg'
            client = MultiVisionClient(
                request_sender=lambda method, url, payload, timeout: ServiceResponse(
                    200,
                    'image/jpeg',
                    b'fake-image',
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    ['snapshot', 'overhead', '--output', str(output_path)],
                    client,
                )

            assert result == 0
            assert output_path.read_bytes() == b'fake-image', f'{output_path=}'
            assert 'Wrote snapshot' in output.getvalue(), f'{output.getvalue()=}'

    def test_service_errors_are_printed_and_return_non_zero(self) -> None:
        client = MultiVisionClient(
            request_sender=lambda method, url, payload, timeout: ServiceResponse(
                503,
                'application/json',
                json.dumps({
                    'error': {
                        'code': 'CAMERA_UNAVAILABLE',
                        'message': 'Camera is unplugged',
                    },
                }).encode('utf-8'),
            ),
        )
        error_output = io.StringIO()
        with redirect_stderr(error_output):
            result = main(['status'], client)

        assert result == 1
        assert 'CAMERA_UNAVAILABLE' in error_output.getvalue()
        assert 'Camera is unplugged' in error_output.getvalue()

    def test_metric_service_errors_are_structured_and_fail_closed(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(
                422,
                'application/json',
                json.dumps({
                    'error': {
                        'code': 'METRIC_STALE',
                        'message': 'Metric calibration is stale',
                    },
                }).encode('utf-8'),
            )

        client = MultiVisionClient(
            'http://service.test',
            request_sender=request_sender,
        )
        error_output = io.StringIO()
        with redirect_stderr(error_output):
            result = main(
                [
                    'metric', 'ruler', '--from-mm', '10,10', '--to-mm', '110,10',
                ],
                client,
            )

        assert result == 1
        assert 'METRIC_STALE' in error_output.getvalue()
        assert 'Metric calibration is stale' in error_output.getvalue()
        assert requests == [
            (
                'POST',
                'http://service.test/metric/ruler',
                {
                    'from': {'x': 10.0, 'y': 10.0},
                    'to': {'x': 110.0, 'y': 10.0},
                    'unit': 'mm',
                },
                30.0,
            ),
        ], f'{requests=}'

    def test_transport_failures_and_malformed_responses_are_non_zero(self) -> None:
        def failing_request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            raise RuntimeError('connection failed')

        cases = [
            (failing_request_sender, 'connection failed'),
            (
                lambda method, url, payload, timeout: ServiceResponse(
                    '200',  # type: ignore[arg-type]
                    'application/json',
                    b'{}',
                ),
                'invalid status code',
            ),
            (
                lambda method, url, payload, timeout: ServiceResponse(
                    200,
                    'application/json',
                    '{}',  # type: ignore[arg-type]
                ),
                'malformed response data',
            ),
        ]
        for request_sender, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                client = MultiVisionClient(request_sender=request_sender)
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    result = main(['status'], client)

                assert result == 1
                assert expected_message in error_output.getvalue(), f'{error_output.getvalue()=}'
                assert 'Traceback' not in error_output.getvalue(), f'{error_output.getvalue()=}'

    def test_falsey_request_sender_is_still_used(self) -> None:
        class FalseyRequestSender:
            was_called = False

            def __bool__(self) -> bool:
                return False

            def __call__(
                self,
                method: str,
                url: str,
                payload: dict[str, Any] | None,
                timeout_seconds: float,
            ) -> ServiceResponse:
                self.was_called = True
                return ServiceResponse(200, 'application/json', b'{"ok": true}')

        request_sender = FalseyRequestSender()
        client = MultiVisionClient(request_sender=request_sender)  # type: ignore[arg-type]
        assert client.get('/health').status_code == 200
        assert request_sender.was_called

    def test_invalid_point_coordinates_are_rejected_before_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"ok": true}')

        client = MultiVisionClient(request_sender=request_sender)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['point', '--camera', 'overhead', '--x', 'nan', '--y', '1'], client)
        assert len(requests) == 0, f'{requests=}'

    def test_invalid_service_urls_fail_before_http(self) -> None:
        for service_url in [
            'service.test',
            'http://',
            'http://service.test/?unexpected=query',
            ' http://service.test',
        ]:
            with self.subTest(service_url=service_url):
                with self.assertRaises(ValueError):
                    MultiVisionClient(service_url)

    def test_cli_client_does_not_import_camera_or_display_modules(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys; import multivision.cli; '
                    "print('cv2' in sys.modules, 'pygame' in sys.modules, "
                    "'multivision.camera' in sys.modules, "
                    "'multivision.display' in sys.modules, "
                    "'multivision.geometry' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f'{result.stdout=}, {result.stderr=}'
        assert result.stdout.strip() == 'False False False False False', f'{result.stdout=}'


if __name__ == '__main__':
    unittest.main()
