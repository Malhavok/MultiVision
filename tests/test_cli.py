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
            ['cameras', 'bind', 'overhead', 'device/one'],
            ['calibrate', '--camera', 'overhead'],
            ['calibration', 'verify', '--camera', 'overhead'],
            ['calibration', 'status'],
            ['point', '--camera', 'overhead', '--x', '12', '--y', '34'],
            ['overlay', 'clear'],
        ]

        for command in commands:
            with self.subTest(command=command):
                assert main(command, client) == 0

        assert [(method, url, payload) for method, url, payload, _ in requests] == [
            ('GET', 'http://service.test/health', None),
            ('GET', 'http://service.test/cameras/discovered', None),
            (
                'POST',
                'http://service.test/cameras/overhead/binding',
                {'device_id': 'device/one'},
            ),
            ('POST', 'http://service.test/calibration', {'camera': 'overhead'}),
            ('POST', 'http://service.test/calibration/verify', {'camera': 'overhead'}),
            ('GET', 'http://service.test/calibration/status', None),
            ('POST', 'http://service.test/overlay/point', {
                'camera': 'overhead',
                'x': 12.0,
                'y': 34.0,
            }),
            ('DELETE', 'http://service.test/overlay', None),
        ]

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
                    "print('cv2' in sys.modules, 'pygame' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f'{result.stdout=}, {result.stderr=}'
        assert result.stdout.strip() == 'False False', f'{result.stdout=}'


if __name__ == '__main__':
    unittest.main()
