"""Thin HTTP client for the running MultiVision service."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple


DEFAULT_SERVICE_URL = 'http://127.0.0.1:8000'
DEFAULT_TIMEOUT_SECONDS = 5.0


class ServiceResponse(NamedTuple):
    status_code: int
    content_type: str
    body: bytes


class ServiceClientError(Exception):
    """Raised when the service cannot fulfil a CLI request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


RequestSender = Callable[
    [str, str, dict[str, Any] | None, float],
    ServiceResponse,
]


class MultiVisionClient:
    """Send capability requests to an already-running MultiVision service."""

    def __init__(
        self,
        service_url: str = DEFAULT_SERVICE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        request_sender: RequestSender | None = None,
    ) -> None:
        _validate_service_url(service_url)
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError('timeout_seconds must be a finite positive number')
        if request_sender is not None and not callable(request_sender):
            raise ValueError('request_sender must be callable')
        self.service_url = service_url.rstrip('/')
        self.timeout_seconds = float(timeout_seconds)
        self._request_sender = (
            _send_http_request
            if request_sender is None
            else request_sender
        )

    def get(self, path: str) -> ServiceResponse:
        return self._request('GET', path)

    def post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> ServiceResponse:
        return self._request('POST', path, payload)

    def delete(self, path: str) -> ServiceResponse:
        return self._request('DELETE', path)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> ServiceResponse:
        if not isinstance(method, str) or len(method.strip()) == 0:
            raise ValueError('method must be a non-empty string')
        if not isinstance(path, str) or len(path.strip()) == 0:
            raise ValueError('path must be a non-empty string')
        if payload is not None and not isinstance(payload, dict):
            raise ValueError('payload must be an object or None')
        try:
            response = self._request_sender(
                method,
                f'{self.service_url}/{path.lstrip("/")}',
                payload,
                self.timeout_seconds,
            )
        except ServiceClientError:
            raise
        except Exception as ex:  # noqa: BLE001 (normalise transport-seam failures).
            raise ServiceClientError(f'Could not contact MultiVision service: {ex}') from ex
        _validate_service_response(response)
        if not 200 <= response.status_code < 300:
            raise ServiceClientError(
                _format_service_failure(response),
                status_code=response.status_code,
            )
        return response


def main(
    argv: Sequence[str] | None = None,
    client: MultiVisionClient | None = None,
) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        service_client = (
            client
            if client is not None
            else MultiVisionClient(
                arguments.service_url,
                arguments.timeout_seconds,
            )
        )
        response = _run_command(service_client, arguments)
        _print_command_response(arguments, response)
    except (ServiceClientError, OSError, ValueError) as ex:
        print(f'multivision: {ex}', file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='multivision',
        description='Control a running MultiVision service.',
    )
    parser.add_argument(
        '--url',
        '--base-url',
        dest='service_url',
        default=DEFAULT_SERVICE_URL,
        help=f'service URL (default: {DEFAULT_SERVICE_URL})',
    )
    parser.add_argument(
        '--timeout-seconds',
        dest='timeout_seconds',
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f'HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    status_parser = subparsers.add_parser('status', help='show service health')
    status_parser.set_defaults(command_handler='status')

    cameras_parser = subparsers.add_parser('cameras', help='list or bind cameras')
    camera_subparsers = cameras_parser.add_subparsers(dest='cameras_command', required=True)
    list_parser = camera_subparsers.add_parser('list', help='list discovered camera devices')
    list_parser.set_defaults(command_handler='cameras_list')
    bind_parser = camera_subparsers.add_parser('bind', help='bind a logical name to a device')
    bind_parser.add_argument('logical_name')
    bind_parser.add_argument('device_id')
    bind_parser.set_defaults(command_handler='cameras_bind')

    calibrate_parser = subparsers.add_parser('calibrate', help='calibrate one or all cameras')
    calibrate_parser.add_argument('--camera', dest='camera', default=None)
    calibrate_parser.set_defaults(command_handler='calibrate')

    calibration_parser = subparsers.add_parser('calibration', help='inspect calibration')
    calibration_subparsers = calibration_parser.add_subparsers(
        dest='calibration_command',
        required=True,
    )
    verify_parser = calibration_subparsers.add_parser('verify', help='verify one or all cameras')
    verify_parser.add_argument('--camera', dest='camera', default=None)
    verify_parser.set_defaults(command_handler='calibration_verify')
    status_parser = calibration_subparsers.add_parser('status', help='show calibration status')
    status_parser.set_defaults(command_handler='calibration_status')

    snapshot_parser = subparsers.add_parser('snapshot', help='request a retained camera frame')
    snapshot_parser.add_argument('logical_name')
    snapshot_parser.add_argument(
        '--output',
        type=pathlib.Path,
        default=None,
        help='write an encoded image response to this file',
    )
    snapshot_parser.set_defaults(command_handler='snapshot')

    point_parser = subparsers.add_parser('point', help='request a camera-space overlay point')
    point_parser.add_argument('--camera', required=True)
    point_parser.add_argument('--x', required=True, type=_parse_finite_float)
    point_parser.add_argument('--y', required=True, type=_parse_finite_float)
    point_parser.set_defaults(command_handler='point')

    overlay_parser = subparsers.add_parser('overlay', help='manage the current overlay')
    overlay_subparsers = overlay_parser.add_subparsers(
        dest='overlay_command',
        required=True,
    )
    clear_parser = overlay_subparsers.add_parser('clear', help='clear the current overlay')
    clear_parser.set_defaults(command_handler='overlay_clear')
    return parser


def _run_command(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    command_handlers: dict[
        str,
        Callable[[MultiVisionClient, argparse.Namespace], ServiceResponse],
    ] = {
        'status': _status,
        'cameras_list': _cameras_list,
        'cameras_bind': _cameras_bind,
        'calibrate': _calibrate,
        'calibration_verify': _calibration_verify,
        'calibration_status': _calibration_status,
        'snapshot': _snapshot,
        'point': _point,
        'overlay_clear': _overlay_clear,
    }
    try:
        handler = command_handlers[arguments.command_handler]
    except KeyError as ex:
        raise ValueError('No handler exists for the requested CLI command') from ex
    return handler(client, arguments)


def _status(client: MultiVisionClient, _arguments: argparse.Namespace) -> ServiceResponse:
    return client.get('/health')


def _cameras_list(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.get('/cameras/discovered')


def _cameras_bind(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    logical_name = _quote_path_component(arguments.logical_name)
    return client.post(
        f'/cameras/{logical_name}/binding',
        {'device_id': arguments.device_id},
    )


def _calibrate(client: MultiVisionClient, arguments: argparse.Namespace) -> ServiceResponse:
    payload = None if arguments.camera is None else {'camera': arguments.camera}
    return client.post('/calibration', payload)


def _calibration_verify(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    payload = None if arguments.camera is None else {'camera': arguments.camera}
    return client.post('/calibration/verify', payload)


def _calibration_status(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.get('/calibration/status')


def _snapshot(client: MultiVisionClient, arguments: argparse.Namespace) -> ServiceResponse:
    logical_name = _quote_path_component(arguments.logical_name)
    return client.get(f'/cameras/{logical_name}/snapshot')


def _point(client: MultiVisionClient, arguments: argparse.Namespace) -> ServiceResponse:
    return client.post(
        '/overlay/point',
        {'camera': arguments.camera, 'x': arguments.x, 'y': arguments.y},
    )


def _overlay_clear(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.delete('/overlay')


def _print_command_response(arguments: argparse.Namespace, response: ServiceResponse) -> None:
    output_path = getattr(arguments, 'output', None)
    if arguments.command == 'snapshot' and output_path is not None:
        output_path.write_bytes(response.body)
        print(f'Wrote snapshot to {output_path}')
        return
    is_json_response = response.content_type.startswith('application/json') or (
        response.body.lstrip().startswith((b'{', b'['))
    )
    if is_json_response:
        try:
            data = json.loads(response.body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as ex:
            raise ValueError('MultiVision service returned malformed JSON') from ex
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    if arguments.command == 'snapshot':
        print(
            f'Received {len(response.body)} bytes ({response.content_type}); '
            'use --output to save the snapshot.',
        )
        return
    raise ValueError('MultiVision service returned an unsupported response')


def _format_service_failure(response: ServiceResponse) -> str:
    description = f'Service request failed with HTTP {response.status_code}'
    try:
        data = json.loads(response.body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return description
    if not isinstance(data, dict):
        return description
    error = data.get('error')
    if not isinstance(error, dict):
        return description
    code = error.get('code')
    message = error.get('message')
    if isinstance(code, str) and isinstance(message, str):
        return f'{description}: {code}: {message}'
    if isinstance(message, str):
        return f'{description}: {message}'
    return description


def _validate_service_url(service_url: str) -> None:
    if (
        not isinstance(service_url, str)
        or len(service_url) == 0
        or service_url != service_url.strip()
    ):
        raise ValueError('service_url must be a trimmed HTTP or HTTPS URL')
    parsed_url = urllib.parse.urlsplit(service_url)
    if parsed_url.scheme not in {'http', 'https'} or parsed_url.hostname is None:
        raise ValueError('service_url must be a valid HTTP or HTTPS URL')
    if '?' in service_url or '#' in service_url:
        raise ValueError('service_url must not contain a query or fragment')
    try:
        parsed_url.port
    except ValueError as ex:
        raise ValueError('service_url must contain a valid port') from ex


def _parse_finite_float(value: str) -> float:
    try:
        parsed_value = float(value)
    except (TypeError, ValueError, OverflowError) as ex:
        raise argparse.ArgumentTypeError('must be a number') from ex
    if not math.isfinite(parsed_value):
        raise argparse.ArgumentTypeError('must be a finite number')
    return parsed_value


def _quote_path_component(value: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError('path component must be a non-empty string')
    return urllib.parse.quote(value, safe='')


def _validate_service_response(response: ServiceResponse) -> None:
    if not isinstance(response, ServiceResponse):
        raise ServiceClientError('MultiVision service returned an invalid response')
    if (
        not isinstance(response.status_code, int)
        or isinstance(response.status_code, bool)
        or not 100 <= response.status_code <= 599
    ):
        raise ServiceClientError('MultiVision service returned an invalid status code')
    if not isinstance(response.content_type, str) or not isinstance(response.body, bytes):
        raise ServiceClientError('MultiVision service returned malformed response data')


def _send_http_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> ServiceResponse:
    request_data = None
    headers = {'Accept': 'application/json'}
    if payload is not None:
        request_data = json.dumps(payload, allow_nan=False).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(
        url,
        data=request_data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return ServiceResponse(
                response.status,
                response.headers.get('Content-Type', ''),
                response.read(),
            )
    except urllib.error.HTTPError as ex:
        return ServiceResponse(
            ex.code,
            ex.headers.get('Content-Type', '') if ex.headers is not None else '',
            ex.read(),
        )
    except OSError as ex:
        raise ServiceClientError(f'Could not contact MultiVision service: {ex}') from ex


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'DEFAULT_SERVICE_URL',
    'MultiVisionClient',
    'ServiceClientError',
    'ServiceResponse',
    'main',
]
