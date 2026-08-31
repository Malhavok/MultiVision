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
DEFAULT_TIMEOUT_SECONDS = 30.0


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
    except (ServiceClientError, OSError, RuntimeError, ValueError) as ex:
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

    cameras_parser = subparsers.add_parser(
        'cameras',
        help='list or manage session cameras',
    )
    camera_subparsers = cameras_parser.add_subparsers(
        dest='cameras_command',
        required=True,
    )
    list_parser = camera_subparsers.add_parser(
        'list',
        help='list session cameras',
    )
    list_parser.set_defaults(command_handler='cameras_list')
    rename_parser = camera_subparsers.add_parser(
        'rename',
        help='rename a session camera',
    )
    rename_parser.add_argument('slot_id')
    rename_parser.add_argument('name')
    rename_parser.set_defaults(command_handler='cameras_rename')
    close_parser = camera_subparsers.add_parser(
        'close',
        help='close a session camera',
    )
    close_parser.add_argument('slot_id')
    close_parser.set_defaults(command_handler='cameras_close')
    open_parser = camera_subparsers.add_parser(
        'open',
        help='open a session camera',
    )
    open_parser.add_argument('slot_id')
    open_parser.set_defaults(command_handler='cameras_open')
    area_parser = camera_subparsers.add_parser(
        'area',
        help='enable or disable a camera diagnostic area',
    )
    area_subparsers = area_parser.add_subparsers(
        dest='area_command',
        required=True,
    )
    area_enable_parser = area_subparsers.add_parser(
        'enable',
        help='enable a camera diagnostic area',
    )
    area_enable_parser.add_argument('slot_id')
    area_enable_parser.set_defaults(
        command_handler='cameras_area',
        desired_area_enabled=True,
    )
    area_disable_parser = area_subparsers.add_parser(
        'disable',
        help='disable a camera diagnostic area',
    )
    area_disable_parser.add_argument('slot_id')
    area_disable_parser.set_defaults(
        command_handler='cameras_area',
        desired_area_enabled=False,
    )
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
    pattern_parser = calibration_subparsers.add_parser(
        'pattern',
        help='show or hide the calibration pattern',
    )
    pattern_subparsers = pattern_parser.add_subparsers(
        dest='pattern_command',
        required=True,
    )
    pattern_show_parser = pattern_subparsers.add_parser('show', help='keep tags projected')
    pattern_show_parser.set_defaults(command_handler='calibration_pattern_show')
    pattern_hide_parser = pattern_subparsers.add_parser('hide', help='hide projected tags')
    pattern_hide_parser.set_defaults(command_handler='calibration_pattern_hide')

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
    for overlay_kind in ('grid', 'circle', 'rect', 'text', 'line', 'ruler'):
        create_overlay_parser = overlay_subparsers.add_parser(
            overlay_kind,
            help=f'create a {overlay_kind} overlay',
        )
        if overlay_kind == 'grid':
            grid_mode = create_overlay_parser.add_mutually_exclusive_group(required=True)
            grid_mode.add_argument(
                '--spec-json',
                dest='spec_json',
                type=_parse_json_object,
                help='validated overlay request as a JSON object',
            )
            grid_mode.add_argument(
                '--fill-projector',
                action='store_true',
                help='fill the projector footprint with a physical grid',
            )
            create_overlay_parser.add_argument(
                '--spacing',
                type=_parse_quantity_argument,
                help='physical spacing such as 35mm or 1in',
            )
            create_overlay_parser.add_argument('--name', default=None)
            create_overlay_parser.add_argument(
                '--angle-deg',
                type=_parse_finite_float,
                default=0.0,
            )
            create_overlay_parser.add_argument('--colour', default='#ffffff')
        else:
            create_overlay_parser.add_argument(
                '--spec-json',
                required=True,
                type=_parse_json_object,
                help='validated overlay request as a JSON object',
            )
        create_overlay_parser.set_defaults(
            command_handler='overlay_create',
            overlay_kind=overlay_kind,
            fill_projector=False,
        )
    for lifecycle_action in ('show', 'hide', 'remove'):
        lifecycle_parser = overlay_subparsers.add_parser(
            lifecycle_action,
            help=f'{lifecycle_action} a generic overlay',
        )
        selector_group = lifecycle_parser.add_mutually_exclusive_group(required=True)
        selector_group.add_argument('--id', dest='overlay_id', type=_parse_non_empty_argument)
        selector_group.add_argument(
            '--name',
            dest='overlay_name',
            type=_parse_non_empty_argument,
        )
        lifecycle_parser.set_defaults(
            command_handler='overlay_lifecycle',
            overlay_lifecycle_action=lifecycle_action,
        )

    overlays_parser = subparsers.add_parser(
        'overlays',
        help='manage generic overlays',
    )
    overlays_subparsers = overlays_parser.add_subparsers(
        dest='overlays_command',
        required=True,
    )
    overlays_list_parser = overlays_subparsers.add_parser(
        'list',
        help='list generic overlays',
    )
    overlays_list_parser.set_defaults(command_handler='overlays_list')
    overlays_clear_parser = overlays_subparsers.add_parser(
        'clear',
        help='clear generic overlays',
    )
    overlays_clear_parser.set_defaults(command_handler='overlays_clear')

    metric_parser = subparsers.add_parser(
        'metric',
        help='manage metric surface calibration and rulers',
    )
    metric_subparsers = metric_parser.add_subparsers(
        dest='metric_command',
        required=True,
    )
    metric_target_parser = metric_subparsers.add_parser(
        'target',
        help='generate metric calibration artifacts',
    )
    metric_target_subparsers = metric_target_parser.add_subparsers(
        dest='metric_target_command',
        required=True,
    )
    metric_target_generate_parser = metric_target_subparsers.add_parser(
        'generate',
        help='write the printable metric target SVG',
    )
    metric_target_generate_parser.add_argument(
        '--output',
        required=True,
        type=_parse_output_path,
        help='write the generated SVG to this path',
    )
    metric_target_generate_parser.set_defaults(
        command_handler='metric_target_generate',
    )

    metric_calibrate_parser = metric_subparsers.add_parser(
        'calibrate',
        help='calibrate the shared metric surface',
    )
    metric_calibrate_parser.add_argument(
        '--camera',
        required=True,
        type=_parse_non_empty_argument,
        help='camera session slot used to observe the target',
    )
    metric_calibrate_parser.set_defaults(command_handler='metric_calibrate')

    metric_status_parser = metric_subparsers.add_parser(
        'status',
        help='show metric calibration status',
    )
    metric_status_parser.set_defaults(command_handler='metric_status')

    metric_clear_parser = metric_subparsers.add_parser(
        'clear',
        help='clear metric calibration',
    )
    metric_clear_parser.set_defaults(command_handler='metric_clear')

    metric_ruler_parser = metric_subparsers.add_parser(
        'ruler',
        help='create or clear the metric ruler',
    )
    metric_ruler_parser.add_argument(
        '--from-mm',
        type=_parse_metric_point,
        dest='from_mm',
        help='surface start point as x,y in millimetres',
    )
    metric_ruler_parser.add_argument(
        '--to-mm',
        type=_parse_metric_point,
        dest='to_mm',
        help='surface end point as x,y in millimetres',
    )
    metric_ruler_parser.add_argument(
        '--unit',
        choices=('mm', 'cm', 'in'),
        default=None,
        help='unit used for the displayed length (default: mm)',
    )
    metric_ruler_parser.add_argument(
        '--observed-length',
        type=_parse_positive_finite_float,
        default=None,
        help='optional physically measured length',
    )
    metric_ruler_parser.add_argument(
        '--observed-unit',
        choices=('mm', 'cm', 'in'),
        default=None,
        help='unit of --observed-length (default: mm)',
    )
    metric_ruler_parser.add_argument(
        'ruler_action',
        nargs='?',
        choices=('clear',),
    )
    metric_ruler_parser.set_defaults(command_handler='metric_ruler')
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
        'cameras_rename': _cameras_rename,
        'cameras_close': _cameras_close,
        'cameras_open': _cameras_open,
        'cameras_area': _cameras_area,
        'calibrate': _calibrate,
        'calibration_verify': _calibration_verify,
        'calibration_status': _calibration_status,
        'calibration_pattern_show': _calibration_pattern_show,
        'calibration_pattern_hide': _calibration_pattern_hide,
        'snapshot': _snapshot,
        'point': _point,
        'overlay_clear': _overlay_clear,
        'overlay_create': _overlay_create,
        'overlay_lifecycle': _overlay_lifecycle,
        'overlays_list': _overlays_list,
        'overlays_clear': _overlays_clear,
        'metric_target_generate': _metric_target_generate,
        'metric_calibrate': _metric_calibrate,
        'metric_status': _metric_status,
        'metric_clear': _metric_clear,
        'metric_ruler': _metric_ruler,
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
    return client.get('/cameras')


def _cameras_rename(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    slot_id = _quote_path_component(arguments.slot_id)
    return client.post(
        f'/cameras/{slot_id}/rename',
        {'name': arguments.name},
    )


def _cameras_close(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    slot_id = _quote_path_component(arguments.slot_id)
    return client.post(f'/cameras/{slot_id}/close')


def _cameras_open(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    slot_id = _quote_path_component(arguments.slot_id)
    return client.post(f'/cameras/{slot_id}/open')


def _cameras_area(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    slot_id = _quote_path_component(arguments.slot_id)
    return client.post(
        f'/cameras/{slot_id}/area',
        {'enabled': arguments.desired_area_enabled},
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


def _calibration_pattern_show(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.post('/calibration/pattern')


def _calibration_pattern_hide(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.delete('/calibration/pattern')


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


def _overlay_create(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    if arguments.overlay_kind == 'grid' and arguments.fill_projector:
        if arguments.spacing is None:
            raise ValueError('--spacing is required with --fill-projector')
        spacing_value, spacing_unit = arguments.spacing
        payload = {
            'name': arguments.name,
            'spacing': {'value': spacing_value, 'unit': spacing_unit},
            'angle_deg': arguments.angle_deg,
            'style': {'colour': arguments.colour},
        }
        return client.post('/overlays/grid/projector-footprint', payload)
    spec = _validate_overlay_spec(arguments.overlay_kind, arguments.spec_json)
    return client.post(f'/overlays/{arguments.overlay_kind}', spec)


def _overlay_lifecycle(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    selector = _get_overlay_selector(arguments)
    selector_kind, selector_value = selector
    path_selector = _quote_path_component(selector_value)
    path = f'/overlays/{selector_kind}/{path_selector}'
    if arguments.overlay_lifecycle_action == 'remove':
        return client.delete(path)
    return client.post(f'{path}/{arguments.overlay_lifecycle_action}')


def _overlays_list(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.get('/overlays')


def _overlays_clear(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.delete('/overlays')


def _validate_overlay_spec(
    overlay_kind: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    request_types: dict[str, type[Any]]
    from multivision.overlays import (
        CircleRequest,
        GridRequest,
        LineRequest,
        RectRequest,
        RulerRequest,
        TextRequest,
    )

    request_types = {
        'grid': GridRequest,
        'circle': CircleRequest,
        'rect': RectRequest,
        'text': TextRequest,
        'line': LineRequest,
        'ruler': RulerRequest,
    }
    try:
        request_types[overlay_kind].model_validate(spec)
    except KeyError as ex:
        raise ValueError(f'Unknown overlay kind: {overlay_kind}') from ex
    except ValueError as ex:
        raise ValueError(f'Invalid {overlay_kind} overlay spec: {ex}') from ex
    return spec


def _get_overlay_selector(
    arguments: argparse.Namespace,
) -> tuple[str, str]:
    overlay_id = getattr(arguments, 'overlay_id', None)
    overlay_name = getattr(arguments, 'overlay_name', None)
    if (overlay_id is None) == (overlay_name is None):
        raise ValueError('exactly one of --id or --name is required')
    if overlay_id is not None:
        return 'id', overlay_id
    assert overlay_name is not None
    return 'name', overlay_name


def _metric_target_generate(
    _client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    _write_metric_target_svg(arguments.output)
    return ServiceResponse(204, '', b'')


def _metric_calibrate(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.post('/metric/calibration', {'camera': arguments.camera})


def _metric_status(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.get('/metric/calibration/status')


def _metric_clear(
    client: MultiVisionClient,
    _arguments: argparse.Namespace,
) -> ServiceResponse:
    return client.delete('/metric/calibration')


def _metric_ruler(
    client: MultiVisionClient,
    arguments: argparse.Namespace,
) -> ServiceResponse:
    if arguments.ruler_action == 'clear':
        if (
            arguments.from_mm is not None
            or arguments.to_mm is not None
            or arguments.unit is not None
            or arguments.observed_length is not None
            or arguments.observed_unit is not None
        ):
            raise ValueError('ruler clear cannot include ruler options')
        return client.delete('/metric/ruler')
    if arguments.from_mm is None or arguments.to_mm is None:
        raise ValueError('--from-mm and --to-mm are required unless clearing the ruler')
    if arguments.observed_length is None and arguments.observed_unit is not None:
        raise ValueError('--observed-unit requires --observed-length')

    payload: dict[str, Any] = {
        'from': {'x': arguments.from_mm[0], 'y': arguments.from_mm[1]},
        'to': {'x': arguments.to_mm[0], 'y': arguments.to_mm[1]},
        'unit': arguments.unit or 'mm',
    }
    if arguments.observed_length is not None:
        payload['observed_length'] = arguments.observed_length
        payload['observed_unit'] = arguments.observed_unit or 'mm'
    return client.post('/metric/ruler', payload)


def _write_metric_target_svg(output_path: pathlib.Path) -> None:
    from multivision.metric_target import write_metric_target_svg

    write_metric_target_svg(output_path)


def _print_command_response(arguments: argparse.Namespace, response: ServiceResponse) -> None:
    if arguments.command_handler == 'metric_target_generate':
        print(f'Wrote metric target to {arguments.output}')
        return
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


def _parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed_value = json.loads(value)
    except (TypeError, json.JSONDecodeError) as ex:
        raise argparse.ArgumentTypeError('must be valid JSON') from ex
    if not isinstance(parsed_value, dict):
        raise argparse.ArgumentTypeError('must be a JSON object')
    return parsed_value


def _parse_finite_float(value: str) -> float:
    try:
        parsed_value = float(value)
    except (TypeError, ValueError, OverflowError) as ex:
        raise argparse.ArgumentTypeError('must be a number') from ex
    if not math.isfinite(parsed_value):
        raise argparse.ArgumentTypeError('must be a finite number')
    return parsed_value


def _parse_positive_finite_float(value: str) -> float:
    parsed_value = _parse_finite_float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed_value


def _parse_quantity_argument(value: str) -> tuple[float, str]:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError('must be a finite number followed by mm, cm or in')
    value = value.strip()
    for unit in ('mm', 'cm', 'in'):
        if not value.endswith(unit):
            continue
        try:
            quantity_value = float(value[:-len(unit)].strip())
        except (TypeError, ValueError, OverflowError) as ex:
            raise argparse.ArgumentTypeError(
                'must be a finite number followed by mm, cm or in',
            ) from ex
        if not math.isfinite(quantity_value):
            raise argparse.ArgumentTypeError(
                'must be a finite number followed by mm, cm or in',
            )
        return quantity_value, unit
    raise argparse.ArgumentTypeError('must be a finite number followed by mm, cm or in')


def _parse_metric_point(value: str) -> tuple[float, float]:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError('must be two comma-separated numbers')
    coordinates = value.split(',')
    if len(coordinates) != 2:
        raise argparse.ArgumentTypeError('must be two comma-separated numbers')
    try:
        return (
            _parse_finite_float(coordinates[0].strip()),
            _parse_finite_float(coordinates[1].strip()),
        )
    except argparse.ArgumentTypeError as ex:
        raise argparse.ArgumentTypeError(
            'must be two comma-separated finite numbers',
        ) from ex


def _parse_non_empty_argument(value: str) -> str:
    if not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError('must be a non-empty value')
    return value.strip()


def _parse_output_path(value: str) -> pathlib.Path:
    if not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError('must be a non-empty path')
    return pathlib.Path(value)


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
