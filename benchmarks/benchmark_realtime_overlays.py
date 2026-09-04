"""Benchmark a running MultiVision service without spawning mutation clients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple


SCHEMA_VERSION = 1
ACCEPTANCE_RATE_PER_SECOND = 30.0
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_WARMUP_REQUESTS = 2
DEFAULT_SAMPLES_PER_WORKLOAD = 10
DEFAULT_BATCH_SIZES = (10, 50, 100)
PRESENTATION_STALL_FRAME_LIMIT = 2
PREVIEW_MODES = ('active', 'low_rate', 'off')
WORKLOAD_NAMES = (
    'single_simple',
    'single_dynamic',
    'batch_10',
    'batch_50',
    'batch_100',
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce an honest report."""


class HttpResult(NamedTuple):
    status_code: int
    body: bytes
    elapsed_seconds: float
    cpu_seconds: float


class DiagnosticSnapshot(NamedTuple):
    counters: dict[str, int]
    timings: dict[str, dict[str, float | int]]
    configuration: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfiguration:
    service_urls: tuple[tuple[str, str], ...]
    warmup_requests: int = DEFAULT_WARMUP_REQUESTS
    samples_per_workload: int = DEFAULT_SAMPLES_PER_WORKLOAD
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    dynamic_group: str | None = None
    cli_command: tuple[str, ...] = (
        sys.executable,
        '-m',
        'multivision.cli',
        '--help',
    )

    def __post_init__(self) -> None:
        if len(self.service_urls) == 0:
            raise ValueError('At least one running service URL is required')
        if (
            not isinstance(self.warmup_requests, int)
            or isinstance(self.warmup_requests, bool)
            or self.warmup_requests < 0
        ):
            raise ValueError('warmup_requests must be a non-negative integer')
        if (
            not isinstance(self.samples_per_workload, int)
            or isinstance(self.samples_per_workload, bool)
            or self.samples_per_workload <= 0
        ):
            raise ValueError('samples_per_workload must be positive')
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError('timeout_seconds must be finite and positive')
        if len(self.batch_sizes) == 0 or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > 100
            for size in self.batch_sizes
        ):
            raise ValueError('batch_sizes must be positive values no greater than 100')
        if self.dynamic_group is not None and len(self.dynamic_group.strip()) == 0:
            raise ValueError('dynamic_group must be non-empty when supplied')
        if len(self.cli_command) == 0:
            raise ValueError('cli_command must not be empty')


class _HttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError('service URL must use HTTP or HTTPS')
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResult:
        body = None
        headers = {'Accept': 'application/json'}
        if payload is not None:
            body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(
            f'{self.base_url}/{path.lstrip("/")}',
            data=body,
            headers=headers,
            method=method,
        )
        started_seconds = time.perf_counter()
        started_cpu_seconds = time.process_time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = HttpResult(
                    response.status,
                    response.read(),
                    time.perf_counter() - started_seconds,
                    max(0.0, time.process_time() - started_cpu_seconds),
                )
        except urllib.error.HTTPError as ex:
            result = HttpResult(
                ex.code,
                ex.read(),
                time.perf_counter() - started_seconds,
                max(0.0, time.process_time() - started_cpu_seconds),
            )
        except urllib.error.URLError as ex:
            raise BenchmarkError(f'HTTP transport failed for {path}: {ex}') from ex
        return result

    def json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[HttpResult, dict[str, Any] | list[Any] | None]:
        result = self.request(method, path, payload)
        try:
            decoded = json.loads(result.body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        return result, decoded


def _build_benchmark_id(label: str) -> str:
    digest = bytearray(hashlib.sha256(label.encode('utf-8')).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(digest)))


def build_simple_overlay(index: int) -> dict[str, Any]:
    """Build a deterministic projector-native line request."""
    return {
        'kind': 'line',
        'id': _build_benchmark_id(f'multivision-benchmark-{index}'),
        'name': f'benchmark-simple-{index}',
        'start': {'type': 'projector', 'x': 100.0, 'y': 100.0, 'unit': 'px'},
        'end': {'type': 'projector', 'x': 300.0 + index % 17, 'y': 220.0, 'unit': 'px'},
        'style': {'colour': '#ffffff', 'fill': False, 'line_width_px': 2, 'intensity': 1.0},
    }


def build_dynamic_overlay(index: int, group: str) -> dict[str, Any]:
    """Build a deterministic fiducial-dependent arrow request."""
    return {
        'kind': 'arrow',
        'id': _build_benchmark_id(f'multivision-benchmark-dynamic-{index}'),
        'name': f'benchmark-dynamic-{index}',
        'start': {'type': 'fiducial', 'group': group, 'id': 0},
        'end': {
            'type': 'fiducial',
            'group': group,
            'id': 0,
            'local_offset': {'x': 30.0, 'y': 0.0, 'unit': 'mm'},
            'follow_rotation': True,
        },
        'geometry_space': 'surface_mm',
        'head_length': {'value': 12.0, 'unit': 'mm'},
        'head_width': {'value': 8.0, 'unit': 'mm'},
        'style': {'colour': '#00ff00', 'fill': False, 'line_width_px': 2, 'intensity': 1.0},
    }


def build_create_operation(
    index: int,
    dynamic_group: str | None,
    include_dynamic: bool = False,
) -> dict[str, Any]:
    request = (
        build_dynamic_overlay(index, dynamic_group or '__missing_benchmark_group__')
        if include_dynamic and index % 2 == 1
        else build_simple_overlay(index)
    )
    return {'op': 'create', 'request': request}


def run_injected_benchmark(
    samples_per_workload: int = DEFAULT_SAMPLES_PER_WORKLOAD,
) -> dict[str, Any]:
    """Return a repeatable schema smoke report, never a performance claim."""
    if (
        not isinstance(samples_per_workload, int)
        or isinstance(samples_per_workload, bool)
        or samples_per_workload <= 0
    ):
        raise ValueError('samples_per_workload must be positive')
    workloads = []
    for workload_name in WORKLOAD_NAMES:
        operation_count = (
            1
            if workload_name.startswith('single_')
            else int(workload_name.rsplit('_', 1)[1])
        )
        request_count = samples_per_workload
        elapsed_seconds = request_count * 0.001
        accepted_operations = request_count * operation_count
        workloads.append(
            _make_workload_report(
                workload_name,
                operation_count,
                request_count * operation_count,
                accepted_operations,
                request_count,
                elapsed_seconds,
                {},
                ['Injected mode is a repeatable smoke benchmark only.'],
            ),
        )
    return _make_report(
        'injected',
        {'injected': 'not a running service'},
        [{'workloads': workloads}],
        {
            'status': 'not_evaluated',
            'threshold_mutations_per_second': ACCEPTANCE_RATE_PER_SECOND,
            'accepted_mutations_per_second': None,
            'published_transactions_per_second': None,
            'criterion_excluded': True,
        },
        {
            'status': 'not_requested',
            'criterion_excluded': True,
            'latency_seconds': None,
            'command': None,
        },
        ['Injected mode cannot satisfy the real-service acceptance target.'],
    )


def run_real_benchmark(configuration: BenchmarkConfiguration) -> dict[str, Any]:
    """Drive each configured HTTP service and report measured evidence."""
    reports: list[dict[str, Any]] = []
    for preview_mode, service_url in configuration.service_urls:
        reports.append(
            _run_service_workloads(
                preview_mode,
                service_url,
                configuration,
            ),
        )
    cold_start = _measure_cli_startup(configuration.cli_command, configuration.timeout_seconds)
    acceptance = _get_acceptance_report(reports)
    report = _make_report(
        'real',
        dict(configuration.service_urls),
        reports,
        acceptance,
        cold_start,
        [
            'The acceptance rate uses only successful single-overlay HTTP mutations.',
            'Batch operation count is reported separately from one atomic publication.',
            'Target hardware results require a manual run with capture, tracking and '
            'projector paths active.',
        ],
    )
    report['benchmark_configuration'] = {
        'warmup_requests': configuration.warmup_requests,
        'samples_per_workload': configuration.samples_per_workload,
        'batch_sizes': list(configuration.batch_sizes),
        'timing_barrier': 'completed HTTP response',
        'presentation_stall_frame_limit': PRESENTATION_STALL_FRAME_LIMIT,
    }
    validate_benchmark_report(report)
    return report


def validate_benchmark_report(report: Mapping[str, Any]) -> None:
    """Validate the stable JSON contract used by smoke tests and result parsers."""
    required_fields = {
        'schema_version',
        'benchmark',
        'mode',
        'services',
        'acceptance',
        'workloads',
        'cold_start',
        'notes',
    }
    missing_fields = required_fields - set(report)
    if len(missing_fields) > 0:
        raise ValueError(f'Benchmark report is missing fields: {sorted(missing_fields)!r}')
    if type(report['schema_version']) is not int or report['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported benchmark report schema version')
    if report['benchmark'] != 'realtime_overlays':
        raise ValueError('Unexpected benchmark report name')
    if report['mode'] not in {'real', 'injected'}:
        raise ValueError('Benchmark mode is invalid')
    if not isinstance(report['services'], Mapping):
        raise ValueError('Benchmark services must be an object')
    if not isinstance(report['acceptance'], Mapping):
        raise ValueError('Benchmark acceptance must be an object')
    if not isinstance(report['cold_start'], Mapping):
        raise ValueError('Benchmark cold_start must be an object')
    if not isinstance(report['notes'], list):
        raise ValueError('Benchmark notes must be an array')
    if not isinstance(report['workloads'], list) or len(report['workloads']) == 0:
        raise ValueError('Benchmark report must contain workloads')
    for workload in report['workloads']:
        if not isinstance(workload, Mapping):
            raise ValueError('Workloads must be objects')
        for field_name in (
            'name',
            'operation_count_per_request',
            'request_count',
            'requested_operations',
            'accepted_operations',
            'published_transactions',
            'tracking_frames',
            'tracking_cycles_with_frames',
            'elapsed_seconds',
            'accepted_operations_per_second',
            'published_transactions_per_second',
            'components',
            'failures',
        ):
            if field_name not in workload:
                raise ValueError(f'Workload is missing {field_name!r}')
        if not isinstance(workload['components'], Mapping):
            raise ValueError('Workload components must be an object')
        for component_name in (
            'http_transport',
            'request_validation',
            'registry_candidate_build_publication',
            'mutation_publication',
            'detection_tracking',
            'spatial_resolution_materialisation',
            'presentation_cadence',
            'projector_presentation',
            'presentation_stalls',
            'preview_conversion',
        ):
            if not isinstance(workload['components'].get(component_name), Mapping):
                raise ValueError(f'Workload is missing component {component_name!r}')
        if not isinstance(workload['failures'], list):
            raise ValueError('Workload failures must be an array')
    _validate_finite_numbers(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Benchmark a running MultiVision HTTP service.',
    )
    parser.add_argument('--injected', action='store_true', help='run schema smoke mode only')
    parser.add_argument('--service-url', default=None)
    parser.add_argument('--preview-mode', choices=PREVIEW_MODES, default='active')
    parser.add_argument(
        '--mode-url',
        action='append',
        default=[],
        metavar='MODE=URL',
        help='add a separately configured running service for active, low_rate or off',
    )
    parser.add_argument('--warmup-requests', type=int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES_PER_WORKLOAD)
    parser.add_argument('--timeout-seconds', type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument('--dynamic-group', default=None)
    parser.add_argument(
        '--cli-command',
        default=None,
        help='optional command to measure for cold-start, parsed with shell quoting',
    )
    args = parser.parse_args(argv)
    try:
        if args.injected:
            report = run_injected_benchmark(args.samples)
        else:
            service_urls = _parse_service_urls(
                args.service_url,
                args.preview_mode,
                args.mode_url,
            )
            cli_command = (
                tuple(shlex.split(args.cli_command))
                if args.cli_command is not None
                else (
                    sys.executable,
                    '-m',
                    'multivision.cli',
                    '--help',
                )
            )
            report = run_real_benchmark(
                BenchmarkConfiguration(
                    service_urls,
                    args.warmup_requests,
                    args.samples,
                    args.timeout_seconds,
                    dynamic_group=args.dynamic_group,
                    cli_command=cli_command,
                ),
            )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (BenchmarkError, ValueError, OSError) as ex:
        print(f'benchmark failed: {ex}', file=sys.stderr)
        return 2
    if report['mode'] == 'real' and report['acceptance']['status'] == 'fail':
        return 1
    return 0


def _run_service_workloads(
    preview_mode: str,
    service_url: str,
    configuration: BenchmarkConfiguration,
) -> dict[str, Any]:
    client = _HttpClient(service_url, configuration.timeout_seconds)
    health_result, health_data = client.json_request('GET', '/health')
    if health_result.status_code != 200:
        raise BenchmarkError(
            f'{preview_mode} service is not healthy: HTTP {health_result.status_code}',
        )
    service_configuration = _get_service_configuration(client, health_data)
    reported_preview_mode = service_configuration['preview_mode']
    if reported_preview_mode is not None and reported_preview_mode != preview_mode:
        raise BenchmarkError(
            f'Service at {service_url} reports preview mode '
            f'{reported_preview_mode!r}, expected {preview_mode!r}',
        )
    dynamic_group = configuration.dynamic_group or _get_first_group(client)
    diagnostics_before = _get_diagnostics(client)
    workloads: list[dict[str, Any]] = []
    workloads.extend(
        _measure_workload(
            client,
            'single_simple',
            1,
            configuration.samples_per_workload,
            configuration.warmup_requests,
            dynamic_group=None,
        ),
    )
    workloads.extend(
        _measure_workload(
            client,
            'single_dynamic',
            1,
            configuration.samples_per_workload,
            configuration.warmup_requests,
            dynamic_group=dynamic_group,
        ),
    )
    for batch_size in configuration.batch_sizes:
        if batch_size > service_configuration['max_batch_operations']:
            workloads.append(
                _make_workload_report(
                    f'batch_{batch_size}',
                    batch_size,
                    0,
                    0,
                    0,
                    0.0,
                    {},
                    [
                        f'Configured max_batch_operations is '
                        f'{service_configuration["max_batch_operations"]}; batch was not sent.',
                    ],
                ),
            )
            continue
        workloads.extend(
            _measure_workload(
                client,
                f'batch_{batch_size}',
                batch_size,
                configuration.samples_per_workload,
                configuration.warmup_requests,
                dynamic_group=dynamic_group,
            ),
        )
    diagnostics_after = _get_diagnostics(client)
    # Do not leave the final batch on the projector after the measurement.
    client.request('DELETE', '/overlays')
    service_run = {
        'preview_mode': preview_mode,
        'service_url': service_url,
        'service_configuration': service_configuration,
        'workloads': workloads,
        'diagnostics': _subtract_diagnostics(diagnostics_after, diagnostics_before),
    }
    service_run['acceptance'] = _get_acceptance_report([service_run])
    return service_run


def _measure_workload(
    client: _HttpClient,
    workload_name: str,
    operation_count: int,
    samples_per_workload: int,
    warmup_requests: int,
    dynamic_group: str | None,
) -> list[dict[str, Any]]:
    client.request('DELETE', '/overlays')
    for warmup_index in range(warmup_requests):
        request_path, payload = _build_workload_request(
            workload_name,
            warmup_index * operation_count
            + (1 if workload_name == 'single_dynamic' else 0),
            operation_count,
            dynamic_group,
        )
        client.json_request('POST', request_path, payload)
    client.request('DELETE', '/overlays')
    measurement_diagnostics_before = _get_diagnostics(client)
    samples: list[HttpResult] = []
    failures: list[str] = []
    workload_started_seconds = time.perf_counter()
    for sample_index in range(samples_per_workload):
        request_path, payload = _build_workload_request(
            workload_name,
            sample_index * operation_count
            + (1 if workload_name == 'single_dynamic' else 0),
            operation_count,
            dynamic_group,
        )
        result, _response_data = client.json_request('POST', request_path, payload)
        samples.append(result)
        if not 200 <= result.status_code < 300:
            failure_body = result.body[:200].decode('utf-8', 'replace')
            failures.append(f'HTTP {result.status_code}: {failure_body}')
    elapsed_seconds = time.perf_counter() - workload_started_seconds
    accepted_request_count = sum(
        200 <= result.status_code < 300
        for result in samples
    )
    diagnostics_after = _get_diagnostics(client)
    component_diagnostics = _subtract_diagnostics(
        diagnostics_after,
        measurement_diagnostics_before,
    )
    counters = component_diagnostics.get('counters', {})
    accepted_operations = counters.get('accepted_mutations')
    published_transactions = counters.get('published_transactions')
    if not isinstance(accepted_operations, int) or isinstance(accepted_operations, bool):
        failures.append('Running service did not report accepted_mutations diagnostics.')
        accepted_operations = 0
    if not isinstance(published_transactions, int) or isinstance(
        published_transactions,
        bool,
    ):
        failures.append('Running service did not report published_transactions diagnostics.')
        published_transactions = 0
    requested_operations = len(samples) * operation_count
    if accepted_operations < 0 or accepted_operations > requested_operations:
        failures.append(
            f'accepted_mutations diagnostics exceeded the workload boundary: '
            f'{accepted_operations} not in [0, {requested_operations}].',
        )
        accepted_operations = 0
    if published_transactions < 0 or published_transactions > len(samples):
        failures.append(
            f'published_transactions diagnostics exceeded the request boundary: '
            f'{published_transactions} not in [0, {len(samples)}].',
        )
        published_transactions = 0
    if published_transactions > accepted_request_count:
        failures.append(
            'published_transactions diagnostics exceeded successful HTTP requests.',
        )
        published_transactions = 0
    return [
        _make_workload_report(
            workload_name,
            operation_count,
            requested_operations,
            accepted_operations,
            published_transactions,
            elapsed_seconds,
            component_diagnostics,
            failures,
            http_samples=samples,
        ),
    ]


def _build_workload_request(
    workload_name: str,
    first_index: int,
    operation_count: int,
    dynamic_group: str | None,
) -> tuple[str, dict[str, Any]]:
    if workload_name == 'single_simple':
        return '/overlays/line', build_simple_overlay(first_index)
    if workload_name == 'single_dynamic':
        return (
            '/overlays/arrow',
            build_dynamic_overlay(
                first_index,
                dynamic_group or '__missing_benchmark_group__',
            ),
        )
    return (
        '/overlays/batch',
        _build_batch_payload(
            first_index,
            operation_count,
            dynamic_group,
            include_dynamic=True,
        ),
    )


def _build_batch_payload(
    first_index: int,
    operation_count: int,
    dynamic_group: str | None,
    include_dynamic: bool = True,
) -> dict[str, Any]:
    return {
        'operations': [
            build_create_operation(
                first_index + index,
                dynamic_group,
                include_dynamic,
            )
            for index in range(operation_count)
        ],
    }


def _get_first_group(client: _HttpClient) -> str | None:
    result, data = client.json_request('GET', '/fiducial-groups')
    if result.status_code != 200 or not isinstance(data, Mapping):
        return None
    groups = data.get('groups')
    if not isinstance(groups, Mapping) or len(groups) == 0:
        return None
    first_group = sorted(str(group) for group in groups)[0]
    return first_group


def _get_service_configuration(
    client: _HttpClient,
    health_data: dict[str, Any] | list[Any] | None,
) -> dict[str, Any]:
    del health_data
    result, data = client.json_request('GET', '/diagnostics/benchmark')
    if result.status_code != 200 or not isinstance(data, Mapping):
        raise BenchmarkError('Running service does not expose benchmark diagnostics')
    configuration = data.get('configuration', {})
    if not isinstance(configuration, Mapping):
        configuration = {}
    max_batch_operations = configuration.get('max_batch_operations', 100)
    if (
        not isinstance(max_batch_operations, int)
        or isinstance(max_batch_operations, bool)
        or max_batch_operations <= 0
    ):
        raise BenchmarkError('Running service returned an invalid batch limit')
    return {
        'max_batch_operations': max_batch_operations,
        'preview_mode': configuration.get('preview_mode'),
        'preview_low_rate_hz': configuration.get('preview_low_rate_hz'),
    }


def _get_diagnostics(client: _HttpClient) -> DiagnosticSnapshot:
    result, data = client.json_request('GET', '/diagnostics/benchmark')
    if result.status_code != 200 or not isinstance(data, Mapping):
        raise BenchmarkError('Running service does not expose benchmark diagnostics')
    if data.get('available') is False:
        raise BenchmarkError('Running service does not expose benchmark diagnostics')
    counters = data.get('counters', {})
    timings = data.get('timings', {})
    if not isinstance(counters, Mapping) or not isinstance(timings, Mapping):
        raise BenchmarkError('Running service returned malformed benchmark diagnostics')
    counter_values: dict[str, int] = {}
    for key, value in counters.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BenchmarkError('Running service returned invalid benchmark counters')
        counter_values[str(key)] = value
    timing_values: dict[str, dict[str, float | int]] = {}
    for component, values in timings.items():
        if not isinstance(values, Mapping):
            raise BenchmarkError('Running service returned malformed benchmark timings')
        timing_values[str(component)] = {
            str(field_name): value
            for field_name, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    configuration = data.get('configuration', {})
    return DiagnosticSnapshot(
        counter_values,
        timing_values,
        dict(configuration) if isinstance(configuration, Mapping) else {},
    )


def _subtract_diagnostics(
    after: DiagnosticSnapshot,
    before: DiagnosticSnapshot,
) -> dict[str, Any]:
    counters = {
        key: after.counters.get(key, 0) - before.counters.get(key, 0)
        for key in set(after.counters) | set(before.counters)
    }
    timings: dict[str, dict[str, float | int]] = {}
    for component in set(after.timings) | set(before.timings):
        after_values = after.timings.get(component, {})
        before_values = before.timings.get(component, {})
        sample_count = int(after_values.get('sample_count', 0)) - int(
            before_values.get('sample_count', 0),
        )
        total_seconds = float(after_values.get('total_seconds', 0.0)) - float(
            before_values.get('total_seconds', 0.0),
        )
        cpu_seconds = float(after_values.get('cpu_seconds', 0.0)) - float(
            before_values.get('cpu_seconds', 0.0),
        )
        maximum_seconds = (
            float(after_values.get('maximum_seconds', 0.0))
            if sample_count > 0
            else 0.0
        )
        stall_count = int(after_values.get('stall_count', 0)) - int(
            before_values.get('stall_count', 0),
        )
        timings[component] = {
            'sample_count': sample_count,
            'total_seconds': max(0.0, total_seconds),
            'mean_seconds': max(0.0, total_seconds) / sample_count
            if sample_count > 0
            else 0.0,
            'maximum_seconds': maximum_seconds,
            'cpu_seconds': max(0.0, cpu_seconds),
            'stall_count': stall_count,
        }
    return {'counters': counters, 'timings': timings}


def _make_workload_report(
    name: str,
    operation_count: int,
    requested_operations: int,
    accepted_operations: int,
    published_transactions: int,
    elapsed_seconds: float,
    diagnostics: Mapping[str, Any],
    failures: Sequence[str],
    http_samples: Sequence[HttpResult] = (),
) -> dict[str, Any]:
    timing_data = diagnostics.get('timings', {})
    presentation_render = timing_data.get('presentation_render', _empty_component())
    projector_presentation = timing_data.get(
        'projector_presentation',
        _empty_component(),
    )
    counters = diagnostics.get('counters', {})
    tracking_frames = int(counters.get('tracking_frames', 0))
    tracking_cycles_with_frames = int(counters.get('tracking_cycles_with_frames', 0))
    components = {
        'http_transport': _summarise_http_samples(http_samples),
        'request_validation': timing_data.get(
            'request_validation_and_dispatch',
            _empty_component(),
        ),
        'registry_candidate_build_publication': timing_data.get(
            'registry_candidate_build_publication',
            _empty_component(),
        ),
        'mutation_publication': timing_data.get(
            'registry_candidate_build_publication',
            _empty_component(),
        ),
        'detection_tracking': timing_data.get('detection_tracking', _empty_component()),
        'spatial_resolution_materialisation': timing_data.get(
            'spatial_resolution_materialisation',
            _empty_component(),
        ),
        'presentation_cadence': presentation_render,
        'projector_presentation': projector_presentation,
        'presentation_stalls': {
            'sample_count': int(projector_presentation.get('sample_count', 0)),
            'total_seconds': 0.0,
            'mean_seconds': 0.0,
            'maximum_seconds': float(presentation_render.get('maximum_seconds', 0.0)),
            'cpu_seconds': 0.0,
            'stall_count': int(presentation_render.get('stall_count', 0))
            + int(projector_presentation.get('stall_count', 0)),
        },
        'preview_conversion': timing_data.get('preview_conversion', _empty_component()),
    }
    return {
        'name': name,
        'operation_count_per_request': operation_count,
        'request_count': (
            len(http_samples)
            if len(http_samples) > 0
            else requested_operations // operation_count
        ),
        'accepted_request_count': (
            sum(200 <= sample.status_code < 300 for sample in http_samples)
            if len(http_samples) > 0
            else published_transactions
        ),
        'failed_request_count': (
            sum(not 200 <= sample.status_code < 300 for sample in http_samples)
            if len(http_samples) > 0
            else 0
        ),
        'requested_operations': requested_operations,
        'accepted_operations': accepted_operations,
        'published_transactions': published_transactions,
        'tracking_frames': tracking_frames,
        'tracking_cycles_with_frames': tracking_cycles_with_frames,
        'elapsed_seconds': elapsed_seconds,
        'accepted_operations_per_second': (
            accepted_operations / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        'published_transactions_per_second': (
            published_transactions / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        'components': components,
        'failures': list(failures),
    }


def _summarise_http_samples(
    samples: Sequence[HttpResult],
) -> dict[str, int | float]:
    elapsed_values = [sample.elapsed_seconds for sample in samples]
    cpu_values = [sample.cpu_seconds for sample in samples]
    if len(samples) == 0:
        return _empty_component()
    return {
        'sample_count': len(samples),
        'total_seconds': sum(elapsed_values),
        'mean_seconds': sum(elapsed_values) / len(elapsed_values),
        'maximum_seconds': max(elapsed_values),
        'cpu_seconds': sum(cpu_values),
        'stall_count': 0,
    }


def _empty_component() -> dict[str, int | float]:
    return {
        'sample_count': 0,
        'total_seconds': 0.0,
        'mean_seconds': 0.0,
        'maximum_seconds': 0.0,
        'cpu_seconds': 0.0,
        'stall_count': 0,
    }


def _get_acceptance_report(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_workloads = {'single_simple', 'single_dynamic'}
    active_reports = [
        report for report in reports if report.get('preview_mode') == 'active'
    ]
    simple_reports_by_service = []
    for report in active_reports:
        service_workloads = {
            workload['name']: workload
            for workload in report.get('workloads', [])
            if workload.get('name') in expected_workloads
        }
        simple_reports_by_service.append(service_workloads)

    accepted_rates: list[float] = []
    published_mutation_rates: list[float] = []
    published_transaction_rates: list[float] = []
    simple_reports: list[Mapping[str, Any]] = []
    missing_workload = len(active_reports) == 0
    for service_workloads in simple_reports_by_service:
        if set(service_workloads) != expected_workloads:
            missing_workload = True
            continue
        for workload_name in sorted(expected_workloads):
            workload = service_workloads[workload_name]
            simple_reports.append(workload)
            elapsed_seconds = float(workload['elapsed_seconds'])
            if elapsed_seconds <= 0:
                accepted_rates.append(0.0)
                published_mutation_rates.append(0.0)
                published_transaction_rates.append(0.0)
                continue
            accepted_rates.append(
                float(workload['accepted_operations']) / elapsed_seconds,
            )
            operation_count = int(workload['operation_count_per_request'])
            published_transactions = float(workload['published_transactions'])
            published_mutation_rates.append(
                published_transactions * operation_count / elapsed_seconds,
            )
            published_transaction_rates.append(
                published_transactions / elapsed_seconds,
            )

    accepted_rate = min(accepted_rates) if len(accepted_rates) > 0 else 0.0
    published_rate = (
        min(published_mutation_rates)
        if len(published_mutation_rates) > 0
        else 0.0
    )
    published_transaction_rate = (
        min(published_transaction_rates)
        if len(published_transaction_rates) > 0
        else 0.0
    )
    presentation_stalls = sum(
        _get_component_count(workload, 'presentation_stalls', 'stall_count')
        for workload in simple_reports
    )
    active_components = (
        len(simple_reports) == len(active_reports) * len(expected_workloads)
        and all(
            int(workload.get('tracking_frames', 0)) > 0
            and int(workload.get('tracking_cycles_with_frames', 0)) > 0
            and _has_component_samples(workload, 'request_validation')
            and _has_component_samples(
                workload,
                'registry_candidate_build_publication',
            )
            and _has_component_samples(workload, 'detection_tracking')
            and _has_component_samples(
                workload,
                'spatial_resolution_materialisation',
            )
            and _has_component_samples(workload, 'presentation_cadence')
            and _has_component_samples(workload, 'projector_presentation')
            and _has_component_samples(workload, 'presentation_stalls')
            and _has_component_samples(workload, 'preview_conversion')            for workload in simple_reports
        )
    )
    has_failures = any(len(workload.get('failures', [])) > 0 for workload in simple_reports)
    status = (
        'pass'
        if not missing_workload
        and not has_failures
        and accepted_rate >= ACCEPTANCE_RATE_PER_SECOND
        and published_rate >= ACCEPTANCE_RATE_PER_SECOND
        and presentation_stalls == 0
        and active_components
        else 'fail'
    )
    return {
        'status': status,
        'threshold_mutations_per_second': ACCEPTANCE_RATE_PER_SECOND,
        'accepted_mutations_per_second': accepted_rate,
        'published_mutations_per_second': published_rate,
        'published_transactions_per_second': published_transaction_rate,
        'presentation_stall_count': presentation_stalls,
        'capture_tracking_projector_observed': active_components,
        'criterion_preview_mode': 'active',
        'criterion_excluded': False,
    }


def _get_component_count(
    workload: Mapping[str, Any],
    component_name: str,
    field_name: str,
) -> int:
    components = workload.get('components', {})
    if not isinstance(components, Mapping):
        return 0
    component = components.get(component_name, {})
    if not isinstance(component, Mapping):
        return 0
    value = component.get(field_name, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return max(0, value)


def _has_component_samples(
    workload: Mapping[str, Any],
    component_name: str,
) -> bool:
    components = workload.get('components', {})
    if not isinstance(components, Mapping):
        return False
    component = components.get(component_name, {})
    if not isinstance(component, Mapping):
        return False
    sample_count = component.get('sample_count', 0)
    return (
        isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0
    )


def _measure_cli_startup(
    command: Sequence[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started_seconds = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as ex:
        return {
            'status': 'failed',
            'criterion_excluded': True,
            'latency_seconds': None,
            'command': list(command),
            'error': str(ex),
        }
    return {
        'status': 'measured' if result.returncode == 0 else 'failed',
        'criterion_excluded': True,
        'latency_seconds': time.perf_counter() - started_seconds,
        'command': list(command),
        'return_code': result.returncode,
    }


def _make_report(
    mode: str,
    services: Mapping[str, str],
    reports: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    cold_start: Mapping[str, Any],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        'schema_version': SCHEMA_VERSION,
        'benchmark': 'realtime_overlays',
        'mode': mode,
        'services': dict(services),
        'acceptance': dict(acceptance),
        'workloads': [
            workload
            for report in reports
            for workload in report.get('workloads', [])
        ],
        'service_runs': list(reports),
        'cold_start': dict(cold_start),
        'notes': list(notes),
    }


def _parse_service_urls(
    service_url: str | None,
    preview_mode: str,
    mode_urls: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    parsed: dict[str, str] = {}
    if service_url is not None:
        parsed[preview_mode] = service_url
    for mode_url in mode_urls:
        if '=' not in mode_url:
            raise ValueError('--mode-url must use MODE=URL')
        mode, url = mode_url.split('=', 1)
        if mode not in PREVIEW_MODES or len(url) == 0:
            raise ValueError('--mode-url must name a valid preview mode and URL')
        if mode in parsed:
            raise ValueError(f'Duplicate preview mode: {mode!r}')
        parsed[mode] = url
    if len(parsed) == 0:
        raise ValueError('Provide --service-url or at least one --mode-url')
    return tuple(sorted(parsed.items()))


def _validate_finite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Benchmark report contains a non-finite number')
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_numbers(child)
        return
    if isinstance(value, list):
        for child in value:
            _validate_finite_numbers(child)


if __name__ == '__main__':
    raise SystemExit(main())
