"""Deterministic benchmark contract coverage; no hardware result is claimed."""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import TypeAdapter

from benchmarks.benchmark_realtime_overlays import (
    DiagnosticSnapshot,
    HttpResult,
    WORKLOAD_NAMES,
    _get_acceptance_report,
    _measure_workload,
    _subtract_diagnostics,
    build_dynamic_overlay,
    build_simple_overlay,
    run_injected_benchmark,
    validate_benchmark_report,
)
from multivision.api import OverlayPayload



def test_injected_benchmark_has_stable_schema_without_target_claim() -> None:
    report = run_injected_benchmark(samples_per_workload=2)

    validate_benchmark_report(report)
    json.dumps(report, allow_nan=False)
    assert report['schema_version'] == 1, f'{report=}'
    assert report['mode'] == 'injected', f'{report=}'
    assert report['acceptance']['status'] == 'not_evaluated', f'{report=}'
    assert report['acceptance']['criterion_excluded'] is True, f'{report=}'
    assert [workload['name'] for workload in report['workloads']] == list(
        WORKLOAD_NAMES,
    ), f'{report=}'
    assert report['cold_start']['criterion_excluded'] is True, f'{report=}'
    assert set(report['workloads'][0]['components']) >= {
        'http_transport',
        'request_validation',
        'registry_candidate_build_publication',
        'detection_tracking',
        'spatial_resolution_materialisation',
        'presentation_cadence',
        'presentation_stalls',
        'preview_conversion',
    }, f'{report=}'


def test_injected_batch_workloads_report_operation_counts_truthfully() -> None:
    report = run_injected_benchmark(samples_per_workload=2)

    batch_reports = {
        workload['name']: workload
        for workload in report['workloads']
        if workload['name'].startswith('batch_')
    }
    assert batch_reports['batch_10']['request_count'] == 2, f'{batch_reports=}'
    assert batch_reports['batch_10']['requested_operations'] == 20, f'{batch_reports=}'
    assert batch_reports['batch_100']['requested_operations'] == 200, f'{batch_reports=}'


def test_overlay_workload_builders_are_strict_service_payloads() -> None:
    simple = build_simple_overlay(3)
    dynamic = build_dynamic_overlay(3, 'benchmark')

    assert simple['kind'] == 'line', f'{simple=}'
    assert dynamic['kind'] == 'arrow', f'{dynamic=}'
    assert uuid.UUID(simple['id']).version == 4, f'{simple=}'
    assert uuid.UUID(dynamic['id']).version == 4, f'{dynamic=}'
    adapter = TypeAdapter(OverlayPayload)
    assert adapter.validate_python(simple).kind == 'line', f'{simple=}'
    assert adapter.validate_python(dynamic).kind == 'arrow', f'{dynamic=}'
    assert dynamic['start']['group'] == 'benchmark', f'{dynamic=}'
    assert dynamic['end']['follow_rotation'] is True, f'{dynamic=}'


def test_diagnostic_maximum_is_scoped_to_the_measured_interval() -> None:
    component_before = {
        'sample_count': 1,
        'total_seconds': 0.5,
        'maximum_seconds': 0.5,
        'cpu_seconds': 0.1,
        'stall_count': 0,
    }
    component_after = {
        'sample_count': 1,
        'total_seconds': 0.5,
        'maximum_seconds': 0.5,
        'cpu_seconds': 0.1,
        'stall_count': 0,
    }

    diagnostics = _subtract_diagnostics(
        DiagnosticSnapshot({}, {'render': component_after}, {}),
        DiagnosticSnapshot({}, {'render': component_before}, {}),
    )

    assert diagnostics['timings']['render']['sample_count'] == 0, f'{diagnostics=}'
    assert diagnostics['timings']['render']['maximum_seconds'] == 0.0, (
        f'{diagnostics=}'
    )


def test_workload_rate_uses_end_to_end_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def request(
            self,
            _method: str,
            _path: str,
            _payload: dict[str, object] | None = None,
        ) -> HttpResult:
            return HttpResult(204, b'', 0.01, 0.0)

        def json_request(
            self,
            _method: str,
            _path: str,
            _payload: dict[str, object] | None = None,
        ) -> tuple[HttpResult, dict[str, object]]:
            return HttpResult(200, b'{}', 0.01, 0.0), {}

    perf_counter_values = iter((10.0, 10.5))
    monkeypatch.setattr(
        'benchmarks.benchmark_realtime_overlays.time.perf_counter',
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        'benchmarks.benchmark_realtime_overlays._get_diagnostics',
        lambda _client: DiagnosticSnapshot({}, {}, {}),
    )

    reports = _measure_workload(
        FakeClient(),
        'single_simple',
        1,
        1,
        0,
        None,
    )

    assert reports[0]['elapsed_seconds'] == 0.5, f'{reports=}'


def test_acceptance_handles_zero_duration_without_crashing() -> None:
    workloads = [
        {
            'name': workload_name,
            'operation_count_per_request': 1,
            'accepted_operations': 0,
            'published_transactions': 0,
            'tracking_frames': 0,
            'elapsed_seconds': 0.0,
            'components': {},
            'failures': [],
        }
        for workload_name in ('single_simple', 'single_dynamic')
    ]

    report = _get_acceptance_report(
        [{'preview_mode': 'active', 'workloads': workloads}],
    )

    assert report['status'] == 'fail', f'{report=}'
    assert report['accepted_mutations_per_second'] == 0.0, f'{report=}'
    assert report['published_mutations_per_second'] == 0.0, f'{report=}'
    assert report['published_transactions_per_second'] == 0.0, f'{report=}'


def test_acceptance_requires_each_single_workload_to_publish_at_target_rate() -> None:
    component = {
        'sample_count': 1,
        'total_seconds': 0.01,
        'mean_seconds': 0.01,
        'maximum_seconds': 0.01,
        'cpu_seconds': 0.01,
        'stall_count': 0,
    }
    active_components = {
        'detection_tracking': component,
        'presentation_cadence': component,
        'projector_presentation': component,
        'presentation_stalls': component,
        'preview_conversion': component,
    }
    workloads = [
        {
            'name': 'single_simple',
            'operation_count_per_request': 1,
            'accepted_operations': 1,
            'published_transactions': 1,
            'tracking_frames': 1,
            'elapsed_seconds': 0.01,
            'components': active_components,
            'failures': [],
        },
        {
            'name': 'single_dynamic',
            'operation_count_per_request': 1,
            'accepted_operations': 0,
            'published_transactions': 0,
            'tracking_frames': 1,
            'elapsed_seconds': 0.01,
            'components': active_components,
            'failures': ['dynamic request rejected'],
        },
    ]

    report = _get_acceptance_report(
        [{'preview_mode': 'active', 'workloads': workloads}],
    )

    assert report['status'] == 'fail', f'{report=}'
    assert report['accepted_mutations_per_second'] == 0.0, f'{report=}'
    assert report['published_mutations_per_second'] == 0.0, f'{report=}'
