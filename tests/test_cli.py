import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
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

    def test_full_calibration_is_one_http_command(self) -> None:
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

        assert main(['full-calibration'], client) == 0
        assert requests[0][:3] == (
            'POST',
            'http://service.test/calibration/full',
            None,
        )

    def test_tag_list_delegates_to_http_and_url_encodes_request_values(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(
                200,
                'application/json',
                b'{"camera": "camera name/1", "tags": []}',
            )

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                [
                    'tags',
                    'list',
                    '--camera',
                    'camera name/1',
                    '--dictionary',
                    'DICT 5/5',
                ],
                client,
            )

        assert result == 0
        assert json.loads(output.getvalue()) == {
            'camera': 'camera name/1',
            'tags': [],
        }
        assert requests == [
            (
                'GET',
                'http://service.test/cameras/camera%20name%2F1/tags?dictionary=DICT+5%2F5',
                None,
                30.0,
            ),
        ], f'{requests=}'

    def test_tag_list_prints_structured_service_failures(self) -> None:
        client = MultiVisionClient(
            request_sender=lambda method, url, payload, timeout: ServiceResponse(
                422,
                'application/json',
                json.dumps({
                    'error': {
                        'code': 'REQUEST_VALIDATION_ERROR',
                        'message': 'Unsupported tag dictionary',
                    },
                }).encode('utf-8'),
            ),
        )
        error_output = io.StringIO()
        with redirect_stderr(error_output):
            result = main(
                ['tags', 'list', '--camera', 'overhead', '--dictionary', 'bad'],
                client,
            )

        assert result == 1
        assert 'REQUEST_VALIDATION_ERROR' in error_output.getvalue()
        assert 'Unsupported tag dictionary' in error_output.getvalue()

    def test_generic_overlay_commands_delegate_to_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"id": "overlay-1"}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        specs = {
            'grid': {
                'origin': {'space': 'projector_px', 'x': 1, 'y': 2},
                'geometry_space': 'projector_px',
                'spacing': {'value': 10, 'unit': 'px'},
                'extent': {
                    'width': {'value': 30, 'unit': 'px'},
                    'height': {'value': 20, 'unit': 'px'},
                },
            },
            'circle': {
                'centre': {'space': 'projector_px', 'x': 10, 'y': 20},
                'geometry_space': 'projector_px',
                'radius': {'value': 5, 'unit': 'px'},
            },
            'rect': {
                'centre': {'space': 'projector_px', 'x': 10, 'y': 20},
                'geometry_space': 'projector_px',
                'width': {'value': 10, 'unit': 'px'},
                'height': {'value': 8, 'unit': 'px'},
                'label': 'card-1',
                'label_angle_deg': 10,
                'label_scale': 1.5,
            },
            'text': {
                'position': {'space': 'projector_px', 'x': 15, 'y': 25},
                'text': 'floating',
                'angle_deg': 20,
                'scale': 2,
            },
            'line': {
                'start': {'space': 'projector_px', 'x': 1, 'y': 2},
                'end': {'space': 'projector_px', 'x': 3, 'y': 4},
            },
            'ruler': {
                'start': {'space': 'projector_px', 'x': 1, 'y': 2},
                'end': {'space': 'projector_px', 'x': 3, 'y': 4},
                'measurement_space': 'projector_px',
                'unit': 'px',
            },
        }
        commands = [
            [
                'overlay', overlay_kind, '--spec-json', json.dumps(spec),
            ]
            for overlay_kind, spec in specs.items()
        ] + [
            ['overlays', 'list'],
            ['overlay', 'show', '--id', 'overlay-1'],
            ['overlay', 'hide', '--name', 'my overlay'],
            ['overlay', 'remove', '--id', 'overlay-1'],
            ['overlays', 'clear'],
        ]

        for command in commands:
            with self.subTest(command=command), redirect_stdout(io.StringIO()):
                assert main(command, client) == 0

        assert [
            (method, url, payload)
            for method, url, payload, _timeout_seconds in requests
        ] == [
            ('POST', f'http://service.test/overlays/{overlay_kind}', spec)
            for overlay_kind, spec in specs.items()
        ] + [
            ('GET', 'http://service.test/overlays', None),
            ('POST', 'http://service.test/overlays/id/overlay-1/show', None),
            ('POST', 'http://service.test/overlays/name/my%20overlay/hide', None),
            ('DELETE', 'http://service.test/overlays/id/overlay-1', None),
            ('DELETE', 'http://service.test/overlays', None),
        ], f'{requests=}'

    def test_runtime_overlay_commands_accept_json_file_and_stdin_payloads(self) -> None:
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
        arrow_spec = {
            'start': {'type': 'projector', 'x': 1, 'y': 2, 'unit': 'px'},
            'end': {'type': 'projector', 'x': 20, 'y': 30, 'unit': 'px'},
            'geometry_space': 'projector_px',
            'head_length': {'value': 4, 'unit': 'px'},
            'head_width': {'value': 2, 'unit': 'px'},
        }
        batch_spec = {
            'operations': [
                {'op': 'create', 'request': {'kind': 'line', 'name': 'first'}},
                {'op': 'remove', 'selector': 'first'},
            ],
        }
        replacement_spec = {
            'kind': 'line',
            'start': {'type': 'projector', 'x': 1, 'y': 2, 'unit': 'px'},
            'end': {'type': 'projector', 'x': 20, 'y': 30, 'unit': 'px'},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_path = pathlib.Path(temporary_directory) / 'batch.json'
            batch_path.write_text(json.dumps(batch_spec))
            with redirect_stdout(io.StringIO()):
                assert main(
                    ['overlay', 'arrow', '--spec-json', json.dumps(arrow_spec)],
                    client,
                ) == 0
                assert main(
                    ['overlays', 'batch', '--spec-json', str(batch_path)],
                    client,
                ) == 0
                with patch('sys.stdin', io.StringIO(json.dumps(batch_spec))):
                    assert main(
                        ['overlays', 'batch', '--spec-json', '-'],
                        client,
                    ) == 0
                assert main(
                    [
                        'overlay',
                        'replace',
                        '--id',
                        '123e4567-e89b-42d3-a456-426614174000',
                        '--spec-json',
                        json.dumps(replacement_spec),
                    ],
                    client,
                ) == 0

        assert [method for method, _url, _payload, _timeout in requests] == [
            'POST', 'POST', 'POST', 'PUT',
        ], f'{requests=}'
        assert requests[0][1:] == (
            'http://service.test/overlays/arrow',
            arrow_spec,
            30.0,
        )
        assert requests[1][2] == batch_spec, f'{requests=}'
        assert requests[2][2] == batch_spec, f'{requests=}'
        assert requests[3][1] == (
            'http://service.test/overlays/id/123e4567-e89b-42d3-a456-426614174000'
        )

    def test_intensity_and_spatial_inspection_commands_delegate_to_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{"intensity": 0.5}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        commands = [
            ['overlays', 'intensity', 'get'],
            ['overlays', 'intensity', 'set', '--intensity', '0.5'],
            ['fiducial-groups'],
            ['spatial-state'],
        ]
        for command in commands:
            with self.subTest(command=command), redirect_stdout(io.StringIO()):
                assert main(command, client) == 0

        assert [
            (method, url, payload)
            for method, url, payload, _timeout in requests
        ] == [
            ('GET', 'http://service.test/overlays/intensity', None),
            ('PUT', 'http://service.test/overlays/intensity', {'intensity': 0.5}),
            ('GET', 'http://service.test/fiducial-groups', None),
            ('GET', 'http://service.test/spatial-state', None),
        ], f'{requests=}'

    def test_runtime_overlay_input_errors_do_not_reach_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []
        client = MultiVisionClient(
            request_sender=lambda method, url, payload, timeout: (
                requests.append((method, url, payload, timeout))
                or ServiceResponse(200, 'application/json', b'{}')
            ),
        )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['overlays', 'batch', '--spec-json', '{not-json}'], client)
            with self.assertRaises(SystemExit):
                main(['overlays', 'intensity', 'set', '--intensity', '2'], client)
        assert requests == [], f'{requests=}'

    def test_projector_grid_command_delegates_physical_spacing_to_service(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            return ServiceResponse(200, 'application/json', b'{}')

        client = MultiVisionClient('http://service.test', request_sender=request_sender)

        assert main(
            [
                'overlay',
                'grid',
                '--fill-projector',
                '--spacing',
                '35mm',
                '--name',
                'whole-projector',
            ],
            client,
        ) == 0
        assert requests[0][:3] == (
            'POST',
            'http://service.test/overlays/grid/projector-footprint',
            {
                'name': 'whole-projector',
                'spacing': {'value': 35.0, 'unit': 'mm'},
                'angle_deg': 0.0,
                'style': {'colour': '#ffffff'},
            },
        ), f'{requests=}'

    def test_invalid_overlay_spec_is_rejected_before_http(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []
        client = MultiVisionClient(
            request_sender=lambda method, url, payload, timeout: (
                requests.append((method, url, payload, timeout))
                or ServiceResponse(200, 'application/json', b'{}')
            ),
        )

        with redirect_stderr(io.StringIO()):
            result = main(
                ['overlay', 'circle', '--spec-json', '{"radius": {"value": 0, "unit": "px"}}'],
                client,
            )

        assert result == 1
        assert requests == [], f'{requests=}'

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

    def test_metric_rotation_calculates_local_camera_view_angle(self) -> None:
        requests: list[tuple[str, str, dict[str, Any] | None, float]] = []
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        def request_sender(
            method: str,
            url: str,
            payload: dict[str, Any] | None,
            timeout_seconds: float,
        ) -> ServiceResponse:
            requests.append((method, url, payload, timeout_seconds))
            if url.endswith('/metric/calibration/status'):
                body = {'surface_to_projector': matrix}
            else:
                body = {
                    'calibrations': {
                        'camera-1': {'projector_to_camera': matrix},
                    },
                }
            return ServiceResponse(
                200,
                'application/json',
                json.dumps(body).encode('utf-8'),
            )

        client = MultiVisionClient('http://service.test', request_sender=request_sender)
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                [
                    'metric',
                    'rotation',
                    '--camera',
                    'camera-1',
                    '--at-mm=-80,80',
                ],
                client,
            )

        assert result == 0
        response = json.loads(output.getvalue())
        assert response['rotation_angle_deg'] == 0.0, f'{response=}'
        assert response['camera_position_px'] == {'x': -80.0, 'y': 80.0}
        assert [url for _method, url, _payload, _timeout in requests] == [
            'http://service.test/metric/calibration/status',
            'http://service.test/calibration/status',
        ], f'{requests=}'

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

    def test_overlay_commands_do_not_load_geometry_or_detection_modules(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys; '
                    'from multivision.cli import MultiVisionClient, ServiceResponse, main; '
                    "spec = {'kind': 'line', "
                    "'start': {'space': 'projector_px', 'x': 1, 'y': 2}, "
                    "'end': {'space': 'projector_px', 'x': 3, 'y': 4}}; "
                    "client = MultiVisionClient(request_sender=lambda *args: "
                    "ServiceResponse(200, 'application/json', b'{}')); "
                    "main(['overlay', 'line', '--spec-json', __import__('json').dumps(spec)], client); "
                    "print('multivision.geometry' in sys.modules, "
                    "'multivision.metric' in sys.modules, "
                    "'multivision.camera' in sys.modules, "
                    "'multivision.display' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f'{result.stdout=}, {result.stderr=}'
        assert result.stdout.splitlines()[-1] == 'False False False False', f'{result.stdout=}'


if __name__ == '__main__':
    unittest.main()
