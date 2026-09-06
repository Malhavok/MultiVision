from __future__ import annotations

import json
from pathlib import Path

import pytest

from multivision.calibration_document import (
    CALIBRATION_DOCUMENT_FORMAT,
    CALIBRATION_DOCUMENT_VERSION,
    build_calibration_document,
    load_calibration_document,
    save_calibration_document,
)


def test_calibration_document_round_trip(tmp_path: Path) -> None:
    path = tmp_path / 'calibration.json'
    camera_status = {
        'calibrations': {
            'camera-0': {'camera_id': 'camera-0', 'version': 1, 'timestamp': 2.0},
        },
    }
    metric_status = {
        'state': 'CALIBRATED',
        'calibration': {
            'state': 'CALIBRATED',
            'observation_camera_slot': 'camera-0',
            'observation_camera_id': 'camera-0',
            'observation_camera_calibration_version': 1,
            'observation_camera_calibration_timestamp': 2.0,
        },
    }

    save_calibration_document(path, camera_status, metric_status)
    document = load_calibration_document(path)

    assert document == {
        'format': CALIBRATION_DOCUMENT_FORMAT,
        'version': CALIBRATION_DOCUMENT_VERSION,
        'camera': camera_status,
        'metric': metric_status,
    }, f'{document=}'


def test_calibration_document_rejects_wrong_contract(tmp_path: Path) -> None:
    path = tmp_path / 'calibration.json'
    path.write_text(
        json.dumps({'format': 'other', 'version': 1, 'camera': {}, 'metric': {}}),
        encoding='utf-8',
    )

    with pytest.raises(ValueError):
        load_calibration_document(path)


def test_build_calibration_document_rejects_non_objects() -> None:
    with pytest.raises(ValueError):
        build_calibration_document([], {})  # type: ignore[arg-type]


def test_build_calibration_document_rejects_missing_metric_source_camera() -> None:
    with pytest.raises(ValueError):
        build_calibration_document(
            {'calibrations': {}},
            {
                'calibration': {
                    'state': 'CALIBRATED',
                    'observation_camera_slot': 'camera-0',
                    'observation_camera_id': 'camera-0',
                    'observation_camera_calibration_version': 1,
                    'observation_camera_calibration_timestamp': 1.0,
                },
            },
        )
