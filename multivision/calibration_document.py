"""Portable JSON snapshots for the session calibration authorities."""

from __future__ import annotations

import json
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Any


CALIBRATION_DOCUMENT_FORMAT = 'multivision-calibration'
CALIBRATION_DOCUMENT_VERSION = 1
DEFAULT_CALIBRATION_DOCUMENT_PATH = pathlib.Path('calibration.json')


def build_calibration_document(
    camera_status: Mapping[str, Any],
    metric_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the versioned document from the two public status responses."""
    if not isinstance(camera_status, Mapping):
        raise ValueError('camera_status must be an object')
    if not isinstance(metric_status, Mapping):
        raise ValueError('metric_status must be an object')
    calibrations = camera_status.get('calibrations')
    if not isinstance(calibrations, Mapping):
        raise ValueError('camera_status must contain calibrations')
    metric_record = metric_status.get('calibration')
    metric_state = metric_status.get('state')
    if metric_state != 'CALIBRATED' or not isinstance(metric_record, Mapping):
        raise ValueError('Only a CALIBRATED metric status can be saved')
    if metric_record.get('state') != 'CALIBRATED':
        raise ValueError('Only a CALIBRATED metric record can be saved')
    if isinstance(metric_record, Mapping) and metric_record.get('state') == 'CALIBRATED':
        source_slot = metric_record.get('observation_camera_slot')
        source_id = metric_record.get('observation_camera_id')
        source_version = metric_record.get('observation_camera_calibration_version')
        source_timestamp = metric_record.get('observation_camera_calibration_timestamp')
        camera_record = calibrations.get(source_slot)
        if not isinstance(camera_record, Mapping):
            raise ValueError('metric_status refers to a camera absent from camera_status')
        if any(
            camera_record.get(field_name) != expected
            for field_name, expected in (
                ('camera_id', source_id),
                ('version', source_version),
                ('timestamp', source_timestamp),
            )
        ):
            raise ValueError('metric_status source-camera provenance does not match camera_status')
    return {
        'format': CALIBRATION_DOCUMENT_FORMAT,
        'version': CALIBRATION_DOCUMENT_VERSION,
        'camera': dict(camera_status),
        'metric': dict(metric_status),
    }


def load_calibration_document(path: pathlib.Path) -> dict[str, Any]:
    """Read and validate one portable calibration document."""
    if not isinstance(path, pathlib.Path):
        raise ValueError('path must be a pathlib.Path')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as ex:
        raise ValueError(f'Calibration document does not exist: {path}') from ex
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise ValueError(f'Could not read calibration document: {path}') from ex
    if not isinstance(data, dict):
        raise ValueError('Calibration document must contain an object')
    if data.get('format') != CALIBRATION_DOCUMENT_FORMAT:
        raise ValueError('Calibration document has an unsupported format')
    if data.get('version') != CALIBRATION_DOCUMENT_VERSION:
        raise ValueError('Calibration document has an unsupported version')
    camera_status = data.get('camera')
    metric_status = data.get('metric')
    if not isinstance(camera_status, dict):
        raise ValueError('Calibration document is missing the camera status object')
    if not isinstance(metric_status, dict):
        raise ValueError('Calibration document is missing the metric status object')
    return data


def save_calibration_document(
    path: pathlib.Path,
    camera_status: Mapping[str, Any],
    metric_status: Mapping[str, Any],
) -> None:
    """Atomically write one portable calibration document."""
    if not isinstance(path, pathlib.Path):
        raise ValueError('path must be a pathlib.Path')
    document = build_calibration_document(camera_status, metric_status)
    temporary_path: pathlib.Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            delete=False,
        ) as temporary_file:
            temporary_path = pathlib.Path(temporary_file.name)
            json.dump(document, temporary_file, indent=2, sort_keys=True, allow_nan=False)
            temporary_file.write('\n')
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as ex:
        raise ValueError(f'Could not write calibration document: {path}') from ex
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


__all__ = [
    'CALIBRATION_DOCUMENT_FORMAT',
    'CALIBRATION_DOCUMENT_VERSION',
    'DEFAULT_CALIBRATION_DOCUMENT_PATH',
    'build_calibration_document',
    'load_calibration_document',
    'save_calibration_document',
]
