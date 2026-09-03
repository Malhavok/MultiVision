import json
import math
from pathlib import Path

import pytest

from multivision.config import (
    Configuration,
    FiducialGroup,
    load_configuration,
    save_configuration,
)
from multivision.errors import ConfigurationError
from multivision.pattern import DICT_5X5_1000


def test_realtime_configuration_defaults() -> None:
    configuration = Configuration()

    assert configuration.fiducial_groups == {}, f'{configuration.fiducial_groups=}'
    assert configuration.fiducial_history_length == 8, f'{configuration=}'
    assert configuration.fiducial_tracking_rate_hz == 30.0, f'{configuration=}'
    assert configuration.fiducial_grace_period_seconds == 5.0, f'{configuration=}'
    assert configuration.fiducial_protection_margin_mm == 5.0, f'{configuration=}'
    assert configuration.max_batch_operations == 100, f'{configuration=}'
    assert configuration.preview_mode == 'active', f'{configuration=}'
    assert configuration.preview_low_rate_hz == 10.0, f'{configuration=}'


def test_realtime_configuration_round_trips_complete_values() -> None:
    configuration = Configuration(
        fiducial_groups={
            'cards': FiducialGroup(DICT_5X5_1000, 38.5),
            'units': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 24.0},
        },
        fiducial_history_length=32,
        fiducial_tracking_rate_hz=60.0,
        fiducial_grace_period_seconds=0.1,
        fiducial_protection_margin_mm=1000.0,
        max_batch_operations=1000,
        preview_mode='low_rate',
        preview_low_rate_hz=15.0,
    )

    round_tripped = Configuration.from_data(configuration.to_data())

    assert round_tripped == configuration, f'{round_tripped=}, {configuration=}'
    assert configuration.to_data()['fiducial_groups'] == {
        'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 38.5},
        'units': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 24.0},
    }


def test_empty_fiducial_groups_are_valid_and_persisted() -> None:
    configuration = Configuration.from_data({'fiducial_groups': {}})

    assert configuration.fiducial_groups == {}, f'{configuration=}'
    assert configuration.to_data()['fiducial_groups'] == {}, f'{configuration=}'


@pytest.mark.parametrize(
    ['data'],
    [
        ({'fiducial_groups': {'': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 10.0}}},),
        ({'fiducial_groups': {'   ': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 10.0}}},),
        ({'fiducial_groups': {1: {'dictionary': DICT_5X5_1000, 'marker_size_mm': 10.0}}},),
        ({'fiducial_groups': {'cards': {'dictionary': 'DICT_UNKNOWN', 'marker_size_mm': 10.0}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 0}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': -1}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': math.nan}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': math.inf}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000}}},),
        ({'fiducial_groups': {'cards': {'dictionary': DICT_5X5_1000, 'marker_size_mm': 10.0, 'extra': True}}},),
        ({'fiducial_groups': []},),
    ],
)
def test_fiducial_groups_reject_invalid_names_dictionaries_sizes_and_fields(
    data: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        Configuration.from_data(data)


@pytest.mark.parametrize(
    ['field_name', 'invalid_value'],
    [
        ('fiducial_history_length', [0, -1, 33, True, 1.0]),
        ('fiducial_tracking_rate_hz', [0.9, 60.1, math.nan, math.inf, True, '30']),
        ('fiducial_grace_period_seconds', [0, 0.09, 60.1, math.nan, math.inf, True]),
        ('fiducial_protection_margin_mm', [0, 0.09, 1000.1, math.nan, math.inf, True]),
        ('max_batch_operations', [0, -1, 1001, True, 1.0]),
        ('preview_mode', ['normal', '', None, 1]),
        ('preview_low_rate_hz', [0.9, 15.1, math.nan, math.inf, True, '10']),
    ],
)
def test_realtime_configuration_rejects_invalid_bounds(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ConfigurationError):
        Configuration.from_data({field_name: invalid_value})


def test_fiducial_group_values_are_immutable() -> None:
    group = FiducialGroup(DICT_5X5_1000, 20.0)

    with pytest.raises((AttributeError, TypeError)):
        group.marker_size_mm = 30.0  # type: ignore[misc]


def test_unknown_root_fields_are_preserved_when_saving(tmp_path: Path) -> None:
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'operator_setting': {'enabled': True}}), encoding='utf-8')

    save_configuration(Configuration(), path)

    saved_data = json.loads(path.read_text(encoding='utf-8'))
    assert saved_data['operator_setting'] == {'enabled': True}, f'{saved_data=}'


def test_save_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'config.json'
    original = {'operator_setting': {'enabled': True}}
    path.write_text(json.dumps(original), encoding='utf-8')

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError('simulated replacement failure')

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(ConfigurationError):
        save_configuration(Configuration(preview_mode='off'), path)

    assert json.loads(path.read_text(encoding='utf-8')) == original
    assert list(tmp_path.glob('.config.json.*')) == []


def test_saved_configuration_loads_with_new_fields(tmp_path: Path) -> None:
    path = tmp_path / 'config.json'
    save_configuration(
        Configuration(
            fiducial_groups={'cards': FiducialGroup(DICT_5X5_1000, 40.0)},
            preview_mode='off',
        ),
        path,
    )

    loaded = load_configuration(path)

    assert loaded.fiducial_groups['cards'].marker_size_mm == 40.0
    assert loaded.preview_mode == 'off'
