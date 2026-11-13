from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fluxthemis.environments.sim import robolab_metrics as module


@pytest.fixture
def recording(tmp_path, monkeypatch):
    path = tmp_path / 'run_0.hdf5'
    with h5py.File(path, 'w') as file:
        demo = file.create_group('data/demo_0')
        demo.attrs['success'] = False
        demo.create_dataset('actions', data=np.zeros((8, 8)))
        # Sparse progress can fall; the final score is not the maximum.
        demo.create_dataset('subtask/score', data=[0.75, 0.25])
    values = {name: 0.0 for name in module.METRIC_NAMES}
    values['ee_sparc'] = np.nan

    def compute(data, dt):
        assert dt == 1 / 15
        return values

    monkeypatch.setattr(
        module, '_native_metrics', lambda: SimpleNamespace(
            load_demo_data=lambda path, key: {'ee_position': np.zeros((8, 3))},
            compute_episode_metrics=compute))
    return path


def test_final_sparse_score_and_undefined_metric(recording):
    score, info = module.collect_episode_metrics(recording, 1 / 15, False, 8)
    assert score == 0.25
    assert info['metrics']['ee_sparc'] is None
    assert info['metric_status']['ee_sparc'] == 'native_undefined'
    assert info['metrics_complete']
    assert info['simulation_duration_s'] == 8 / 15
    assert info['metrics_compute_s'] >= 0


@pytest.mark.parametrize('success,steps', [(True, 8), (False, 7)])
def test_recording_must_match_native_outcome(recording, success, steps):
    with pytest.raises(ValueError, match='disagree'):
        module.collect_episode_metrics(recording, 1 / 15, success, steps)


def test_missing_score_is_not_zero(recording):
    with h5py.File(recording, 'a') as file:
        del file['data/demo_0/subtask/score']
    with pytest.raises(KeyError):
        module.collect_episode_metrics(recording, 1 / 15, False, 8)


def test_short_trajectory_is_explicitly_unavailable(recording, monkeypatch):
    monkeypatch.setattr(
        module, '_native_metrics', lambda: SimpleNamespace(
            load_demo_data=lambda path, key: {'ee_position': np.zeros((2, 3))},
            compute_episode_metrics=lambda data, dt: None))
    _, info = module.collect_episode_metrics(recording, 1 / 15, False, 8)
    assert set(info['metrics'].values()) == {None}
    assert set(info['metric_status'].values()) == {'too_few_samples'}


def test_demo_identity_is_not_assumed_zero(recording, monkeypatch):
    with h5py.File(recording, 'a') as file:
        file.copy('data/demo_0', 'data/demo_7')
        file['data/demo_7/subtask/score'][-1] = 0.75
    seen = []
    monkeypatch.setattr(
        module, '_native_metrics', lambda: SimpleNamespace(
            load_demo_data=lambda path, key: seen.append(
                key) or {'ee_position': np.zeros((8, 3))},
            compute_episode_metrics=lambda data, dt:
            {name: 0
             for name in module.METRIC_NAMES}))
    score, info = module.collect_episode_metrics(
        recording, 1 / 15, False, 8, demo_key='demo_7')
    assert score == 0.75 and info['native_demo_key'] == 'demo_7'
    assert seen == ['demo_7']
    with pytest.raises(KeyError):
        module.collect_episode_metrics(
            recording, 1 / 15, False, 8, demo_key='demo_9')
