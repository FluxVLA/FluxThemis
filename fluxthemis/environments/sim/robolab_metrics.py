"""Read native terminal records; simulator dependencies remain optional."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

METRIC_NAMES = ('ee_sparc', 'joint_sparc_mean', 'ee_isj', 'joint_isj',
                'ee_path_length', 'joint_rmse_mean', 'ee_speed_max',
                'ee_speed_mean')


def _native_metrics():
    return importlib.import_module('robolab.core.metrics.compute_metrics')


def collect_episode_metrics(hdf5_path,
                            dt,
                            expected_success,
                            expected_steps,
                            demo_key='demo_0'):
    """Return final score and JSON-safe metrics after native auto-export.

    A missing/corrupt recording is an error. A native mathematically undefined
    trajectory metric is null with a reason, preserving the native outcome.
    """
    started = time.perf_counter()
    import h5py

    path = Path(hdf5_path)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('RoboLab control dt must be positive and finite')
    with h5py.File(path, 'r') as file:
        demo = file[f'data/{demo_key}']
        if 'success' not in demo.attrs:
            raise ValueError('Native recording is not finalized: no success')
        if bool(demo.attrs['success']) != expected_success:
            raise ValueError(
                'Native HDF5 success disagrees with frozen result')
        if len(demo['actions']) != expected_steps:
            raise ValueError('Native HDF5 steps disagree with frozen result')
        scores = demo['subtask/score']
        if len(scores) == 0:
            raise ValueError('Native recording has no final score')
        score = float(np.asarray(scores[-1]).reshape(()))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Native final score must be finite and in [0, 1]')

    native = _native_metrics()
    data = native.load_demo_data(str(path), demo_key)
    if data is None:
        raise ValueError('Native trajectory fields could not be loaded')
    for key, value in data.items():
        if value is not None and not np.isfinite(value).all():
            raise ValueError(f'Native trajectory contains nonfinite {key}')
    values = native.compute_episode_metrics(data, dt=dt)
    if values is None and len(data['ee_position']) >= 3:
        raise ValueError('Native metrics unexpectedly unavailable')
    metrics, status = {}, {}
    for name in METRIC_NAMES:
        if values is None:
            metrics[name], status[name] = None, 'too_few_samples'
        else:
            value = float(values[name])
            metrics[name] = value if np.isfinite(value) else None
            status[name] = 'valid' if np.isfinite(
                value) else 'native_undefined'
    return score, dict(
        native_hdf5_path=str(path),
        native_demo_key=demo_key,
        control_dt_s=float(dt),
        simulation_duration_s=expected_steps * dt,
        metrics=metrics,
        metric_status=status,
        metrics_complete=True,
        metrics_source='robolab.core.metrics.compute_metrics',
        metrics_schema_version='1.0',
        duration_includes_terminal_metrics=True,
        metrics_compute_s=time.perf_counter() - started)
