from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmengine.utils import digit_version

from fluxthemis.engines.utils import ENVIRONMENTS

from .libero import (ROBOSUITE_MAXIMUM_VERSION, ROBOSUITE_MINIMUM_VERSION,
                     LiberoEnvironment, _LiberoRuntime,
                     configure_mujoco_egl_defaults)

LIBERO_PLUS_SUITE_TASKS = {
    'libero_spatial': 2402,
    'libero_object': 2518,
    'libero_goal': 2591,
    'libero_10': 2519,
}
LIBERO_PLUS_CATEGORIES = frozenset({
    'Background Textures',
    'Camera Viewpoints',
    'Language Instructions',
    'Light Conditions',
    'Objects Layout',
    'Robot Initial States',
    'Sensor Noise',
})
LIBERO_PLUS_MAX_ENV_CREATION_ATTEMPTS = 20


@dataclass(frozen=True)
class _LiberoPlusRuntime(_LiberoRuntime):
    classification: Mapping[str, tuple[Mapping[str, Any], ...]]
    randomization_error: type[Exception]


@ENVIRONMENTS.register_module()
class LiberoPlusEnvironment(LiberoEnvironment):
    """LIBERO-Plus adapter with official robustness classification.

    The inherited adapter owns the raw observation, 7D action, warm-up,
    success, fixed-initial-state and lazy environment lifecycle semantics.
    This variant only selects the isolated ``libero_plus`` package, attaches
    benchmark classification, and retries stochastic placement failures while
    constructing an environment.
    """

    def __init__(self,
                 task_suite_name: str,
                 task_order_index: int = 0,
                 resolution: int = 256,
                 controller: str = 'OSC_POSE',
                 simulator_seed: int = 0,
                 num_steps_wait: int = 10,
                 max_env_creation_attempts:
                 int = LIBERO_PLUS_MAX_ENV_CREATION_ATTEMPTS,
                 env_kwargs: Mapping[str, Any] | None = None) -> None:
        if task_suite_name not in LIBERO_PLUS_SUITE_TASKS:
            raise ValueError(
                f'Unsupported LIBERO-Plus task suite {task_suite_name!r}; '
                f'expected one of {sorted(LIBERO_PLUS_SUITE_TASKS)}')
        if task_order_index != 0:
            raise ValueError('LIBERO-Plus requires task_order_index=0')
        super().__init__(
            task_suite_name=task_suite_name,
            task_order_index=task_order_index,
            resolution=resolution,
            controller=controller,
            simulator_seed=simulator_seed,
            num_steps_wait=num_steps_wait,
            env_kwargs=env_kwargs,
        )
        self.max_env_creation_attempts = self._positive_integer(
            max_env_creation_attempts, 'max_env_creation_attempts')

    def setup(self) -> None:
        super().setup()
        expected = LIBERO_PLUS_SUITE_TASKS[self.task_suite_name]
        actual = len(self._tasks)
        if actual != expected:
            self.close()
            raise ValueError(
                f'LIBERO-Plus suite {self.task_suite_name!r} exposes '
                f'{actual} tasks; expected {expected}')

    def _load_runtime(self) -> _LiberoPlusRuntime:
        return _load_libero_plus_runtime()

    def _task_metadata(self, task: Any, task_index: int,
                       runtime: _LiberoPlusRuntime) -> dict[str, Any]:
        rows = runtime.classification.get(self.task_suite_name)
        if rows is None or task_index >= len(rows):
            raise ValueError(f'LIBERO-Plus classification is incomplete for '
                             f'{self.task_suite_name!r} task {task_index}')
        row = rows[task_index]
        classification_id = row.get('id')
        if classification_id != task_index + 1:
            raise ValueError(
                f'LIBERO-Plus classification IDs must be contiguous and '
                f'one-based; task {task_index} has id {classification_id!r}')
        task_name = str(task.name)
        if row.get('name') != task_name:
            raise ValueError(
                f'LIBERO-Plus classification name mismatch for task '
                f'{task_index}: {row.get("name")!r} != {task_name!r}')
        category = row.get('category')
        if category not in LIBERO_PLUS_CATEGORIES:
            raise ValueError(
                f'LIBERO-Plus task {task_index} has invalid category '
                f'{category!r}')
        metadata = super()._task_metadata(task, task_index, runtime)
        metadata.update({
            'benchmark_variant': 'libero_plus',
            'classification_id': classification_id,
            'category': category,
            'difficulty_level': row.get('difficulty_level'),
        })
        return metadata

    def _create_environment(self, kwargs: Mapping[str, Any]) -> Any:
        for attempt in range(1, self.max_env_creation_attempts + 1):
            try:
                return super()._create_environment(kwargs)
            except self._runtime.randomization_error:
                if attempt == self.max_env_creation_attempts:
                    raise
        raise AssertionError('unreachable')


def _load_libero_plus_runtime() -> _LiberoPlusRuntime:
    configure_mujoco_egl_defaults()
    try:
        robosuite = importlib.import_module('robosuite')
        libero = importlib.import_module('libero_plus.libero')
        benchmark = importlib.import_module('libero_plus.libero.benchmark')
        envs = importlib.import_module('libero_plus.libero.envs')
        errors = importlib.import_module('robosuite.utils.errors')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'LiberoPlusEnvironment requires the pinned LIBERO-Plus '
            'simulation stack. Install it with '
            '`bash scripts/install_libero_plus.sh`.') from exc

    version = getattr(robosuite, '__version__', '')
    if not (digit_version(version) >= digit_version(ROBOSUITE_MINIMUM_VERSION)
            and digit_version(version)
            < digit_version(ROBOSUITE_MAXIMUM_VERSION)):
        raise RuntimeError(
            f'Robosuite=={version or "unknown"} is incompatible; install the '
            'pinned FluxVLA fork with '
            '`bash scripts/install_libero_plus.sh`.')
    if not hasattr(robosuite, 'load_controller_config'):
        raise RuntimeError(
            'Robosuite is missing the legacy LIBERO compatibility API; '
            'install the pinned e293cc3 FluxVLA fork.')
    randomization_error = getattr(errors, 'RandomizationError', None)
    if not isinstance(randomization_error, type) or not issubclass(
            randomization_error, Exception):
        raise RuntimeError(
            'Robosuite does not expose utils.errors.RandomizationError')

    classification_path = (
        Path(
            benchmark.__file__).resolve().with_name('task_classification.json')
    )
    try:
        classification_value = json.loads(
            classification_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot load LIBERO-Plus classification from '
                           f'{classification_path}') from exc
    classification = _validate_classification(classification_value)
    return _LiberoPlusRuntime(
        benchmark=benchmark,
        libero=libero,
        envs=envs,
        classification=classification,
        randomization_error=randomization_error,
    )


def _validate_classification(
        value: Any) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if not isinstance(value, Mapping):
        raise TypeError('LIBERO-Plus classification must be a mapping')
    classification = {}
    for suite_name, expected_count in LIBERO_PLUS_SUITE_TASKS.items():
        rows = value.get(suite_name)
        if not isinstance(rows, list) or len(rows) != expected_count:
            actual_count = len(rows) if isinstance(rows, list) else None
            raise ValueError(
                f'LIBERO-Plus classification for {suite_name!r} contains '
                f'{actual_count} rows; expected {expected_count}')
        if not all(isinstance(row, Mapping) for row in rows):
            raise TypeError(
                f'LIBERO-Plus classification rows for {suite_name!r} must '
                'be mappings')
        for task_index, row in enumerate(rows):
            classification_id = row.get('id')
            if (isinstance(classification_id, bool)
                    or classification_id != task_index + 1):
                raise ValueError(
                    f'LIBERO-Plus classification IDs for {suite_name!r} '
                    'must be contiguous and one-based')
            if row.get('category') not in LIBERO_PLUS_CATEGORIES:
                raise ValueError(
                    f'LIBERO-Plus classification for {suite_name!r} has '
                    f'unknown category {row.get("category")!r}')
            difficulty = row.get('difficulty_level')
            if (difficulty is not None and (isinstance(difficulty, bool)
                                            or not isinstance(difficulty, int)
                                            or difficulty not in range(1, 6))):
                raise ValueError(
                    f'LIBERO-Plus classification for {suite_name!r} has '
                    f'invalid difficulty {difficulty!r}')
        classification[suite_name] = tuple(dict(row) for row in rows)
    return classification
