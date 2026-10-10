from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


class EvaluationStatus(str, Enum):
    CREATED = 'created'
    RUNNING = 'running'
    FINISHED = 'finished'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'


@dataclass(frozen=True)
class EvaluationSettings:
    """Validated, immutable settings for one evaluation run."""

    base_seed: int = 7
    episodes_per_task: int = 1
    max_episode_steps: int = 1000
    execute_horizon: int | None = None
    task_ids: Sequence[str] | str | None = None
    stop_on_success: bool = True
    run_name: str | None = None
    episodes_per_task_overrides: Mapping[str,
                                         int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.base_seed,
                      bool) or not isinstance(self.base_seed, Integral):
            raise TypeError('base_seed must be an integer')
        object.__setattr__(self, 'base_seed', int(self.base_seed))
        object.__setattr__(
            self, 'episodes_per_task',
            _positive_integer(self.episodes_per_task, 'episodes_per_task'))
        object.__setattr__(
            self, 'max_episode_steps',
            _positive_integer(self.max_episode_steps, 'max_episode_steps'))
        if self.execute_horizon is not None:
            object.__setattr__(
                self, 'execute_horizon',
                _positive_integer(self.execute_horizon, 'execute_horizon'))
        object.__setattr__(self, 'task_ids',
                           _normalize_task_ids(self.task_ids))
        if not isinstance(self.stop_on_success, bool):
            raise TypeError('stop_on_success must be a boolean')
        run_name = self.run_name or _default_run_name()
        _validate_run_name(run_name)
        object.__setattr__(self, 'run_name', run_name)
        object.__setattr__(
            self, 'episodes_per_task_overrides',
            _normalize_episode_overrides(self.episodes_per_task_overrides))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return value


def _normalize_episode_overrides(
        value: Mapping[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError('episodes_per_task_overrides must be a mapping')
    normalized = {}
    for task_id, count in value.items():
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(
                'episodes_per_task_overrides keys must be non-empty strings')
        normalized[task_id] = _positive_integer(
            count, f'episodes_per_task_overrides[{task_id!r}]')
    return normalized


def _normalize_task_ids(
        task_ids: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if task_ids is None:
        return None
    if isinstance(task_ids, str):
        task_ids = [
            value.strip() for value in task_ids.split(',') if value.strip()
        ]
    normalized = tuple(str(task_id) for task_id in task_ids)
    if not normalized:
        raise ValueError('task_ids cannot be empty')
    if len(normalized) != len(set(normalized)):
        raise ValueError(f'task_ids contains duplicates: {normalized}')
    return normalized


def _default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    return f'eval_{timestamp}_{os.getpid()}'


def _validate_run_name(run_name: str) -> None:
    if (not isinstance(run_name, str) or not run_name
            or run_name in {'.', '..'} or Path(run_name).name != run_name
            or Path(run_name).is_absolute()):
        raise ValueError('run_name must be a non-empty path component')
