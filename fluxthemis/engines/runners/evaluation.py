from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

from fluxthemis.engines.utils import to_jsonable
from fluxthemis.evaluators.base import BaseEvaluator
from fluxthemis.reporters.base import BaseEvaluationReporter
from fluxthemis.structures import EpisodeResult, TaskSpec


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


@dataclass
class _EvaluationState:
    status: EvaluationStatus = EvaluationStatus.CREATED
    selected_tasks: tuple[TaskSpec, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float = 0.0
    completed_episodes: int = 0
    total_episodes: int = 0


class EvaluationRun:
    """Track metrics and reporting for one coordinator run."""

    def __init__(self, settings: EvaluationSettings, evaluator: BaseEvaluator,
                 reporter: BaseEvaluationReporter) -> None:
        self.settings = settings
        self._state = _EvaluationState()
        self.evaluator = evaluator
        self.reporter = reporter
        self._started_monotonic: float | None = None

    @property
    def run_session_id(self) -> str:
        return self.reporter.run_session_id

    @property
    def report_run_dir(self) -> str | None:
        return self.reporter.evaluation_report_run_dir

    def __enter__(self) -> EvaluationRun:
        self.reporter.setup()
        try:
            self.evaluator.setup()
        except Exception:
            self.reporter.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        for component in (self.evaluator, self.reporter):
            try:
                component.close()
            except Exception:
                pass

    def start(self, selected_tasks: Sequence[TaskSpec],
              total_episodes: int) -> None:
        if self._state.status is not EvaluationStatus.CREATED:
            raise RuntimeError(f'cannot start evaluation while status is '
                               f'{self._state.status.value!r}')
        self._state.selected_tasks = tuple(selected_tasks)
        self._state.total_episodes = total_episodes
        self._state.started_at = utc_now()
        self._state.status = EvaluationStatus.RUNNING
        self._started_monotonic = time.perf_counter()
        self.evaluator.on_run_start(self._state.selected_tasks)
        self.reporter.report_evaluation(
            'run_start', _build_run_start_event(self.settings, self._state))

    def on_episode_start(self, payload: Mapping[str, Any]) -> None:
        self._require_running()
        self.reporter.report_evaluation('episode_start', payload)

    def on_episode_end(self, result: EpisodeResult, episode_id: str) -> None:
        self._require_running()
        self.evaluator.on_episode_end(result)
        self._state.completed_episodes += 1
        self.reporter.report_evaluation(
            'episode_end',
            _build_episode_end_event(
                result,
                self._state,
                episode_id,
                finished_at=utc_now(),
            ),
        )

    def finish(self, last_result: EpisodeResult | None) -> dict[str, Any]:
        self._require_running()
        metrics = self.evaluator.summarize()
        self._stop(EvaluationStatus.FINISHED)
        summary = _build_summary(
            self.settings,
            self._state,
            metrics,
            self.report_run_dir,
        )
        self.reporter.report_evaluation(
            'run_end',
            _build_run_end_event(
                self.settings,
                self._state,
                metrics,
                last_result=last_result,
                summary=summary,
            ),
        )
        return summary

    def fail(self, status: EvaluationStatus,
             current_episode: Mapping[str, Any] | None,
             last_result: EpisodeResult | None, error: Mapping[str,
                                                               Any]) -> None:
        if self._state.status not in (EvaluationStatus.RUNNING,
                                      EvaluationStatus.FINISHED):
            return
        self._stop(status)
        metrics, metrics_error = self._summarize_safely()
        try:
            self.reporter.report_evaluation(
                'run_end',
                _build_run_end_event(
                    self.settings,
                    self._state,
                    metrics,
                    current_episode=current_episode,
                    last_result=last_result,
                    error=error,
                    metrics_error=metrics_error,
                ),
            )
        except Exception as report_exc:
            warnings.warn(
                'Could not report terminal evaluation state to FluxVLA: '
                f'{report_exc}',
                RuntimeWarning,
                stacklevel=2,
            )

    def _stop(self, status: EvaluationStatus) -> None:
        self._state.status = status
        self._state.finished_at = utc_now()
        if self._started_monotonic is not None:
            self._state.duration_s = max(
                0.0,
                time.perf_counter() - self._started_monotonic)

    def _summarize_safely(
            self) -> tuple[dict[str, Any], dict[str, str]
                           | None]:
        try:
            return self.evaluator.summarize(), None
        except Exception as exc:
            return {}, {
                'error_type': type(exc).__name__,
                'message': str(exc),
            }

    def _require_running(self) -> None:
        if self._state.status is not EvaluationStatus.RUNNING:
            raise RuntimeError(
                f'evaluation is not running: {self._state.status.value!r}')


def _build_summary(settings: EvaluationSettings, state: _EvaluationState,
                   metrics: Mapping[str, Any],
                   remote_run_dir: str | None) -> dict[str, Any]:
    artifacts = {}
    if remote_run_dir:
        artifacts['remote_run_dir'] = remote_run_dir
    return {
        'schema_version': '1.0',
        'run': {
            'name':
            settings.run_name,
            'started_at':
            state.started_at,
            'finished_at':
            state.finished_at,
            'duration_s':
            state.duration_s,
            'base_seed':
            settings.base_seed,
            'tasks': [task.task_id for task in state.selected_tasks],
            'episodes_per_task':
            settings.episodes_per_task,
            'episodes_per_task_by_task':
            _episodes_per_task_by_task(settings, state.selected_tasks),
            'max_episode_steps':
            settings.max_episode_steps,
            'execute_horizon':
            settings.execute_horizon,
            'completed_episodes':
            state.completed_episodes,
        },
        'metrics': to_jsonable(dict(metrics)),
        'artifacts': artifacts,
    }


def _build_run_start_event(settings: EvaluationSettings,
                           state: _EvaluationState) -> dict[str, Any]:
    return {
        'schema_version':
        '1.0',
        'run_name':
        settings.run_name,
        'started_at':
        state.started_at,
        'base_seed':
        settings.base_seed,
        'episodes_per_task':
        settings.episodes_per_task,
        'episodes_per_task_by_task':
        _episodes_per_task_by_task(settings, state.selected_tasks),
        'max_episode_steps':
        settings.max_episode_steps,
        'execute_horizon':
        settings.execute_horizon,
        'stop_on_success':
        settings.stop_on_success,
        'task_ids': [task.task_id for task in state.selected_tasks],
        'total_tasks':
        len(state.selected_tasks),
        'total_episodes':
        state.total_episodes,
        'tasks': [
            _task_report(task, task_index)
            for task_index, task in enumerate(state.selected_tasks)
        ],
    }


def _build_episode_end_event(result: EpisodeResult, state: _EvaluationState,
                             episode_id: str,
                             finished_at: str) -> dict[str, Any]:
    return {
        'completed_episodes': state.completed_episodes,
        'total_episodes': state.total_episodes,
        'finished_at': finished_at,
        'episode_id': episode_id,
        **to_jsonable(result),
    }


def _build_run_end_event(
        settings: EvaluationSettings,
        state: _EvaluationState,
        metrics: Mapping[str, Any],
        current_episode: Mapping[str, Any] | None = None,
        last_result: EpisodeResult | None = None,
        summary: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        metrics_error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'schema_version':
        '1.0',
        'run_name':
        settings.run_name,
        'status':
        state.status.value,
        'finished_at':
        state.finished_at,
        'duration_s':
        state.duration_s,
        'completed_episodes':
        state.completed_episodes,
        'total_episodes':
        state.total_episodes,
        'current_episode': (to_jsonable(dict(current_episode))
                            if current_episode is not None else None),
        'last_result':
        to_jsonable(last_result),
        'metrics':
        to_jsonable(dict(metrics)),
    }
    if metrics_error is not None:
        payload['metrics_error'] = to_jsonable(dict(metrics_error))
    if summary is not None:
        payload['summary'] = to_jsonable(dict(summary))
    if error is not None:
        payload['error'] = to_jsonable(dict(error))
    return payload


def _episodes_per_task_by_task(settings: EvaluationSettings,
                               tasks: Sequence[TaskSpec]) -> dict[str, int]:
    return {
        task.task_id:
        settings.episodes_per_task_overrides.get(task.task_id,
                                                 settings.episodes_per_task)
        for task in tasks
    }


def _task_report(task: TaskSpec, task_index: int) -> dict[str, Any]:
    native_task_index = task.metadata.get('task_index', task_index)
    if (isinstance(native_task_index, bool)
            or not isinstance(native_task_index, int)
            or native_task_index < 0):
        raise ValueError(
            f'Task {task.task_id!r} has invalid metadata.task_index '
            f'{native_task_index!r}')
    return {
        'task_id': task.task_id,
        'task_index': native_task_index,
        'description': task.description,
        'metadata': to_jsonable(dict(task.metadata)),
    }
