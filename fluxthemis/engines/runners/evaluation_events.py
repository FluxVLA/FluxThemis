from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from fluxthemis.engines.utils import to_jsonable
from fluxthemis.structures import EpisodeResult, TaskSpec

from .evaluation import EvaluationSettings, utc_now

if TYPE_CHECKING:
    from .evaluation_run import _EvaluationState


def build_summary(settings: EvaluationSettings, state: _EvaluationState,
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


def build_run_start_event(settings: EvaluationSettings,
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


def build_episode_end_event(result: EpisodeResult, state: _EvaluationState,
                            episode_id: str) -> dict[str, Any]:
    return {
        'completed_episodes': state.completed_episodes,
        'total_episodes': state.total_episodes,
        'finished_at': utc_now(),
        'episode_id': episode_id,
        **to_jsonable(result),
    }


def build_run_end_event(
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
