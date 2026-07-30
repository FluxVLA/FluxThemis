from __future__ import annotations

import copy
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from queue import Empty
from typing import Any

import numpy as np

from fluxthemis.engines.utils import (build_environment_from_cfg,
                                      build_model_client_from_cfg)
from fluxthemis.environments.base import BaseEnvironment
from fluxthemis.evaluators.base import BaseEvaluator
from fluxthemis.model_clients.base import BaseModelClient
from fluxthemis.structures import (ActionChunk, EpisodeResult, StepResult,
                                   TaskSpec)


@dataclass(frozen=True)
class EpisodeJob:
    """One canonical episode scheduled onto a simulator worker."""

    ordinal: int
    task: TaskSpec
    task_position: int
    total_tasks: int
    episode_index: int
    episodes_per_task: int
    seed: int
    episode_id: str

    def progress(self, started_at: str) -> dict[str, Any]:
        return {
            'ordinal': self.ordinal,
            'task_id': self.task.task_id,
            'task_position': self.task_position,
            'total_tasks': self.total_tasks,
            'episode_index': self.episode_index,
            'episode_number': self.episode_index + 1,
            'episodes_per_task': self.episodes_per_task,
            'seed': self.seed,
            'episode_id': self.episode_id,
            'started_at': started_at,
        }


def build_episode_jobs(tasks: Sequence[TaskSpec], episodes_per_task: int,
                       seed: int,
                       report_run_id: str) -> tuple[EpisodeJob, ...]:
    """Build stable task-major jobs independent of worker completion order."""
    if isinstance(episodes_per_task,
                  bool) or not isinstance(episodes_per_task, Integral):
        raise TypeError('episodes_per_task must be an integer')
    if episodes_per_task < 1:
        raise ValueError('episodes_per_task must be positive')
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError('seed must be an integer')
    if not isinstance(report_run_id, str) or not report_run_id.strip():
        raise ValueError('report_run_id must be a non-empty string')

    canonical_tasks = tuple(TaskSpec.from_value(task) for task in tasks)
    jobs = []
    global_index = 0
    for task_position, task in enumerate(canonical_tasks, start=1):
        for episode_index in range(int(episodes_per_task)):
            ordinal = global_index + 1
            jobs.append(
                EpisodeJob(
                    ordinal=ordinal,
                    task=task,
                    task_position=task_position,
                    total_tasks=len(canonical_tasks),
                    episode_index=episode_index,
                    episodes_per_task=int(episodes_per_task),
                    seed=int(seed) + global_index,
                    episode_id=f'{report_run_id}:{ordinal}',
                ))
            global_index += 1
    return tuple(jobs)


def episode_worker_index(job: EpisodeJob, parallel_workers: int) -> int:
    """Return the stable simulator-worker shard for a canonical job."""
    if not isinstance(job, EpisodeJob):
        raise TypeError('job must be an EpisodeJob')
    if isinstance(parallel_workers,
                  bool) or not isinstance(parallel_workers, Integral):
        raise TypeError('parallel_workers must be an integer')
    if parallel_workers < 1:
        raise ValueError('parallel_workers must be positive')
    return (job.ordinal - 1) % int(parallel_workers)


def make_inference_only_model_config(
        model_client_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a ROS client config and remove its parent-only report channel."""
    if not isinstance(model_client_cfg, Mapping):
        raise TypeError(
            'parallel model_client must be configured as a mapping')
    cfg = copy.deepcopy(dict(model_client_cfg))
    transport = cfg.get('transport')
    if isinstance(transport, Mapping):
        transport = copy.deepcopy(dict(transport))
        transport.pop('report_service_name', None)
        cfg['transport'] = transport
    return cfg


def configure_simulator_gpu(simulator_gpu_id: str | None) -> None:
    """Pin one spawned simulator process to one physical render GPU."""
    if simulator_gpu_id is None:
        return
    if not isinstance(simulator_gpu_id, str) or not simulator_gpu_id.strip():
        raise ValueError('simulator_gpu_id must be a non-empty string')
    gpu_id = simulator_gpu_id.strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['MUJOCO_EGL_DEVICE_ID'] = gpu_id


class _NullEpisodeEvaluator(BaseEvaluator):
    """Run step hooks in workers without owning aggregate result state."""

    def on_episode_end(self, result: EpisodeResult) -> None:
        del result

    def summarize(self) -> dict:
        return {}


def execute_episode(environment: BaseEnvironment,
                    model_client: BaseModelClient,
                    evaluator: BaseEvaluator,
                    task: TaskSpec,
                    episode_index: int,
                    seed: int,
                    max_episode_steps: int,
                    execute_horizon: int | None,
                    stop_on_success: bool,
                    episode_id: str | None = None) -> EpisodeResult:
    """Execute one episode using components local to the current process."""
    model_client.reset(task, episode_index, seed)
    if episode_id is not None:
        if not model_client.supports_explicit_episode_id:
            raise RuntimeError(
                f'{type(model_client).__name__} does not support explicit '
                'episode ids required by parallel evaluation')
        model_client.set_episode_id(episode_id)
    observation, reset_info = environment.reset(task, episode_index, seed)
    if not isinstance(reset_info, Mapping):
        raise TypeError('Environment reset info must be a mapping')
    evaluator.on_episode_start(task, episode_index, seed)

    started = time.perf_counter()
    episode_return = 0.0
    steps = 0
    model_calls = 0
    prediction_metadata: list[Mapping[str, Any]] = []
    success = False
    termination_reason = 'max_steps'
    latest_info = dict(reset_info)
    episode_done = False

    while steps < max_episode_steps and not episode_done:
        context = build_context(
            task=task,
            episode_index=episode_index,
            seed=seed,
            info=latest_info,
            episode_id=episode_id,
        )
        action_chunk = model_client.predict_action(observation, context)
        actions = validate_action_chunk(action_chunk)
        if not isinstance(action_chunk.metadata, Mapping):
            raise TypeError('ActionChunk.metadata must be a mapping')
        prediction_metadata.append(dict(action_chunk.metadata))
        model_calls += 1
        horizon = (
            len(actions) if execute_horizon is None else min(
                len(actions), execute_horizon))

        for action in actions[:horizon]:
            transition = environment.step(action.copy())
            if not isinstance(transition, StepResult):
                raise TypeError('Environment.step() must return a StepResult')
            if not isinstance(transition.info, Mapping):
                raise TypeError('StepResult.info must be a mapping')
            evaluator.on_step(transition)
            observation = transition.observation
            episode_return += float(transition.reward)
            steps += 1
            latest_info = dict(transition.info)
            if transition.success is not None:
                success = success or bool(transition.success)

            if success and stop_on_success:
                termination_reason = 'success'
                episode_done = True
            elif transition.terminated:
                termination_reason = 'terminated'
                episode_done = True
            elif transition.truncated:
                termination_reason = 'truncated'
                episode_done = True
            elif steps >= max_episode_steps:
                termination_reason = 'max_steps'
                episode_done = True
            if episode_done:
                break

    return EpisodeResult(
        task_id=task.task_id,
        episode_index=episode_index,
        seed=seed,
        success=success,
        episode_return=episode_return,
        steps=steps,
        duration_s=time.perf_counter() - started,
        termination_reason=termination_reason,
        model_calls=model_calls,
        info=latest_info,
        prediction_metadata=prediction_metadata,
    )


def build_context(task: TaskSpec,
                  episode_index: int,
                  seed: int,
                  info: Mapping[str, Any],
                  episode_id: str | None = None) -> dict[str, Any]:
    task_description = info.get('task_description', task.description)
    if not isinstance(task_description, str):
        raise TypeError('environment task_description must be a string')
    context = {
        'task_id': task.task_id,
        'task_description': task_description,
        'task_metadata': dict(task.metadata),
        'episode_index': episode_index,
        'seed': seed,
        'info': dict(info),
    }
    if episode_id is not None:
        context['episode_id'] = episode_id
    for key, value in task.metadata.items():
        context.setdefault(str(key), value)
    return context


def validate_action_chunk(action_chunk: ActionChunk) -> np.ndarray:
    if not isinstance(action_chunk, ActionChunk):
        raise TypeError(
            'ModelClient.predict_action() must return an ActionChunk')
    actions = np.asarray(action_chunk.actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(
            f'ModelClient actions must have shape [T, A], got {actions.shape}')
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError('ModelClient returned an empty action chunk')
    if not np.isfinite(actions).all():
        raise ValueError('ModelClient returned NaN or infinite actions')
    return actions


def rollout_worker_main(worker_id: int, environment_cfg: Mapping[str, Any],
                        model_client_cfg: Mapping[str, Any], job_queue: Any,
                        control_queue: Any, event_queue: Any,
                        simulator_gpu_id: str | None, max_episode_steps: int,
                        execute_horizon: int | None, stop_on_success: bool,
                        acknowledgement_timeout_s: float) -> None:
    """Spawn target for a persistent environment and inference client."""
    environment = None
    model_client = None
    active_job: EpisodeJob | None = None
    phase = 'startup'
    try:
        configure_simulator_gpu(simulator_gpu_id)
        environment = build_environment_from_cfg(environment_cfg)
        model_client = build_model_client_from_cfg(model_client_cfg)
        if not model_client.supports_explicit_episode_id:
            raise RuntimeError(
                f'{type(model_client).__name__} does not support explicit '
                'episode ids required by parallel evaluation')
        model_client.setup()
        environment.setup()
        tasks = tuple(
            TaskSpec.from_value(task) for task in environment.tasks())
        event_queue.put({
            'type':
            'ready',
            'worker_id':
            worker_id,
            'tasks':
            tasks,
            'simulator_gpu_id':
            simulator_gpu_id,
            'cuda_visible_devices':
            os.environ.get('CUDA_VISIBLE_DEVICES'),
            'mujoco_egl_device_id':
            os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        })

        while True:
            phase = 'waiting_for_job'
            command = job_queue.get()
            if command is None:
                break
            if not isinstance(command, EpisodeJob):
                raise TypeError(
                    f'worker received invalid job {type(command).__name__}')
            active_job = command
            phase = 'waiting_for_episode_start_ack'
            event_queue.put({
                'type': 'episode_start',
                'worker_id': worker_id,
                'job': active_job,
            })
            _wait_for_ack(
                control_queue,
                event_type='episode_start',
                episode_id=active_job.episode_id,
                timeout_s=acknowledgement_timeout_s,
            )

            phase = 'rollout'
            result = execute_episode(
                environment=environment,
                model_client=model_client,
                evaluator=_NullEpisodeEvaluator(),
                task=active_job.task,
                episode_index=active_job.episode_index,
                seed=active_job.seed,
                max_episode_steps=max_episode_steps,
                execute_horizon=execute_horizon,
                stop_on_success=stop_on_success,
                episode_id=active_job.episode_id,
            )
            phase = 'waiting_for_episode_end_ack'
            event_queue.put({
                'type': 'episode_end',
                'worker_id': worker_id,
                'job': active_job,
                'result': result,
            })
            _wait_for_ack(
                control_queue,
                event_type='episode_end',
                episode_id=active_job.episode_id,
                timeout_s=acknowledgement_timeout_s,
            )
            active_job = None
    except BaseException as exc:  # noqa: BLE001 - preserve child diagnostics
        event_queue.put({
            'type':
            'error',
            'worker_id':
            worker_id,
            'phase':
            phase,
            'episode_id':
            (active_job.episode_id if active_job is not None else None),
            'error_type':
            type(exc).__name__,
            'message':
            str(exc),
            'traceback':
            traceback.format_exc(),
        })
    finally:
        for component in (environment, model_client):
            if component is None:
                continue
            try:
                component.close()
            except Exception:
                pass
        event_queue.put({'type': 'stopped', 'worker_id': worker_id})


def _wait_for_ack(control_queue: Any, event_type: str, episode_id: str,
                  timeout_s: float) -> None:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
        raise TypeError('acknowledgement_timeout_s must be numeric')
    try:
        acknowledgement = control_queue.get(timeout=float(timeout_s))
    except Empty as exc:
        raise TimeoutError(
            f'parent did not acknowledge {event_type} for {episode_id!r} '
            f'within {float(timeout_s):g} seconds') from exc
    expected = {
        'type': f'{event_type}_ack',
        'episode_id': episode_id,
    }
    if acknowledgement != expected:
        raise RuntimeError(
            f'invalid parent acknowledgement: expected {expected!r}, got '
            f'{acknowledgement!r}')
