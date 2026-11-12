"""Episode planning and execution."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable

import numpy as np

from fluxthemis.environments.base import (BaseEnvironment,
                                          EnvironmentCaseRejected,
                                          EnvironmentReset)
from fluxthemis.model_clients.base import BasePolicyClient
from fluxthemis.structures import (ActionChunk, EpisodeResult, StepResult,
                                   TaskSpec)

from .assignment import EpisodeAssignment, ExecutionPolicy
from .video import EpisodeVideoRecorder


@dataclass(frozen=True)
class EpisodeJob:
    """One canonical episode scheduled onto a simulator worker.

    Seeds control policy and simulator randomness.  The environment case index
    selects a task-local initial condition such as a LIBERO state or RoboDojo
    layout.
    """

    global_episode_number: int
    task: TaskSpec
    total_tasks: int
    episode_index: int
    episodes_per_task: int
    policy_seed: int
    environment_seed: int | None
    environment_case_index: int | None
    episode_id: str

    def episode_start_payload(self, started_at: str) -> dict[str, Any]:
        return {
            'global_episode_number': self.global_episode_number,
            'task_id': self.task.task_id,
            'total_tasks': self.total_tasks,
            'episode_index': self.episode_index,
            'task_episode_number': self.episode_index + 1,
            'episodes_per_task': self.episodes_per_task,
            'policy_seed': self.policy_seed,
            'environment_seed': self.environment_seed,
            'environment_case_index': self.environment_case_index,
            'episode_id': self.episode_id,
            'started_at': started_at,
        }

    def validate_result(self, result: EpisodeResult) -> None:
        identity = (
            result.task_id,
            result.episode_index,
            result.policy_seed,
            result.environment_seed,
            result.environment_case_index,
        )
        expected = (
            self.task.task_id,
            self.episode_index,
            self.policy_seed,
            self.environment_seed,
            self.environment_case_index,
        )
        if identity != expected:
            raise RuntimeError(
                f'episode result identity {identity!r} does not match job '
                f'{expected!r}')


def select_tasks(task_manifest: Sequence[TaskSpec],
                 task_ids: Sequence[str] | None) -> tuple[TaskSpec, ...]:
    """Validate an environment manifest and apply an optional task filter."""
    tasks = tuple(TaskSpec.from_value(task) for task in task_manifest)
    if not tasks:
        raise ValueError('Environment exposes no evaluation tasks')
    available_ids = [task.task_id for task in tasks]
    if len(available_ids) != len(set(available_ids)):
        raise ValueError(
            f'Environment returned duplicate task ids: {available_ids}')
    if task_ids is None:
        return tasks
    tasks_by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError(
            f'Unknown task ids {missing}; available tasks: {available_ids}')
    return tuple(tasks_by_id[task_id] for task_id in task_ids)


def build_episode_jobs(
    tasks: Sequence[TaskSpec],
    episodes_per_task: int,
    base_seed: int,
    run_session_id: str,
    episodes_per_task_overrides: Mapping[str, int] | None = None
) -> tuple[EpisodeJob, ...]:
    """Build stable task-major jobs independent of worker completion order.

    Seeds advance globally, while environment cases restart at zero for every
    task.
    """
    overrides = episodes_per_task_overrides or {}
    jobs = []
    global_index = 0
    for task in tasks:
        task_episode_count = overrides.get(task.task_id, episodes_per_task)
        for episode_index in range(task_episode_count):
            global_episode_number = global_index + 1
            policy_seed = int(base_seed) + global_index
            episode_id = f'{run_session_id}:{global_episode_number}'
            jobs.append(
                EpisodeJob(
                    global_episode_number=global_episode_number,
                    task=task,
                    total_tasks=len(tasks),
                    episode_index=episode_index,
                    episodes_per_task=task_episode_count,
                    policy_seed=policy_seed,
                    environment_seed=policy_seed,
                    environment_case_index=episode_index,
                    episode_id=episode_id,
                ))
            global_index += 1
    return tuple(jobs)


def configure_worker_lifetime(parent_pid: int | None = None) -> None:
    """Isolate a Linux worker and terminate it if its parent disappears."""
    if sys.platform != 'linux':
        return
    if parent_pid is None:
        parent_pid = os.getppid()
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, Integral):
        raise TypeError('parent_pid must be an integer')
    if parent_pid < 1:
        raise ValueError('parent_pid must be positive')

    if os.getsid(0) != os.getpid():
        os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != int(parent_pid):
        os.kill(os.getpid(), signal.SIGKILL)


def execute_episode(environment: BaseEnvironment,
                    policy_client: BasePolicyClient,
                    job: EpisodeJob,
                    max_episode_steps: int,
                    execute_horizon: int | None,
                    stop_on_success: bool,
                    on_episode_started: Callable[[], None] | None = None,
                    save_videos: bool = False,
                    video_dir: str | os.PathLike | None = None,
                    video_fps: int = 20,
                    video_worker_id: str | None = None) -> EpisodeResult:
    """Compatibility entry point for a capacity-one assignment."""
    return execute_assignment(
        environment, [policy_client],
        EpisodeAssignment((job, )),
        max_episode_steps,
        execute_horizon,
        stop_on_success,
        on_episode_started=((lambda _job: on_episode_started())
                            if on_episode_started is not None else None),
        save_videos=save_videos,
        video_dir=video_dir,
        video_fps=video_fps,
        video_worker_id=video_worker_id)[0]


@dataclass
class _EpisodeState:
    job: EpisodeJob
    policy_client: BasePolicyClient
    observation: Any
    info: dict
    started: float = field(default_factory=time.perf_counter)
    actions: deque = field(default_factory=deque)
    episode_return: float = 0.0
    steps: int = 0
    model_calls: int = 0
    prediction_metadata: list = field(default_factory=list)
    success: bool = False
    score: float | None = None
    termination_reason: str | None = None
    result: EpisodeResult | None = None

    def next_action(self, execute_horizon: int | None) -> np.ndarray:
        if not self.actions:
            context = build_context(self.job.task, self.job.episode_index,
                                    self.job.policy_seed, self.info,
                                    self.job.episode_id, self.observation)
            chunk = self.policy_client.predict_action(self.observation,
                                                      context)
            actions = validate_action_chunk(chunk)
            if not isinstance(chunk.metadata, Mapping):
                raise TypeError('ActionChunk.metadata must be a mapping')
            self.prediction_metadata.append(dict(chunk.metadata))
            self.model_calls += 1
            horizon = (
                len(actions) if execute_horizon is None else min(
                    len(actions), execute_horizon))
            self.actions.extend(action.copy() for action in actions[:horizon])
        return self.actions.popleft()

    def advance(self, transition: StepResult, max_episode_steps: int,
                stop_on_success: bool) -> None:
        if not isinstance(transition, StepResult):
            raise TypeError('Active environment slot must return a StepResult')
        if not isinstance(transition.info, Mapping):
            raise TypeError('StepResult.info must be a mapping')
        self.observation = transition.observation
        self.policy_client.observe(self.observation)
        self.episode_return += float(transition.reward)
        self.steps += 1
        self.info = dict(transition.info)
        if transition.success is not None:
            self.success = self.success or bool(transition.success)
        if transition.score is not None:
            score = float(transition.score)
            if not np.isfinite(score):
                raise ValueError(
                    'Environment returned a non-finite episode score')
            if not 0.0 <= score <= 1.0:
                raise ValueError('Environment episode score must be in [0, 1]')
            self.score = score
        if self.success and stop_on_success:
            self.termination_reason = 'success'
        elif transition.terminated:
            self.termination_reason = 'terminated'
        elif transition.truncated:
            self.termination_reason = 'truncated'
        elif self.steps >= max_episode_steps:
            self.termination_reason = 'max_steps'

    def finish(self) -> EpisodeResult:
        self.result = EpisodeResult(
            task_id=self.job.task.task_id,
            episode_index=self.job.episode_index,
            policy_seed=self.job.policy_seed,
            environment_seed=self.job.environment_seed,
            environment_case_index=self.job.environment_case_index,
            success=self.success,
            episode_return=self.episode_return,
            steps=self.steps,
            duration_s=time.perf_counter() - self.started,
            termination_reason=self.termination_reason,
            model_calls=self.model_calls,
            score=self.score,
            info=self.info,
            prediction_metadata=self.prediction_metadata)
        return self.result


def execute_assignment(
    environment: BaseEnvironment,
    policy_clients: Sequence[BasePolicyClient],
    assignment: EpisodeAssignment,
    max_episode_steps: int,
    execute_horizon: int | None,
    stop_on_success: bool,
    on_episode_started: Callable[[EpisodeJob], None] | None = None,
    on_episode_finished: (Callable[[EpisodeJob, EpisodeResult], None]
                          | None) = None,
    save_videos: bool = False,
    video_dir: str | os.PathLike | None = None,
    video_fps: int = 20,
    video_worker_id: str | None = None,
    on_case_rejected: Callable[[EpisodeJob, Mapping[str, Any]], None]
    | None = None
) -> tuple[EpisodeResult, ...]:
    """Advance one assignment with independent per-slot episode state.

    Policy requests remain per episode; this is simulator batching, not a
    model-inference batching protocol. Results are emitted when each slot
    finishes, while inactive slots receive None until the whole group ends.
    """
    policy = ExecutionPolicy(environment.num_envs)
    policy.validate_options(save_videos=save_videos)
    policy.validate_assignment(assignment)
    jobs = assignment.jobs
    if len(policy_clients) != environment.num_envs:
        raise ValueError(
            'Assignment, environment and policy client sizes differ')
    if len({id(client) for client in policy_clients}) != len(policy_clients):
        raise ValueError('Each environment slot needs its own policy client')
    if max_episode_steps < 1 or (execute_horizon is not None
                                 and execute_horizon < 1):
        raise ValueError(
            'Episode steps and execution horizon must be positive')
    requests = [
        EnvironmentReset(job.task, job.episode_index, job.environment_seed,
                         job.environment_case_index) for job in jobs
    ]
    requests.extend([None] * (environment.num_envs - len(jobs)))
    resets = environment.reset_batch(requests)
    if len(resets) != environment.num_envs:
        raise ValueError('reset_batch returned the wrong number of slots')
    states = []
    for slot, (client, reset) in enumerate(zip(policy_clients, resets)):
        if slot >= len(jobs):
            if reset is not None:
                raise ValueError('Unused reset slots must return None')
            states.append(None)
            continue
        job = jobs[slot]
        if reset is None:
            raise ValueError('Active reset slots must return an observation')
        observation, info = reset
        if not isinstance(info, Mapping):
            raise TypeError('Environment reset info must be a mapping')
        client.reset(job.task, job.episode_index, job.policy_seed)
        client.set_episode_id(job.episode_id)
        if on_episode_started is not None:
            on_episode_started(job)
        states.append(_EpisodeState(job, client, observation, dict(info)))

    recorder = None
    if save_videos and not environment.supports_native_video_recording:
        if video_dir is None:
            raise ValueError('video_dir is required when saving videos')
        recorder = EpisodeVideoRecorder(
            video_dir, fps=video_fps, worker_id=video_worker_id)
        recorder.start(jobs[0].task.task_id, jobs[0].episode_index)
    try:
        if recorder is not None:
            recorder.capture(environment)
        while any(state is not None and state.result is None
                  for state in states):
            actions = [
                state.next_action(execute_horizon)
                if state is not None and state.result is None else None
                for state in states
            ]
            transitions = environment.step_batch(actions)
            if len(transitions) != len(states):
                raise ValueError(
                    'step_batch returned the wrong number of slots')
            for slot, (state,
                       transition) in enumerate(zip(states, transitions)):
                if state is None or state.result is not None:
                    if transition is not None:
                        raise ValueError('Inactive slots must return None')
                    continue
                if transition.case_rejected:
                    if on_case_rejected is None:
                        raise EnvironmentCaseRejected(
                            'Environment invalidated an active case',
                            slots=[slot])
                    on_case_rejected(state.job, transition.info)
                    if recorder is not None:
                        recorder.abort()
                        recorder = None
                    states[slot] = None
                    continue
                state.advance(transition, max_episode_steps, stop_on_success)
                if recorder is not None:
                    recorder.capture(environment)
                if state.termination_reason is not None:
                    if recorder is not None:
                        paths = recorder.finish(state.success)
                        if paths:
                            existing = state.info.get('video_paths', [])
                            if isinstance(existing, str):
                                existing = [existing]
                            elif not isinstance(existing, (list, tuple)):
                                raise TypeError(
                                    'video_paths must be a sequence of paths')
                            state.info['video_paths'] = [*existing, *paths]
                    result = state.finish()
                    if on_episode_finished is not None:
                        on_episode_finished(state.job, result)
    except BaseException:
        if recorder is not None:
            recorder.abort()
        raise
    return tuple(state.result for state in states if state is not None)


def build_context(task: TaskSpec,
                  episode_index: int,
                  policy_seed: int,
                  info: Mapping[str, Any],
                  episode_id: str | None = None,
                  observation: Any = None) -> dict[str, Any]:
    prompt = info.get('task_description', task.description)
    if not isinstance(prompt, str):
        raise TypeError('environment task_description must be a string')
    if isinstance(observation, Mapping):
        observed_prompt = observation.get('task_description')
        if isinstance(observed_prompt, str) and observed_prompt:
            prompt = observed_prompt
    context = {
        'task_id': task.task_id,
        'task_description': prompt,
        'task_metadata': dict(task.metadata),
        'episode_index': episode_index,
        'policy_seed': policy_seed,
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
            'PolicyClient.predict_action() must return an ActionChunk')
    actions = np.asarray(action_chunk.actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(
            f'PolicyClient actions must have shape [T, A], got {actions.shape}'
        )
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError('PolicyClient returned an empty action chunk')
    if not np.isfinite(actions).all():
        raise ValueError('PolicyClient returned NaN or infinite actions')
    return actions
