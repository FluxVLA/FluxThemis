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
                                          EnvironmentResetRequest)
from fluxthemis.model_clients.base import BasePolicyClient
from fluxthemis.structures import (ActionChunk, CaseRejection,
                                   EnvironmentReset, EpisodeInterruption,
                                   EpisodeResult, PolicyStop, StepResult,
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
) -> list[EpisodeJob]:
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
    return jobs


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

    def next_action(self, execute_horizon: int | None) -> np.ndarray | None:
        if not self.actions:
            context = build_context(
                self.job.task,
                self.job.episode_index,
                self.job.policy_seed,
                self.info,
                self.job.episode_id,
                self.observation,
                executed_steps=self.steps)
            chunk = self.policy_client.predict_action(self.observation,
                                                      context)
            if isinstance(chunk, PolicyStop):
                self.model_calls += 1
                self.prediction_metadata.append(dict(chunk.metadata))
                self.info['policy_stop'] = dict(
                    status=chunk.status, reason=chunk.reason)
                self.termination_reason = f'policy_{chunk.status}'
                if chunk.status == 'completed':
                    # Agent completion is separate from benchmark success.
                    self.info['agent_success'] = True
                    self.info['completion_source'] = 'agent_visual'
                print(
                    f'[FluxThemis] episode {self.job.episode_id} stopped: '
                    f'{self.termination_reason} ({chunk.reason}), '
                    f'steps={self.steps}; recording outcome and continuing',
                    flush=True)
                return None
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
        policy_clients: list[BasePolicyClient],
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
        on_case_rejected: (Callable[[EpisodeJob, str, Mapping[str, Any]], None]
                           | None) = None,
        *,
        on_episode_interrupted: (Callable[[EpisodeJob, str, Mapping[str, Any]],
                                          None] | None) = None,
        execution_policy: ExecutionPolicy) -> list[EpisodeResult | None]:
    """Advance one assignment with independent per-slot episode state.

    Policy requests remain per episode. All environment inputs and results
    are lists in slot order; unused and discarded attempts return None.
    Emit each result when it finishes, keeping inactive slots frozen until
    the group ends. Discard callbacks distinguish rejected cases from
    interrupted attempts. Each slot owns its video recorder. Use the
    execution policy constructed after environment setup for all assignments.
    """
    execution_policy.validate_assignment(assignment, policy_clients)
    if max_episode_steps < 1 or (execute_horizon is not None
                                 and execute_horizon < 1):
        raise ValueError(
            'Episode steps and execution horizon must be positive')
    jobs = assignment.jobs
    requests = [
        EnvironmentResetRequest(job.task, job.episode_index,
                                job.environment_seed,
                                job.environment_case_index) for job in jobs
    ]
    requests.extend([None] * (execution_policy.capacity - len(jobs)))
    states: list[_EpisodeState | None] = [None] * execution_policy.capacity
    recorders: list[EpisodeVideoRecorder
                    | None] = [None] * execution_policy.capacity

    def discard_episode(slot: int, job: EpisodeJob, phase: str,
                        result: CaseRejection | EpisodeInterruption) -> None:
        if not isinstance(result.info, Mapping):
            raise TypeError('Environment discard info must be a mapping')
        if isinstance(result, EpisodeInterruption):
            reason, callback = 'interrupted', on_episode_interrupted
        else:
            reason, callback = 'case_rejected', on_case_rejected
        recorder = recorders[slot]
        if recorder is not None:
            recorder.abort()
            recorders[slot] = None
        environment.finalize_episode(
            slot, success=False, termination_reason=reason)
        states[slot] = None
        if callback is not None:
            callback(job, phase=phase, info=result.info)

    def finish_episode(slot: int, state: _EpisodeState) -> None:
        final_info = environment.finalize_episode(
            slot,
            success=state.success,
            termination_reason=state.termination_reason)
        if not isinstance(final_info, Mapping):
            raise TypeError('Environment finalization info must be a mapping')
        state.info.update(final_info)
        recorder = recorders[slot]
        if recorder is not None:
            paths = recorder.finish(state.success)
            recorders[slot] = None
            if paths:
                existing = state.info.get('video_paths', [])
                if isinstance(existing, str):
                    existing = [existing]
                elif not isinstance(existing, (list, tuple)):
                    raise TypeError('video_paths must be a sequence of paths')
                state.info['video_paths'] = [*existing, *paths]
        result = state.finish()
        if on_episode_finished is not None:
            on_episode_finished(state.job, result)

    try:
        resets = environment.reset_batch(requests)
        if len(resets) != execution_policy.capacity:
            raise ValueError('reset_batch returned the wrong number of slots')
        for slot, (client, reset) in enumerate(zip(policy_clients, resets)):
            if slot >= len(jobs):
                if reset is not None:
                    raise ValueError('Unused reset slots must return None')
                continue
            job = jobs[slot]
            if isinstance(reset, (CaseRejection, EpisodeInterruption)):
                discard_episode(slot, job, 'reset', reset)
                continue
            if not isinstance(reset, EnvironmentReset):
                raise TypeError(
                    'Active reset slots must return EnvironmentReset, '
                    'CaseRejection or EpisodeInterruption')
            if not isinstance(reset.info, Mapping):
                raise TypeError('Environment reset info must be a mapping')
            client.reset(job.task, job.episode_index, job.policy_seed)
            client.set_episode_id(job.episode_id)
            if on_episode_started is not None:
                on_episode_started(job)
            states[slot] = _EpisodeState(job, client, reset.observation,
                                         dict(reset.info))
            if save_videos and not environment.supports_native_video_recording:
                if video_dir is None:
                    raise ValueError(
                        'video_dir is required when saving videos')
                recorder = EpisodeVideoRecorder(
                    video_dir, fps=video_fps, worker_id=video_worker_id)
                recorders[slot] = recorder
                recorder.start(job.task.task_id, job.episode_index)
                recorder.capture(environment, slot)
        while any(state is not None and state.result is None
                  for state in states):
            actions = [None] * execution_policy.capacity
            for slot, state in enumerate(states):
                if state is None or state.result is not None:
                    continue
                action = state.next_action(execute_horizon)
                if state.termination_reason is not None:
                    finish_episode(slot, state)
                else:
                    actions[slot] = action
            if not any(action is not None for action in actions):
                continue
            transitions = environment.step_batch(actions)
            if len(transitions) != execution_policy.capacity:
                raise ValueError(
                    'step_batch returned the wrong number of slots')
            for slot, (state,
                       transition) in enumerate(zip(states, transitions)):
                if state is None or state.result is not None:
                    if transition is not None:
                        raise ValueError('Inactive slots must return None')
                    continue
                if isinstance(transition,
                              (CaseRejection, EpisodeInterruption)):
                    discard_episode(slot, state.job, 'step', transition)
                    continue
                if not isinstance(transition, StepResult):
                    raise TypeError(
                        'Active environment slot must return StepResult, '
                        'CaseRejection or EpisodeInterruption')
                state.advance(transition, max_episode_steps, stop_on_success)
                recorder = recorders[slot]
                if recorder is not None:
                    recorder.capture(environment, slot)
                if state.termination_reason is not None:
                    finish_episode(slot, state)
    except BaseException:
        for recorder in recorders:
            if recorder is not None:
                recorder.abort()
        raise
    return [None if state is None else state.result for state in states]


def build_context(task: TaskSpec,
                  episode_index: int,
                  policy_seed: int,
                  info: Mapping[str, Any],
                  episode_id: str | None = None,
                  observation: Any = None,
                  executed_steps: int | None = None) -> dict[str, Any]:
    """Build policy context using actions executed by this episode runner.

    Simulator counters can include warm-up or expert motions, or be absent.
    The runner's count starts at zero after reset and advances only after
    an environment step returns successfully. Keep native info unchanged.
    """
    if executed_steps is not None and (type(executed_steps) is not int
                                       or executed_steps < 0):
        raise ValueError('executed_steps must be a nonnegative integer')
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
    if executed_steps is not None:
        context['observation_id'] = executed_steps
    for key, value in task.metadata.items():
        context.setdefault(str(key), value)
    return context


def validate_action_chunk(action_chunk: ActionChunk) -> np.ndarray:
    if not isinstance(action_chunk, ActionChunk):
        raise TypeError(
            'PolicyClient.predict_action() must return an ActionChunk')
    _validate_action_type(action_chunk.actions)
    actions = np.asarray(action_chunk.actions)
    if actions.ndim != 2:
        raise ValueError(
            f'PolicyClient actions must have shape [T, A], got {actions.shape}'
        )
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError('PolicyClient returned an empty action chunk')
    if not np.isfinite(actions).all():
        raise ValueError('PolicyClient returned NaN or infinite actions')
    with np.errstate(over='ignore'):
        actions = actions.astype(np.float32, copy=False)
    if not np.isfinite(actions).all():
        raise ValueError('PolicyClient actions are outside the FP32 range')
    return actions


def _validate_action_type(value: Any) -> None:
    # Check before coercion so mixed bool/float sequences cannot become valid.
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_action_type(item)
    elif np.asarray(value).dtype.kind not in 'fiu':
        raise TypeError('PolicyClient actions must contain real numbers')
