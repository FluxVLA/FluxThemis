from __future__ import annotations

import copy
import json
import math
import multiprocessing
import os
import queue
import sys
import time
import warnings
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fluxthemis.environments.base import BaseEnvironment
from fluxthemis.evaluators.base import BaseEvaluator
from fluxthemis.model_clients.base import BaseModelClient
from fluxthemis.structures import ActionChunk, EpisodeResult, TaskSpec

from ..utils import (RUNNERS, build_environment_from_cfg,
                     build_evaluator_from_cfg, build_model_client_from_cfg,
                     to_jsonable, write_json_atomic)
from .parallel_rollout import (EpisodeJob, build_context, build_episode_jobs,
                               configure_simulator_gpu, episode_worker_index,
                               execute_episode,
                               make_inference_only_model_config,
                               rollout_worker_main, validate_action_chunk)


@RUNNERS.register_module()
class EvalRunner:
    """Environment-agnostic task and episode evaluation loop."""

    def __init__(self,
                 environment: Mapping[str, Any] | BaseEnvironment,
                 model_client: Mapping[str, Any] | BaseModelClient,
                 evaluator: Mapping[str, Any] | BaseEvaluator,
                 seed: int = 7,
                 episodes_per_task: int = 1,
                 max_episode_steps: int = 1000,
                 execute_horizon: int | None = None,
                 task_ids: Sequence[str] | str | None = None,
                 task_metadata_filters: Mapping[str, Any] | None = None,
                 stop_on_success: bool = True,
                 work_dir: str | os.PathLike | None = None,
                 run_name: str | None = None,
                 overwrite: bool = False,
                 parallel_workers: int = 1,
                 simulator_gpu_ids: Sequence[str | int] | str | None = None,
                 parallel_startup_timeout_s: float = 300.0,
                 parallel_event_timeout_s: float = 3600.0,
                 parallel_ack_timeout_s: float = 300.0,
                 parallel_start_method: str = 'spawn') -> None:
        if episodes_per_task < 1:
            raise ValueError('episodes_per_task must be positive')
        if max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        if execute_horizon is not None and execute_horizon < 1:
            raise ValueError('execute_horizon must be positive or None')
        if work_dir is None:
            raise ValueError(
                'work_dir must be injected from the FluxVLA configuration')
        if not isinstance(work_dir, (str, os.PathLike)):
            raise TypeError('work_dir must be a path')
        if not str(work_dir).strip():
            raise ValueError('work_dir cannot be empty')
        resolved_work_dir = Path(work_dir).expanduser().resolve()
        if not Path(work_dir).expanduser().is_absolute():
            raise ValueError('work_dir must be an absolute path')

        self.parallel_workers = self._positive_integer(parallel_workers,
                                                       'parallel_workers')
        self.simulator_gpu_ids = self._normalize_simulator_gpu_ids(
            simulator_gpu_ids)
        self.parallel_startup_timeout_s = self._positive_float(
            parallel_startup_timeout_s, 'parallel_startup_timeout_s')
        self.parallel_event_timeout_s = self._positive_float(
            parallel_event_timeout_s, 'parallel_event_timeout_s')
        self.parallel_ack_timeout_s = self._positive_float(
            parallel_ack_timeout_s, 'parallel_ack_timeout_s')
        if parallel_start_method != 'spawn':
            raise ValueError(
                "parallel_start_method must be 'spawn' so simulator and ROS "
                'state are never inherited across workers')
        self.parallel_start_method = parallel_start_method

        self._environment_cfg = (
            self._copy_component_cfg(environment, 'environment') if isinstance(
                environment, Mapping) else None)
        self._model_client_cfg = (
            self._copy_component_cfg(model_client, 'model_client')
            if isinstance(model_client, Mapping) else None)

        self.environment = (
            environment if isinstance(environment, BaseEnvironment) else
            build_environment_from_cfg(environment))
        self.model_client = (
            model_client if isinstance(model_client, BaseModelClient) else
            build_model_client_from_cfg(model_client))
        self.evaluator = (
            evaluator if isinstance(evaluator, BaseEvaluator) else
            build_evaluator_from_cfg(evaluator))
        if self._environment_cfg is not None:
            self._environment_cfg['type'] = type(self.environment)
        if self._model_client_cfg is not None:
            self._model_client_cfg['type'] = type(self.model_client)
        if self.parallel_workers > 1:
            if self._environment_cfg is None or self._model_client_cfg is None:
                raise TypeError(
                    'parallel evaluation requires environment and '
                    'model_client configuration mappings so each spawned '
                    'worker can build independent components')
            if not self.model_client.evaluation_reporting_enabled:
                raise ValueError(
                    'parallel evaluation requires FluxVLA evaluation '
                    'reporting so the parent process can own lifecycle and '
                    'result aggregation')
            if not self.model_client.supports_explicit_episode_id:
                raise ValueError(
                    f'{type(self.model_client).__name__} cannot provide the '
                    'globally unique episode ids required by parallel '
                    'evaluation')
            if not self.evaluator.supports_parallel_rollouts:
                raise ValueError(
                    f'{type(self.evaluator).__name__} does not declare '
                    'supports_parallel_rollouts=True; parallel workers do '
                    'not forward on_episode_start/on_step hooks to the '
                    'parent evaluator')

        self.seed = int(seed)
        self.episodes_per_task = int(episodes_per_task)
        self.max_episode_steps = int(max_episode_steps)
        self.execute_horizon = execute_horizon
        self.task_ids = self._normalize_task_ids(task_ids)
        self.task_metadata_filters = self._normalize_task_metadata_filters(
            task_metadata_filters)
        self.stop_on_success = bool(stop_on_success)
        self.work_dir = resolved_work_dir
        self.run_name = run_name or self._default_run_name()
        self._validate_run_name(self.work_dir, self.run_name)
        self.run_dir = self.work_dir / self.run_name
        self.overwrite = bool(overwrite)

        self.summary_path = self.run_dir / 'summary.json'
        self.episodes_path = self.run_dir / 'episodes.jsonl'
        self.progress_path = self.run_dir / 'progress.json'
        self.failure_path = self.run_dir / 'failure.json'
        self.resolved_config_path = self.run_dir / 'resolved_config.py'
        self.tasks_to_run: list[TaskSpec] = []
        self._state = 'created'
        self._started_at: str | None = None
        self._run_started_monotonic: float | None = None
        self._resolved_config_source: str | None = None
        self._parallel_context = None
        self._parallel_processes: list[Any] = []
        self._parallel_job_queues: list[Any] = []
        self._parallel_event_queue = None
        self._parallel_control_queues: list[Any] = []

    @property
    def local_artifacts_enabled(self) -> bool:
        """Whether this process owns result serialization for the run."""
        return not self.model_client.evaluation_reporting_enabled

    @property
    def authoritative_run_dir(self) -> str | None:
        """Return the acknowledged remote run directory, when available."""
        if self.local_artifacts_enabled:
            return str(self.run_dir)
        return self.model_client.evaluation_report_run_dir

    def set_resolved_config(self, source: str, text: str) -> None:
        """Attach safe provenance without transmitting config contents."""
        if self._state != 'created':
            raise RuntimeError(
                'set_resolved_config() must be called before run_setup()')
        if not isinstance(source, str) or not source:
            raise ValueError(
                'resolved config source must be a non-empty string')
        if not isinstance(text, str) or not text:
            raise ValueError('resolved config text must be a non-empty string')
        self._resolved_config_source = source

    def run_setup(self) -> None:
        if self._state != 'created':
            raise RuntimeError(
                f'run_setup() is invalid while runner is {self._state!r}')
        if self.local_artifacts_enabled:
            self._prepare_run_dir()
        try:
            self.model_client.setup()
            self.evaluator.setup(self.run_dir)
            if self.parallel_workers > 1:
                tasks = self._start_parallel_workers()
            else:
                if self.simulator_gpu_ids:
                    self._configure_serial_simulator_gpu(
                        self.simulator_gpu_ids[0])
                self.environment.setup()
                tasks = [
                    TaskSpec.from_value(task)
                    for task in self.environment.tasks()
                ]
            self.tasks_to_run = self._select_tasks(tasks)
            self.evaluator.on_run_start(self.tasks_to_run)
        except Exception:
            self.cleanup()
            raise
        self._started_at = self._utc_now()
        self._state = 'setup'

    def run(self) -> dict:
        if self.model_client.evaluation_reporting_enabled:
            if self.parallel_workers > 1:
                return self._run_remote_parallel()
            return self._run_remote()
        return self._run_local()

    def _run_local(self) -> dict:
        if self._state != 'setup':
            raise RuntimeError('run_setup() must complete before run()')
        self._state = 'running'
        self._run_started_monotonic = time.perf_counter()
        results: list[EpisodeResult] = []
        total_episodes = len(self.tasks_to_run) * self.episodes_per_task
        current_episode: dict[str, Any] | None = None
        last_result: EpisodeResult | None = None
        try:
            self._write_progress(
                results,
                total_episodes=total_episodes,
                status='running',
            )
            print(
                f'[FluxThemis] evaluation_start total={total_episodes} '
                f'tasks={len(self.tasks_to_run)} '
                f'progress={json.dumps(str(self.progress_path))}',
                file=sys.stderr,
                flush=True,
            )
            with self.episodes_path.open('w', encoding='utf-8') as result_file:
                global_episode_index = 0
                for task_position, task in enumerate(
                        self.tasks_to_run, start=1):
                    for episode_index in range(self.episodes_per_task):
                        episode_seed = self.seed + global_episode_index
                        current_episode = {
                            'ordinal': global_episode_index + 1,
                            'task_id': task.task_id,
                            'task_position': task_position,
                            'total_tasks': len(self.tasks_to_run),
                            'episode_index': episode_index,
                            'episode_number': episode_index + 1,
                            'episodes_per_task': self.episodes_per_task,
                            'seed': episode_seed,
                            'started_at': self._utc_now(),
                        }
                        self._write_progress(
                            results,
                            total_episodes=total_episodes,
                            status='running',
                            current_episode=current_episode,
                            last_result=last_result,
                        )
                        self._print_episode_start(current_episode,
                                                  total_episodes)
                        result = self._run_episode(
                            task=task,
                            episode_index=episode_index,
                            seed=episode_seed,
                        )
                        self.evaluator.on_episode_end(result)
                        result_file.write(
                            json.dumps(
                                to_jsonable(result), ensure_ascii=False) +
                            '\n')
                        result_file.flush()
                        results.append(result)
                        global_episode_index += 1
                        last_result = result
                        current_episode = None
                        progress = self._write_progress(
                            results,
                            total_episodes=total_episodes,
                            status='running',
                            last_result=last_result,
                        )
                        self._print_episode_done(progress)
            summary = self._build_summary(results)
            write_json_atomic(self.summary_path, summary)
            self._state = 'finished'
            self._write_progress(
                results,
                total_episodes=total_episodes,
                status='finished',
                last_result=last_result,
            )
            return summary
        except KeyboardInterrupt:
            self._state = 'interrupted'
            self._write_terminal_progress(
                results,
                total_episodes=total_episodes,
                status='interrupted',
                current_episode=current_episode,
                last_result=last_result,
            )
            raise
        except Exception as exc:
            self._state = 'failed'
            try:
                write_json_atomic(
                    self.failure_path, {
                        'status': 'failed',
                        'failed_at': self._utc_now(),
                        'error_type': type(exc).__name__,
                        'message': str(exc),
                        'completed_episodes': len(results),
                        'current_episode': current_episode,
                    })
            except Exception as artifact_exc:  # pragma: no cover - defensive
                self._warn_artifact_write_failure(self.failure_path,
                                                  artifact_exc)
            self._write_terminal_progress(
                results,
                total_episodes=total_episodes,
                status='failed',
                current_episode=current_episode,
                last_result=last_result,
                error={
                    'error_type': type(exc).__name__,
                    'message': str(exc),
                },
            )
            raise

    def _run_remote(self) -> dict:
        """Run with FluxVLA as the acknowledged result/artifact owner."""
        if self._state != 'setup':
            raise RuntimeError('run_setup() must complete before run()')
        self._state = 'running'
        self._run_started_monotonic = time.perf_counter()
        results: list[EpisodeResult] = []
        total_episodes = len(self.tasks_to_run) * self.episodes_per_task
        current_episode: dict[str, Any] | None = None
        last_result: EpisodeResult | None = None
        run_end_attempted = False
        try:
            self.model_client.report_evaluation(
                'run_start', self._build_run_start_report(total_episodes))
            report_run_id = self.model_client.evaluation_report_run_id
            if not isinstance(report_run_id, str) or not report_run_id:
                raise RuntimeError(
                    'FluxVLA did not provide a report_run_id for episode '
                    'identity')
            if not self.model_client.supports_explicit_episode_id:
                raise RuntimeError(
                    f'{type(self.model_client).__name__} cannot use the '
                    'globally unique episode ids required by reported '
                    'evaluation')
            global_episode_index = 0
            for task_position, task in enumerate(self.tasks_to_run, start=1):
                for episode_index in range(self.episodes_per_task):
                    episode_seed = self.seed + global_episode_index
                    current_episode = {
                        'ordinal': global_episode_index + 1,
                        'task_id': task.task_id,
                        'task_position': task_position,
                        'total_tasks': len(self.tasks_to_run),
                        'episode_index': episode_index,
                        'episode_number': episode_index + 1,
                        'episodes_per_task': self.episodes_per_task,
                        'seed': episode_seed,
                        'episode_id':
                        f'{report_run_id}:{global_episode_index + 1}',
                        'started_at': self._utc_now(),
                    }
                    self.model_client.report_evaluation(
                        'episode_start',
                        self._build_episode_start_report(
                            task, current_episode, total_episodes),
                    )
                    result = self._run_episode(
                        task=task,
                        episode_index=episode_index,
                        seed=episode_seed,
                        episode_id=current_episode['episode_id'],
                    )
                    self.evaluator.on_episode_end(result)
                    self.model_client.report_evaluation(
                        'episode_end',
                        self._build_episode_end_report(
                            task,
                            result,
                            completed_episodes=len(results) + 1,
                            total_episodes=total_episodes,
                            episode_id=current_episode['episode_id'],
                        ),
                    )
                    results.append(result)
                    global_episode_index += 1
                    last_result = result
                    current_episode = None

            summary = self._build_summary(results, local_artifacts=False)
            self._state = 'finished'
            run_end_attempted = True
            self.model_client.report_evaluation(
                'run_end',
                self._build_run_end_report(
                    status='finished',
                    results=results,
                    total_episodes=total_episodes,
                    last_result=last_result,
                    summary=summary,
                ),
            )
            return summary
        except KeyboardInterrupt:
            self._state = 'interrupted'
            if not run_end_attempted:
                self._report_run_end_safely(
                    status='interrupted',
                    results=results,
                    total_episodes=total_episodes,
                    current_episode=current_episode,
                    last_result=last_result,
                    error={
                        'error_type': 'KeyboardInterrupt',
                        'message': '',
                    },
                )
            raise
        except Exception as exc:
            self._state = 'failed'
            if not run_end_attempted:
                self._report_run_end_safely(
                    status='failed',
                    results=results,
                    total_episodes=total_episodes,
                    current_episode=current_episode,
                    last_result=last_result,
                    error={
                        'error_type': type(exc).__name__,
                        'message': str(exc),
                    },
                )
            raise

    def _run_remote_parallel(self) -> dict:
        """Coordinate spawned rollouts while FluxVLA owns all artifacts."""
        if self._state != 'setup':
            raise RuntimeError('run_setup() must complete before run()')
        if (not self._parallel_job_queues
                or self._parallel_event_queue is None):
            raise RuntimeError('parallel rollout workers are not available')

        report_run_id = self.model_client.evaluation_report_run_id
        if not isinstance(report_run_id, str) or not report_run_id:
            raise RuntimeError(
                'FluxVLA did not provide a report_run_id for parallel '
                'episode identity')
        jobs = build_episode_jobs(
            self.tasks_to_run,
            episodes_per_task=self.episodes_per_task,
            seed=self.seed,
            report_run_id=report_run_id,
        )
        jobs_by_id = {job.episode_id: job for job in jobs}
        total_episodes = len(jobs)

        self._state = 'running'
        self._run_started_monotonic = time.perf_counter()
        results: list[EpisodeResult] = []
        results_by_id: dict[str, EpisodeResult] = {}
        aggregation_attempted = False
        last_result: EpisodeResult | None = None
        active_episodes: dict[str, dict[str, Any]] = {}
        started_episode_ids: set[str] = set()
        completed_episode_ids: set[str] = set()
        stopped_worker_ids: set[int] = set()
        run_end_attempted = False
        try:
            self.model_client.report_evaluation(
                'run_start', self._build_run_start_report(total_episodes))
            for job in jobs:
                worker_id = episode_worker_index(job, self.parallel_workers)
                self._parallel_job_queues[worker_id].put(job)
            for job_queue in self._parallel_job_queues:
                job_queue.put(None)

            while len(stopped_worker_ids) < self.parallel_workers:
                event = self._next_parallel_event(stopped_worker_ids)
                event_type = event.get('type')
                worker_id = self._parallel_worker_id(event)
                if event_type == 'episode_start':
                    job = self._validate_parallel_job_event(event, jobs_by_id)
                    if job.episode_id in started_episode_ids:
                        raise RuntimeError(
                            f'duplicate episode_start for {job.episode_id!r}')
                    current_episode = job.progress(self._utc_now())
                    self.model_client.report_evaluation(
                        'episode_start',
                        self._build_episode_start_report(
                            job.task, current_episode, total_episodes),
                    )
                    active_episodes[job.episode_id] = current_episode
                    started_episode_ids.add(job.episode_id)
                    self._acknowledge_parallel_event(worker_id,
                                                     'episode_start',
                                                     job.episode_id)
                elif event_type == 'episode_end':
                    job = self._validate_parallel_job_event(event, jobs_by_id)
                    if job.episode_id not in active_episodes:
                        raise RuntimeError(
                            f'episode_end arrived without an acknowledged '
                            f'episode_start for {job.episode_id!r}')
                    if job.episode_id in completed_episode_ids:
                        raise RuntimeError(
                            f'duplicate episode_end for {job.episode_id!r}')
                    result = event.get('result')
                    if not isinstance(result, EpisodeResult):
                        raise TypeError(
                            'parallel worker episode_end must contain an '
                            'EpisodeResult')
                    self._validate_parallel_result(job, result)
                    self.model_client.report_evaluation(
                        'episode_end',
                        self._build_episode_end_report(
                            job.task,
                            result,
                            completed_episodes=len(results_by_id) + 1,
                            total_episodes=total_episodes,
                            episode_id=job.episode_id,
                        ),
                    )
                    results_by_id[job.episode_id] = result
                    last_result = result
                    completed_episode_ids.add(job.episode_id)
                    active_episodes.pop(job.episode_id, None)
                    self._acknowledge_parallel_event(worker_id, 'episode_end',
                                                     job.episode_id)
                elif event_type == 'error':
                    raise self._parallel_worker_error(event)
                elif event_type == 'stopped':
                    stopped_worker_ids.add(worker_id)
                    if len(completed_episode_ids) != total_episodes:
                        alive = sum(process.is_alive()
                                    for process in self._parallel_processes)
                        if alive == 0:
                            raise RuntimeError(
                                'all parallel rollout workers stopped after '
                                f'{len(completed_episode_ids)}/'
                                f'{total_episodes} episodes')
                elif event_type == 'ready':
                    raise RuntimeError(
                        f'worker {worker_id} emitted duplicate ready event')
                else:
                    raise RuntimeError(
                        f'worker {worker_id} emitted unknown event '
                        f'{event_type!r}')

            if completed_episode_ids != set(jobs_by_id):
                missing = sorted(set(jobs_by_id) - completed_episode_ids)
                raise RuntimeError(
                    f'parallel rollout finished with missing episodes: '
                    f'{missing}')
            results = self._canonical_parallel_results(jobs, results_by_id)
            aggregation_attempted = True
            for result in results:
                self.evaluator.on_episode_end(result)
            last_result = results[-1] if results else None
            summary = self._build_summary(results, local_artifacts=False)
            self._state = 'finished'
            run_end_attempted = True
            self.model_client.report_evaluation(
                'run_end',
                self._build_run_end_report(
                    status='finished',
                    results=results,
                    total_episodes=total_episodes,
                    last_result=last_result,
                    summary=summary,
                ),
            )
            return summary
        except KeyboardInterrupt:
            self._state = 'interrupted'
            self._quiesce_parallel_workers_for_terminal_report()
            results = self._canonical_parallel_results(jobs, results_by_id)
            if not aggregation_attempted:
                self._aggregate_parallel_results_safely(results)
            last_result = results[-1] if results else None
            if not run_end_attempted:
                self._report_run_end_safely(
                    status='interrupted',
                    results=results,
                    total_episodes=total_episodes,
                    current_episode=self._parallel_current_episode(
                        active_episodes),
                    last_result=last_result,
                    error={
                        'error_type': 'KeyboardInterrupt',
                        'message': '',
                    },
                )
            raise
        except Exception as exc:
            self._state = 'failed'
            self._quiesce_parallel_workers_for_terminal_report()
            results = self._canonical_parallel_results(jobs, results_by_id)
            if not aggregation_attempted:
                self._aggregate_parallel_results_safely(results)
            last_result = results[-1] if results else None
            if not run_end_attempted:
                self._report_run_end_safely(
                    status='failed',
                    results=results,
                    total_episodes=total_episodes,
                    current_episode=self._parallel_current_episode(
                        active_episodes),
                    last_result=last_result,
                    error={
                        'error_type': type(exc).__name__,
                        'message': str(exc),
                    },
                )
            raise

    def cleanup(self) -> None:
        if self._state == 'closed':
            return
        errors = []
        try:
            self._stop_parallel_workers()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            errors.append(f'parallel workers: {exc}')
        for component in (self.evaluator, self.environment, self.model_client):
            try:
                component.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                errors.append(f'{type(component).__name__}: {exc}')
        self._state = 'closed'
        if errors:
            warnings.warn(
                'Errors occurred during evaluation cleanup: ' +
                '; '.join(errors),
                RuntimeWarning,
                stacklevel=2,
            )

    def _run_episode(self,
                     task: TaskSpec,
                     episode_index: int,
                     seed: int,
                     episode_id: str | None = None) -> EpisodeResult:
        return execute_episode(
            environment=self.environment,
            model_client=self.model_client,
            evaluator=self.evaluator,
            task=task,
            episode_index=episode_index,
            seed=seed,
            max_episode_steps=self.max_episode_steps,
            execute_horizon=self.execute_horizon,
            stop_on_success=self.stop_on_success,
            episode_id=episode_id,
        )

    def _build_context(self, task: TaskSpec, episode_index: int, seed: int,
                       info: Mapping[str, Any]) -> dict:
        return build_context(task, episode_index, seed, info)

    @staticmethod
    def _validate_action_chunk(action_chunk: ActionChunk) -> np.ndarray:
        return validate_action_chunk(action_chunk)

    def _start_parallel_workers(self) -> list[TaskSpec]:
        report_run_id = self.model_client.evaluation_report_run_id
        if not isinstance(report_run_id, str) or not report_run_id:
            raise RuntimeError(
                'FluxVLA reporting setup did not provide a report_run_id')
        if self._environment_cfg is None or self._model_client_cfg is None:
            raise RuntimeError('parallel component configs are unavailable')

        context = multiprocessing.get_context(self.parallel_start_method)
        event_queue = context.Queue()
        job_queues = [context.Queue() for _ in range(self.parallel_workers)]
        control_queues = [
            context.Queue() for _ in range(self.parallel_workers)
        ]
        inference_model_cfg = make_inference_only_model_config(
            self._model_client_cfg)
        processes = []
        try:
            for worker_id in range(self.parallel_workers):
                simulator_gpu_id = (
                    self.simulator_gpu_ids[worker_id %
                                           len(self.simulator_gpu_ids)]
                    if self.simulator_gpu_ids else None)
                process = context.Process(
                    name=f'fluxthemis-rollout-{worker_id}',
                    target=rollout_worker_main,
                    args=(
                        worker_id,
                        self._environment_cfg,
                        inference_model_cfg,
                        job_queues[worker_id],
                        control_queues[worker_id],
                        event_queue,
                        simulator_gpu_id,
                        self.max_episode_steps,
                        self.execute_horizon,
                        self.stop_on_success,
                        self.parallel_ack_timeout_s,
                    ),
                )
                process.start()
                processes.append(process)
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
            raise

        self._parallel_context = context
        self._parallel_processes = processes
        self._parallel_job_queues = job_queues
        self._parallel_event_queue = event_queue
        self._parallel_control_queues = control_queues

        ready_tasks: dict[int, tuple[TaskSpec, ...]] = {}
        deadline = time.monotonic() + self.parallel_startup_timeout_s
        while len(ready_tasks) < self.parallel_workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'{len(ready_tasks)}/{self.parallel_workers} parallel '
                    'rollout workers became ready within '
                    f'{self.parallel_startup_timeout_s:g} seconds')
            event = self._next_parallel_event(
                stopped_worker_ids=set(),
                timeout_s=min(remaining, self.parallel_startup_timeout_s),
            )
            event_type = event.get('type')
            worker_id = self._parallel_worker_id(event)
            if event_type == 'ready':
                if worker_id in ready_tasks:
                    raise RuntimeError(
                        f'worker {worker_id} emitted duplicate ready event')
                tasks = event.get('tasks')
                if (isinstance(tasks, (str, bytes))
                        or not isinstance(tasks, Sequence)):
                    raise TypeError(
                        f'worker {worker_id} returned an invalid task list')
                canonical = tuple(TaskSpec.from_value(task) for task in tasks)
                if not canonical:
                    raise ValueError(
                        f'worker {worker_id} exposed no evaluation tasks')
                ready_tasks[worker_id] = canonical
            elif event_type == 'error':
                raise self._parallel_worker_error(event)
            elif event_type == 'stopped':
                raise RuntimeError(
                    f'worker {worker_id} stopped before becoming ready')
            else:
                raise RuntimeError(
                    f'worker {worker_id} emitted {event_type!r} before ready')

        reference = ready_tasks[0]
        for worker_id, tasks in sorted(ready_tasks.items()):
            if tasks != reference:
                raise RuntimeError(
                    f'parallel worker {worker_id} returned a task manifest '
                    'different from worker 0')
        return list(reference)

    def _stop_parallel_workers(self) -> None:
        processes = self._parallel_processes
        if not processes:
            return
        for job_queue in self._parallel_job_queues:
            try:
                job_queue.put_nowait(None)
            except Exception:
                continue

        deadline = time.monotonic() + 2.0
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover - defensive OS path
                kill = getattr(process, 'kill', None)
                if callable(kill):
                    kill()
                    process.join(timeout=5.0)

        queues = [
            *self._parallel_job_queues,
            self._parallel_event_queue,
            *self._parallel_control_queues,
        ]
        for process_queue in queues:
            if process_queue is None:
                continue
            cancel_join_thread = getattr(process_queue, 'cancel_join_thread',
                                         None)
            if callable(cancel_join_thread):
                cancel_join_thread()
            close = getattr(process_queue, 'close', None)
            if callable(close):
                close()
        self._parallel_processes = []
        self._parallel_job_queues = []
        self._parallel_event_queue = None
        self._parallel_control_queues = []
        self._parallel_context = None

    def _quiesce_parallel_workers_for_terminal_report(self) -> None:
        try:
            self._stop_parallel_workers()
        except Exception as exc:  # Preserve the original run failure.
            warnings.warn(
                'Could not fully stop parallel rollout workers before the '
                f'terminal report: {exc}',
                RuntimeWarning,
                stacklevel=2,
            )

    def _next_parallel_event(
            self,
            stopped_worker_ids: set[int],
            timeout_s: float | None = None) -> Mapping[str, Any]:
        if self._parallel_event_queue is None:
            raise RuntimeError('parallel event queue is unavailable')
        timeout = (
            self.parallel_event_timeout_s
            if timeout_s is None else float(timeout_s))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    'parallel rollout workers emitted no lifecycle event '
                    f'within {timeout:g} seconds')
            try:
                event = self._parallel_event_queue.get(
                    timeout=min(remaining, 0.5))
            except queue.Empty:
                failed = [(worker_id, process.exitcode)
                          for worker_id, process in enumerate(
                              self._parallel_processes)
                          if worker_id not in stopped_worker_ids
                          and process.exitcode is not None]
                if failed:
                    try:
                        event = self._parallel_event_queue.get_nowait()
                    except queue.Empty as exc:
                        raise RuntimeError(
                            f'parallel rollout worker exited without a '
                            f'terminal event: {failed}') from exc
                    else:
                        if not isinstance(event, Mapping):
                            raise TypeError(
                                'parallel worker event must be a mapping')
                        return event
                continue
            if not isinstance(event, Mapping):
                raise TypeError('parallel worker event must be a mapping')
            return event

    def _acknowledge_parallel_event(self, worker_id: int, event_type: str,
                                    episode_id: str) -> None:
        self._parallel_control_queues[worker_id].put({
            'type': f'{event_type}_ack',
            'episode_id': episode_id,
        })

    def _parallel_worker_id(self, event: Mapping[str, Any]) -> int:
        worker_id = event.get('worker_id')
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            raise TypeError('parallel worker event has an invalid worker_id')
        if worker_id < 0 or worker_id >= self.parallel_workers:
            raise ValueError(f'parallel worker_id {worker_id} is outside [0, '
                             f'{self.parallel_workers})')
        return worker_id

    @staticmethod
    def _validate_parallel_job_event(
            event: Mapping[str, Any],
            jobs_by_id: Mapping[str, EpisodeJob]) -> EpisodeJob:
        job = event.get('job')
        if not isinstance(job, EpisodeJob):
            raise TypeError('parallel worker event must contain an EpisodeJob')
        expected = jobs_by_id.get(job.episode_id)
        if expected is None or job != expected:
            raise RuntimeError(
                f'parallel worker returned unknown or modified job '
                f'{job.episode_id!r}')
        return job

    @staticmethod
    def _validate_parallel_result(job: EpisodeJob,
                                  result: EpisodeResult) -> None:
        identity = (result.task_id, result.episode_index, result.seed)
        expected = (job.task.task_id, job.episode_index, job.seed)
        if identity != expected:
            raise RuntimeError(
                f'parallel result identity {identity!r} does not match job '
                f'{expected!r}')

    @staticmethod
    def _canonical_parallel_results(
            jobs: Sequence[EpisodeJob],
            results_by_id: Mapping[str, EpisodeResult]) -> list[EpisodeResult]:
        return [
            results_by_id[job.episode_id] for job in jobs
            if job.episode_id in results_by_id
        ]

    def _aggregate_parallel_results_safely(
            self, results: Sequence[EpisodeResult]) -> None:
        try:
            for result in results:
                self.evaluator.on_episode_end(result)
        except Exception as exc:  # Preserve the worker/interrupt failure.
            warnings.warn(
                'Could not aggregate all completed parallel episodes after '
                f'the run failed: {exc}',
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _parallel_worker_error(event: Mapping[str, Any]) -> RuntimeError:
        worker_id = event.get('worker_id')
        phase = event.get('phase', 'unknown')
        error_type = event.get('error_type', 'Error')
        message = event.get('message', '')
        child_traceback = event.get('traceback', '')
        detail = (
            f'parallel rollout worker {worker_id} failed during {phase}: '
            f'{error_type}: {message}')
        if child_traceback:
            detail += f'\n{child_traceback}'
        return RuntimeError(detail)

    @staticmethod
    def _parallel_current_episode(
            active: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
        if not active:
            return None
        return {
            'active_episodes':
            [dict(active[episode_id]) for episode_id in sorted(active)],
        }

    @staticmethod
    def _configure_serial_simulator_gpu(simulator_gpu_id: str) -> None:
        configure_simulator_gpu(simulator_gpu_id)

    def _prepare_run_dir(self) -> None:
        artifacts = (
            self.summary_path,
            self.episodes_path,
            self.progress_path,
            self.failure_path,
            self.resolved_config_path,
        )
        if not self.overwrite and any(path.exists() for path in artifacts):
            raise FileExistsError(
                f'Run directory already contains evaluation artifacts: '
                f'{self.run_dir}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            for path in artifacts:
                path.unlink(missing_ok=True)

    def _select_tasks(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        if not tasks:
            raise ValueError('Environment exposes no evaluation tasks')
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                f'Environment returned duplicate task ids: {task_ids}')
        selected = tasks
        if self.task_ids is not None:
            task_map = {task.task_id: task for task in tasks}
            missing = [
                task_id for task_id in self.task_ids if task_id not in task_map
            ]
            if missing:
                raise ValueError(
                    f'Unknown task ids {missing}; available tasks: '
                    f'{task_ids}')
            selected = [task_map[task_id] for task_id in self.task_ids]

        if self.task_metadata_filters is None:
            return selected
        known_fields = {str(key) for task in tasks for key in task.metadata}
        unknown_fields = sorted(set(self.task_metadata_filters) - known_fields)
        if unknown_fields:
            raise ValueError(
                f'Unknown task metadata filter fields {unknown_fields}; '
                f'available fields: {sorted(known_fields)}')
        missing_value = object()
        selected = [
            task for task in selected if all(
                (value := task.metadata.get(field, missing_value))
                is not missing_value and any(value == expected
                                             for expected in expected_values)
                for field, expected_values in
                self.task_metadata_filters.items())
        ]
        if not selected:
            raise ValueError('Task metadata filters selected no tasks')
        return selected

    def _build_summary(self,
                       results: list[EpisodeResult],
                       local_artifacts: bool = True) -> dict:
        elapsed = time.perf_counter() - self._run_started_monotonic
        if local_artifacts:
            artifacts = {
                'episodes': str(self.episodes_path),
                'progress': str(self.progress_path),
                'summary': str(self.summary_path),
            }
            if self.resolved_config_path.exists():
                artifacts['resolved_config'] = str(self.resolved_config_path)
        else:
            artifacts = {}
            remote_run_dir = self.model_client.evaluation_report_run_dir
            if remote_run_dir:
                artifacts['remote_run_dir'] = remote_run_dir
        return {
            'schema_version': '1.0',
            'run': {
                'name': self.run_name,
                'runner': type(self).__name__,
                'started_at': self._started_at,
                'finished_at': self._utc_now(),
                'duration_s': elapsed,
                'seed': self.seed,
                'tasks': [task.task_id for task in self.tasks_to_run],
                'episodes_per_task': self.episodes_per_task,
                'max_episode_steps': self.max_episode_steps,
                'execute_horizon': self.execute_horizon,
                'parallel_workers': self.parallel_workers,
                'simulator_gpu_ids': list(self.simulator_gpu_ids),
                'completed_episodes': len(results),
            },
            'metrics': self.evaluator.summarize(),
            'artifacts': artifacts,
        }

    def _build_run_start_report(self, total_episodes: int) -> dict:
        payload = {
            'schema_version':
            '1.0',
            'run_name':
            self.run_name,
            'runner':
            type(self).__name__,
            'started_at':
            self._started_at,
            'seed':
            self.seed,
            'episodes_per_task':
            self.episodes_per_task,
            'max_episode_steps':
            self.max_episode_steps,
            'execute_horizon':
            self.execute_horizon,
            'stop_on_success':
            self.stop_on_success,
            'parallel_workers':
            self.parallel_workers,
            'simulator_gpu_ids':
            list(self.simulator_gpu_ids),
            'task_ids': [task.task_id for task in self.tasks_to_run],
            'total_tasks':
            len(self.tasks_to_run),
            'total_episodes':
            total_episodes,
            'full_suite':
            self._is_full_suite,
            'tasks': [
                self._task_report(task, task_index)
                for task_index, task in enumerate(self.tasks_to_run)
            ],
        }
        if self._resolved_config_source is not None:
            payload['resolved_config'] = {
                'source': self._resolved_config_source,
            }
        if self.task_metadata_filters is not None:
            payload['task_metadata_filters'] = to_jsonable(
                self.task_metadata_filters)
        return payload

    def _build_episode_start_report(self, task: TaskSpec,
                                    episode: Mapping[str, Any],
                                    total_episodes: int) -> dict:
        task_index = int(episode['task_position']) - 1
        return {
            'schema_version': '1.0',
            'run_name': self.run_name,
            'total_episodes': total_episodes,
            **self._task_report(task, task_index),
            **to_jsonable(dict(episode)),
        }

    def _build_episode_end_report(self,
                                  task: TaskSpec,
                                  result: EpisodeResult,
                                  completed_episodes: int,
                                  total_episodes: int,
                                  episode_id: str | None = None) -> dict:
        task_index = self.tasks_to_run.index(task)
        payload = {
            'schema_version': '1.0',
            'run_name': self.run_name,
            **self._task_report(task, task_index),
            'completed_episodes': completed_episodes,
            'total_episodes': total_episodes,
            'finished_at': self._utc_now(),
            **to_jsonable(result),
        }
        if episode_id is not None:
            payload['episode_id'] = episode_id
        return payload

    def _build_run_end_report(self,
                              status: str,
                              results: list[EpisodeResult],
                              total_episodes: int,
                              current_episode: Mapping[str, Any] | None = None,
                              last_result: EpisodeResult | None = None,
                              summary: Mapping[str, Any] | None = None,
                              error: Mapping[str, Any] | None = None) -> dict:
        metrics_error = None
        if summary is not None and isinstance(summary.get('metrics'), Mapping):
            metrics = summary['metrics']
        else:
            try:
                metrics = self.evaluator.summarize()
            except Exception as exc:  # Preserve the original run failure.
                metrics = {}
                metrics_error = {
                    'error_type': type(exc).__name__,
                    'message': str(exc),
                }
        payload = {
            'schema_version':
            '1.0',
            'run_name':
            self.run_name,
            'status':
            status,
            'finished_at':
            self._utc_now(),
            'duration_s':
            self._elapsed_s(),
            'completed_episodes':
            len(results),
            'total_episodes':
            total_episodes,
            'full_suite':
            self._is_full_suite,
            'current_episode': (to_jsonable(dict(current_episode))
                                if current_episode is not None else None),
            'last_result':
            to_jsonable(last_result),
            'metrics':
            to_jsonable(metrics),
        }
        if metrics_error is not None:
            payload['metrics_error'] = metrics_error
        if summary is not None:
            payload['summary'] = to_jsonable(dict(summary))
        if error is not None:
            payload['error'] = to_jsonable(dict(error))
        return payload

    @staticmethod
    def _task_report(task: TaskSpec, task_index: int) -> dict:
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

    def _report_run_end_safely(self, status: str, results: list[EpisodeResult],
                               total_episodes: int,
                               current_episode: Mapping[str, Any] | None,
                               last_result: EpisodeResult | None,
                               error: Mapping[str, Any]) -> None:
        try:
            payload = self._build_run_end_report(
                status=status,
                results=results,
                total_episodes=total_episodes,
                current_episode=current_episode,
                last_result=last_result,
                error=error,
            )
            self.model_client.report_evaluation('run_end', payload)
        except Exception as report_exc:  # pragma: no cover - defensive path
            warnings.warn(
                'Could not report terminal evaluation state to FluxVLA: '
                f'{report_exc}',
                RuntimeWarning,
                stacklevel=2,
            )

    def _write_progress(self,
                        results: list[EpisodeResult],
                        total_episodes: int,
                        status: str,
                        current_episode: Mapping[str, Any] | None = None,
                        last_result: EpisodeResult | None = None,
                        error: Mapping[str, Any] | None = None) -> dict:
        completed = len(results)
        successes = sum(bool(result.success) for result in results)
        elapsed_s = self._elapsed_s()
        if status == 'finished':
            eta_s = 0.0
        elif status != 'running' or completed == 0:
            eta_s = None
        else:
            eta_s = elapsed_s / completed * (total_episodes - completed)

        progress = dict(
            schema_version='1.0',
            run_name=self.run_name,
            status=status,
            pid=os.getpid(),
            started_at=self._started_at,
            updated_at=self._utc_now(),
            total_tasks=len(self.tasks_to_run),
            completed_episodes=completed,
            total_episodes=total_episodes,
            remaining_episodes=max(0, total_episodes - completed),
            progress=(completed / total_episodes if total_episodes else 1.0),
            current_episode=(dict(current_episode)
                             if current_episode is not None else None),
            last_episode=self._episode_result_progress(last_result),
            successes=successes,
            success_rate=successes / completed if completed else 0.0,
            elapsed_s=elapsed_s,
            eta_s=eta_s,
        )
        if error is not None:
            progress['error'] = dict(error)
        write_json_atomic(self.progress_path, progress)
        return progress

    def _write_terminal_progress(
            self,
            results: list[EpisodeResult],
            total_episodes: int,
            status: str,
            current_episode: Mapping[str, Any] | None,
            last_result: EpisodeResult | None,
            error: Mapping[str, Any] | None = None) -> None:
        try:
            self._write_progress(
                results,
                total_episodes=total_episodes,
                status=status,
                current_episode=current_episode,
                last_result=last_result,
                error=error,
            )
        except Exception as artifact_exc:  # pragma: no cover - defensive
            self._warn_artifact_write_failure(self.progress_path, artifact_exc)

    @staticmethod
    def _warn_artifact_write_failure(path: Path, error: Exception) -> None:
        warnings.warn(
            f'Could not write evaluation artifact {path}: {error}',
            RuntimeWarning,
            stacklevel=2,
        )

    def _print_episode_start(self, episode: Mapping[str, Any],
                             total_episodes: int) -> None:
        task_id = json.dumps(str(episode['task_id']), ensure_ascii=False)
        print(
            f'[FluxThemis] episode_start '
            f'current={episode["ordinal"]}/{total_episodes} '
            f'task={task_id} '
            f'task_position={episode["task_position"]}/'
            f'{episode["total_tasks"]} '
            f'episode={episode["episode_number"]}/'
            f'{episode["episodes_per_task"]} seed={episode["seed"]}',
            file=sys.stderr,
            flush=True,
        )

    def _print_episode_done(self, progress: Mapping[str, Any]) -> None:
        completed = int(progress['completed_episodes'])
        total = int(progress['total_episodes'])
        percentage = float(progress['progress']) * 100.0
        episode = progress['last_episode']
        task_id = json.dumps(str(episode['task_id']), ensure_ascii=False)
        success = 'true' if episode['success'] else 'false'
        successes = int(progress['successes'])
        success_rate = float(progress['success_rate']) * 100.0
        print(
            f'[FluxThemis] episode_done completed={completed}/{total} '
            f'progress={percentage:.1f}% task={task_id} '
            f'episode={episode["episode_number"]}/'
            f'{self.episodes_per_task} '
            f'success={success} overall={successes}/{completed} '
            f'({success_rate:.1f}%) steps={episode["steps"]} '
            f'episode_time={float(episode["duration_s"]):.1f}s '
            f'elapsed={self._format_duration(progress["elapsed_s"])} '
            f'eta={self._format_duration(progress["eta_s"])}',
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _episode_result_progress(
            result: EpisodeResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            'task_id': result.task_id,
            'episode_index': result.episode_index,
            'episode_number': result.episode_index + 1,
            'seed': result.seed,
            'success': bool(result.success),
            'termination_reason': result.termination_reason,
            'steps': result.steps,
            'duration_s': result.duration_s,
        }

    def _elapsed_s(self) -> float:
        if self._run_started_monotonic is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._run_started_monotonic)

    @staticmethod
    def _format_duration(value: float | None) -> str:
        if value is None:
            return '--:--'
        seconds = max(0, int(round(float(value))))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    @staticmethod
    def _copy_component_cfg(value: Mapping[str, Any],
                            name: str) -> dict[str, Any]:
        try:
            copied = copy.deepcopy(dict(value))
        except Exception as exc:
            raise TypeError(
                f'{name} config must be copyable for spawned workers') \
                from exc
        if 'type' not in copied:
            raise KeyError(f'{name} config must define type')
        return copied

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        value = int(value)
        if value < 1:
            raise ValueError(f'{name} must be positive')
        return value

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{name} must be numeric')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    @staticmethod
    def _normalize_simulator_gpu_ids(
            value: Sequence[str | int] | str | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            values: Sequence[str | int] = value.split(',')
        elif isinstance(value, Sequence) and not isinstance(value, bytes):
            values = value
        else:
            raise TypeError(
                'simulator_gpu_ids must be a comma-separated string or a '
                'sequence')
        normalized = []
        for item in values:
            if isinstance(item, bool):
                raise TypeError('simulator_gpu_ids entries cannot be bools')
            if isinstance(item, Integral):
                if int(item) < 0:
                    raise ValueError(
                        'simulator_gpu_ids integers must be non-negative')
                gpu_id = str(int(item))
            elif isinstance(item, str):
                gpu_id = item.strip()
            else:
                raise TypeError(
                    'simulator_gpu_ids entries must be strings or integers')
            if not gpu_id or ',' in gpu_id:
                raise ValueError(
                    'simulator_gpu_ids entries must be non-empty GPU ids')
            normalized.append(gpu_id)
        if not normalized:
            raise ValueError('simulator_gpu_ids cannot be empty')
        return tuple(normalized)

    @staticmethod
    def _normalize_task_ids(
            task_ids: Sequence[str] | str | None) -> tuple[str, ...] | None:
        if task_ids is None:
            return None
        if isinstance(task_ids, str):
            values = [value.strip() for value in task_ids.split(',')]
            task_ids = [value for value in values if value]
        normalized = tuple(str(task_id) for task_id in task_ids)
        if not normalized:
            raise ValueError('task_ids cannot be empty')
        if len(normalized) != len(set(normalized)):
            raise ValueError(f'task_ids contains duplicates: {normalized}')
        return normalized

    @staticmethod
    def _normalize_task_metadata_filters(
        filters: Mapping[str, Any] | None
    ) -> dict[str, tuple[Any, ...]] | None:
        if filters is None:
            return None
        if not isinstance(filters, Mapping):
            raise TypeError('task_metadata_filters must be a mapping')
        if not filters:
            raise ValueError('task_metadata_filters cannot be empty')
        normalized = {}
        for field, expected in filters.items():
            if not isinstance(field, str) or not field:
                raise ValueError(
                    'task_metadata_filters fields must be non-empty strings')
            if isinstance(expected,
                          Sequence) and not isinstance(expected, (str, bytes)):
                values = tuple(expected)
            else:
                values = (expected, )
            if not values:
                raise ValueError(
                    f'task_metadata_filters[{field!r}] cannot be empty')
            normalized[field] = values
        return normalized

    @property
    def _is_full_suite(self) -> bool:
        return (self.task_ids is None and self.task_metadata_filters is None)

    @staticmethod
    def _default_run_name() -> str:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        return f'eval_{timestamp}_{os.getpid()}'

    @staticmethod
    def _validate_run_name(work_dir: Path, run_name: str) -> None:
        if (not run_name or run_name in {'.', '..'}
                or Path(run_name).name != run_name
                or Path(run_name).is_absolute()):
            raise ValueError('run_name must be a non-empty path component')
        resolved_work_dir = work_dir.resolve()
        resolved_run_dir = (work_dir / run_name).resolve()
        if resolved_run_dir.parent != resolved_work_dir:
            raise ValueError('run_name must stay directly inside work_dir')

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
