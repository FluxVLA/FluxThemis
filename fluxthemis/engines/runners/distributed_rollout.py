"""Multi-node ZMQ rollout coordinator and simulator workers."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mmengine import Config
from mmengine.utils import import_modules_from_strings

from fluxthemis.engines.utils import (build_environment_from_cfg,
                                      build_evaluation_reporter_from_cfg,
                                      build_evaluator_from_cfg,
                                      build_policy_client_from_cfg,
                                      to_jsonable)
from fluxthemis.environments.base import EnvironmentCaseRejected
from fluxthemis.structures import EpisodeResult, TaskSpec

from ...transports.fluxvla_zmq import connection_transport
from ...transports.zmq_protocol import (decode_frame, decode_header,
                                        empty_payload, encode_frame,
                                        encode_header)
from .evaluation import EvaluationSettings, EvaluationStatus, utc_now
from .evaluation_run import EvaluationRun
from .rollout import (EpisodeJob, build_episode_jobs, configure_simulator_gpu,
                      configure_worker_lifetime, execute_episode, select_tasks)

DEFAULT_ARTIFACT_SUBDIR = 'fluxthemis'
_WORKER_ONLY_EVALUATION_KEYS = {
    'environment',
    'policy_client',
    'save_videos',
    'video_dir',
    'video_fps',
    'work_dir',
}


@dataclass
class _WorkerSlot:
    routing_id: bytes
    worker_id: str
    device: str
    task_manifest: tuple[TaskSpec, ...] = ()
    current_job: EpisodeJob | None = None
    episode_deadline: float | None = None

    @property
    def is_ready(self) -> bool:
        return bool(self.task_manifest)

    @property
    def is_idle(self) -> bool:
        return self.is_ready and self.current_job is None


def _next_environment_case_index_by_task(
        jobs: tuple[EpisodeJob, ...]) -> dict[str, int]:
    next_index_by_task: dict[str, int] = {}
    for job in jobs:
        if job.environment_case_index is None:
            continue
        task_id = job.task.task_id
        next_index_by_task[task_id] = max(
            next_index_by_task.get(task_id, 0), job.environment_case_index + 1)
    return next_index_by_task


@dataclass
class _SchedulingState:
    jobs: tuple[EpisodeJob, ...]
    pending_jobs: deque[EpisodeJob]
    jobs_by_episode_id: dict[str, EpisodeJob]
    results_by_episode_id: dict[str, EpisodeResult]
    started_episode_ids: set[str]
    next_environment_case_index_by_task: dict[str, int]

    @classmethod
    def from_jobs(cls, jobs: tuple[EpisodeJob, ...]) -> _SchedulingState:
        return cls(
            jobs=jobs,
            pending_jobs=deque(jobs),
            jobs_by_episode_id={job.episode_id: job
                                for job in jobs},
            results_by_episode_id={},
            started_episode_ids=set(),
            next_environment_case_index_by_task=(
                _next_environment_case_index_by_task(jobs)),
        )

    @property
    def is_complete(self) -> bool:
        return len(self.results_by_episode_id) == len(self.jobs)

    def ordered_results(self) -> list[EpisodeResult]:
        return [
            self.results_by_episode_id[job.episode_id] for job in self.jobs
            if job.episode_id in self.results_by_episode_id
        ]


class DistributedEvaluationCoordinator:
    """Dynamically schedule canonical episodes across remote simulators."""

    def __init__(self,
                 rollout_bind: str,
                 expected_workers: int,
                 startup_timeout_s: float = 900.0,
                 episode_timeout_s: float = 3600.0,
                 deployment_id: str | None = None) -> None:
        self.rollout_bind = _endpoint(rollout_bind, 'rollout_bind')
        self.expected_workers = _positive_int(expected_workers,
                                              'expected_workers')
        self.startup_timeout_s = _positive_float(startup_timeout_s,
                                                 'startup_timeout_s')
        self.episode_timeout_s = _positive_float(episode_timeout_s,
                                                 'episode_timeout_s')
        self.deployment_id = deployment_id or uuid.uuid4().hex
        self._worker_slots_by_routing_id: dict[bytes, _WorkerSlot] = {}
        self._routing_id_by_worker_id: dict[str, bytes] = {}
        self._idle_worker_routing_ids: deque[bytes] = deque()
        self._scheduling_state: _SchedulingState | None = None
        self._worker_router = None
        self._context = None
        self._stopping = False

    def run(self, evaluation_run: EvaluationRun) -> dict[str, Any]:
        import zmq

        self._context = zmq.Context()
        self._worker_router = self._context.socket(zmq.ROUTER)
        self._worker_router.setsockopt(zmq.LINGER, 0)
        self._worker_router.bind(self.rollout_bind)
        poller = zmq.Poller()
        poller.register(self._worker_router, zmq.POLLIN)
        self._install_signal_handlers()
        try:
            canonical_task_manifest = self._wait_for_workers(
                poller, evaluation_run)
            return self._run_jobs(poller, canonical_task_manifest,
                                  evaluation_run)
        finally:
            self._shutdown_all_workers()
            self._worker_router.close(linger=1000)
            self._context.term()

    def _wait_for_workers(
            self, poller: Any,
            evaluation_run: EvaluationRun) -> tuple[TaskSpec, ...]:
        deadline = time.monotonic() + self.startup_timeout_s
        next_log_at = 0.0
        while not self._all_workers_ready():
            if self._stopping:
                raise KeyboardInterrupt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'Only {self._ready_worker_count()}/'
                    f'{self.expected_workers} simulator workers '
                    'became ready')
            now = time.monotonic()
            if now >= next_log_at:
                print(
                    '[FluxThemis] waiting for simulator workers: '
                    f'{self._ready_worker_count()}/'
                    f'{self.expected_workers}',
                    file=sys.stderr,
                    flush=True,
                )
                next_log_at = now + 10.0
            self._poll_once(poller, min(1.0, remaining), evaluation_run)

        task_manifests = [
            slot.task_manifest
            for slot in self._worker_slots_by_routing_id.values()
            if slot.is_ready
        ]
        if not task_manifests:
            raise RuntimeError(
                'No simulator worker task manifest is available')
        canonical_task_manifest = task_manifests[0]
        if any(task_manifest != canonical_task_manifest
               for task_manifest in task_manifests[1:]):
            raise RuntimeError('Simulator workers returned different task '
                               'manifests')
        return canonical_task_manifest

    def _run_jobs(self, poller: Any, canonical_task_manifest: tuple[TaskSpec,
                                                                    ...],
                  evaluation_run: EvaluationRun) -> dict[str, Any]:
        settings = evaluation_run.settings
        selected_tasks = select_tasks(canonical_task_manifest,
                                      settings.task_ids)
        jobs = build_episode_jobs(
            selected_tasks,
            episodes_per_task=settings.episodes_per_task,
            base_seed=settings.base_seed,
            run_session_id=evaluation_run.run_session_id,
            episodes_per_task_overrides=settings.episodes_per_task_overrides,
        )
        scheduling_state = _SchedulingState.from_jobs(jobs)
        self._scheduling_state = scheduling_state

        try:
            evaluation_run.start(selected_tasks, len(jobs))
            self._dispatch_jobs()
            while not scheduling_state.is_complete:
                if self._stopping:
                    raise KeyboardInterrupt
                now = time.monotonic()
                expired_slots = [
                    slot for slot in self._worker_slots_by_routing_id.values()
                    if slot.episode_deadline is not None
                    and slot.episode_deadline <= now
                ]
                if expired_slots:
                    worker_ids = ', '.join(slot.worker_id
                                           for slot in expired_slots)
                    raise TimeoutError(
                        f'simulator episode timed out on {worker_ids}')
                self._poll_once(poller, 0.5, evaluation_run)
                self._dispatch_jobs()

            results = scheduling_state.ordered_results()
            return evaluation_run.finish(results[-1] if results else None)
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            results = scheduling_state.ordered_results()
            evaluation_run.fail(
                EvaluationStatus.INTERRUPTED
                if interrupted else EvaluationStatus.FAILED,
                current_episode=self._active_episode_progress(),
                last_result=results[-1] if results else None,
                error={
                    'error_type': type(exc).__name__,
                    'message': str(exc),
                },
            )
            raise

    def _poll_once(self, poller: Any, timeout_s: float,
                   evaluation_run: EvaluationRun) -> None:
        events = dict(poller.poll(timeout=max(1, int(timeout_s * 1000))))
        if self._worker_router in events:
            self._receive_worker_message(evaluation_run)

    def _receive_worker_message(self, evaluation_run: EvaluationRun) -> None:
        frames = self._worker_router.recv_multipart()
        if len(frames) < 2:
            raise RuntimeError('simulator worker message is missing a header')
        routing_id = frames[0]
        header = decode_header(frames[1])
        self._validate_deployment(header)
        payload = frames[2] if len(frames) > 2 else empty_payload()
        message_type = header['type']
        if message_type == 'register_worker':
            self._register_worker(routing_id, header)
            return
        slot = self._require_worker_slot(routing_id, header)
        if message_type == 'ready':
            self._mark_worker_ready(slot, payload)
            return
        if message_type == 'worker_error':
            raise RuntimeError(f'simulator worker {slot.worker_id} failed: '
                               f'{header.get("message") or message_type}')
        scheduling_state = self._require_scheduling_state()
        if message_type == 'episode_start':
            job = self._require_assigned_job(slot, header)
            slot.episode_deadline = time.monotonic() + self.episode_timeout_s
            if job.episode_id not in scheduling_state.started_episode_ids:
                episode_start_payload = job.episode_start_payload(utc_now())
                evaluation_run.on_episode_start(episode_start_payload)
                scheduling_state.started_episode_ids.add(job.episode_id)
            self._send_worker_message(
                'episode_start_ack', slot, episode_id=job.episode_id)
        elif message_type == 'episode_end':
            job = self._require_assigned_job(slot, header)
            result = _episode_result_from_wire(decode_frame(payload))
            job.validate_result(result)
            if job.episode_id in scheduling_state.results_by_episode_id:
                raise RuntimeError(
                    f'duplicate episode result {job.episode_id!r}')
            scheduling_state.results_by_episode_id[job.episode_id] = result
            evaluation_run.on_episode_end(result, job.episode_id)
            self._send_worker_message(
                'episode_end_ack', slot, episode_id=job.episode_id)
            self._release_worker_slot(slot)
        elif message_type == 'environment_case_rejected':
            job = self._require_assigned_job(slot, header)
            if job.episode_id in scheduling_state.started_episode_ids:
                raise RuntimeError(
                    f'worker {slot.worker_id} rejected started episode '
                    f'{job.episode_id!r}')
            next_case_index = (
                scheduling_state.next_environment_case_index_by_task.get(
                    job.task.task_id))
            if next_case_index is None:
                raise RuntimeError(
                    f'environment rejected a case for job {job.episode_id!r} '
                    'without an environment_case_index')
            scheduling_state.next_environment_case_index_by_task[
                job.task.task_id] = next_case_index + 1
            retry = replace(job, environment_case_index=next_case_index)
            scheduling_state.jobs_by_episode_id[job.episode_id] = retry
            scheduling_state.pending_jobs.appendleft(retry)
            self._send_worker_message(
                'environment_case_rejected_ack',
                slot,
                episode_id=job.episode_id,
            )
            self._release_worker_slot(slot)
            print(
                f'[FluxThemis] simulator worker {slot.worker_id} rejected '
                f'environment case {job.environment_case_index} for '
                f'{job.task.task_id}; retrying with case {next_case_index}',
                file=sys.stderr,
                flush=True,
            )
        else:
            raise RuntimeError(
                f'Unknown simulator-worker message {message_type!r}')

    def _register_worker(self, routing_id: bytes,
                         header: Mapping[str, Any]) -> None:
        worker_id = _required_string(header, 'worker_id')
        if (routing_id in self._worker_slots_by_routing_id
                or worker_id in self._routing_id_by_worker_id):
            raise RuntimeError(f'duplicate simulator worker {worker_id!r}')
        if len(self._worker_slots_by_routing_id) >= self.expected_workers:
            raise RuntimeError('received more simulator workers than expected')
        slot = _WorkerSlot(
            routing_id=routing_id,
            worker_id=worker_id,
            device=_required_string(header, 'device'),
        )
        self._worker_slots_by_routing_id[routing_id] = slot
        self._routing_id_by_worker_id[worker_id] = routing_id

    def _mark_worker_ready(self, slot: _WorkerSlot, payload: bytes) -> None:
        if slot.is_ready:
            raise RuntimeError(
                f'simulator worker {slot.worker_id!r} is already ready')
        body = decode_frame(payload)
        if not isinstance(body, Mapping):
            raise TypeError('simulator ready payload must be a mapping')
        slot.task_manifest = tuple(
            TaskSpec.from_value(value) for value in body.get('tasks', ()))
        if not slot.task_manifest:
            raise ValueError('simulator task manifest cannot be empty')
        self._idle_worker_routing_ids.append(slot.routing_id)
        print(
            f'[FluxThemis] simulator worker ready id={slot.worker_id} '
            f'device={slot.device}',
            file=sys.stderr,
            flush=True,
        )

    def _dispatch_jobs(self) -> None:
        scheduling_state = self._require_scheduling_state()
        while scheduling_state.pending_jobs:
            slot = self._next_idle_worker_slot()
            if slot is None:
                return
            job = scheduling_state.pending_jobs.popleft()
            slot.current_job = job
            slot.episode_deadline = time.monotonic() + self.episode_timeout_s
            self._send_worker_message(
                'episode_job',
                slot,
                episode_id=job.episode_id,
                payload=encode_frame(to_jsonable(job)),
            )

    def _next_idle_worker_slot(self) -> _WorkerSlot | None:
        while self._idle_worker_routing_ids:
            routing_id = self._idle_worker_routing_ids.popleft()
            slot = self._worker_slots_by_routing_id.get(routing_id)
            if slot is not None and slot.is_idle:
                return slot
        return None

    def _release_worker_slot(self, slot: _WorkerSlot) -> None:
        slot.current_job = None
        slot.episode_deadline = None
        self._idle_worker_routing_ids.append(slot.routing_id)

    def _require_assigned_job(self, slot: _WorkerSlot,
                              header: Mapping[str, Any]) -> EpisodeJob:
        scheduling_state = self._require_scheduling_state()
        episode_id = _required_string(header, 'episode_id')
        job = scheduling_state.jobs_by_episode_id.get(episode_id)
        if (job is None or slot.current_job is None
                or slot.current_job != job):
            raise RuntimeError(
                f'worker {slot.worker_id} emitted an unassigned episode '
                f'{episode_id!r}')
        return job

    def _require_worker_slot(self, routing_id: bytes,
                             header: Mapping[str, Any]) -> _WorkerSlot:
        slot = self._worker_slots_by_routing_id.get(routing_id)
        if slot is None:
            raise RuntimeError(
                'simulator worker must register before sending data')
        if _required_string(header, 'worker_id') != slot.worker_id:
            raise RuntimeError('simulator worker_id mismatch')
        return slot

    def _require_scheduling_state(self) -> _SchedulingState:
        if self._scheduling_state is None:
            raise RuntimeError('simulator emitted an episode event before '
                               'the evaluation started')
        return self._scheduling_state

    def _validate_deployment(self, header: Mapping[str, Any]) -> None:
        if header.get('deployment_id') != self.deployment_id:
            raise RuntimeError('ZMQ deployment_id mismatch')

    def _send_worker_message(self,
                             message_type: str,
                             slot: _WorkerSlot,
                             payload: bytes | None = None,
                             **values: Any) -> None:
        self._worker_router.send_multipart([
            slot.routing_id,
            encode_header(
                message_type,
                deployment_id=self.deployment_id,
                worker_id=slot.worker_id,
                **values,
            ),
            payload if payload is not None else empty_payload(),
        ])

    def _ready_worker_count(self) -> int:
        return sum(slot.is_ready
                   for slot in self._worker_slots_by_routing_id.values())

    def _all_workers_ready(self) -> bool:
        return self._ready_worker_count() == self.expected_workers

    def _active_episode_progress(self) -> dict[str, Any] | None:
        jobs = [
            slot.current_job.episode_start_payload(utc_now())
            for slot in self._worker_slots_by_routing_id.values()
            if slot.current_job is not None
        ]
        return {'active_episodes': jobs} if jobs else None

    def _shutdown_all_workers(self) -> None:
        self._stopping = True
        if self._worker_router is not None:
            for slot in self._worker_slots_by_routing_id.values():
                try:
                    self._send_worker_message('shutdown_worker', slot)
                except Exception:
                    pass

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def stop(_signum, _frame):
            self._stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)


def run_simulator_worker(
        config_path: str,
        rollout_endpoint: str,
        worker_id: str,
        deployment_id: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None) -> None:
    """Build one simulator and execute jobs received from the coordinator."""
    import zmq

    evaluation_cfg = load_distributed_evaluation_config(
        config_path, config_overrides=config_overrides)
    save_videos, video_dir, video_fps = _distributed_video_settings(
        evaluation_cfg, deployment_id)
    environment = build_environment_from_cfg(evaluation_cfg['environment'])
    environment.configure_worker(worker_id, deployment_id)
    configure_simulator_gpu(environment, device)
    environment.configure_video_recording(save_videos, video_dir)
    policy_client = build_policy_client_from_cfg(
        evaluation_cfg['policy_client'])
    context = zmq.Context()
    coordinator_socket = context.socket(zmq.DEALER)
    coordinator_socket.setsockopt(zmq.IDENTITY, worker_id.encode())
    coordinator_socket.setsockopt(zmq.LINGER, 0)
    coordinator_socket.connect(_endpoint(rollout_endpoint, 'rollout_endpoint'))
    message_identity = {
        'deployment_id': deployment_id,
        'worker_id': worker_id,
    }
    coordinator_socket.send_multipart([
        encode_header(
            'register_worker',
            **message_identity,
            device=device,
        ),
        empty_payload(),
    ])
    try:
        policy_client.setup()
        environment.setup()
        task_manifest = tuple(
            TaskSpec.from_value(task) for task in environment.tasks())
        coordinator_socket.send_multipart([
            encode_header('ready', **message_identity),
            encode_frame({
                'tasks': [to_jsonable(task) for task in task_manifest],
            }),
        ])
        while True:
            frames = coordinator_socket.recv_multipart()
            header = decode_header(frames[0])
            if header.get('deployment_id') != deployment_id:
                raise RuntimeError('ZMQ deployment_id mismatch')
            if header.get('worker_id') != worker_id:
                raise RuntimeError('simulator worker_id mismatch')
            payload = frames[1] if len(frames) > 1 else empty_payload()
            if header['type'] == 'shutdown_worker':
                break
            if header['type'] != 'episode_job':
                raise RuntimeError(
                    f'Unexpected simulator command {header["type"]!r}')
            job = _episode_job_from_wire(decode_frame(payload))

            def report_episode_start() -> None:
                _send_worker_event(coordinator_socket, message_identity,
                                   'episode_start', job.episode_id)

            try:
                result = execute_episode(
                    environment=environment,
                    policy_client=policy_client,
                    job=job,
                    max_episode_steps=int(evaluation_cfg['max_episode_steps']),
                    execute_horizon=evaluation_cfg.get('execute_horizon'),
                    stop_on_success=bool(
                        evaluation_cfg.get('stop_on_success', True)),
                    on_episode_started=report_episode_start,
                    save_videos=save_videos,
                    video_dir=video_dir,
                    video_fps=video_fps,
                    video_worker_id=worker_id,
                )
            except EnvironmentCaseRejected as exc:
                _send_worker_event(
                    coordinator_socket,
                    message_identity,
                    'environment_case_rejected',
                    job.episode_id,
                    message=str(exc),
                )
                continue
            _send_worker_event(
                coordinator_socket,
                message_identity,
                'episode_end',
                job.episode_id,
                payload=encode_frame(to_jsonable(result)),
            )
    except BaseException as exc:
        try:
            coordinator_socket.send_multipart([
                encode_header(
                    'worker_error',
                    **message_identity,
                    message=str(exc),
                ),
                empty_payload(),
            ])
            time.sleep(0.1)
        except Exception:
            pass
        raise
    finally:
        for component in (environment, policy_client):
            try:
                component.close()
            except Exception:
                pass
        coordinator_socket.close(linger=0)
        context.term()


def load_distributed_evaluation_config(
        config_path: str,
        config_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load evaluation config and inject its canonical ZMQ settings."""
    path = Path(config_path).expanduser().resolve(strict=True)
    cfg = Config.fromfile(str(path), import_custom_modules=False)
    if config_overrides:
        cfg.merge_from_dict(dict(config_overrides))
    if 'themis' not in cfg or not isinstance(cfg.themis, Mapping):
        raise KeyError('FluxVLA config must define a `themis` mapping')
    custom_imports = cfg.themis.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    if ('evaluation' not in cfg.themis
            or not isinstance(cfg.themis.evaluation, Mapping)):
        raise KeyError('FluxVLA config must define `themis.evaluation`')
    evaluation_cfg = copy.deepcopy(dict(cfg.themis.evaluation))
    transport = cfg.themis.get('transport')
    if not isinstance(transport, Mapping):
        raise KeyError('FluxVLA config must define `themis.transport`')
    policy_client_cfg = evaluation_cfg.get('policy_client')
    if not isinstance(policy_client_cfg, Mapping):
        raise TypeError('themis.evaluation.policy_client must be a mapping')
    if policy_client_cfg.get('type') != 'FluxVLAPolicyClient':
        raise ValueError('distributed evaluation requires FluxVLAPolicyClient')
    policy_client_cfg = copy.deepcopy(dict(policy_client_cfg))
    policy_client_cfg['transport'] = copy.deepcopy(dict(transport))
    evaluation_cfg['policy_client'] = policy_client_cfg

    reporter_cfg = evaluation_cfg.get('reporter')
    if not isinstance(reporter_cfg, Mapping):
        raise TypeError('themis.evaluation.reporter must be a mapping')
    if reporter_cfg.get('type') != 'FluxVLAEvaluationReporter':
        raise ValueError('distributed evaluation requires '
                         'FluxVLAEvaluationReporter')
    reporter_cfg = copy.deepcopy(dict(reporter_cfg))
    reporter_cfg['transport'] = connection_transport(transport)
    evaluation_cfg['reporter'] = reporter_cfg
    root = _find_fluxvla_root(path)
    evaluation_cfg['work_dir'] = str(
        _resolve_work_dir(root, evaluation_cfg.get('work_dir')))
    return evaluation_cfg


def _distributed_video_settings(
        evaluation_cfg: Mapping[str, Any],
        deployment_id: str) -> tuple[bool, str | None, int]:
    save_videos = evaluation_cfg.get('save_videos', False)
    if not isinstance(save_videos, bool):
        raise TypeError('themis.evaluation.save_videos must be a boolean')
    video_fps = _positive_int(
        evaluation_cfg.get('video_fps', 20), 'themis.evaluation.video_fps')
    if not save_videos:
        return False, None, video_fps
    value = evaluation_cfg.get('video_dir')
    if value is None:
        value = Path(evaluation_cfg['work_dir']) / deployment_id / 'videos'
    elif not isinstance(value, (str, os.PathLike)):
        raise TypeError('themis.evaluation.video_dir must be a path or None')
    elif not str(value).strip():
        raise ValueError('themis.evaluation.video_dir cannot be empty')
    return True, str(Path(value).expanduser().resolve()), video_fps


def _find_fluxvla_root(config_path: Path) -> Path:
    for candidate in config_path.parent.parents:
        package_init = candidate / 'fluxvla' / '__init__.py'
        configs_root = candidate / 'configs'
        if not package_init.is_file() or not configs_root.is_dir():
            continue
        try:
            config_path.relative_to(configs_root.resolve())
        except ValueError:
            continue
        return candidate.resolve()
    raise ValueError(
        f'Config must be located under a FluxVLA `configs` directory: '
        f'{config_path}')


def _resolve_work_dir(fluxvla_root: Path,
                      value: str | os.PathLike | None) -> Path:
    work_dirs_root = (fluxvla_root / 'work_dirs').resolve()
    if value is None:
        candidate = work_dirs_root / DEFAULT_ARTIFACT_SUBDIR
    else:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError('themis.evaluation.work_dir must be a path')
        if not str(value).strip():
            raise ValueError('themis.evaluation.work_dir cannot be empty')
        requested = Path(value).expanduser()
        if requested.is_absolute():
            candidate = requested
        elif requested.parts and requested.parts[0] == 'work_dirs':
            candidate = fluxvla_root / requested
        else:
            candidate = work_dirs_root / requested

    resolved = candidate.resolve()
    try:
        resolved.relative_to(work_dirs_root)
    except ValueError as exc:
        raise ValueError(
            f'FluxThemis artifacts must stay inside FluxVLA work_dirs: '
            f'{work_dirs_root}') from exc
    return resolved


def build_distributed_evaluation_run(
        config_path: str,
        config_overrides: Mapping[str, Any] | None = None) -> EvaluationRun:
    evaluation_cfg = load_distributed_evaluation_config(
        config_path, config_overrides=config_overrides)
    settings_cfg = copy.deepcopy(evaluation_cfg)
    for key in _WORKER_ONLY_EVALUATION_KEYS:
        settings_cfg.pop(key, None)
    reporter = build_evaluation_reporter_from_cfg(settings_cfg.pop('reporter'))
    evaluator = build_evaluator_from_cfg(settings_cfg.pop('evaluator'))
    return EvaluationRun(
        settings=EvaluationSettings(**settings_cfg),
        evaluator=evaluator,
        reporter=reporter,
    )


def run_distributed_evaluation_rank(*,
                                    config_path: str,
                                    device: str,
                                    rollout_bind: str,
                                    rollout_endpoint: str,
                                    rank: int,
                                    world_size: int,
                                    config_overrides: Mapping[str, Any]
                                    | None = None,
                                    startup_timeout_s: float = 900.0,
                                    episode_timeout_s: float = 3600.0,
                                    deployment_id: str | None = None) -> None:
    """Run one torchrun simulator worker and the rank-zero coordinator."""
    if rank < 0 or rank >= world_size:
        raise ValueError('rank must be in [0, world_size)')
    deployment_id = (
        deployment_id or os.environ.get('FLUXTHEMIS_DEPLOYMENT_ID')
        or os.environ.get('TORCHELASTIC_RUN_ID'))
    if not deployment_id:
        stable = (f'{os.environ.get("MASTER_ADDR", "127.0.0.1")}:'
                  f'{os.environ.get("MASTER_PORT", "29500")}:'
                  f'{Path(config_path).resolve()}')
        import hashlib
        deployment_id = hashlib.sha256(stable.encode()).hexdigest()[:32]
    coordinator_process = None
    coordinator_monitor = None
    if rank == 0:
        context = multiprocessing.get_context('spawn')
        coordinator_process = context.Process(
            target=_run_coordinator_process,
            kwargs={
                'config_path': config_path,
                'config_overrides': dict(config_overrides or {}),
                'rollout_bind': rollout_bind,
                'expected_workers': world_size,
                'startup_timeout_s': startup_timeout_s,
                'episode_timeout_s': episode_timeout_s,
                'deployment_id': deployment_id,
                'parent_pid': os.getpid(),
            },
            name='fluxthemis-coordinator',
        )
        coordinator_process.start()
        coordinator_monitor = threading.Thread(
            target=_monitor_coordinator_process,
            args=(coordinator_process, os.getpid()),
            daemon=True,
        )
        coordinator_monitor.start()

    try:
        run_simulator_worker(
            config_path=config_path,
            rollout_endpoint=rollout_endpoint,
            worker_id=f'simulator-worker-{rank}',
            deployment_id=deployment_id,
            device=device,
            config_overrides=config_overrides,
        )
    except BaseException:
        if coordinator_process is not None:
            if coordinator_process.is_alive():
                coordinator_process.terminate()
            coordinator_monitor.join()
        raise
    else:
        if coordinator_monitor is not None:
            coordinator_monitor.join()


def _monitor_coordinator_process(process: multiprocessing.Process,
                                 parent_pid: int) -> None:
    process.join()
    if process.exitcode not in {0, -signal.SIGTERM}:
        print(
            f'[FluxThemis] coordinator exited with code {process.exitcode}',
            file=sys.stderr,
            flush=True,
        )
        os.kill(parent_pid, signal.SIGTERM)


def _run_coordinator_process(*, config_path: str,
                             config_overrides: Mapping[str,
                                                       Any], rollout_bind: str,
                             expected_workers: int, startup_timeout_s: float,
                             episode_timeout_s: float, deployment_id: str,
                             parent_pid: int) -> None:
    configure_worker_lifetime(parent_pid)
    evaluation_run = build_distributed_evaluation_run(
        config_path=config_path,
        config_overrides=config_overrides,
    )
    coordinator = DistributedEvaluationCoordinator(
        rollout_bind=rollout_bind,
        expected_workers=expected_workers,
        startup_timeout_s=startup_timeout_s,
        episode_timeout_s=episode_timeout_s,
        deployment_id=deployment_id,
    )
    with evaluation_run:
        summary = coordinator.run(evaluation_run)
    print(json.dumps(summary['metrics'], indent=2), flush=True)
    print(
        f'Evaluation reported to {evaluation_run.report_run_dir}',
        flush=True,
    )


def _send_worker_event(coordinator_socket: Any,
                       message_identity: Mapping[str, str],
                       event_type: str,
                       episode_id: str,
                       payload: bytes | None = None,
                       **values: Any) -> None:
    coordinator_socket.send_multipart([
        encode_header(
            event_type,
            **message_identity,
            episode_id=episode_id,
            **values,
        ),
        payload if payload is not None else empty_payload(),
    ])
    frames = coordinator_socket.recv_multipart()
    header = decode_header(frames[0])
    if header['type'] == 'shutdown_worker':
        raise KeyboardInterrupt
    expected_type = f'{event_type}_ack'
    if (header['type'] != expected_type
            or header.get('episode_id') != episode_id
            or header.get('deployment_id') != message_identity['deployment_id']
            or header.get('worker_id') != message_identity['worker_id']):
        raise RuntimeError(
            f'expected {expected_type} for episode {episode_id!r}, got '
            f'{header.get("type")!r} for {header.get("episode_id")!r}')


def _episode_job_from_wire(value: Any) -> EpisodeJob:
    if not isinstance(value, Mapping):
        raise TypeError('episode job payload must be a mapping')
    data = dict(value)
    data['task'] = TaskSpec.from_value(data['task'])
    return EpisodeJob(**data)


def _episode_result_from_wire(value: Any) -> EpisodeResult:
    if not isinstance(value, Mapping):
        raise TypeError('episode result payload must be a mapping')
    return EpisodeResult(**dict(value))


def _endpoint(value: Any, name: str) -> str:
    value = _nonempty(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


def _required_string(header: Mapping[str, Any], name: str) -> str:
    return _nonempty(header.get(name), f'header.{name}')


__all__ = [
    'DistributedEvaluationCoordinator',
    'build_distributed_evaluation_run',
    'load_distributed_evaluation_config',
    'run_distributed_evaluation_rank',
    'run_simulator_worker',
]
