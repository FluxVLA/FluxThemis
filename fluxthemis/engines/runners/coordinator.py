"""Coordinator and scheduling state for distributed evaluation."""

from __future__ import annotations

import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from fluxthemis.engines.utils import to_jsonable
from fluxthemis.structures import EpisodeResult, TaskSpec

from ...transports.zmq_protocol import (decode_frame, decode_header,
                                        empty_payload, encode_frame,
                                        encode_header)
from .episode import EpisodeJob, build_episode_jobs, select_tasks
from .evaluation import EvaluationRun, EvaluationStatus, utc_now

_SIMULATOR_RESTART_GRACE_S = 30.0


@dataclass
class _WorkerSlot:
    worker_id: str
    routing_id: bytes
    generation: int
    device: str
    task_manifest: tuple[TaskSpec, ...] = ()
    current_job: EpisodeJob | None = None
    episode_deadline: float | None = None
    recovery_deadline: float | None = None
    simulator_ready: bool = False
    startup_restart_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.simulator_ready and bool(self.task_manifest)

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
    restart_counts_by_episode_id: dict[str, int]

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
            restart_counts_by_episode_id={},
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
                 simulator_restart_limit: int = 3,
                 deployment_id: str | None = None) -> None:
        self.rollout_bind = _endpoint(rollout_bind, 'rollout_bind')
        self.expected_workers = _positive_int(expected_workers,
                                              'expected_workers')
        self.startup_timeout_s = _positive_float(startup_timeout_s,
                                                 'startup_timeout_s')
        self.episode_timeout_s = _positive_float(episode_timeout_s,
                                                 'episode_timeout_s')
        self.simulator_restart_limit = _nonnegative_int(
            simulator_restart_limit, 'simulator_restart_limit')
        self.deployment_id = deployment_id or uuid.uuid4().hex
        self._worker_slots: dict[str, _WorkerSlot] = {}
        self._idle_worker_ids: deque[str] = deque()
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
            slot.task_manifest for slot in self._worker_slots.values()
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
                    slot for slot in self._worker_slots.values()
                    if slot.episode_deadline is not None
                    and slot.episode_deadline <= now
                ]
                if expired_slots:
                    worker_ids = ', '.join(slot.worker_id
                                           for slot in expired_slots)
                    raise TimeoutError(
                        f'simulator episode timed out on {worker_ids}')
                stalled_slots = [
                    slot for slot in self._worker_slots.values()
                    if slot.recovery_deadline is not None
                    and slot.recovery_deadline <= now
                ]
                if stalled_slots:
                    worker_ids = ', '.join(slot.worker_id
                                           for slot in stalled_slots)
                    raise TimeoutError(
                        'replacement simulator processes did not become '
                        f'ready on {worker_ids}')
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
        slot = self._worker_slot_for_message(routing_id, header)
        if slot is None:
            return
        if message_type == 'ready':
            self._mark_worker_ready(slot, payload)
            return
        scheduling_state = self._require_scheduling_state()
        if message_type == 'episode_start':
            job = self._require_assigned_job(slot, header)
            slot.episode_deadline = self._new_episode_deadline()
            if job.episode_id not in scheduling_state.started_episode_ids:
                episode_start_payload = job.episode_start_payload(utc_now())
                evaluation_run.on_episode_start(episode_start_payload)
                scheduling_state.started_episode_ids.add(job.episode_id)
        elif message_type == 'episode_end':
            job = self._require_assigned_job(slot, header)
            result = _episode_result_from_wire(decode_frame(payload))
            job.validate_result(result)
            if job.episode_id in scheduling_state.results_by_episode_id:
                raise RuntimeError(
                    f'duplicate episode result {job.episode_id!r}')
            scheduling_state.results_by_episode_id[job.episode_id] = result
            evaluation_run.on_episode_end(result, job.episode_id)
            self._release_worker_slot(slot)
        elif message_type == 'environment_case_rejected':
            job = self._require_assigned_job(slot, header)
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
            scheduling_state.started_episode_ids.discard(job.episode_id)
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
        generation = _required_nonnegative_int(header, 'generation')
        device = _required_string(header, 'device')
        slot = self._worker_slots.get(worker_id)
        if slot is not None:
            if generation < slot.generation:
                return
            if generation == slot.generation:
                raise RuntimeError(
                    f'duplicate simulator worker generation {worker_id!r} '
                    f'generation={generation}')
            if device != slot.device:
                raise RuntimeError(
                    f'simulator worker {worker_id!r} changed device from '
                    f'{slot.device!r} to {device!r}')
            self._replace_worker_connection(
                slot,
                routing_id=routing_id,
                generation=generation,
            )
            return
        if len(self._worker_slots) >= self.expected_workers:
            raise RuntimeError('received more simulator workers than expected')
        self._worker_slots[worker_id] = _WorkerSlot(
            worker_id=worker_id,
            routing_id=routing_id,
            generation=generation,
            device=device,
        )

    def _replace_worker_connection(self, slot: _WorkerSlot, *,
                                   routing_id: bytes, generation: int) -> None:
        job = slot.current_job
        if job is not None:
            self._requeue_restarted_job(job)
        elif not slot.simulator_ready:
            slot.startup_restart_count += 1
            if slot.startup_restart_count > self.simulator_restart_limit:
                raise RuntimeError(
                    f'simulator worker {slot.worker_id!r} exceeded startup '
                    f'restart limit {self.simulator_restart_limit}')
        self._idle_worker_ids = deque(worker_id
                                      for worker_id in self._idle_worker_ids
                                      if worker_id != slot.worker_id)
        slot.routing_id = routing_id
        slot.generation = generation
        slot.current_job = None
        slot.episode_deadline = None
        slot.recovery_deadline = time.monotonic() + self.startup_timeout_s
        slot.simulator_ready = False
        print(
            f'[FluxThemis] simulator worker {slot.worker_id} reconnecting'
            f'{f" episode={job.episode_id}" if job else ""}: '
            f'generation={generation}',
            file=sys.stderr,
            flush=True,
        )

    def _requeue_restarted_job(self, job: EpisodeJob) -> None:
        scheduling_state = self._require_scheduling_state()
        restart_count = scheduling_state.restart_counts_by_episode_id.get(
            job.episode_id, 0) + 1
        if restart_count > self.simulator_restart_limit:
            raise RuntimeError(
                f'episode {job.episode_id!r} exceeded simulator restart '
                f'limit {self.simulator_restart_limit}')
        scheduling_state.restart_counts_by_episode_id[
            job.episode_id] = restart_count
        scheduling_state.pending_jobs.appendleft(job)

    def _mark_worker_ready(self, slot: _WorkerSlot, payload: bytes) -> None:
        if slot.is_ready:
            raise RuntimeError(
                f'simulator worker {slot.worker_id!r} is already ready')
        body = decode_frame(payload)
        if not isinstance(body, Mapping):
            raise TypeError('simulator ready payload must be a mapping')
        task_manifest = tuple(
            TaskSpec.from_value(value) for value in body.get('tasks', ()))
        if not task_manifest:
            raise ValueError('simulator task manifest cannot be empty')
        if slot.task_manifest and task_manifest != slot.task_manifest:
            raise RuntimeError(
                f'simulator worker {slot.worker_id!r} changed its task '
                'manifest after restart')
        slot.task_manifest = task_manifest
        slot.recovery_deadline = None
        slot.simulator_ready = True
        slot.startup_restart_count = 0
        self._idle_worker_ids.append(slot.worker_id)
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
            slot.episode_deadline = self._new_episode_deadline()
            self._send_worker_message(
                'episode_job',
                slot,
                episode_id=job.episode_id,
                payload=encode_frame(to_jsonable(job)),
            )

    def _next_idle_worker_slot(self) -> _WorkerSlot | None:
        while self._idle_worker_ids:
            worker_id = self._idle_worker_ids.popleft()
            slot = self._worker_slots.get(worker_id)
            if slot is not None and slot.is_idle:
                return slot
        return None

    def _release_worker_slot(self, slot: _WorkerSlot) -> None:
        slot.current_job = None
        slot.episode_deadline = None
        self._idle_worker_ids.append(slot.worker_id)

    def _new_episode_deadline(self) -> float:
        return (time.monotonic() + self.episode_timeout_s +
                _SIMULATOR_RESTART_GRACE_S)

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

    def _worker_slot_for_message(
            self, routing_id: bytes,
            header: Mapping[str, Any]) -> _WorkerSlot | None:
        worker_id = _required_string(header, 'worker_id')
        generation = _required_nonnegative_int(header, 'generation')
        slot = self._worker_slots.get(worker_id)
        if slot is None:
            raise RuntimeError(
                'simulator worker must register before sending data')
        if generation < slot.generation:
            return None
        if generation != slot.generation or routing_id != slot.routing_id:
            raise RuntimeError('simulator worker connection mismatch')
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
                generation=slot.generation,
                **values,
            ),
            payload if payload is not None else empty_payload(),
        ])

    def _ready_worker_count(self) -> int:
        return sum(slot.is_ready for slot in self._worker_slots.values())

    def _all_workers_ready(self) -> bool:
        return self._ready_worker_count() == self.expected_workers

    def _active_episode_progress(self) -> dict[str, Any] | None:
        jobs = [
            slot.current_job.episode_start_payload(utc_now())
            for slot in self._worker_slots.values()
            if slot.current_job is not None
        ]
        return {'active_episodes': jobs} if jobs else None

    def _shutdown_all_workers(self) -> None:
        self._stopping = True
        if self._worker_router is not None:
            for slot in self._worker_slots.values():
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


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


def _required_string(header: Mapping[str, Any], name: str) -> str:
    return _nonempty(header.get(name), f'header.{name}')


def _required_nonnegative_int(header: Mapping[str, Any], name: str) -> int:
    value = header.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'header.{name} must be a non-negative integer')
    return value


__all__ = ['DistributedEvaluationCoordinator']
