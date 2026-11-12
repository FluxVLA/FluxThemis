"""Coordinator and scheduling state for distributed evaluation."""

from __future__ import annotations

import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from fluxthemis.engines.utils import to_jsonable
from fluxthemis.structures import EpisodeResult, TaskSpec

from ...transports.zmq_protocol import (decode_frame, decode_header,
                                        empty_payload, encode_frame,
                                        encode_header)
from .assignment import EpisodeAssignment, ExecutionPolicy
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
    capacity: int = 1
    assignment: EpisodeAssignment | None = None
    completed_episode_ids: set[str] = field(default_factory=set)
    episode_deadline: float | None = None
    recovery_deadline: float | None = None
    simulator_ready: bool = False
    restart_pending: bool = False
    startup_restart_count: int = 0

    @property
    def is_ready(self) -> bool:
        return (self.simulator_ready and not self.restart_pending
                and bool(self.task_manifest))

    @property
    def is_idle(self) -> bool:
        return self.is_ready and self.assignment is None


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
    rejection_counts_by_episode_id: dict[str,
                                         int] = field(default_factory=dict)
    restored_started_episode_ids: set[str] = field(default_factory=set)
    replaced_cases_by_episode_id: dict[str, int] = field(default_factory=dict)

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
        self._strict_worker_assignment = False
        self._execution_policy = ExecutionPolicy()

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
        self._strict_worker_assignment = settings.strict_worker_assignment
        self._execution_policy = ExecutionPolicy(settings.episode_batch_size)
        for slot in self._worker_slots.values():
            if slot.capacity != self._execution_policy.batch_size:
                raise ValueError(
                    "Worker capacity differs from episode_batch_size")
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
            evaluation_run.start(selected_tasks, len(jobs),
                                 self.expected_workers)
            # Rebuild the new report session before dispatching, so saved
            # results are counted without rerunning completed episodes.
            self._restore_jobs(evaluation_run)
            self._replay_results(evaluation_run)
            self._dispatch_jobs()
            while (not scheduling_state.is_complete
                   or any(slot.assignment is not None
                          or slot.recovery_deadline is not None
                          for slot in self._worker_slots.values())):
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

    def _restore_jobs(self, evaluation_run: EvaluationRun) -> None:
        data = evaluation_run.resume_data
        if not data:
            return
        state = self._require_scheduling_state()
        saved = {j['global_episode_number']: j for j in data['jobs']}
        # Keep the new session's job IDs and canonical seeds. Only layout
        # choices come from recovery, including replacements of rejected cases.
        for key, job in list(state.jobs_by_episode_id.items()):
            if saved[job.global_episode_number].get('started', False):
                state.restored_started_episode_ids.add(key)
            state.jobs_by_episode_id[key] = replace(
                job,
                environment_case_index=saved[job.global_episode_number]
                ['environment_case_index'])
        state.pending_jobs = deque(state.jobs_by_episode_id.values())
        state.next_environment_case_index_by_task = dict(
            data['next_environment_case_index_by_task'])

    def _replay_results(self, evaluation_run: EvaluationRun) -> None:
        data = evaluation_run.resume_data
        if not data:
            return
        result_fields = {f.name for f in fields(EpisodeResult)}
        state = self._require_scheduling_state()
        # Session IDs change on resume; task ID and episode index do not.
        by_key = {
            (j.task.task_id, j.episode_index): j
            for j in state.jobs_by_episode_id.values()
        }
        for pair in data['completed']:
            end = pair['end']
            job = by_key[(end['task_id'], end['episode_index'])]
            result = EpisodeResult(**{
                k: v
                for k, v in end.items() if k in result_fields
            })
            job.validate_result(result)
            payload = job.episode_start_payload(pair['start']['started_at'])
            payload['resumed_from'] = data['source']
            # Replay report events with the original timestamps and results;
            # no simulator reset or policy inference is needed for these jobs.
            evaluation_run.on_episode_start(payload)
            evaluation_run.on_episode_end(
                result,
                job.episode_id,
                finished_at=end['finished_at'],
                resumed_from=data['source'])
            state.results_by_episode_id[job.episode_id] = result
        state.pending_jobs = deque(
            j for j in state.pending_jobs
            if j.episode_id not in state.results_by_episode_id)
        print(
            f'[FluxThemis] restored {len(state.results_by_episode_id)}'
            f'/{len(state.jobs)} '
            f'episodes; remaining={len(state.pending_jobs)}',
            flush=True)

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
        if message_type == 'worker_exit':
            self._handle_worker_exit(routing_id, header)
            return
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
        assignment_id = _required_string(header, 'assignment_id')
        if (slot.assignment is None
                or assignment_id != slot.assignment.assignment_id):
            # A delayed message cannot complete or release a newer attempt,
            # including retries of exactly the same episode in this process.
            return
        if message_type == 'assignment_end':
            assignment = slot.assignment
            expected = {job.episode_id for job in assignment.jobs}
            if slot.completed_episode_ids != expected:
                raise RuntimeError(
                    'Worker ended assignment before all results')
            self._release_worker_slot(slot)
        elif message_type == 'episode_start':
            if header.get('episode_id') in slot.completed_episode_ids:
                return
            job = self._require_assigned_job(slot, header)
            if job.episode_id not in scheduling_state.started_episode_ids:
                episode_start_payload = job.episode_start_payload(utc_now())
                replacements = scheduling_state.replaced_cases_by_episode_id
                previous_case = replacements.get(job.episode_id)
                if previous_case is not None:
                    episode_start_payload[
                        'replaces_environment_case_index'] = previous_case
                evaluation_run.on_episode_start(episode_start_payload)
                scheduling_state.started_episode_ids.add(job.episode_id)
                replacements.pop(job.episode_id, None)
        elif message_type == 'episode_end':
            job = self._require_assigned_job(
                slot, header, allow_completed=True)
            result = _episode_result_from_wire(decode_frame(payload))
            job.validate_result(result)
            episode_id = _required_string(header, 'episode_id')
            saved = scheduling_state.results_by_episode_id.get(episode_id)
            if saved is not None:
                if to_jsonable(saved) != to_jsonable(result):
                    raise RuntimeError(
                        f'Conflicting duplicate episode result {episode_id!r}')
                return
            if job.episode_id not in scheduling_state.started_episode_ids:
                raise RuntimeError(
                    'Episode result received before episode start')
            # Report acknowledgement must precede completion and slot reuse;
            # a failed report must not leave an apparently finished job.
            evaluation_run.on_episode_end(result, job.episode_id)
            scheduling_state.results_by_episode_id[job.episode_id] = result
            slot.completed_episode_ids.add(job.episode_id)
        elif message_type == 'environment_case_rejected':
            if header.get('episode_id') in slot.completed_episode_ids:
                return
            job = self._require_assigned_job(slot, header)
            was_started = (
                job.episode_id in scheduling_state.started_episode_ids
                or job.episode_id
                in scheduling_state.restored_started_episode_ids)
            replace_case = header.get('replace_case', True)
            if type(replace_case) is not bool:
                raise ValueError('replace_case must be a boolean')
            invalidated = header.get('invalidated', False)
            if type(invalidated) is not bool:
                raise ValueError('invalidated must be a boolean')
            if invalidated and (not was_started or not replace_case):
                raise ValueError(
                    'Invalid case invalidation before episode start')
            retry = job
            if was_started and replace_case and not invalidated:
                # A previously acknowledged start fixes the initial condition.
                # A reset failure during recovery retries it within the same
                # infrastructure budget, without choosing a different layout.
                self._requeue_restarted_job(job)
            elif replace_case:
                count = scheduling_state.rejection_counts_by_episode_id.get(
                    job.episode_id, 0) + 1
                settings = evaluation_run.settings
                limit = settings.environment_case_rejection_limit
                if count > limit:
                    raise RuntimeError(
                        f'Episode {job.episode_id!r} exceeded environment '
                        f'case rejection limit {limit}')
                scheduling_state.rejection_counts_by_episode_id[
                    job.episode_id] = count
                next_case_index = (
                    scheduling_state.next_environment_case_index_by_task.get(
                        job.task.task_id))
                if next_case_index is None:
                    raise RuntimeError(f'environment rejected a case for '
                                       f'job {job.episode_id!r} '
                                       'without an environment_case_index')
                scheduling_state.next_environment_case_index_by_task[
                    job.task.task_id] = next_case_index + 1
                retry = replace(job, environment_case_index=next_case_index)
                if invalidated:
                    scheduling_state.replaced_cases_by_episode_id[
                        job.episode_id] = job.environment_case_index
                    scheduling_state.started_episode_ids.discard(
                        job.episode_id)
                    scheduling_state.restored_started_episode_ids.discard(
                        job.episode_id)
                print(
                    f'[FluxThemis] simulator worker {slot.worker_id} '
                    'rejected '
                    f'environment case {job.environment_case_index} for '
                    f'{job.task.task_id}; retrying with case '
                    f'{next_case_index}',
                    file=sys.stderr,
                    flush=True,
                )
            scheduling_state.jobs_by_episode_id[job.episode_id] = retry
            if invalidated or not (was_started and replace_case):
                scheduling_state.pending_jobs.append(retry)
            slot.completed_episode_ids.add(job.episode_id)
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
                if slot.restart_pending:
                    return
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

    def _handle_worker_exit(self, routing_id: bytes,
                            header: Mapping[str, Any]) -> None:
        """Decide recovery before a supervisor starts a replacement process."""
        worker_id = _required_string(header, 'worker_id')
        generation = _required_nonnegative_int(header, 'generation')
        slot = self._worker_slots.get(worker_id)
        if slot is None:
            # A child can fail before its first registration reaches us.
            self._register_worker(routing_id, header)
            slot = self._worker_slots[worker_id]
        if generation < slot.generation:
            return
        if (generation != slot.generation or slot.restart_pending
                or header.get('device') != slot.device):
            raise RuntimeError('Invalid simulator exit notification')
        # Route the eventual shutdown to the waiting supervisor, including
        # when recovery is rejected. run.fail() persists results first.
        slot.routing_id = routing_id
        slot.restart_pending = True
        self._recover_worker(slot)
        slot.assignment = None
        slot.completed_episode_ids.clear()
        slot.episode_deadline = None
        slot.simulator_ready = False
        slot.recovery_deadline = time.monotonic() + self.startup_timeout_s
        self._idle_worker_ids = deque(value for value in self._idle_worker_ids
                                      if value != worker_id)
        self._send_worker_message('restart_worker', slot)

    def _recover_worker(self, slot: _WorkerSlot) -> None:
        assignment = slot.assignment
        if assignment is not None:
            for job in reversed(assignment.jobs):
                if job.episode_id not in slot.completed_episode_ids:
                    self._requeue_restarted_job(job)
        elif not slot.simulator_ready:
            slot.startup_restart_count += 1
            if slot.startup_restart_count > self.simulator_restart_limit:
                raise RuntimeError(
                    f'simulator worker {slot.worker_id!r} exceeded startup '
                    f'restart limit {self.simulator_restart_limit}')

    def _replace_worker_connection(self, slot: _WorkerSlot, *,
                                   routing_id: bytes, generation: int) -> None:
        assignment = slot.assignment
        if not slot.restart_pending:
            self._recover_worker(slot)
        slot.restart_pending = False
        self._idle_worker_ids = deque(worker_id
                                      for worker_id in self._idle_worker_ids
                                      if worker_id != slot.worker_id)
        slot.routing_id = routing_id
        slot.generation = generation
        slot.assignment = None
        slot.completed_episode_ids.clear()
        slot.episode_deadline = None
        slot.recovery_deadline = time.monotonic() + self.startup_timeout_s
        slot.simulator_ready = False
        assignment_label = (f' assignment={assignment.assignment_id}'
                            if assignment else '')
        print(
            f'[FluxThemis] simulator worker {slot.worker_id} reconnecting'
            f'{assignment_label}: '
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
        capacity = _positive_int(body.get('capacity', 1), 'worker capacity')
        if slot.task_manifest and slot.capacity != capacity:
            raise RuntimeError(
                'Simulator worker changed capacity after restart')
        slot.capacity = capacity
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
        if self._strict_worker_assignment:
            # Use the stable task-major ordinal, not completion order, so
            # retries stay on the same worker for a given worker count. Scan
            # past busy workers to let other fixed assignments proceed.
            for job in tuple(scheduling_state.pending_jobs):
                index = (job.global_episode_number - 1) % self.expected_workers
                slot = self._worker_slots.get(f'simulator-worker-{index}')
                if slot is None or not slot.is_idle:
                    continue
                scheduling_state.pending_jobs.remove(job)
                self._idle_worker_ids.remove(slot.worker_id)
                self._assign_jobs(slot, (job, ))
            return
        # Default scheduling still gives the next job to any idle worker.
        while scheduling_state.pending_jobs:
            slot = self._next_idle_worker_slot()
            if slot is None:
                return
            first = scheduling_state.pending_jobs.popleft()
            jobs = [first]
            retries = scheduling_state.restart_counts_by_episode_id
            while (not retries.get(first.episode_id)
                   and len(jobs) < slot.capacity
                   and scheduling_state.pending_jobs
                   and scheduling_state.pending_jobs[0].task == first.task
                   and not retries.get(
                       scheduling_state.pending_jobs[0].episode_id)):
                jobs.append(scheduling_state.pending_jobs.popleft())
            self._assign_jobs(slot, tuple(jobs))

    def _assign_jobs(self, slot: _WorkerSlot, jobs: tuple[EpisodeJob,
                                                          ...]) -> None:
        assignment = EpisodeAssignment(jobs)
        ExecutionPolicy(slot.capacity).validate_assignment(assignment)
        slot.assignment = assignment
        slot.completed_episode_ids.clear()
        slot.episode_deadline = self._new_episode_deadline()
        self._send_worker_message(
            'episode_assignment',
            slot,
            assignment_id=assignment.assignment_id,
            payload=encode_frame({
                'assignment_id': assignment.assignment_id,
                'jobs': [to_jsonable(job) for job in jobs]
            }))

    def _next_idle_worker_slot(self) -> _WorkerSlot | None:
        while self._idle_worker_ids:
            worker_id = self._idle_worker_ids.popleft()
            slot = self._worker_slots.get(worker_id)
            if slot is not None and slot.is_idle:
                return slot
        return None

    def _release_worker_slot(self, slot: _WorkerSlot) -> None:
        slot.assignment = None
        slot.completed_episode_ids.clear()
        slot.episode_deadline = None
        self._idle_worker_ids.append(slot.worker_id)

    def _new_episode_deadline(self) -> float:
        return (time.monotonic() + self.episode_timeout_s +
                _SIMULATOR_RESTART_GRACE_S)

    def _require_assigned_job(self,
                              slot: _WorkerSlot,
                              header: Mapping[str, Any],
                              *,
                              allow_completed: bool = False) -> EpisodeJob:
        episode_id = _required_string(header, 'episode_id')
        job = next(
            (job
             for job in slot.assignment.jobs if job.episode_id == episode_id),
            None) if slot.assignment else None
        if (job is None or (not allow_completed
                            and episode_id in slot.completed_episode_ids)):
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
        if generation < slot.generation or slot.restart_pending:
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
            job.episode_start_payload(utc_now())
            for slot in self._worker_slots.values()
            if slot.assignment is not None for job in slot.assignment.jobs
            if job.episode_id not in slot.completed_episode_ids
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
