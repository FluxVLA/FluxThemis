import pytest

from fluxthemis.engines.runners.assignment import EpisodeAssignment
from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState, _WorkerSlot)
from fluxthemis.engines.runners.episode import build_episode_jobs
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import empty_payload, encode_header


class _RouterSocket:

    def __init__(self, incoming):
        self.incoming = incoming
        self.sent = []

    def recv_multipart(self):
        return self.incoming

    def send_multipart(self, frames):
        self.sent.append(frames)


def _coordinator_with_active_job(job):
    evaluation_run = EvaluationRun(
        EvaluationSettings(run_name='test-run'),
        SuccessRateEvaluator(),
        MockEvaluationReporter(),
    )
    coordinator = DistributedEvaluationCoordinator(
        rollout_bind='tcp://127.0.0.1:15558',
        expected_workers=1,
        deployment_id='deployment',
    )
    routing_id = b'worker-0'
    slot = _WorkerSlot(
        worker_id='worker-0',
        routing_id=routing_id,
        generation=0,
        device='0',
        task_manifest=(job.task, ),
        assignment=EpisodeAssignment((job, )),
        simulator_ready=True,
    )
    coordinator._worker_slots[slot.worker_id] = slot
    coordinator._worker_router = _RouterSocket([
        routing_id,
        encode_header(
            'environment_case_rejected',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=slot.generation,
            episode_id=job.episode_id,
            assignment_id=slot.assignment.assignment_id,
        ),
        empty_payload(),
    ])
    return coordinator, slot, evaluation_run


def test_rejected_job_requeues_same_slot_with_next_environment_case():
    jobs = build_episode_jobs(
        [TaskSpec('task')],
        episodes_per_task=2,
        base_seed=7,
        run_session_id='run',
    )
    job = jobs[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    scheduling_state = _SchedulingState.from_jobs(jobs)
    scheduling_state.pending_jobs.clear()
    coordinator._scheduling_state = scheduling_state

    coordinator._receive_worker_message(evaluation_run)

    retry = scheduling_state.pending_jobs.popleft()
    assert retry.episode_id == job.episode_id
    assert retry.episode_index == job.episode_index
    assert retry.policy_seed == job.policy_seed
    assert retry.environment_seed == job.environment_seed
    assert retry.environment_case_index == 2
    assert scheduling_state.jobs_by_episode_id[job.episode_id] == retry
    assert scheduling_state.results_by_episode_id == {}
    assert scheduling_state.started_episode_ids == set()
    assert scheduling_state.next_environment_case_index_by_task == {'task': 3}
    assert not slot.is_idle
    coordinator._worker_router.incoming[1] = encode_header(
        "assignment_end",
        deployment_id="deployment",
        worker_id=slot.worker_id,
        generation=slot.generation,
        assignment_id=slot.assignment.assignment_id)
    coordinator._receive_worker_message(evaluation_run)
    assert slot.is_idle
    assert coordinator._worker_router.sent == []


def test_reconnecting_worker_requeues_active_episode():
    jobs = build_episode_jobs(
        [TaskSpec('task')],
        episodes_per_task=1,
        base_seed=7,
        run_session_id='run',
    )
    job = jobs[0]
    coordinator, slot, evaluation_run = _coordinator_with_active_job(job)
    scheduling_state = _SchedulingState.from_jobs(jobs)
    scheduling_state.pending_jobs.clear()
    scheduling_state.started_episode_ids.add(job.episode_id)
    coordinator._scheduling_state = scheduling_state
    replacement_routing_id = b'worker-0:1'
    coordinator._worker_router.incoming = [
        replacement_routing_id,
        encode_header(
            'register_worker',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=1,
            device='0',
        ),
        empty_payload(),
    ]

    coordinator._receive_worker_message(evaluation_run)

    assert scheduling_state.pending_jobs.popleft() == job
    assert scheduling_state.restart_counts_by_episode_id == {job.episode_id: 1}
    assert not slot.is_ready
    assert slot.assignment is None
    assert slot.routing_id == replacement_routing_id
    assert slot.generation == 1
    assert slot.recovery_deadline is not None
    assert coordinator._worker_router.sent == []


def test_case_replacement_has_a_per_episode_limit():
    jobs = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')
    coordinator, slot, _ = _coordinator_with_active_job(jobs[0])
    coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    coordinator._scheduling_state.pending_jobs.clear()
    run = EvaluationRun(
        EvaluationSettings(environment_case_rejection_limit=1),
        SuccessRateEvaluator(), MockEvaluationReporter())
    coordinator._receive_worker_message(run)
    coordinator._receive_worker_message(run)
    assert len(coordinator._scheduling_state.pending_jobs) == 1
    retry = coordinator._scheduling_state.pending_jobs.popleft()
    slot.assignment = EpisodeAssignment((retry, ))
    slot.completed_episode_ids.clear()
    coordinator._worker_router.incoming[1] = encode_header(
        'environment_case_rejected',
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=0,
        episode_id=retry.episode_id,
        assignment_id=slot.assignment.assignment_id)
    with pytest.raises(RuntimeError, match='rejection limit'):
        coordinator._receive_worker_message(run)
    assert (
        coordinator._scheduling_state.next_environment_case_index_by_task == {
            'task': 2
        })


def test_case_reset_failure_after_start_retries_same_layout():
    jobs = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')
    coordinator, slot, run = _coordinator_with_active_job(jobs[0])
    coordinator.simulator_restart_limit = 1
    state = coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    state.pending_jobs.clear()
    # Restored starts fix the layout without suppressing the new session's
    # first episode_start event.
    state.restored_started_episode_ids.add(jobs[0].episode_id)
    coordinator._receive_worker_message(run)
    assert state.pending_jobs.popleft() == jobs[0]
    assert state.next_environment_case_index_by_task == {'task': 1}
    assert not state.rejection_counts_by_episode_id
    assert state.restart_counts_by_episode_id == {jobs[0].episode_id: 1}
    slot.completed_episode_ids.clear()
    with pytest.raises(RuntimeError, match='restart limit'):
        coordinator._receive_worker_message(run)


def test_native_invalid_sample_can_replace_started_case_without_restart():
    jobs = build_episode_jobs([TaskSpec('task')], 2, 7, 'run')
    coordinator, slot, run = _coordinator_with_active_job(jobs[0])
    state = coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    state.pending_jobs.clear()
    state.started_episode_ids.add(jobs[0].episode_id)
    state.restored_started_episode_ids.add(jobs[0].episode_id)
    starts = []
    run.on_episode_start = starts.append
    coordinator._worker_router.incoming[1] = encode_header(
        'environment_case_rejected',
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=0,
        episode_id=jobs[0].episode_id,
        assignment_id=slot.assignment.assignment_id,
        replace_case=True,
        invalidated=True)
    coordinator._receive_worker_message(run)
    retry = state.pending_jobs[0]
    assert retry.environment_case_index == 2
    assert retry.policy_seed == jobs[0].policy_seed
    assert retry.episode_id == jobs[0].episode_id
    assert state.restart_counts_by_episode_id == {}
    assert state.rejection_counts_by_episode_id == {retry.episode_id: 1}
    assert not state.started_episode_ids
    assert not state.restored_started_episode_ids
    coordinator._worker_router.incoming[1] = encode_header(
        'assignment_end',
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=0,
        assignment_id=slot.assignment.assignment_id)
    coordinator._receive_worker_message(run)
    coordinator._dispatch_jobs()
    coordinator._worker_router.incoming[1] = encode_header(
        'episode_start',
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=0,
        assignment_id=slot.assignment.assignment_id,
        episode_id=retry.episode_id)
    coordinator._receive_worker_message(run)
    assert starts[0]['replaces_environment_case_index'] == 0
    assert starts[0]['environment_case_index'] == 2
