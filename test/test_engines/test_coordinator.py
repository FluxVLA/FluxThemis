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
        current_job=job,
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
    scheduling_state.started_episode_ids.add(job.episode_id)
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
    assert scheduling_state.started_episode_ids == {job.episode_id}
    assert scheduling_state.next_environment_case_index_by_task == {'task': 3}
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
    assert slot.current_job is None
    assert slot.routing_id == replacement_routing_id
    assert slot.generation == 1
    assert slot.recovery_deadline is not None
    assert coordinator._worker_router.sent == []
