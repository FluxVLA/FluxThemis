from test.test_engines.helpers import RouterSocket, make_coordinator, make_jobs

import pytest

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState, _WorkerSlot)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.engines.utils import to_jsonable
from fluxthemis.environments import MockVectorEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import (empty_payload, encode_frame,
                                                encode_header)


@pytest.mark.parametrize('phase', ['reset', 'step'])
@pytest.mark.parametrize('capacity', [1, 3])
def test_rejection_replaces_layout_before_pending_jobs(phase, capacity,
                                                       capsys):
    jobs = make_jobs(capacity + 1)
    job = jobs[0]
    coordinator, slot = make_coordinator(capacity, jobs, active=True)
    coordinator.simulator_restart_limit = 0
    state = coordinator._scheduling_state
    state.started_episode_ids.add(job.episode_id)
    _receive_event(
        coordinator, slot, 'environment_case_rejected', job=job, phase=phase)
    coordinator._receive_worker_message(None)

    retry = state.pending_jobs[0]
    assert retry.episode_id == job.episode_id
    assert retry.episode_index == job.episode_index
    assert retry.policy_seed == job.policy_seed
    assert retry.environment_seed == job.environment_seed
    assert retry.environment_case_index == capacity + 1
    assert state.jobs_by_episode_id[job.episode_id] == retry
    assert state.next_environment_case_index_by_task == {'task': capacity + 2}
    assert list(state.pending_jobs) == [retry, jobs[-1]]
    assert slot.completed_episode_ids == {job.episode_id}
    assert not slot.is_idle
    assert state.started_episode_ids == set()
    assert state.restart_counts_by_episode_id == {}
    assert state.results_by_episode_id == {}
    assert f'during {phase}' in capsys.readouterr().err

    slot.completed_episode_ids.update(job.episode_id
                                      for job in slot.assignment.jobs)
    _receive_event(coordinator, slot, 'assignment_end')
    coordinator._receive_worker_message(None)
    assert slot.is_idle
    assert coordinator._worker_router.sent == []


@pytest.mark.parametrize('capacity', [1, 3])
def test_reconnecting_worker_requeues_only_unfinished_episodes(capacity):
    jobs = make_jobs(capacity)
    coordinator, slot = make_coordinator(capacity, jobs, active=True)
    state = coordinator._scheduling_state
    job = jobs[-1]
    state.started_episode_ids.add(job.episode_id)
    saved = {completed.episode_id: 'saved' for completed in jobs[:-1]}
    slot.completed_episode_ids.update(saved)
    state.results_by_episode_id.update(saved)
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

    coordinator._receive_worker_message(None)

    assert list(state.pending_jobs) == [job]
    assert state.restart_counts_by_episode_id == {job.episode_id: 1}
    assert state.results_by_episode_id == saved
    assert not slot.is_ready
    assert slot.assignment is None
    assert slot.routing_id == replacement_routing_id
    assert slot.generation == 1
    assert slot.recovery_deadline is not None
    assert coordinator._worker_router.sent == []


@pytest.mark.parametrize('phase', [None, 'unknown', True])
def test_invalid_rejection_phase_is_rejected(phase):
    jobs = build_episode_jobs([TaskSpec('task')], 1, 7, 'run')
    coordinator, slot = make_coordinator(1, jobs, active=True)
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header(
            'environment_case_rejected',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=0,
            episode_id=jobs[0].episode_id,
            assignment_id=slot.assignment.assignment_id,
            phase=phase),
        empty_payload()
    ]
    with pytest.raises(ValueError, match='phase'):
        coordinator._receive_worker_message(None)


def _receive_event(coordinator,
                   slot,
                   kind,
                   *,
                   job=None,
                   result=None,
                   assignment_id=None,
                   generation=None,
                   phase=None):
    values = dict(
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=slot.generation if generation is None else generation,
        assignment_id=assignment_id or slot.assignment.assignment_id)
    if job is not None:
        values['episode_id'] = job.episode_id
    if phase is not None:
        values['phase'] = phase
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header(kind, **values),
        empty_payload() if result is None else encode_frame(result)
    ]


@pytest.mark.parametrize('strict', [False, True])
def test_dispatch_groups_jobs_by_declared_task_limit(strict):
    tasks = [TaskSpec('stack_bowls_random'), TaskSpec('stack_bowls')]
    jobs = build_episode_jobs(tasks, 25, 7, 'run')
    coordinator, slot = make_coordinator(
        8,
        jobs,
        requires_single_task=True,
        task_capacities={'stack_bowls_random': 5})
    coordinator._strict_worker_assignment = strict
    batches = {task.task_id: [] for task in tasks}
    dispatched = []
    while coordinator._scheduling_state.pending_jobs:
        coordinator._dispatch_jobs()
        assignment = slot.assignment
        batches[assignment.jobs[0].task.task_id].append(len(assignment.jobs))
        dispatched.extend(assignment.jobs)
        coordinator._release_worker_slot(slot)
    assert batches == {
        'stack_bowls_random': [5] * 5,
        'stack_bowls': [8, 8, 8, 1]
    }
    assert dispatched == jobs
    assert slot.capacity == 8


@pytest.mark.parametrize('task_capacities', [{}, {
    'task': 4
}, {
    'task': 5,
    'another': 2
}])
def test_worker_restart_rejects_changed_task_capacities(task_capacities):
    coordinator, slot = make_coordinator(
        8, make_jobs(1), task_capacities={'task': 5})
    coordinator._replace_worker_connection(
        slot, routing_id=b'new', generation=1)
    with pytest.raises(
            ValueError, match='task capacities changed after restart'):
        coordinator._mark_worker_ready(
            slot,
            encode_frame({
                'tasks': [to_jsonable(task) for task in slot.task_manifest],
                'capacity':
                8,
                'task_capacities':
                task_capacities,
            }))
    assert not slot.is_ready


def test_retried_jobs_fill_worker_capacity():
    jobs = make_jobs(4)
    coordinator, slot = make_coordinator(3, jobs)
    coordinator._dispatch_jobs()
    coordinator._replace_worker_connection(
        slot, routing_id=b'replacement', generation=1)
    slot.simulator_ready = True
    slot.recovery_deadline = None
    coordinator._idle_worker_ids.append(slot.worker_id)
    coordinator._dispatch_jobs()
    assert slot.assignment.jobs == jobs[:3]
    assert list(coordinator._scheduling_state.pending_jobs) == jobs[3:]


def test_retry_budget_is_per_episode_not_worker_generation():
    jobs = make_jobs(4)
    coordinator, slot = make_coordinator(1, jobs)
    coordinator.simulator_restart_limit = 1
    for index, job in enumerate(jobs):
        slot.assignment = EpisodeAssignment((job, ))
        slot.simulator_ready = True
        coordinator._replace_worker_connection(
            slot,
            routing_id=f'worker:{index + 1}'.encode(),
            generation=index + 1)
    assert list(coordinator._scheduling_state.restart_counts_by_episode_id.
                values()) == [1] * 4
    slot.assignment = EpisodeAssignment((jobs[0], ))
    with pytest.raises(RuntimeError, match='restart limit'):
        coordinator._replace_worker_connection(
            slot, routing_id=b'new', generation=5)


def test_duplicate_result_and_early_assignment_end_are_rejected():
    jobs = make_jobs(2)
    coordinator, slot = make_coordinator(2, jobs)
    coordinator._dispatch_jobs()
    slot.completed_episode_ids.add(jobs[0].episode_id)
    with pytest.raises(RuntimeError, match='unassigned episode'):
        coordinator._require_assigned_job(slot,
                                          {'episode_id': jobs[0].episode_id})
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header(
            'assignment_end',
            deployment_id='deployment',
            worker_id=slot.worker_id,
            generation=0,
            assignment_id=slot.assignment.assignment_id),
        empty_payload()
    ]
    with pytest.raises(RuntimeError, match='before all results'):
        coordinator._receive_worker_message(None)
    assert not slot.is_idle


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('event, phase',
                         [('episode_end', None),
                          ('environment_case_rejected', 'reset'),
                          ('environment_case_rejected', 'step'),
                          ('episode_interrupted', 'reset'),
                          ('episode_interrupted', 'step')])
def test_terminal_episode_rejects_further_events(capacity, event, phase):
    jobs = make_jobs(capacity)
    coordinator, slot = make_coordinator(capacity, jobs)
    coordinator._dispatch_jobs()
    env = MockVectorEnvironment(capacity, success_after=1)
    env.setup()
    results = execute_assignment(
        env, [MockPolicyClient() for _ in jobs],
        slot.assignment,
        5,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(env))
    reporter = MockEvaluationReporter()
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       reporter) as run:
        run.start([jobs[0].task], capacity)
        if phase != 'reset':
            _receive_event(coordinator, slot, 'episode_start', job=jobs[0])
            coordinator._receive_worker_message(run)
        result = to_jsonable(results[0])
        if phase is None:
            _receive_event(
                coordinator, slot, 'episode_end', job=jobs[0], result=result)
        else:
            _receive_event(
                coordinator,
                slot,
                event,
                job=jobs[0],
                result={'reason': 'unstable layout'},
                phase=phase)
        coordinator._receive_worker_message(run)
        state = coordinator._scheduling_state
        pending = list(state.pending_jobs)
        next_case = state.next_environment_case_index_by_task.copy()
        for kind, payload in [('episode_start', None),
                              ('episode_interrupted', {}),
                              ('environment_case_rejected', {
                                  'reason': 'unstable layout'
                              }), ('episode_end', result),
                              ('episode_end', {
                                  **result, 'success': False
                              })]:
            _receive_event(
                coordinator,
                slot,
                kind,
                job=jobs[0],
                result=payload,
                phase='step')
            with pytest.raises(RuntimeError, match='unassigned episode'):
                coordinator._receive_worker_message(run)
        assert list(state.pending_jobs) == pending
        assert state.next_environment_case_index_by_task == next_case
        assert slot.completed_episode_ids == {jobs[0].episode_id}
        assert len([
            kind for kind, _ in reporter.events if kind == 'episode_end'
        ]) == int(phase is None)
        assert len(state.results_by_episode_id) == int(phase is None)


@pytest.mark.parametrize('capacity', [1, 3])
def test_reconnected_worker_deduplicates_start_and_completes_retry(capacity):
    jobs = make_jobs(capacity)
    coordinator, slot = make_coordinator(capacity, jobs)
    coordinator._dispatch_jobs()
    reporter = MockEvaluationReporter()
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       reporter) as run:
        run.start([jobs[0].task], capacity)
        _receive_event(coordinator, slot, 'episode_start', job=jobs[0])
        coordinator._receive_worker_message(run)
        original_id = slot.assignment.assignment_id
        coordinator._replace_worker_connection(
            slot, routing_id=b'worker:1', generation=1)
        coordinator._mark_worker_ready(
            slot,
            encode_frame({
                'tasks': [to_jsonable(jobs[0].task)],
                'capacity': capacity,
            }))
        coordinator._dispatch_jobs()
        assert slot.assignment.assignment_id != original_id
        env = MockVectorEnvironment(capacity, success_after=1)
        env.setup()
        results = execute_assignment(
            env, [MockPolicyClient() for _ in jobs],
            slot.assignment,
            5,
            None,
            True,
            execution_policy=slot.execution_policy)
        for job, result in zip(slot.assignment.jobs, results):
            _receive_event(coordinator, slot, 'episode_start', job=job)
            coordinator._receive_worker_message(run)
            _receive_event(
                coordinator,
                slot,
                'episode_end',
                job=job,
                result=to_jsonable(result))
            coordinator._receive_worker_message(run)
        _receive_event(coordinator, slot, 'assignment_end')
        coordinator._receive_worker_message(run)
        assert slot.is_idle
        assert len([
            kind for kind, _ in reporter.events if kind == 'episode_start'
        ]) == capacity
        assert run.finish(results[-1])['run']['completed_episodes'] == capacity


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('phase', ['reset', 'step'])
def test_interruption_requeues_same_jobs_without_restart_or_duplicate_start(
        capacity, phase):
    jobs = make_jobs(capacity + 1)
    coordinator, slot = make_coordinator(capacity, jobs)
    coordinator.simulator_restart_limit = 0
    coordinator._dispatch_jobs()
    original = slot.assignment
    state = coordinator._scheduling_state
    next_cases = dict(state.next_environment_case_index_by_task)
    reporter = MockEvaluationReporter()
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       reporter) as run:
        run.start([jobs[0].task], len(jobs))
        for job in original.jobs:
            if phase == 'step':
                _receive_event(coordinator, slot, 'episode_start', job=job)
                coordinator._receive_worker_message(run)
            _receive_event(
                coordinator, slot, 'episode_interrupted', job=job, phase=phase)
            coordinator._receive_worker_message(run)
        assert slot.assignment is original
        assert not slot.is_idle
        assert list(state.pending_jobs) == [*reversed(original.jobs), jobs[-1]]
        assert state.next_environment_case_index_by_task == next_cases
        assert not state.replaced_cases_by_episode_id
        assert not state.restart_counts_by_episode_id
        assert not state.results_by_episode_id
        _receive_event(coordinator, slot, 'assignment_end')
        coordinator._receive_worker_message(run)
        coordinator._dispatch_jobs()
        assert slot.assignment.assignment_id != original.assignment_id
        assert slot.generation == 0
        assert slot.assignment.jobs == list(reversed(original.jobs))
        env = MockVectorEnvironment(capacity, success_after=1)
        env.setup()
        results = execute_assignment(
            env, [MockPolicyClient() for _ in range(capacity)],
            slot.assignment,
            2,
            None,
            True,
            execution_policy=slot.execution_policy)
        for job, result in zip(slot.assignment.jobs, results):
            _receive_event(coordinator, slot, 'episode_start', job=job)
            coordinator._receive_worker_message(run)
            _receive_event(
                coordinator,
                slot,
                'episode_end',
                job=job,
                result=to_jsonable(result))
            coordinator._receive_worker_message(run)
        _receive_event(coordinator, slot, 'assignment_end')
        coordinator._receive_worker_message(run)
        starts = [
            body for kind, body in reporter.events if kind == 'episode_start'
        ]
        assert len(starts) == capacity
        assert all('replaces_environment_case_index' not in body
                   for body in starts)
        assert run.finish(results[-1])['run']['completed_episodes'] == capacity


def test_old_assignment_and_generation_messages_cannot_complete_a_retry():
    jobs = make_jobs(2)
    coordinator, slot = make_coordinator(2, jobs)
    coordinator._dispatch_jobs()
    original = slot.assignment
    coordinator._replace_worker_connection(
        slot, routing_id=b'new', generation=1)
    slot.simulator_ready = True
    slot.recovery_deadline = None
    coordinator._idle_worker_ids.append(slot.worker_id)
    coordinator._dispatch_jobs()
    assert slot.assignment.assignment_id != original.assignment_id
    for kind in [
            'episode_start', 'episode_end', 'assignment_end',
            'environment_case_rejected', 'episode_interrupted'
    ]:
        _receive_event(
            coordinator,
            slot,
            kind,
            job=jobs[0],
            assignment_id=original.assignment_id)
        coordinator._receive_worker_message(None)
        assert not slot.is_idle
        _receive_event(coordinator, slot, kind, job=jobs[0], generation=0)
        coordinator._receive_worker_message(None)
    assert not coordinator._scheduling_state.results_by_episode_id
    assert not coordinator._scheduling_state.started_episode_ids


def test_report_failure_does_not_commit_completion():

    class Reporter(MockEvaluationReporter):

        def report_evaluation(self, event_type, payload):
            if event_type == 'episode_end':
                raise RuntimeError('report unavailable')
            return super().report_evaluation(event_type, payload)

    jobs = make_jobs(1)
    coordinator, slot = make_coordinator(2, jobs)
    coordinator._dispatch_jobs()
    env = MockVectorEnvironment(2, success_after=1)
    env.setup()
    result = execute_assignment(
        env, [MockPolicyClient(), MockPolicyClient()],
        slot.assignment,
        5,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(env))[0]
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       Reporter()) as run:
        run.start([jobs[0].task], 1)
        _receive_event(coordinator, slot, 'episode_start', job=jobs[0])
        coordinator._receive_worker_message(run)
        _receive_event(
            coordinator,
            slot,
            'episode_end',
            job=jobs[0],
            result=to_jsonable(result))
        with pytest.raises(RuntimeError, match='report unavailable'):
            coordinator._receive_worker_message(run)
        assert not slot.completed_episode_ids
        assert not coordinator._scheduling_state.results_by_episode_id
        assert run.finish(None)['run']['completed_episodes'] == 0


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_strict_scheduling_uses_capacity_and_fixed_worker_pairing(
        capacity, requires_single_task):
    jobs = make_jobs(capacity + 1, ('a', 'b'))
    coordinator = DistributedEvaluationCoordinator(
        'inproc://strict-assignments', 2, deployment_id='deployment')
    coordinator._strict_worker_assignment = True
    state = coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    state.restart_counts_by_episode_id = {job.episode_id: 1 for job in jobs}
    coordinator._worker_router = RouterSocket()
    for index in range(2):
        slot = _WorkerSlot(
            f'simulator-worker-{index}',
            f'worker-{index}'.encode(),
            0,
            str(index),
            task_manifest=(jobs[0].task, ),
            execution_policy=ExecutionPolicy(capacity, requires_single_task),
            simulator_ready=True)
        coordinator._worker_slots[slot.worker_id] = slot
        coordinator._idle_worker_ids.append(slot.worker_id)
    dispatched = []
    while state.pending_jobs:
        coordinator._dispatch_jobs()
        for index, slot in enumerate(coordinator._worker_slots.values()):
            assignment = slot.assignment
            if assignment is None:
                continue
            assert 0 < len(assignment.jobs) <= capacity
            assert all((job.global_episode_number - 1) % 2 == index
                       for job in assignment.jobs)
            if requires_single_task:
                assert len({job.task.task_id for job in assignment.jobs}) == 1
            dispatched.extend(assignment.jobs)
            coordinator._release_worker_slot(slot)
    assert sorted(job.episode_id
                  for job in dispatched) == sorted(job.episode_id
                                                   for job in jobs)
