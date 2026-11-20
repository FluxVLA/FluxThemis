from contextlib import nullcontext

import numpy as np
import pytest

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState, _WorkerSlot)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.engines.runners.worker import _SimulatorWorker
from fluxthemis.engines.utils import to_jsonable
from fluxthemis.environments import MockEnvironment, MockVectorEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import (CaseRejection, EpisodeInterruption,
                                   PolicyStop, StepResult, TaskSpec)
from fluxthemis.transports.zmq_protocol import (decode_frame, decode_header,
                                                empty_payload, encode_frame,
                                                encode_header)


def _jobs(count, tasks=('task', )):
    return build_episode_jobs([TaskSpec(t) for t in tasks], count, 7, 'run')


@pytest.mark.parametrize('capacity', [1, 3])
def test_shared_execution_preserves_slot_identity_and_chunk_progress(capacity):
    environment = MockVectorEnvironment(capacity, success_after=None)
    for index, slot in enumerate(environment.environments):
        slot.success_after = index * 2 + 1
    environment.setup()
    jobs = _jobs(capacity)
    clients = [
        MockPolicyClient(chunk_size=i + 1, fill_value=i)
        for i in range(capacity)
    ]
    completed = []
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        10,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment),
        on_episode_finished=lambda job, result: completed.append(job.episode_id
                                                                 ))
    assert completed == [job.episode_id for job in jobs]
    for index, (job, result, slot) in enumerate(
            zip(jobs, results, environment.environments)):
        job.validate_result(result)
        assert result.steps == index * 2 + 1
        assert result.model_calls == (result.steps + index) // (index + 1)
        assert result.success
        assert result.termination_reason == 'success'
        assert slot.step_count == result.steps
        assert all(np.all(action == index) for action in slot.received_actions)
        assert all(meta['episode_id'] == job.episode_id
                   for meta in result.prediction_metadata)


def test_horizon_and_max_steps_apply_per_slot():
    environment = MockVectorEnvironment(2, success_after=None)
    environment.environments[0].max_internal_steps = 2
    environment.setup()
    results = execute_assignment(
        environment, [MockPolicyClient(chunk_size=8) for _ in range(2)],
        EpisodeAssignment(_jobs(2)),
        5,
        2,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment))
    assert [r.steps for r in results] == [2, 5]
    assert [r.model_calls for r in results] == [1, 3]
    assert [r.termination_reason
            for r in results] == ['truncated', 'max_steps']


@pytest.mark.parametrize('capacity', [1, 2])
def test_policy_completion_without_actions_finalizes_each_slot(capacity):
    finalized, completed = [], []

    class Environment(MockVectorEnvironment):

        def step_batch(self, actions):
            raise AssertionError('A policy stop must not step the simulator')

        def finalize_episode(self, slot, *, success, termination_reason):
            finalized.append((slot, success, termination_reason))
            return {'video_paths': [f'slot-{slot}.mp4']}

    class Client(MockPolicyClient):

        def predict_action(self, observation, context):
            self.call_count += 1
            assert context['observation_id'] == 0
            return PolicyStop('completed', 'Agent declared completion',
                              {'source': 'agent'})

    environment = Environment(capacity, success_after=None)
    environment.setup()
    clients = [Client() for _ in range(capacity)]
    jobs = _jobs(capacity)
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        10,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment),
        on_episode_finished=lambda job, result: completed.append(result))
    assert finalized == [(slot, False, 'policy_completed')
                         for slot in range(capacity)]
    assert completed == list(results)
    assert all(slot.step_count == 0 for slot in environment.environments)
    assert all(client.call_count == 1 for client in clients)
    for slot, (job, result) in enumerate(zip(jobs, results)):
        job.validate_result(result)
        assert not result.success and result.score is None
        assert result.steps == 0 and result.model_calls == 1
        assert result.termination_reason == 'policy_completed'
        assert result.info['agent_success'] is True
        assert result.info['completion_source'] == 'agent_visual'
        assert result.info['video_paths'] == [f'slot-{slot}.mp4']
        assert result.prediction_metadata == [{'source': 'agent'}]


def test_policy_stop_keeps_peer_running_and_uses_executed_step_counts():
    observation_ids, finalized = [], []

    class Environment(MockVectorEnvironment):

        def reset_batch(self, requests):
            resets = super().reset_batch(requests)
            for reset in resets:
                reset.info['step'] = 1000
            return resets

        def step_batch(self, actions):
            transitions = super().step_batch(actions)
            for slot, transition in zip(self.environments, transitions):
                if transition is not None:
                    transition.info['step'] = 1000 + slot.step_count * 10
            return transitions

        def finalize_episode(self, slot, *, success, termination_reason):
            finalized.append((slot, success, termination_reason))
            return {'finalized_slot': slot}

    class Client(MockPolicyClient):

        def predict_action(self, observation, context):
            observation_ids.append(context['observation_id'])
            if len(observation_ids) == 3:
                return PolicyStop('blocked', 'Unable to continue')
            return super().predict_action(observation, context)

    environment = Environment(2, success_after=None, max_internal_steps=20)
    environment.environments[1].success_after = 6
    environment.setup()
    results = execute_assignment(
        environment, [Client(), MockPolicyClient()],
        EpisodeAssignment(_jobs(2)),
        10,
        2,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment))
    assert observation_ids == [0, 2, 4]
    assert [slot.step_count for slot in environment.environments] == [4, 6]
    assert [result.steps for result in results] == [4, 6]
    assert results[0].model_calls == 3 and not results[0].success
    assert results[0].termination_reason == 'policy_blocked'
    assert results[0].info['policy_stop'] == {
        'status': 'blocked',
        'reason': 'Unable to continue'
    }
    assert 'agent_success' not in results[0].info
    assert results[1].success and results[1].termination_reason == 'success'
    assert finalized == [(0, False, 'policy_blocked'), (1, True, 'success')]
    assert [result.info['finalized_slot'] for result in results] == [0, 1]


@pytest.mark.parametrize('capacity', [1, 2])
def test_finalization_always_receives_its_slot(capacity):
    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.setup()
    finalized = []

    def finalize_episode(slot, *, success, termination_reason):
        finalized.append((slot, success, termination_reason))
        return {'artifact': f'slot-{slot}'}

    environment.finalize_episode = finalize_episode
    results = execute_assignment(
        environment, [MockPolicyClient() for _ in range(capacity)],
        EpisodeAssignment(_jobs(capacity)),
        10,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment))
    assert finalized == [(slot, True, 'success') for slot in range(capacity)]
    assert [result.info['artifact'] for result in results
            ] == [f'slot-{slot}' for slot in range(capacity)]


@pytest.mark.parametrize('capacity', [1, 2])
def test_score_validation_and_sticky_success_are_shared(capacity):

    class Environment(MockVectorEnvironment):

        def step_batch(self, actions):
            transitions = super().step_batch(actions)
            for slot, transition in zip(self.environments, transitions):
                if transition is not None:
                    transition.success = slot.step_count == 1
                    transition.terminated = slot.step_count == 3
                    transition.score = slot.step_count / 4
            return transitions

    environment = Environment(capacity, success_after=None)
    environment.setup()
    results = execute_assignment(
        environment, [MockPolicyClient() for _ in range(capacity)],
        EpisodeAssignment(_jobs(capacity)),
        10,
        None,
        False,
        execution_policy=ExecutionPolicy.from_environment(environment))
    assert all(r.success and r.steps == 3 and r.score == 0.75 for r in results)
    assert all(r.termination_reason == 'terminated' for r in results)


def test_partial_results_survive_later_slot_failure():

    class FailingClient(MockPolicyClient):

        def predict_action(self, observation, context):
            if self.call_count:
                raise RuntimeError('inference failed')
            return super().predict_action(observation, context)

    environment = MockVectorEnvironment(2, success_after=1)
    environment.environments[1].success_after = 4
    environment.setup()
    results = []
    with pytest.raises(RuntimeError, match='inference failed'):
        execute_assignment(
            environment, [MockPolicyClient(),
                          FailingClient(chunk_size=1)],
            EpisodeAssignment(_jobs(2)),
            10,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment),
            on_episode_finished=lambda job, result: results.append(result))
    assert len(results) == 1
    assert results[0].episode_index == 0
    assert environment.environments[0].step_count == 1


@pytest.mark.parametrize('capacity', [1, 3])
def test_runtime_invalid_case_does_not_drop_healthy_progress(capacity):

    class InvalidCase(MockVectorEnvironment):

        def step_batch(self, actions):
            transitions = super().step_batch(actions)
            if actions[0] is not None:
                transitions[0] = CaseRejection(transitions[0].info)
            return transitions

    environment = InvalidCase(capacity, success_after=3)
    environment.setup()
    jobs = _jobs(capacity)
    clients = [MockPolicyClient(chunk_size=1) for _ in range(capacity)]
    rejected, completed = [], []
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        10,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment),
        on_case_rejected=lambda job, phase, info: rejected.append(
            (job.episode_id, phase, info)),
        on_episode_finished=lambda job, result: completed.append(job.episode_id
                                                                 ))
    assert rejected == [(jobs[0].episode_id, 'step', {
        'task_id': 'task',
        'step': 1
    })]
    assert completed == [job.episode_id for job in jobs[1:]]
    assert results[0] is None
    assert all(result.success and result.steps == 3 for result in results[1:])
    assert [client.call_count for client in clients] == [1] + [3] * (
        capacity - 1)
    assert environment.environments[0].step_count == 1


@pytest.mark.parametrize('capacity', [1, 3])
def test_interruption_keeps_completed_results_and_discards_attempt_metrics(
        capacity):

    class Environment(MockVectorEnvironment):

        calls = 0

        def step_batch(self, actions):
            self.calls += 1
            if self.calls == 2:
                return [
                    None if action is None else EpisodeInterruption()
                    for action in actions
                ]
            return super().step_batch(actions)

    environment = Environment(capacity, success_after=None)
    for peer in environment.environments[:-1]:
        peer.success_after = 1
    environment.setup()
    jobs = _jobs(capacity)
    clients = [MockPolicyClient(chunk_size=1) for _ in jobs]
    finished, interrupted = [], []
    policy = ExecutionPolicy.from_environment(environment)
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        2,
        None,
        True,
        execution_policy=policy,
        on_episode_finished=lambda job, result: finished.append(result),
        on_episode_interrupted=lambda job, phase, info: interrupted.append(
            (job, phase, info)))
    assert finished == results[:-1]
    assert all(result.steps == result.model_calls == 1 for result in finished)
    assert results[-1] is None
    assert interrupted == [(jobs[-1], 'step', {})]
    assert [peer.step_count
            for peer in environment.environments] == [1] * capacity
    for peer in environment.environments:
        peer.success_after = 1
    retried = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs[-1:]),
        2,
        None,
        True,
        execution_policy=policy)
    assert retried[0].steps == retried[0].model_calls == 1
    assert retried[1:] == [None] * (capacity - 1)


@pytest.mark.parametrize('capacity', [1, 2])
def test_nonfinite_score_is_rejected(capacity):

    class BadScore(MockVectorEnvironment):

        def step_batch(self, actions):
            return [
                StepResult(observation={}, score=float('nan')) for _ in actions
            ]

    environment = BadScore(capacity)
    environment.setup()
    with pytest.raises(ValueError, match='non-finite'):
        execute_assignment(
            environment, [MockPolicyClient() for _ in range(capacity)],
            EpisodeAssignment(_jobs(capacity)),
            10,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment))


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_capacity_validation(value):
    with pytest.raises(ValueError, match='positive integer'):
        ExecutionPolicy(value)


def test_capacity_and_declared_task_capability_validation():
    policy = ExecutionPolicy(2)
    policy.validate_assignment(EpisodeAssignment(_jobs(1)))
    mixed = EpisodeAssignment(_jobs(1, ('a', 'b')))
    policy.validate_assignment(mixed)
    with pytest.raises(ValueError, match='one task'):
        ExecutionPolicy(
            2, requires_single_task=True).validate_assignment(mixed)
    with pytest.raises(ValueError, match='exceeds worker capacity'):
        ExecutionPolicy(1).validate_assignment(mixed)
    with pytest.raises(ValueError, match='duplicate'):
        EpisodeAssignment([_jobs(1)[0]] * 2)
    with pytest.raises(ValueError, match='sizes differ'):
        policy.validate_assignment(mixed, [MockPolicyClient()])
    with pytest.raises(ValueError, match='own policy client'):
        policy.validate_assignment(mixed, [MockPolicyClient()] * 2)
    with pytest.raises(ValueError, match='capacity changed'):
        policy.validate_capacity(3, {})
    with pytest.raises(TypeError, match='requires_single_task'):
        ExecutionPolicy(2, requires_single_task='yes')


def test_assignment_respects_task_limit_with_full_worker_client_container():
    task = TaskSpec('limited')
    jobs = build_episode_jobs([task], 8, 7, 'run')
    policy = ExecutionPolicy(
        8,
        requires_single_task=True,
        task_capacities={
            'limited': 5,
            'full': 8
        })
    clients = [MockPolicyClient() for _ in range(8)]
    policy.validate_assignment(EpisodeAssignment(jobs[:5]), clients)
    with pytest.raises(ValueError, match='task slot limit'):
        policy.validate_assignment(EpisodeAssignment(jobs[:6]), clients)
    with pytest.raises(ValueError, match='sizes differ'):
        policy.validate_assignment(EpisodeAssignment(jobs[:5]), clients[:5])
    assert policy.task_capacity(TaskSpec('unlimited')) == 8
    assert policy.task_capacity(TaskSpec('full')) == 8
    assert policy.task_capacities == {'limited': 5}
    policy.validate_capacity(8, {'limited': 5})


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, None])
def test_task_slot_limit_validation(limit):
    with pytest.raises(
            ValueError, match='task capacity must be a positive integer'):
        ExecutionPolicy(8, task_capacities={'task': limit})


def test_task_capacity_cannot_exceed_worker_capacity():
    with pytest.raises(
            ValueError, match='Task capacity exceeds worker capacity'):
        ExecutionPolicy(8, task_capacities={'task': 9})


@pytest.mark.parametrize('limit', [0, -1, True, 8.0, 9.0, 9, None])
def test_environment_task_capacities_are_validated_before_filtering(
        monkeypatch, limit):
    environment = MockVectorEnvironment(8)
    environment.setup()
    monkeypatch.setattr(environment, 'task_capacity', lambda task: limit)
    with pytest.raises(ValueError, match='[Tt]ask capacity'):
        ExecutionPolicy.from_environment(environment)


@pytest.mark.parametrize('capacity', [1, 8])
def test_execute_assignment_reuses_declared_capabilities(
        monkeypatch, capacity):
    limit = min(capacity, 5)
    environment = MockVectorEnvironment(
        capacity, tasks=['limited'], success_after=1)
    environment.setup()
    monkeypatch.setattr(environment, 'tasks', lambda: ['limited'])
    monkeypatch.setattr(environment, 'task_capacity',
                        lambda task: {'limited': limit}[task.task_id])
    policy = ExecutionPolicy.from_environment(environment)

    def read_capabilities_again(*args):
        raise AssertionError('Capabilities must only be read during setup')

    monkeypatch.setattr(environment, 'tasks', read_capabilities_again)
    monkeypatch.setattr(environment, 'task_capacity', read_capabilities_again)
    jobs = _jobs(limit + 1, ('limited', ))
    clients = [MockPolicyClient() for _ in range(capacity)]
    with pytest.raises(ValueError, match='exceeds'):
        execute_assignment(
            environment,
            clients,
            EpisodeAssignment(jobs),
            2,
            None,
            True,
            execution_policy=policy)
    assert all(slot.current_task is None for slot in environment.environments)
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs[:limit]),
        2,
        None,
        True,
        execution_policy=policy)
    assert len(results) == capacity
    assert results[limit:] == [None] * (capacity - limit)
    assert all(result.success for result in results[:limit])
    assert [client.call_count for client in clients] == [1] * limit + [0] * (
        capacity - limit)
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs[-1:]),
        2,
        None,
        True,
        execution_policy=policy)
    assert results[0].success
    assert results[1:] == [None] * (capacity - 1)


class _Socket:

    def __init__(self):
        self.incoming = None
        self.sent = []

    def recv_multipart(self):
        return self.incoming

    def send_multipart(self, frames):
        self.sent.append(frames)


def _coordinator(capacity,
                 jobs,
                 requires_single_task=False,
                 task_capacities=None):
    coordinator = DistributedEvaluationCoordinator(
        'inproc://assignments', 1, deployment_id='deployment')
    coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    slot = _WorkerSlot('simulator-worker-0', b'worker', 0, '0')
    coordinator._worker_slots[slot.worker_id] = slot
    coordinator._worker_router = _Socket()
    tasks = {job.task.task_id: job.task for job in jobs}
    coordinator._mark_worker_ready(
        slot,
        encode_frame({
            'tasks': [to_jsonable(task) for task in tasks.values()],
            'capacity': capacity,
            'requires_single_task': requires_single_task,
            'task_capacities': task_capacities or {},
        }))
    return coordinator, slot


@pytest.mark.parametrize('strict', [False, True])
def test_dispatch_batches_by_declared_task_limit(strict):
    tasks = [TaskSpec('stack_bowls_random'), TaskSpec('stack_bowls')]
    jobs = build_episode_jobs(tasks, 25, 7, 'run')
    coordinator, slot = _coordinator(
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
    coordinator, slot = _coordinator(8, _jobs(1), task_capacities={'task': 5})
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


@pytest.mark.parametrize('capacity', [1, 2, 10])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_worker_coordinator_roundtrip_waits_for_assignment_end(
        monkeypatch, capacity, requires_single_task):
    # Exercise the actual wire codec, worker execution and coordinator events
    # without launching a simulator or connecting a model service.
    monkeypatch.setattr('fluxthemis.engines.runners.worker._episode_timeout',
                        lambda timeout: nullcontext())
    count = 25 if capacity == 10 else 3
    jobs = _jobs(count, ('a', 'b'))
    coordinator, slot = _coordinator(capacity, jobs, requires_single_task)
    socket = _Socket()
    worker = _SimulatorWorker(
        config_path='',
        rollout_endpoint='',
        worker_id=slot.worker_id,
        deployment_id='deployment',
        generation=0,
        device='0',
        episode_timeout_s=10,
        config_overrides={},
        ready_event=None,
        stop_received=None)
    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.setup()
    run = EvaluationRun(
        EvaluationSettings(episodes_per_task=count), SuccessRateEvaluator(),
        MockEvaluationReporter())
    with run:
        run.start([TaskSpec('a'), TaskSpec('b')], len(jobs))
        batch_sizes = []
        while coordinator._scheduling_state.pending_jobs:
            coordinator._dispatch_jobs()
            socket.incoming = coordinator._worker_router.sent[-1][1:]
            assignment = worker._receive_assignment(socket)
            assert 0 < len(assignment.jobs) <= capacity
            batch_sizes.append(len(assignment.jobs))
            if requires_single_task:
                assert len({job.task.task_id for job in assignment.jobs}) == 1
            socket.sent.clear()
            deadline = slot.episode_deadline
            worker._execute_assignment(
                socket=socket,
                environment=environment,
                execution_policy=slot.execution_policy,
                policy_clients=[MockPolicyClient() for _ in range(capacity)],
                evaluation_cfg={'max_episode_steps': 10},
                assignment=assignment,
                save_videos=False,
                video_dir=None,
                video_fps=20)
            assert decode_header(
                socket.sent[-1][0])['type'] == 'assignment_end'
            for frames in socket.sent[:-1]:
                coordinator._worker_router.incoming = [
                    slot.routing_id, *frames
                ]
                coordinator._receive_worker_message(run)
                assert not slot.is_idle
                assert slot.episode_deadline == deadline
            assert coordinator._active_episode_progress() is None
            coordinator._worker_router.incoming = [
                slot.routing_id, *socket.sent[-1]
            ]
            coordinator._receive_worker_message(run)
            assert slot.is_idle
        expected_sizes = []
        group_lengths = [count] * 2 if requires_single_task else [count * 2]
        for length in group_lengths:
            expected_sizes.extend([capacity] * (length // capacity))
            expected_sizes.extend([length % capacity] *
                                  bool(length % capacity))
        assert batch_sizes == expected_sizes
        assert coordinator._scheduling_state.is_complete
        results = coordinator._scheduling_state.ordered_results()
        assert [r.policy_seed
                for r in results] == list(range(7, 7 + len(jobs)))
        assert run.finish(
            results[-1])['run']['completed_episodes'] == len(jobs)


def test_batch_reconnect_only_requeues_unfinished_episodes():
    jobs = _jobs(2)
    coordinator, slot = _coordinator(2, jobs)
    coordinator._dispatch_jobs()
    slot.completed_episode_ids.add(jobs[0].episode_id)
    coordinator._scheduling_state.results_by_episode_id[
        jobs[0].episode_id] = 'saved'
    coordinator._replace_worker_connection(
        slot, routing_id=b'new', generation=1)
    assert list(coordinator._scheduling_state.pending_jobs) == [jobs[1]]
    assert coordinator._scheduling_state.restart_counts_by_episode_id == {
        jobs[1].episode_id: 1
    }
    assert coordinator._scheduling_state.results_by_episode_id == {
        jobs[0].episode_id: 'saved'
    }


def test_retried_jobs_fill_worker_capacity():
    jobs = _jobs(4)
    coordinator, slot = _coordinator(3, jobs)
    coordinator._dispatch_jobs()
    coordinator._replace_worker_connection(
        slot, routing_id=b'replacement', generation=1)
    slot.simulator_ready = True
    slot.recovery_deadline = None
    coordinator._idle_worker_ids.append(slot.worker_id)
    coordinator._dispatch_jobs()
    assert slot.assignment.jobs == jobs[:3]
    assert list(coordinator._scheduling_state.pending_jobs) == jobs[3:]


def test_scalar_retry_budget_is_per_episode_not_worker_generation():
    jobs = _jobs(4)
    coordinator, slot = _coordinator(1, jobs)
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
    jobs = _jobs(2)
    coordinator, slot = _coordinator(2, jobs)
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


def test_scalar_environment_does_not_silently_run_a_batch():
    environment = MockEnvironment()
    environment.setup()
    with pytest.raises(ValueError, match='exceeds worker capacity'):
        execute_assignment(
            environment,
            [MockPolicyClient(), MockPolicyClient()],
            EpisodeAssignment(_jobs(2)),
            10,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment))


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('fail', [False, True])
def test_video_finishes_or_aborts_each_slot(monkeypatch, capacity, fail):
    events = [[] for _ in range(capacity)]

    class Recorder:

        def __init__(self, *args, **kwargs):
            self.slot = None

        def start(self, task_id, episode_index):
            self.slot = episode_index
            events[self.slot].append('start')

        def capture(self, environment, slot):
            assert slot == self.slot
            events[slot].append('capture')

        def finish(self, success):
            events[self.slot].append('finish')
            return [f'episode-{self.slot}.mp4']

        def abort(self):
            events[self.slot].append('abort')

    class Environment(MockVectorEnvironment):

        def step_batch(self, actions):
            if fail:
                raise RuntimeError('simulator failed')
            results = super().step_batch(actions)
            for result in results:
                result.info['video_paths'] = 'native.mp4'
            return results

    monkeypatch.setattr(
        'fluxthemis.engines.runners.episode.EpisodeVideoRecorder', Recorder)
    environment = Environment(capacity, success_after=1)
    environment.setup()

    def execute():
        return execute_assignment(
            environment, [MockPolicyClient() for _ in range(capacity)],
            EpisodeAssignment(_jobs(capacity)),
            10,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment),
            save_videos=True,
            video_dir='/unused')

    if fail:
        with pytest.raises(RuntimeError, match='simulator failed'):
            execute()
        assert events == [['start', 'capture', 'abort']] * capacity
    else:
        results = execute()
        assert [result.info['video_paths'] for result in results
                ] == [['native.mp4', f'episode-{slot}.mp4']
                      for slot in range(capacity)]
        assert events == [['start', 'capture', 'capture', 'finish']] * capacity


def test_assignment_identity_is_checked_at_worker_boundary():
    worker = _SimulatorWorker(
        config_path='',
        rollout_endpoint='',
        worker_id='worker',
        deployment_id='deployment',
        generation=0,
        device='0',
        episode_timeout_s=10,
        config_overrides={},
        ready_event=None,
        stop_received=None)
    coordinator, slot = _coordinator(1, _jobs(1))
    coordinator._dispatch_jobs()
    socket = _Socket()
    socket.incoming = coordinator._worker_router.sent[-1][1:]
    header = decode_header(socket.incoming[0])
    body = decode_frame(socket.incoming[1])
    assert body['jobs'][0]['episode_id'] == 'run:1'
    header.update(worker_id='worker', assignment_id='wrong')
    socket.incoming[0] = encode_header(header.pop('type'), **header)
    with pytest.raises(RuntimeError, match='assignment_id mismatch'):
        worker._receive_assignment(socket)


@pytest.mark.parametrize('capacity', [1, 3])
def test_unused_slots_do_not_reset_or_infer(capacity):
    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.setup()
    clients = [MockPolicyClient() for _ in range(capacity)]
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(_jobs(1)),
        5,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment))
    assert len(results) == capacity
    assert results[1:] == [None] * (capacity - 1)
    assert [client.call_count for client in clients] == [1] + [0] * (
        capacity - 1)
    assert all(slot.current_task is None
               for slot in environment.environments[1:])


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


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('event, phase',
                         [('episode_end', None),
                          ('environment_case_rejected', 'reset'),
                          ('environment_case_rejected', 'step'),
                          ('episode_interrupted', 'reset'),
                          ('episode_interrupted', 'step')])
def test_terminal_episode_rejects_further_events(capacity, event, phase):
    jobs = _jobs(capacity)
    coordinator, slot = _coordinator(capacity, jobs)
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
    jobs = _jobs(capacity)
    coordinator, slot = _coordinator(capacity, jobs)
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
    jobs = _jobs(capacity + 1)
    coordinator, slot = _coordinator(capacity, jobs)
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
    jobs = _jobs(2)
    coordinator, slot = _coordinator(2, jobs)
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

    jobs = _jobs(1)
    coordinator, slot = _coordinator(2, jobs)
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
@pytest.mark.parametrize('interrupted', [False, True])
def test_discarded_reset_does_not_reset_or_infer_its_policy_client(
        capacity, interrupted):

    class DiscardedReset(MockEnvironment):

        def reset(self, *args, **kwargs):
            info = {'reason': 'incomplete reset'}
            return (EpisodeInterruption(info)
                    if interrupted else CaseRejection(info))

    class Client(MockPolicyClient):

        was_reset = False

        def reset(self, *args, **kwargs):
            self.was_reset = True
            super().reset(*args, **kwargs)

    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.environments[0] = DiscardedReset()
    environment.setup()
    jobs = _jobs(capacity)
    clients = [Client() for _ in range(capacity)]
    rejected, interruptions, started = [], [], []
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        5,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment),
        on_case_rejected=lambda job, phase, info: rejected.append(
            (job, phase, info)),
        on_episode_interrupted=lambda job, phase, info: interruptions.append(
            (job, phase, info)),
        on_episode_started=started.append)
    expected = [(jobs[0], 'reset', {'reason': 'incomplete reset'})]
    assert rejected == ([] if interrupted else expected)
    assert interruptions == (expected if interrupted else [])
    assert started == jobs[1:]
    assert results[0] is None
    assert all(result.success for result in results[1:])
    assert [client.call_count for client in clients] == [0] + [1] * (
        capacity - 1)
    assert not clients[0].was_reset
    assert all(client.was_reset for client in clients[1:])


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('interrupted', [False, True])
def test_discarded_slot_aborts_only_its_recorder(monkeypatch, capacity,
                                                 interrupted):
    events = [[] for _ in range(capacity)]

    class Recorder:

        def __init__(self, *args, **kwargs):
            self.slot = None

        def start(self, task_id, episode_index):
            self.slot = episode_index
            events[self.slot].append('start')

        def capture(self, environment, slot):
            assert slot == self.slot
            events[slot].append('capture')

        def finish(self, success):
            events[self.slot].append('finish')
            return [f'slot-{self.slot}.mp4']

        def abort(self):
            events[self.slot].append('abort')

    class DiscardedStep(MockEnvironment):

        def step(self, action):
            return EpisodeInterruption() if interrupted else CaseRejection()

    monkeypatch.setattr(
        'fluxthemis.engines.runners.episode.EpisodeVideoRecorder', Recorder)
    environment = MockVectorEnvironment(capacity, success_after=2)
    environment.environments[0] = DiscardedStep()
    environment.setup()
    results = execute_assignment(
        environment, [MockPolicyClient() for _ in range(capacity)],
        EpisodeAssignment(_jobs(capacity)),
        5,
        None,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment),
        save_videos=True,
        video_dir='/unused')
    assert results[0] is None
    assert events[0] == ['start', 'capture', 'abort']
    assert events[1:] == [['start', 'capture', 'capture', 'capture', 'finish']
                          ] * (
                              capacity - 1)
    assert [result.info['video_paths'] for result in results[1:]
            ] == [[f'slot-{slot}.mp4'] for slot in range(1, capacity)]


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_strict_scheduling_uses_capacity_and_fixed_worker_pairing(
        capacity, requires_single_task):
    jobs = _jobs(capacity + 1, ('a', 'b'))
    coordinator = DistributedEvaluationCoordinator(
        'inproc://strict-assignments', 2, deployment_id='deployment')
    coordinator._strict_worker_assignment = True
    state = coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    state.restart_counts_by_episode_id = {job.episode_id: 1 for job in jobs}
    coordinator._worker_router = _Socket()
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
