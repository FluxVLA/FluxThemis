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
from fluxthemis.structures import StepResult, TaskSpec
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
        EpisodeAssignment(_jobs(2)), 5, 2, True)
    assert [r.steps for r in results] == [2, 5]
    assert [r.model_calls for r in results] == [1, 3]
    assert [r.termination_reason
            for r in results] == ['truncated', 'max_steps']


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
    results = execute_assignment(environment,
                                 [MockPolicyClient() for _ in range(capacity)],
                                 EpisodeAssignment(_jobs(capacity)), 10, None,
                                 False)
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
            on_episode_finished=lambda job, result: results.append(result))
    assert len(results) == 1
    assert results[0].episode_index == 0
    assert environment.environments[0].step_count == 1


def test_runtime_invalid_case_does_not_drop_healthy_progress():

    class InvalidCase(MockVectorEnvironment):

        def step_batch(self, actions):
            transitions = super().step_batch(actions)
            if actions[0] is not None:
                transitions[0].case_rejected = True
            return transitions

    environment = InvalidCase(2, success_after=3)
    environment.setup()
    jobs = _jobs(2)
    clients = [MockPolicyClient(chunk_size=1) for _ in range(2)]
    rejected, completed = [], []
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(jobs),
        10,
        None,
        True,
        on_case_rejected=lambda job, info: rejected.append(job.episode_id),
        on_episode_finished=lambda job, result: completed.append(job.episode_id
                                                                 ))
    assert rejected == [jobs[0].episode_id]
    assert completed == [jobs[1].episode_id]
    assert [result.episode_index for result in results] == [1]
    assert results[0].success and results[0].steps == 3
    assert [client.call_count for client in clients] == [1, 3]
    assert environment.environments[0].step_count == 1


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
        execute_assignment(environment,
                           [MockPolicyClient() for _ in range(capacity)],
                           EpisodeAssignment(_jobs(capacity)), 10, None, True)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_batch_size_validation(value):
    with pytest.raises(ValueError, match='positive integer'):
        EvaluationSettings(episode_batch_size=value)


def test_batch_limits_and_scalar_recovery_compatibility():
    policy = ExecutionPolicy(2)
    policy.validate_assignment(EpisodeAssignment(_jobs(1)))
    for options in ({'save_videos': True}, {'strict_worker_assignment': True}):
        with pytest.raises(ValueError):
            policy.validate_options(**options)
        ExecutionPolicy().validate_options(**options)
    mixed = EpisodeAssignment(_jobs(1, ('a', 'b')))
    with pytest.raises(ValueError, match='one task'):
        policy.validate_assignment(mixed)
    with pytest.raises(ValueError, match='duplicate'):
        EpisodeAssignment((_jobs(1)[0], ) * 2)


class _Socket:

    def __init__(self):
        self.incoming = None
        self.sent = []

    def recv_multipart(self):
        return self.incoming

    def send_multipart(self, frames):
        self.sent.append(frames)


def _coordinator(capacity, jobs):
    coordinator = DistributedEvaluationCoordinator(
        'inproc://assignments', 1, deployment_id='deployment')
    coordinator._execution_policy = ExecutionPolicy(capacity)
    coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    slot = _WorkerSlot(
        'simulator-worker-0',
        b'worker',
        0,
        '0',
        task_manifest=(jobs[0].task, ),
        capacity=capacity,
        simulator_ready=True)
    coordinator._worker_slots[slot.worker_id] = slot
    coordinator._idle_worker_ids.append(slot.worker_id)
    coordinator._worker_router = _Socket()
    return coordinator, slot


@pytest.mark.parametrize('capacity', [1, 2, 10])
def test_worker_coordinator_roundtrip_waits_for_assignment_end(
        monkeypatch, capacity):
    # Exercise the actual wire codec, worker execution and coordinator events
    # without launching a simulator or connecting a model service.
    monkeypatch.setattr('fluxthemis.engines.runners.worker._episode_timeout',
                        lambda timeout: nullcontext())
    count = 25 if capacity == 10 else 3
    jobs = _jobs(count, ('a', 'b'))
    coordinator, slot = _coordinator(capacity, jobs)
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
        EvaluationSettings(
            episode_batch_size=capacity, episodes_per_task=count),
        SuccessRateEvaluator(), MockEvaluationReporter())
    with run:
        run.start([TaskSpec('a'), TaskSpec('b')], len(jobs))
        batch_sizes = []
        while coordinator._scheduling_state.pending_jobs:
            coordinator._dispatch_jobs()
            socket.incoming = coordinator._worker_router.sent[-1][1:]
            assignment = worker._receive_assignment(socket)
            assert 0 < len(assignment.jobs) <= capacity
            batch_sizes.append(len(assignment.jobs))
            assert len({job.task.task_id for job in assignment.jobs}) == 1
            socket.sent.clear()
            worker._execute_assignment(
                socket=socket,
                environment=environment,
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
            assert coordinator._active_episode_progress() is None
            coordinator._worker_router.incoming = [
                slot.routing_id, *socket.sent[-1]
            ]
            coordinator._receive_worker_message(run)
            assert slot.is_idle
        expected_sizes = {
            1: [1] * 6,
            2: [2, 1, 2, 1],
            10: [10, 10, 5, 10, 10, 5]
        }
        assert batch_sizes == expected_sizes[capacity]
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


def test_process_recovery_does_not_regroup_affected_episodes():
    jobs = _jobs(4)
    coordinator, slot = _coordinator(3, jobs)
    coordinator._dispatch_jobs()
    coordinator._replace_worker_connection(
        slot, routing_id=b'replacement', generation=1)
    recovered = []
    for _ in range(3):
        slot.simulator_ready = True
        slot.recovery_deadline = None
        coordinator._idle_worker_ids.append(slot.worker_id)
        coordinator._dispatch_jobs()
        recovered.append(slot.assignment.jobs)
        slot.assignment = None
    assert recovered == [(jobs[0], ), (jobs[1], ), (jobs[2], )]


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
            EpisodeAssignment(_jobs(2)), 10, None, True)


@pytest.mark.parametrize('fail', [False, True])
def test_scalar_video_finishes_or_aborts_in_shared_runner(monkeypatch, fail):
    events = []

    class Recorder:

        def __init__(self, *args, **kwargs):
            pass

        def start(self, task_id, episode_index):
            events.append('start')

        def capture(self, environment):
            events.append('capture')

        def finish(self, success):
            events.append('finish')
            return ['episode.mp4']

        def abort(self):
            events.append('abort')

    class Environment(MockEnvironment):

        def step(self, action):
            if fail:
                raise RuntimeError('simulator failed')
            result = super().step(action)
            result.info['video_paths'] = 'native.mp4'
            return result

    monkeypatch.setattr(
        'fluxthemis.engines.runners.episode.EpisodeVideoRecorder', Recorder)
    environment = Environment(success_after=1)
    environment.setup()

    def execute():
        return execute_assignment(
            environment, [MockPolicyClient()],
            EpisodeAssignment(_jobs(1)),
            10,
            None,
            True,
            save_videos=True,
            video_dir='/unused')

    if fail:
        with pytest.raises(RuntimeError, match='simulator failed'):
            execute()
        assert events == ['start', 'capture', 'abort']
    else:
        assert execute()[0].info['video_paths'] == [
            'native.mp4', 'episode.mp4'
        ]
        assert events == ['start', 'capture', 'capture', 'finish']


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
    results = execute_assignment(environment, clients,
                                 EpisodeAssignment(_jobs(1)), 5, None, True)
    assert len(results) == 1
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
                   generation=None):
    values = dict(
        deployment_id='deployment',
        worker_id=slot.worker_id,
        generation=slot.generation if generation is None else generation,
        assignment_id=assignment_id or slot.assignment.assignment_id)
    if job is not None:
        values['episode_id'] = job.episode_id
    coordinator._worker_router.incoming = [
        slot.routing_id,
        encode_header(kind, **values),
        empty_payload() if result is None else encode_frame(result)
    ]


def test_duplicate_results_are_idempotent_and_conflicts_fail():
    jobs = _jobs(2)
    coordinator, slot = _coordinator(2, jobs)
    coordinator._dispatch_jobs()
    env = MockVectorEnvironment(2, success_after=1)
    env.setup()
    results = execute_assignment(env, [MockPolicyClient() for _ in jobs],
                                 slot.assignment, 5, None, True)
    reporter = MockEvaluationReporter()
    with EvaluationRun(
            EvaluationSettings(episode_batch_size=2), SuccessRateEvaluator(),
            reporter) as run:
        run.start([jobs[0].task], 2)
        _receive_event(coordinator, slot, 'episode_start', job=jobs[0])
        coordinator._receive_worker_message(run)
        result = to_jsonable(results[0])
        for _ in range(2):
            _receive_event(
                coordinator, slot, 'episode_end', job=jobs[0], result=result)
            coordinator._receive_worker_message(run)
        assert len([
            kind for kind, _ in reporter.events if kind == 'episode_end'
        ]) == 1
        assert len(coordinator._scheduling_state.results_by_episode_id) == 1
        _receive_event(
            coordinator,
            slot,
            'episode_end',
            job=jobs[0],
            result={
                **result, 'success': False
            })
        with pytest.raises(RuntimeError, match='Conflicting duplicate'):
            coordinator._receive_worker_message(run)
        assert run.finish(results[0])['run']['completed_episodes'] == 1


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
            'environment_case_rejected'
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
        env, [MockPolicyClient(), MockPolicyClient()], slot.assignment, 5,
        None, True)[0]
    with EvaluationRun(
            EvaluationSettings(episode_batch_size=2), SuccessRateEvaluator(),
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
