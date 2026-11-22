from test.test_engines.helpers import make_jobs

import numpy as np
import pytest

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment,
                                                validate_action_chunk)
from fluxthemis.environments import MockEnvironment, MockVectorEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import (ActionChunk, CaseRejection,
                                   EpisodeInterruption, PolicyStop, StepResult,
                                   TaskSpec)


@pytest.mark.parametrize('capacity', [1, 3])
def test_episode_jobs_and_execution_contract(capacity):
    jobs = build_episode_jobs([TaskSpec('a'), TaskSpec('b')], 2, 7, 'run')
    assert [job.policy_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_case_index for job in jobs] == [0, 1, 0, 1]
    environment = MockVectorEnvironment(capacity, success_after=2)
    environment.setup()
    results = execute_assignment(
        environment, [
            MockPolicyClient(action_dim=7, chunk_size=1)
            for _ in range(capacity)
        ],
        EpisodeAssignment(jobs[:capacity]),
        3,
        1,
        True,
        execution_policy=ExecutionPolicy.from_environment(environment))
    for job, result in zip(jobs, results):
        job.validate_result(result)
        assert result.success and result.steps == 2


@pytest.mark.parametrize('value', [
    np.array([[np.nan]]),
    np.array([[np.inf]]),
    np.array([[1e300]]),
    np.array([[1 + 2j]]), [[True, 1.]], [['1']],
    np.array([[object()]], dtype=object)
])
@pytest.mark.parametrize('capacity', [1, 3])
def test_action_chunk_rejects_invalid_values(value, capacity):
    with pytest.raises((TypeError, ValueError)):
        validate_action_chunk(ActionChunk(value))

    class InvalidPolicy(MockPolicyClient):

        def predict_action(self, observation, context):
            return ActionChunk(value)

    env = MockVectorEnvironment(capacity, success_after=1)
    env.setup()
    jobs = build_episode_jobs([TaskSpec('a')], capacity, 0, 'r')
    with pytest.raises((TypeError, ValueError)):
        execute_assignment(
            env, [InvalidPolicy() for _ in jobs],
            EpisodeAssignment(jobs),
            5,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(env))
    assert all(peer.step_count == 0 for peer in env.environments)


@pytest.mark.parametrize('capacity', [1, 3])
def test_action_dimensions_remain_generic_and_each_action_is_observed(
        capacity):
    value = validate_action_chunk(ActionChunk(np.zeros((4, 29))))
    assert value.shape == (4, 29)
    env = MockVectorEnvironment(capacity, tasks=['a'], success_after=3)
    policies = [
        MockPolicyClient(action_dim=7, chunk_size=4) for _ in range(capacity)
    ]
    observed = [[] for _ in policies]
    for policy, observations in zip(policies, observed):
        policy.observe = observations.append
    env.setup()
    jobs = build_episode_jobs([TaskSpec('a')], capacity, 0, 'r')
    results = execute_assignment(
        env,
        policies,
        EpisodeAssignment(jobs),
        5,
        4,
        True,
        execution_policy=ExecutionPolicy.from_environment(env))
    assert all(result.steps == 3 for result in results)
    assert all(len(observations) == 3 for observations in observed)


@pytest.mark.parametrize('capacity', [1, 3])
def test_shared_execution_preserves_slot_identity_and_chunk_progress(capacity):
    environment = MockVectorEnvironment(capacity, success_after=None)
    for index, slot in enumerate(environment.environments):
        slot.success_after = index * 2 + 1
    environment.setup()
    jobs = make_jobs(capacity)
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
        EpisodeAssignment(make_jobs(2)),
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
    jobs = make_jobs(capacity)
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
        EpisodeAssignment(make_jobs(2)),
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
        EpisodeAssignment(make_jobs(capacity)),
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
        EpisodeAssignment(make_jobs(capacity)),
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
            EpisodeAssignment(make_jobs(2)),
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
    jobs = make_jobs(capacity)
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
    jobs = make_jobs(capacity)
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
            EpisodeAssignment(make_jobs(capacity)),
            10,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment))


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
    jobs = make_jobs(limit + 1, ('limited', ))
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


def test_single_slot_environment_rejects_larger_assignment():
    environment = MockEnvironment()
    environment.setup()
    with pytest.raises(ValueError, match='exceeds worker capacity'):
        execute_assignment(
            environment,
            [MockPolicyClient(), MockPolicyClient()],
            EpisodeAssignment(make_jobs(2)),
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
            EpisodeAssignment(make_jobs(capacity)),
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


@pytest.mark.parametrize('capacity', [1, 3])
def test_unused_slots_do_not_reset_or_infer(capacity):
    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.setup()
    clients = [MockPolicyClient() for _ in range(capacity)]
    results = execute_assignment(
        environment,
        clients,
        EpisodeAssignment(make_jobs(1)),
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
    jobs = make_jobs(capacity)
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
        EpisodeAssignment(make_jobs(capacity)),
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
