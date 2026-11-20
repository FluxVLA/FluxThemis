import numpy as np
import pytest

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment,
                                                validate_action_chunk)
from fluxthemis.environments.mock import MockVectorEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import ActionChunk, TaskSpec


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
