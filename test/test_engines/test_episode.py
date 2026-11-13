import numpy as np
import pytest

from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode,
                                                validate_action_chunk)
from fluxthemis.environments.mock import MockEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import ActionChunk, TaskSpec


def test_episode_jobs_and_execution_contract():
    jobs = build_episode_jobs(
        [TaskSpec('a'), TaskSpec('b')],
        episodes_per_task=2,
        base_seed=7,
        run_session_id='run',
    )

    assert [job.policy_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_case_index for job in jobs] == [0, 1, 0, 1]

    environment = MockEnvironment(tasks=['a'], success_after=2)
    policy_client = MockPolicyClient(action_dim=7, chunk_size=1)
    environment.setup()
    result = execute_episode(
        environment=environment,
        policy_client=policy_client,
        job=jobs[0],
        max_episode_steps=3,
        execute_horizon=1,
        stop_on_success=True,
    )

    assert result.success
    assert result.steps == 2
    assert result.policy_seed == 7
    assert result.environment_seed == 7
    assert result.environment_case_index == 0


@pytest.mark.parametrize('value', [
    np.array([[np.nan]]),
    np.array([[np.inf]]),
    np.array([[1e300]]), [[True, 1.]], [['1']],
    np.array([[object()]], dtype=object)
])
def test_action_chunk_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        validate_action_chunk(ActionChunk(value))


def test_action_dimensions_remain_generic_and_each_action_is_observed():
    value = validate_action_chunk(ActionChunk(np.zeros((4, 29))))
    assert value.shape == (4, 29)
    env = MockEnvironment(tasks=['a'], success_after=3)
    policy = MockPolicyClient(action_dim=7, chunk_size=4)
    observed = []
    policy.observe = lambda observation: observed.append(observation)
    env.setup()
    job = build_episode_jobs([TaskSpec('a')], 1, 0, 'r')[0]
    result = execute_episode(env, policy, job, 5, 4, True)
    assert result.steps == 3 and len(observed) == 3
