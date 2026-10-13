from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode)
from fluxthemis.environments.mock import MockEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import TaskSpec


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
