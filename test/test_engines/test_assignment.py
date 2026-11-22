from test.test_engines.helpers import make_jobs

import pytest

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.episode import build_episode_jobs
from fluxthemis.environments import MockVectorEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import TaskSpec


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_capacity_validation(value):
    with pytest.raises(ValueError, match='positive integer'):
        ExecutionPolicy(value)


def test_capacity_and_declared_task_capability_validation():
    policy = ExecutionPolicy(2)
    policy.validate_assignment(EpisodeAssignment(make_jobs(1)))
    mixed = EpisodeAssignment(make_jobs(1, ('a', 'b')))
    policy.validate_assignment(mixed)
    with pytest.raises(ValueError, match='one task'):
        ExecutionPolicy(
            2, requires_single_task=True).validate_assignment(mixed)
    with pytest.raises(ValueError, match='exceeds worker capacity'):
        ExecutionPolicy(1).validate_assignment(mixed)
    with pytest.raises(ValueError, match='duplicate'):
        EpisodeAssignment([make_jobs(1)[0]] * 2)
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
