import pytest

from fluxthemis.environments import EnvironmentCaseRejected
from fluxthemis.environments.sim.robodojo import RoboDojoEnvironment
from fluxthemis.structures import TaskSpec


class UnStableError(RuntimeError):
    pass


class _UnstableNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.close_calls = 0

    def reset(self, seed):
        self.reset_calls.append(seed)
        raise UnStableError('unstable scene')

    def close(self):
        self.close_calls += 1


def test_robodojo_translates_unstable_layout_to_case_rejection():
    task = TaskSpec('task')
    native = _UnstableNativeEnvironment()
    environment = object.__new__(RoboDojoEnvironment)
    environment._env = native
    environment._task = task
    environment._task_ids = {task.task_id}
    environment._episode_active = False
    environment._episode_task = None
    environment._episode_index = None
    environment._episode_layout_index = None
    environment._physx_monitor = None
    environment._video_finalized = True

    with pytest.raises(
            EnvironmentCaseRejected, match='RoboDojo layout 17 is unstable'):
        environment.reset(
            task,
            episode_index=0,
            environment_seed=41,
            environment_case_index=17,
        )

    assert native.reset_calls == [[17]]
    assert native.close_calls == 1
