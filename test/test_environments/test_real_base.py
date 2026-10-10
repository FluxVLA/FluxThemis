import numpy as np
import pytest

from fluxthemis.environments.real import BaseRealRobotEnvironment
from fluxthemis.structures import StepResult, TaskSpec


class _RobotEnvironment(BaseRealRobotEnvironment):

    def connect(self):
        self.connected = True

    def tasks(self):
        return [TaskSpec('robot')]

    def reset(self, task, episode_index, *, environment_seed,
              environment_case_index):
        return {}, {}

    def execute_action(self, action):
        self.last_action = action
        return StepResult(observation={})

    def on_emergency_stop(self):
        self.stopped = True

    def disconnect(self):
        self.connected = False


def test_real_robot_safety_boundary():
    with pytest.raises(TypeError, match='allow_actuation'):
        _RobotEnvironment(allow_actuation='false')

    environment = _RobotEnvironment(
        allow_actuation=True,
        action_low=[-1.0, -2.0],
        action_high=[1.0, 2.0],
    )
    environment.setup()
    environment.step([0.5, -1.0])
    np.testing.assert_array_equal(environment.last_action, [0.5, -1.0])

    with pytest.raises(ValueError, match='bounds'):
        environment.step([1.5, 0.0])
    environment.emergency_stop()
    with pytest.raises(RuntimeError, match='emergency-stopped'):
        environment.step([0.0, 0.0])

    environment.close()
    assert not environment.connected
