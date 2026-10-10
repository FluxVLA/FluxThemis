import sys
from types import SimpleNamespace

import numpy as np

from fluxthemis.environments.sim import GymnasiumEnvironment


class _FakeGymEnvironment:

    def __init__(self):
        self.reset_call = None
        self.last_action = None
        self.closed = False

    def reset(self, *, seed, options):
        self.reset_call = (seed, options)
        return np.asarray([0.0]), {'reset': True}

    def step(self, action):
        self.last_action = action
        return np.asarray([1.0]), 1.5, True, False, {'solved': True}

    def render(self):
        return 'frame'

    def close(self):
        self.closed = True


def test_gymnasium_adapter_maps_task_options_and_success(monkeypatch):
    fake_env = _FakeGymEnvironment()
    fake_module = SimpleNamespace(make=lambda *args, **kwargs: fake_env)
    monkeypatch.setitem(sys.modules, 'gymnasium', fake_module)
    environment = GymnasiumEnvironment(
        env_id='Fake-v0',
        tasks=[
            dict(
                task_id='task-a',
                metadata={'reset_options': {
                    'difficulty': 2
                }},
            )
        ],
        success_key='solved',
        squeeze_scalar_action=True,
        scalar_action_type='int',
        task_option_key='task_id',
    )

    environment.setup()
    task = environment.tasks()[0]
    observation, info = environment.reset(
        task,
        episode_index=0,
        environment_seed=11,
        environment_case_index=0,
    )
    transition = environment.step(np.asarray([1.0], dtype=np.float32))

    assert observation.tolist() == [0.0]
    assert info == {
        'reset': True,
        'environment_seed': 11,
        'environment_case_index': 0,
    }
    assert fake_env.reset_call == (11, {'difficulty': 2, 'task_id': 'task-a'})
    assert fake_env.last_action == 1
    assert isinstance(fake_env.last_action, int)
    assert transition.terminated
    assert transition.success is True
    assert environment.render() == 'frame'
    environment.close()
    assert fake_env.closed
