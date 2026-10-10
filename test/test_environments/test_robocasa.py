from types import SimpleNamespace

import numpy as np
import pytest

from fluxthemis.environments.sim import robocasa as robocasa_module
from fluxthemis.environments.sim.robocasa import (ROBOCASA_ACTION_DIM,
                                                  ROBOCASA_ACTION_ORDERS,
                                                  RoboCasaEnvironment)


def _observation(prompt='unlocked_waist: move the bottle'):
    return {
        'video.ego_view_pad_res256_freq20': np.zeros((2, 3, 3),
                                                     dtype=np.uint8),
        'annotation.human.coarse_action': prompt,
    }


class _NativeEnvironment:

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.reset_kwargs = None
        self.action = None
        self.closed = False

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        return _observation(), {}

    def step(self, action):
        self.action = action
        return _observation('changed prompt'), 1.0, True, False, {
            'success': True
        }

    def close(self):
        self.closed = True


@pytest.mark.parametrize('action_order', ['fluxvla', 'n15'])
def test_robocasa_episode_contract(monkeypatch, action_order):
    native_environments = []

    def make(_env_name, **kwargs):
        native = _NativeEnvironment(kwargs)
        native_environments.append(native)
        return native

    runtime = SimpleNamespace(
        gymnasium=SimpleNamespace(make=make),
        robocasa=SimpleNamespace(),
        gym_utils=SimpleNamespace(GrootRoboCasaEnv=object),
    )
    monkeypatch.setattr(robocasa_module, '_load_robocasa_runtime',
                        lambda: runtime)
    environment = RoboCasaEnvironment(
        task_list=['gr1_unified/PnPBottleToCabinetClose_Env'],
        action_order=action_order,
    )

    environment.setup()
    task = environment.tasks()[0]
    _, info = environment.reset(
        task,
        episode_index=0,
        environment_seed=19,
        environment_case_index=0,
    )
    action = np.arange(ROBOCASA_ACTION_DIM, dtype=np.float32)
    transition = environment.step(action)
    native = native_environments[0]

    assert native.kwargs == {'seed': 19}
    assert native.reset_kwargs == {'seed': 19}
    for key, (start, end) in ROBOCASA_ACTION_ORDERS[action_order].items():
        np.testing.assert_array_equal(native.action[key], action[start:end])
    assert info['task_description'] == 'unlocked_waist: move the bottle'
    assert transition.info['task_description'] == info['task_description']
    assert transition.success is True

    environment.close()
    assert native.closed
