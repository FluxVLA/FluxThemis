from types import SimpleNamespace

import numpy as np

from fluxthemis.environments.sim import libero as libero_module
from fluxthemis.environments.sim.libero import (LIBERO_DUMMY_ACTION,
                                                LiberoEnvironment)


def _observation(marker):
    return {
        'agentview_image': np.full((2, 3, 3), marker, dtype=np.uint8),
        'state': np.asarray([marker], dtype=np.float32),
    }


class _NativeEnvironment:

    def __init__(self):
        self.seed_value = None
        self.initial_state = None
        self.actions = []
        self.closed = False

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        return _observation(-1)

    def set_init_state(self, value):
        self.initial_state = value
        return _observation(0)

    def step(self, action):
        self.actions.append(list(action))
        done = len(self.actions) == 2
        return _observation(len(self.actions)), float(done), done, {}

    def close(self):
        self.closed = True


def test_libero_episode_contract(monkeypatch):
    task = SimpleNamespace(
        name='pick_up_the_black_bowl',
        language='pick up the black bowl',
        problem_folder='libero_10',
        bddl_file='task.bddl',
    )
    suite = SimpleNamespace(
        n_tasks=1,
        get_task=lambda index: task,
        get_task_init_states=lambda index: ('state-0', 'state-1'),
    )
    native = _NativeEnvironment()
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {
        'libero_10': lambda task_order_index: suite,
    })
    runtime = SimpleNamespace(
        benchmark=benchmark,
        libero=SimpleNamespace(get_libero_path=lambda key: '/bddl'),
        envs=SimpleNamespace(OffScreenRenderEnv=lambda **kwargs: native),
    )
    monkeypatch.setattr(libero_module, '_load_libero_runtime', lambda: runtime)

    environment = LiberoEnvironment(
        task_suite_name='libero_10', simulator_seed=5, num_steps_wait=1)
    environment.setup()
    exposed_task = environment.tasks()[0]
    observation, info = environment.reset(
        exposed_task,
        episode_index=0,
        environment_seed=11,
        environment_case_index=3,
    )
    transition = environment.step(np.arange(7, dtype=np.float32))

    assert exposed_task.description == 'pick up the black bowl'
    assert native.seed_value == 5
    assert native.initial_state == 'state-1'
    assert native.actions[0] == list(LIBERO_DUMMY_ACTION)
    assert native.actions[1] == list(np.arange(7, dtype=np.float32))
    assert info['init_state_index'] == 1
    assert observation['state'].tolist() == [1.0]
    assert transition.success is True

    environment.close()
    assert native.closed
