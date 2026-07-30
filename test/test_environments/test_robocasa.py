from types import SimpleNamespace

import numpy as np
import pytest

from fluxthemis.environments.sim import robocasa as robocasa_module
from fluxthemis.environments.sim.robocasa import (ROBOCASA_ACTION_DIM,
                                                  ROBOCASA_ACTION_ORDERS,
                                                  RoboCasaEnvironment,
                                                  robocasa_group_name)


def _observation(marker=0, prompt='unlocked_waist: move the bottle'):
    image = np.full((4, 5, 3), marker, dtype=np.uint8)
    return {
        'video.ego_view_pad_res256_freq20': image,
        'video.ego_view_bg_crop_pad_res256_freq20': image.copy(),
        'state.left_arm': np.arange(7, dtype=np.float64),
        'state.left_hand': np.arange(6, dtype=np.float64),
        'state.right_arm': np.arange(7, dtype=np.float64),
        'state.right_hand': np.arange(6, dtype=np.float64),
        'state.waist': np.arange(3, dtype=np.float64),
        'annotation.human.coarse_action': prompt,
        'untouched': {
            'marker': marker
        },
    }


class _FakeEnvironment:

    def __init__(self, env_name, kwargs, reset_result=None):
        self.env_name = env_name
        self.kwargs = dict(kwargs)
        self.reset_result = reset_result or (_observation(), {
            'success': False
        })
        self.reset_calls = []
        self.step_calls = []
        self.step_results = []
        self.close_calls = 0

    def reset(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return self.reset_result

    def step(self, action):
        self.step_calls.append(action)
        if self.step_results:
            return self.step_results.pop(0)
        return _observation(1), 0.0, False, False, {'success': False}

    def close(self):
        self.close_calls += 1


class _FakeGymnasium:

    def __init__(self):
        self.instances = []
        self.next_reset_result = None

    def make(self, env_name, **kwargs):
        environment = _FakeEnvironment(
            env_name, kwargs, reset_result=self.next_reset_result)
        self.next_reset_result = None
        self.instances.append(environment)
        return environment


@pytest.fixture
def fake_runtime(monkeypatch):
    gymnasium = _FakeGymnasium()
    runtime = robocasa_module._RoboCasaRuntime(
        gymnasium=gymnasium,
        robocasa=SimpleNamespace(),
        gym_utils=SimpleNamespace(GrootRoboCasaEnv=object),
    )
    monkeypatch.setattr(robocasa_module, '_load_robocasa_runtime',
                        lambda: runtime)
    return SimpleNamespace(gymnasium=gymnasium, runtime=runtime)


def _environment(fake_runtime, **kwargs):
    environment = RoboCasaEnvironment(
        task_list=[
            'gr1_unified/PnPBottleToCabinetClose_Env',
            'gr1_unified/PosttrainNovelToPlate_Env',
        ],
        **kwargs,
    )
    environment.setup()
    return environment


def test_setup_exposes_stable_task_manifest_without_creating_mujoco(
        fake_runtime):
    environment = _environment(fake_runtime, action_order='n15')

    environment.setup()
    tasks = environment.tasks()

    assert [task.task_id for task in tasks] == ['0', '1']
    assert tasks[0].description == \
        'gr1_unified/PnPBottleToCabinetClose_Env'
    assert tasks[0].metadata == {
        'benchmark': 'robocasa',
        'task_suite_name': 'robocasa',
        'task_index': 0,
        'task_name': 'gr1_unified/PnPBottleToCabinetClose_Env',
        'env_name': 'gr1_unified/PnPBottleToCabinetClose_Env',
        'group': 'Cabinet',
        'prompt_source': 'annotation.human.coarse_action',
        'action_order': 'n15',
    }
    assert tasks[1].metadata['group'] == 'Generalization'
    assert fake_runtime.gymnasium.instances == []


def test_reset_recreates_seeded_environment_and_preserves_raw_observation(
        fake_runtime):
    source = _observation(
        3, prompt='unlocked_waist: put the mug in the drawer')
    fake_runtime.gymnasium.next_reset_result = (source, {'success': False})
    environment = _environment(
        fake_runtime, env_kwargs={'enable_render': True})

    observation, info = environment.reset(
        environment.tasks()[0], episode_index=2, seed=19)

    simulator = fake_runtime.gymnasium.instances[0]
    assert simulator.env_name == \
        'gr1_unified/PnPBottleToCabinetClose_Env'
    assert simulator.kwargs == {'enable_render': True, 'seed': 19}
    assert simulator.reset_calls == [{'seed': 19}]
    assert observation is not source
    assert observation['untouched'] is source['untouched']
    assert observation['state.left_arm'] is source['state.left_arm']
    assert info['task_description'] == \
        'unlocked_waist: put the mug in the drawer'
    assert info['episode_index'] == 2
    assert info['episode_seed'] == 19
    assert info['action_order'] == 'fluxvla'

    environment.reset(environment.tasks()[1], episode_index=0, seed=20)
    assert simulator.close_calls == 1
    assert len(fake_runtime.gymnasium.instances) == 2


@pytest.mark.parametrize('action_order', ['fluxvla', 'n15'])
def test_step_splits_denormalized_action_in_configured_order(
        fake_runtime, action_order):
    environment = _environment(fake_runtime, action_order=action_order)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    simulator = fake_runtime.gymnasium.instances[0]
    success_observation = _observation(
        9, prompt='unlocked_waist: finish the task')
    simulator.step_results.append((success_observation, 1.0, False, False, {
        'success': True
    }))
    action = np.arange(ROBOCASA_ACTION_DIM, dtype=np.float64)

    transition = environment.step(action)

    forwarded = simulator.step_calls[0]
    assert set(forwarded) == set(ROBOCASA_ACTION_ORDERS[action_order])
    for key, (start, end) in ROBOCASA_ACTION_ORDERS[action_order].items():
        np.testing.assert_array_equal(forwarded[key],
                                      action[start:end].astype(np.float32))
    assert transition.success is True
    assert transition.terminated is False
    assert transition.info['task_description'] == \
        'unlocked_waist: move the bottle'
    np.testing.assert_array_equal(
        environment.render(),
        success_observation['video.ego_view_pad_res256_freq20'])


@pytest.mark.parametrize(
    'action,exception,message',
    [
        (np.zeros(ROBOCASA_ACTION_DIM - 1), ValueError, 'shape'),
        (np.zeros((1, ROBOCASA_ACTION_DIM)), ValueError, 'shape'),
        (['bad'] * ROBOCASA_ACTION_DIM, TypeError, 'numeric'),
        ([False] * ROBOCASA_ACTION_DIM, TypeError, 'numeric'),
        ([0.0] * (ROBOCASA_ACTION_DIM - 1) + [np.nan], ValueError, 'NaN'),
        ([0.0] * (ROBOCASA_ACTION_DIM - 1) + [np.inf], ValueError, 'infinity'),
    ],
)
def test_step_rejects_invalid_actions(fake_runtime, action, exception,
                                      message):
    environment = _environment(fake_runtime)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    simulator = fake_runtime.gymnasium.instances[0]

    with pytest.raises(exception, match=message):
        environment.step(action)

    assert simulator.step_calls == []


def test_reset_requires_dynamic_prompt_and_closes_partial_environment(
        fake_runtime):
    fake_runtime.gymnasium.next_reset_result = (_observation(prompt=''), {})
    environment = _environment(fake_runtime)

    with pytest.raises(ValueError, match='coarse_action'):
        environment.reset(environment.tasks()[0], episode_index=0, seed=7)

    assert fake_runtime.gymnasium.instances[0].close_calls == 1


def test_close_is_idempotent(fake_runtime):
    environment = _environment(fake_runtime)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    simulator = fake_runtime.gymnasium.instances[0]

    environment.close()
    environment.close()

    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match=r'setup\(\) was not called'):
        environment.tasks()


def test_missing_runtime_has_actionable_installer_message(monkeypatch):
    monkeypatch.setattr(robocasa_module, 'configure_mujoco_egl_defaults',
                        lambda: None)

    def missing_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(robocasa_module.importlib, 'import_module',
                        missing_import)

    with pytest.raises(
            ModuleNotFoundError, match=r'bash scripts/install_robocasa\.sh'):
        robocasa_module._load_robocasa_runtime()


@pytest.mark.parametrize(
    'name,group',
    [
        ('A/PnPToCabinet', 'Cabinet'),
        ('A/PnPToDrawer', 'Drawer'),
        ('A/PnPToMicrowave', 'Microwave'),
        ('A/PnPToBowl', 'Generalization'),
    ],
)
def test_native_group_mapping(name, group):
    assert robocasa_group_name(name) == group
