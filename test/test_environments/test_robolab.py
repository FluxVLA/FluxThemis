"""CPU contract checks; the runtime doubles do not simulate physics."""

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from mmengine.config import Config

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.environments.sim import robolab as module
from fluxthemis.environments.sim.robolab import RoboLabEnvironment
from fluxthemis.structures import TaskSpec


@pytest.mark.parametrize('token,visible,expected_cuda',
                         [('2', '2', 0), ('GPU-bb', 'GPU-bb', 0),
                          ('2', '5,2', 1), ('GPU-bb', 'GPU-cc,GPU-bb', 1)])
def test_explicit_render_map_preserves_cuda_numbering(monkeypatch, token,
                                                      visible, expected_cuda):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    env = RoboLabEnvironment('TaskA', render_device_map={token: 4})
    env.configure_simulator_device(token)
    assert env.device_id == expected_cuda
    assert env.render_device_id == 4


@pytest.mark.parametrize('mapping', [{}, {
    '2': -1
}, {
    '2': True
}, {
    2: 1
}, {
    'bad': 1
}, [('2', 1)]])
def test_render_map_rejects_invalid_configuration(mapping):
    with pytest.raises((TypeError, ValueError)):
        RoboLabEnvironment('TaskA', render_device_map=mapping)


def test_render_map_requires_assignment_and_rejects_conflicts(monkeypatch):
    with pytest.raises(ValueError, match='render_device'):
        RoboLabEnvironment(
            'TaskA', render_device_id=1, render_device_map={'2': 4})
    env = RoboLabEnvironment('TaskA', render_device_map={'2': 4})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(ValueError, match='render_device_map'):
        env.configure_simulator_device('3')
    monkeypatch.setattr(module, '_load_prelaunch_runtime',
                        lambda: pytest.fail('Kit must not be imported'))
    with pytest.raises(ValueError, match='assignment'):
        env.setup()


@pytest.mark.parametrize('kit_args', [
    '--portable-root /manual', '--portable-root=/manual',
    '--/app/tokens/omni_global_logs=/manual'
])
def test_managed_kit_cache_rejects_manual_paths(tmp_path, kit_args):
    with pytest.raises(ValueError, match='kit_cache_root'):
        RoboLabEnvironment('TaskA', kit_cache_root=tmp_path, kit_args=kit_args)


def test_managed_kit_cache_rejects_whitespace(tmp_path):
    with pytest.raises(ValueError, match='whitespace'):
        RoboLabEnvironment('TaskA', kit_cache_root=tmp_path / 'has space')


def test_managed_kit_cache_is_ready_before_import_and_worker_isolated(
        runtime, tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    keys = ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
            'OPTIX_CACHE_PATH', '__GL_SHADER_DISK_CACHE_PATH')
    for key in keys:
        monkeypatch.setenv(key, str(tmp_path / 'original' / key))
    original = {key: os.environ[key] for key in keys}
    loader = module._load_prelaunch_runtime
    observed = []

    def checked_loader():
        paths = {key: Path(os.environ[key]) for key in keys}
        assert all(path.is_dir() and root in path.parents
                   for path in paths.values())
        observed.append(paths)
        return loader()

    monkeypatch.setattr(module, '_load_prelaunch_runtime', checked_loader)
    for worker in ('worker-a', 'worker-b'):
        env = RoboLabEnvironment(
            'TaskA',
            output_dir=tmp_path / 'native',
            kit_cache_root=root,
            kit_args='--allow-root')
        env.configure_worker(worker, 'same-run')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
        env.configure_simulator_device('2')
        env.setup()
        args = runtime.launches[-1].kit_args.split()
        portable = Path(args[args.index('--portable-root') + 1])
        assert portable.is_dir() and '--allow-root' in args
        assert root in portable.parents
        assert root not in env._worker_dir.parents
        env.close()
        assert {key: os.environ[key] for key in keys} == original
        assert portable.exists()  # Cleanup is an explicit deployment action.
    assert observed[0]['XDG_CACHE_HOME'] != observed[1]['XDG_CACHE_HOME']


def test_managed_kit_cache_restores_environment_after_launch_failure(
        tmp_path, monkeypatch):
    monkeypatch.delenv('OPTIX_CACHE_PATH', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', 'original-cache')

    def fail_loader():
        assert Path(os.environ['OPTIX_CACHE_PATH']).is_dir()
        raise RuntimeError('test launch failed')

    monkeypatch.setattr(module, '_load_prelaunch_runtime', fail_loader)
    env = RoboLabEnvironment('TaskA', kit_cache_root=tmp_path)
    with pytest.raises(RuntimeError, match='test launch failed'):
        env.setup()
    assert 'OPTIX_CACHE_PATH' not in os.environ
    assert os.environ['XDG_CACHE_HOME'] == 'original-cache'


def _observation(marker=0, dtype=np.float64):
    return {
        'image_obs': {
            'over_shoulder_left_camera':
            np.full((1, 2, 3, 3), marker, dtype=np.uint8),
            'wrist_cam':
            np.full((1, 3, 2, 3), marker + 1, dtype=np.uint8),
        },
        'proprio_obs': {
            'arm_joint_pos': np.arange(7, dtype=dtype)[None] + 0.123456789,
            'gripper_pos': np.asarray([[0.25]], dtype=dtype),
        },
    }


class _NativeEnvironment:

    def __init__(self):
        self.num_envs = 1
        self.device = 'cpu'
        self.action_manager = SimpleNamespace(total_action_dim=8)
        self.max_episode_length = 6
        self.episode_length_buf = np.zeros(1, dtype=np.int64)
        self.all_terminated = False
        self.result = {'env_id': 0, 'success': None, 'step': None}
        self.events = []
        self.actions = []
        self.raw = _observation()
        self.closed = 0
        self.resets = 0
        self.next_success = None
        self.raw_term = False
        self.raw_trunc = False
        self.recorder_manager = SimpleNamespace(
            set_hdf5_file=lambda name: self.events.append(('file', name)),
            set_episode_index=lambda index, **kwargs: self.events.append(
                ('index', index, kwargs)))

    def reset_eval_state(self):
        self.events.append('clear')
        self.all_terminated = False
        self.result = {'env_id': 0, 'success': None, 'step': None}

    def reset(self, **kwargs):
        self.events.append(('reset', kwargs))
        self.resets += 1
        self.episode_length_buf[:] = 0
        self.raw = _observation(self.resets)
        return self.raw, {}

    def get_env_results(self):
        return [self.result.copy()]

    def step(self, action):
        self.actions.append(action.copy())
        self.episode_length_buf += 1
        if self.next_success is not None:
            self.all_terminated = True
            self.result = dict(env_id=0, success=self.next_success, step=3)
        return (self.raw, np.asarray([0.125]), np.asarray([self.raw_term]),
                np.asarray([self.raw_trunc]), {})

    def close(self):
        self.closed += 1


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        created=[],
        app_closed=0,
        launches=[],
        registrations=[],
        output_dirs=[],
        events=[])

    def close_app():
        state.app_closed += 1

    def launcher(args, **kwargs):
        assert kwargs == {'multi_gpu': False}
        state.events.append('launch')
        state.launches.append(args)
        return SimpleNamespace(app=SimpleNamespace(close=close_app))

    launcher.add_app_launcher_args = lambda parser: None

    class TaskAConfig:

        def __init__(self):
            self.sim = SimpleNamespace(dt=1 / 120)
            self.decimation = 8
            self.instruction = {'default': '  Pick A.\n'}

    class TaskBConfig:

        def __init__(self):
            self.sim = SimpleNamespace(dt=1 / 120)
            self.decimation = 8
            self.instruction = {'default': 'Pick B.'}

    configs = {
        'TaskA': TaskAConfig,
        'TaskB': TaskBConfig,
    }

    def create_env(scene, **kwargs):
        env = _NativeEnvironment()
        state.created.append((scene, kwargs, env))
        return env, configs[scene]()

    def register(**kwargs):
        state.registrations.append(kwargs)

    factory = SimpleNamespace(
        get_envs=lambda task=None: list(configs) if task is None else task,
        get_global_env_factory=lambda: SimpleNamespace(
            get_env_info=lambda name: dict(
                task_name=name, config_class=configs[name])))
    state.runtime = SimpleNamespace(
        create_env=create_env,
        factory=factory,
        register=register,
        constants=SimpleNamespace(set_output_dir=state.output_dirs.append),
        instruction_text=lambda fields: fields['default'],
        select_instruction_fields=lambda fields, requested:
        {requested: fields[requested]},
        torch=SimpleNamespace(
            float32=np.float32,
            as_tensor=lambda values, dtype, device: np.asarray(
                values, dtype=dtype),
            inference_mode=contextlib.nullcontext))

    def postlaunch():
        state.events.append('postlaunch')
        return state.runtime

    monkeypatch.setattr(module, '_load_prelaunch_runtime', lambda: launcher)
    monkeypatch.setattr(module, '_load_postlaunch_runtime', postlaunch)
    state.metrics_calls = []

    def metrics(path, dt, success, steps):
        state.metrics_calls.append((path, dt, success, steps))
        return (1.0 if success else 0.5), dict(
            metrics={
                'ee_speed_mean': 0.03,
                'ee_sparc': -8.0
            },
            metrics_complete=True,
            metrics_compute_s=0.01)

    monkeypatch.setattr(module, 'collect_episode_metrics', metrics)
    return state


@pytest.fixture
def environment(runtime, tmp_path):
    env = RoboLabEnvironment(
        task_name=['TaskA', 'TaskB'], simulator_seed=5, output_dir=tmp_path)
    env.configure_worker('worker-0', 'run-1')
    env.setup()
    yield env
    env.close()


def _reset(env, index=0, task=0):
    return env.reset(
        env.tasks()[task],
        index,
        environment_seed=11 + index,
        environment_case_index=index)


def test_reuse_switch_seed_and_reset_order(environment, runtime):
    env = environment
    obs, first = _reset(env)
    native = runtime.created[0][2]
    assert obs['over_shoulder_left_camera'][0, 0, 0] == 2
    assert native.events[:3] == ['clear', ('reset', {}), ('reset', {})]
    assert native.events[3:] == [('file', 'run_0.hdf5'),
                                 ('index', 0, {
                                     'env_ids': [0]
                                 })]
    native.all_terminated = True
    _, second = _reset(env, 1)
    assert len(runtime.created) == 1
    assert not native.all_terminated
    assert native.events[5:8] == ['clear', ('reset', {}), ('reset', {})]
    assert ('file', 'run_1.hdf5') in native.events
    assert first['environment_seed'] == 11
    assert second['environment_seed'] == 12
    assert first['simulator_seed'] == second['simulator_seed'] == 5
    assert first['environment_case_index'] == 0
    assert second['environment_case_index'] == 1
    assert first['environment_instance'] == second['environment_instance']
    assert first['instance_episode'] == 0
    assert second['instance_episode'] == 1
    _, switched = _reset(env, 0, task=1)
    assert native.closed == 1
    assert len(runtime.created) == 2
    assert switched['environment_instance'] != first['environment_instance']
    assert switched['instance_episode'] == 0
    assert runtime.created[0][1]['seed'] == 5
    assert runtime.created[1][1]['seed'] == 5
    assert runtime.created[0][1]['num_envs'] == 1
    assert len(set(runtime.output_dirs)) == 2
    assert runtime.events == ['launch', 'postlaunch']
    env.setup()
    assert len(runtime.launches) == 1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_observation_preserves_values_and_owns_memory(environment, runtime,
                                                      dtype):
    _reset(environment)
    native = runtime.created[0][2]
    native.raw = _observation(dtype=dtype)
    original = native.raw['proprio_obs']['arm_joint_pos'].copy()
    obs = environment.step(np.zeros(8)).observation
    assert obs['states'].dtype == dtype
    np.testing.assert_array_equal(obs['states'][:7], original[0])
    assert obs['states'][7] == 0.25
    assert obs['task_description'] == '  Pick A.\n'
    assert obs['wrist_cam'].shape == (3, 2, 3)
    assert obs['wrist_cam'].dtype == np.uint8
    native.raw['proprio_obs']['arm_joint_pos'][:] = 99
    native.raw['image_obs']['wrist_cam'][:] = 99
    np.testing.assert_array_equal(obs['states'][:7], original[0])
    assert obs['wrist_cam'][0, 0, 0] == 1
    frame = environment.render()
    frame[:] = 99
    assert obs['over_shoulder_left_camera'][0, 0, 0] == 0


@pytest.mark.parametrize('gripper', [0.0, 0.5, 0.5001, 1.0])
def test_absolute_actions_are_passed_once(environment, runtime, gripper):
    _reset(environment)
    action = np.arange(8, dtype=np.float64) + 0.25
    action[7] = gripper
    result = environment.step(action)
    received = runtime.created[0][2].actions[0]
    assert received.shape == (1, 8)
    assert received.dtype == np.float32
    np.testing.assert_array_equal(received[0], action.astype(np.float32))
    assert result.reward == 0.125
    assert result.success is None
    assert result.score is None


@pytest.mark.parametrize('action, error', [
    (np.zeros((1, 8)), ValueError),
    (np.zeros(7), ValueError),
    (np.zeros(8, dtype=bool), TypeError),
    ([0.] * 7 + [True], TypeError),
    (np.full(8, np.nan), ValueError),
    (np.full(8, np.inf), ValueError),
    (['0'] * 8, TypeError),
    (np.zeros(8, dtype=complex), TypeError),
    (np.full(8, 1e100), ValueError),
])
def test_bad_action_never_reaches_sim(environment, runtime, action, error):
    _reset(environment)
    with pytest.raises(error):
        environment.step(action)
    assert runtime.created[0][2].actions == []


@pytest.mark.parametrize('success', [True, False])
def test_native_aggregate_is_authoritative(environment, runtime, success):
    _reset(environment)
    native = runtime.created[0][2]
    native.raw_term = True
    early = environment.step(np.zeros(8))
    assert early.info['raw_terminated'] is True
    assert early.success is None
    assert not early.terminated and not early.truncated
    assert native.resets == 2  # No wrapper reset for an artifact signal.
    native.next_success = success
    native.raw_term = False
    native.raw_trunc = not success
    final = environment.step(np.zeros(8))
    assert final.success is success
    assert final.terminated is success
    assert final.truncated is (not success)
    assert final.info['native_result']['step'] == 3
    assert final.info['wrapper_steps'] == 2
    with pytest.raises(RuntimeError, match='ended'):
        environment.step(np.zeros(8))


def test_missing_frozen_result_is_error(environment, runtime):
    _reset(environment)
    runtime.created[0][2].all_terminated = True
    with pytest.raises(RuntimeError, match='result'):
        environment.step(np.zeros(8))
    with pytest.raises(RuntimeError):
        environment.step(np.zeros(8))


def test_outer_native_budget_without_result_is_error(environment, runtime):
    _reset(environment)
    for _ in range(5):
        environment.step(np.zeros(8))
    with pytest.raises(RuntimeError, match='limit'):
        environment.step(np.zeros(8))
    assert len(runtime.created[0][2].actions) == 6


@pytest.mark.parametrize('group, key, value, error', [
    ('image_obs', 'wrist_cam', None, KeyError),
    ('image_obs', 'wrist_cam', np.zeros((1, 2, 3, 4), np.uint8), ValueError),
    ('image_obs', 'wrist_cam', np.zeros((1, 2, 3, 3)), ValueError),
    ('proprio_obs', 'arm_joint_pos', np.zeros((1, 6)), ValueError),
    ('proprio_obs', 'gripper_pos', np.asarray([[np.nan]]), ValueError),
])
def test_bad_observation_is_rejected(environment, runtime, group, key, value,
                                     error):
    _reset(environment)
    native = runtime.created[0][2]
    if value is None:
        del native.raw[group][key]
    else:
        native.raw[group][key] = value
    with pytest.raises(error):
        environment.step(np.zeros(8))


def test_setup_failure_closes_app(runtime, tmp_path):

    def fail(**kwargs):
        raise RuntimeError('registration failed')

    runtime.runtime.register = fail
    env = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    with pytest.raises(RuntimeError, match='registration failed'):
        env.setup()
    env.close()
    assert runtime.app_closed == 1


def test_activation_failure_closes_native_and_app(runtime, tmp_path):
    original = runtime.runtime.create_env

    def invalid(*args, **kwargs):
        env, cfg = original(*args, **kwargs)
        env.num_envs = 2
        return env, cfg

    runtime.runtime.create_env = invalid
    env = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    env.setup()
    with pytest.raises(ValueError, match='num_envs'):
        _reset(env)
    assert runtime.created[0][2].closed == 1
    assert runtime.app_closed == 1
    env.close()
    assert runtime.app_closed == 1


def test_close_is_idempotent_and_state_guards(environment, runtime):
    with pytest.raises(RuntimeError, match='reset'):
        environment.step(np.zeros(8))
    with pytest.raises(KeyError):
        environment.reset(
            TaskSpec('unknown'),
            0,
            environment_seed=0,
            environment_case_index=0)
    assert not runtime.created
    _reset(environment)
    environment.close()
    environment.close()
    assert runtime.created[0][2].closed == 1
    assert runtime.app_closed == 1
    with pytest.raises(RuntimeError, match='setup'):
        environment.tasks()


def test_optional_dependencies_and_config_in_fresh_process():
    root = Path(__file__).resolve().parents[2]
    script = '''
import importlib.abc
import os
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            # MMEngine logging probes Torch and handles ImportError itself.
            raise ModuleNotFoundError('Torch is unavailable', name=fullname)
        if fullname.split('.')[0] in {
            'robolab', 'isaaclab', 'isaacsim', 'omni', 'carb', 'openpi',
            'cv2', 'libero', 'robosuite', 'robocasa', 'sapien'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
before = dict(os.environ)
import fluxthemis
from fluxthemis.engines.utils import ENVIRONMENTS
from mmengine.config import Config
cfg = Config.fromfile('configs/robolab/droid.py')
env = ENVIRONMENTS.build(cfg.themis.evaluation.environment)
assert type(env).__name__ == 'RoboLabEnvironment'
assert before == dict(os.environ)
for name in ['LiberoEnvironment', 'RoboCasaEnvironment',
             'RoboDojoEnvironment', 'RoboTwinEnvironment']:
    assert ENVIRONMENTS.get(name) is not None
old = ENVIRONMENTS.build(dict(type='MockEnvironment', success_after=1))
old.setup()
old.reset(old.tasks()[0], 0, environment_seed=0, environment_case_index=0)
assert old.step([0.0]).success
old.close()
gym = ENVIRONMENTS.build(dict(type='GymnasiumEnvironment',
                             env_id='FrozenLake-v1',
                             make_kwargs=dict(is_slippery=False)))
gym.setup()
gym.reset(gym.tasks()[0], 0, environment_seed=0, environment_case_index=0)
gym.step(0)
gym.close()
'''
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=root,
                            env=dict(os.environ),
                            text=True,
                            capture_output=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_contract():
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(str(root / 'configs/robolab/droid.py'))
    env = ENVIRONMENTS.build(cfg.themis.evaluation.environment)
    assert isinstance(env, RoboLabEnvironment)
    assert env.task_names == ('all', )
    assert cfg.themis.evaluation.episodes_per_task == 10
    assert cfg.themis.transport.state_keys == ['states']
    assert cfg.themis.transport.unnorm_key == 'robolab_droid'
    assert cfg.themis.evaluation.max_episode_steps >= 4500
    assert cfg.themis.evaluation.execute_horizon is None


def test_prelaunch_import_order_and_missing_dependency(monkeypatch):
    imports = []

    def load(name):
        imports.append(name)

    # Match the packaged IsaacLab bootstrap: the class is registered as a
    # from-import entry, without an attribute on the parent module.
    app = ModuleType('isaaclab.app')
    monkeypatch.setitem(sys.modules, 'isaaclab.app', app)
    monkeypatch.setitem(sys.modules, 'isaaclab.app.AppLauncher', 'launcher')

    monkeypatch.setattr(module.importlib, 'import_module', load)
    assert module._load_prelaunch_runtime() == 'launcher'
    assert imports == ['cv2']
    assert not hasattr(app, 'AppLauncher')

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError, match='docs/robolab.md'):
        module._load_prelaunch_runtime()


def test_postlaunch_failure_releases_app(runtime, monkeypatch, tmp_path):

    def fail():
        raise ModuleNotFoundError('RoboLab is missing')

    monkeypatch.setattr(module, '_load_postlaunch_runtime', fail)
    env = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    with pytest.raises(ModuleNotFoundError):
        env.setup()
    assert runtime.app_closed == 1
    env.close()
    assert runtime.app_closed == 1


def test_worker_output_is_unique(runtime, tmp_path):
    env1 = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    env2 = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    try:
        env1.setup()
        env2.setup()
        _, first = _reset(env1)
        _, second = _reset(env2)
        assert first['native_output_dir'] != second['native_output_dir']
    finally:
        env1.close()
        env2.close()


def test_camera_mapping(runtime, tmp_path):
    env = RoboLabEnvironment(
        task_name='TaskA',
        output_dir=tmp_path,
        camera_keys={'front': 'wrist_cam'})
    try:
        env.setup()
        obs, _ = _reset(env)
        assert set(obs) == {'front', 'states', 'task_description'}
        assert obs['front'].shape == (3, 2, 3)
    finally:
        env.close()


def test_native_close_error_still_closes_app(environment, runtime):
    _reset(environment)

    def fail():
        raise RuntimeError('native close failed')

    runtime.created[0][2].close = fail
    with pytest.raises(RuntimeError, match='native close failed'):
        environment.close()
    assert runtime.app_closed == 1
    environment.close()
    assert runtime.app_closed == 1


def test_revisit_creates_new_instance(environment, runtime):
    _, first = _reset(environment)
    _reset(environment, task=1)
    _, revisited = _reset(environment)
    assert [entry[0]
            for entry in runtime.created] == ['TaskA', 'TaskB', 'TaskA']
    assert runtime.created[0][2].closed == 1
    assert runtime.created[1][2].closed == 1
    assert revisited['native_output_dir'] != first['native_output_dir']
    assert revisited['instance_episode'] == 0
    assert runtime.created[-1][1]['seed'] == 5


def test_task_ids_use_native_registration(runtime, tmp_path):
    original_factory = runtime.runtime.factory.get_global_env_factory()
    runtime.runtime.factory.get_envs = lambda task: ['TaskAVariant']
    runtime.runtime.factory.get_global_env_factory = lambda: SimpleNamespace(
        get_env_info=lambda name: original_factory.get_env_info('TaskA'))
    create = runtime.runtime.create_env
    requests = []

    def create_variant(scene, **kwargs):
        requests.append(scene)
        return create('TaskA', **kwargs)

    runtime.runtime.create_env = create_variant
    env = RoboLabEnvironment(task_name='TaskA', output_dir=tmp_path)
    try:
        env.setup()
        task = env.tasks()[0]
        assert task.task_id == 'TaskAVariant'
        assert task.metadata['task_name'] == 'TaskA'
        _reset(env)
        assert requests == ['TaskAVariant']
    finally:
        env.close()


def test_reset_failure_releases_resources(environment, runtime):
    _reset(environment)

    def fail():
        raise RuntimeError('reset failed')

    runtime.created[0][2].reset = fail
    with pytest.raises(RuntimeError, match='reset failed'):
        _reset(environment, 1)
    assert runtime.created[0][2].closed == 1
    assert runtime.app_closed == 1


def test_source_path_is_added_only_at_setup_and_removed(runtime, tmp_path):
    root = tmp_path / 'checkout'
    (root / 'robolab').mkdir(parents=True)
    env = RoboLabEnvironment(
        task_name='TaskA', output_dir=tmp_path, robolab_root=root)
    assert str(root) not in sys.path
    env.setup()
    assert str(root) in sys.path
    env.close()
    assert str(root) not in sys.path


def test_render_device_is_overridden_before_app_creation():
    allocated = []

    class Launcher:

        def __init__(self, args, **kwargs):
            self._sim_app_config = dict(active_gpu=0, physics_gpu=0, **kwargs)
            self._create_app()

        def _create_app(self):
            allocated.append(self._sim_app_config.copy())

    module._launch_app(Launcher, SimpleNamespace(), 6)
    assert allocated == [
        dict(active_gpu=6, physics_gpu=0, multi_gpu=False, max_gpu_count=1)
    ]
    module._launch_app(Launcher, SimpleNamespace(), None)
    assert allocated[-1] == dict(active_gpu=0, physics_gpu=0, multi_gpu=False)


def test_render_device_rejects_unknown_launcher_interface():
    with pytest.raises(RuntimeError, match='interface'):
        module._launch_app(lambda args: None, SimpleNamespace(), 6)


def test_kit_arguments_reach_launcher(runtime, tmp_path):
    env = RoboLabEnvironment(
        task_name='TaskA',
        output_dir=tmp_path,
        kit_args='--/plugins/carb.tasking.plugin/threadCount=8')
    try:
        env.setup()
        assert runtime.launches[0].kit_args == env.kit_args
    finally:
        env.close()


@pytest.mark.parametrize('value', [False, True, 0, -1, 2.5, '6'])
def test_cpu_thread_limit_validation(value):
    with pytest.raises((TypeError, ValueError), match='cpu_thread_limit'):
        RoboLabEnvironment(task_name='TaskA', cpu_thread_limit=value)


@pytest.mark.parametrize('render_device', [None, 6])
def test_cpu_limit_reaches_simulation_app_before_allocation(render_device):
    allocated = []

    class Launcher:

        def __init__(self, args, **kwargs):
            self._sim_app_config = dict(active_gpu=0, physics_gpu=0, **kwargs)
            self._create_app()

        def _create_app(self):
            allocated.append(self._sim_app_config.copy())

    module._launch_app(Launcher, SimpleNamespace(), render_device, 6)
    assert allocated[0]['limit_cpu_threads'] == 6
    assert allocated[0]['active_gpu'] == (render_device or 0)
    assert allocated[0]['physics_gpu'] == 0
    assert allocated[0]['multi_gpu'] is False


def test_cpu_limit_is_optional_and_preserves_default():
    assert RoboLabEnvironment(task_name='TaskA').cpu_thread_limit is None
    assert RoboLabEnvironment(
        task_name='TaskA', cpu_thread_limit=6).cpu_thread_limit == 6


def test_cpu_limit_rejects_unknown_launcher_interface():
    with pytest.raises(RuntimeError, match='interface'):
        module._launch_app(lambda args: None, SimpleNamespace(), None, 6)


def test_terminal_score_and_metrics_are_collected_once(environment, runtime):
    _reset(environment)
    native = runtime.created[0][2]
    assert environment.step(np.zeros(8)).score is None
    assert runtime.metrics_calls == []
    native.next_success = False
    result = environment.step(np.zeros(8))
    assert result.score == 0.5 and result.success is False
    assert result.info['metrics']['ee_speed_mean'] == 0.03
    assert len(runtime.metrics_calls) == 1
    path, dt, success, steps = runtime.metrics_calls[0]
    assert path.endswith('run_0.hdf5') and dt == 1 / 15
    assert not success and steps == 3


@pytest.mark.parametrize('artifact_step', [1, 2])
@pytest.mark.parametrize('eventual_success', [False, True])
def test_internal_early_reset_never_extends_outer_budget(
        environment, runtime, monkeypatch, artifact_step, eventual_success):
    _reset(environment)
    native = runtime.created[0][2]
    original = native.step

    def step(action):
        result = original(action)
        if len(native.actions) == artifact_step:
            native.episode_length_buf[:] = 0
            return (*result[:2], np.array([True]), np.array([False]), {})
        return result

    monkeypatch.setattr(native, 'step', step)
    for _ in range(5):
        environment.step(np.zeros(8))
    if eventual_success:
        native.next_success = True
        result = environment.step(np.zeros(8))
        assert result.success and result.info['wrapper_steps'] == 6
    else:
        with pytest.raises(
                RuntimeError, match='native_episode_step=.*raw_terminated'):
            environment.step(np.zeros(8))
        assert environment._steps == 6
        assert native.episode_length_buf[0] == 6 - artifact_step


@pytest.mark.parametrize('value', [True, False, 0, -1, 2, 10])
def test_single_environment_rejected_before_runtime(value):
    with pytest.raises((ValueError, TypeError), match='num_envs'):
        RoboLabEnvironment(task_name='all', num_envs=value)


def test_two_tasks_ten_scalar_episodes_keep_native_rules(environment, runtime):
    outcomes = []
    for task_index in range(2):
        for episode in range(10):
            _reset(environment, episode, task_index)
            native = runtime.created[-1][2]
            native.next_success = episode % 2 == 0
            result = environment.step(np.zeros(8))
            outcomes.append(result.success)
            assert result.score == (1.0 if result.success else 0.5)
    assert outcomes.count(True) == outcomes.count(False) == 10
    assert len(runtime.launches) == 1
    assert len(runtime.created) == 2
    assert all(item[1]['num_envs'] == 1 for item in runtime.created)
    assert [item[2].resets for item in runtime.created] == [20, 20]
    assert len(runtime.metrics_calls) == 20
    assert len({str(item[0]) for item in runtime.metrics_calls}) == 20


def test_full_scalar_manifest_keeps_120_times_ten_jobs():
    from fluxthemis.engines.runners.episode import build_episode_jobs
    tasks = tuple(TaskSpec(f'task-{i}') for i in range(120))
    jobs = build_episode_jobs(tasks, 10, 0, 'scalar-full')
    assert len(jobs) == 1200
    assert len({job.episode_id for job in jobs}) == 1200
    assert all(
        sum(job.task.task_id == task.task_id for job in jobs) == 10
        for task in tasks)
