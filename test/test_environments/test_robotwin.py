"""CPU contracts against a fake checkout with RNG-sensitive observations."""

import importlib
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode)
from fluxthemis.environments import (EnvironmentCaseRejected,
                                     RoboTwinEnvironment)
from fluxthemis.environments.sim.robotwin import CAMERA_KEYS, SEED_BLOCK
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import TaskSpec

FAKE_TASK = '''
import numpy as np
from envs.utils.create_actor import UnStableError

class FakeBaseTask:
    failure = None
    success_at = 3
    limit = 5

    def __init__(self):
        self.calls = []
        self.obs_calls = 0

    def setup_demo(self, **args):
        self.calls.append(('setup', args))
        np.random.seed(args['seed'])
        np.random.uniform()
        self.take_action_cnt = 0
        self.step_lim = self.limit
        self.now_obs = {}
        self.eval_success = False
        self.plan_success = self.failure != 'plan'

    def play_once(self):
        self.calls.append(('play',))
        if self.failure == 'unstable':
            raise UnStableError('unstable')
        if self.failure == 'error':
            raise ValueError('broken expert')
        return {'info': {'object': 'bottle'}}

    def check_success(self):
        return self.failure != 'check'

    def close_env(self, **kwargs):
        self.calls.append(('close', kwargs))

    def set_instruction(self, instruction):
        self.calls.append(('instruction', instruction))

    def get_obs(self):
        self.obs_calls += 1
        light = int(np.random.uniform() * 255)
        self.now_obs = dict(
            observation={name: {'rgb': np.full((4, 6, 3), light, np.uint8)}
                         for name in ['head_camera', 'left_camera',
                                      'right_camera']},
            joint_action={'vector': np.arange(14)})
        return self.now_obs

    def take_action(self, action, action_type):
        assert action.shape == (14,) and action_type == 'qpos'
        self.calls.append(('action',))
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at
'''


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'RoboTwin'
    files = {
        'envs/__init__.py':
        f'CONFIGS_PATH = {str(root / "task_config")!r}',
        'envs/_base_task.py':
        FAKE_TASK,
        'envs/utils/__init__.py':
        '',
        'envs/utils/create_actor.py':
        'class UnStableError(Exception): pass',
        'description/utils/generate_episode_instructions.py':
        '''
import random
def generate_episode_descriptions(task, episodes, count):
    assert count == 100 and len(episodes) == 1
    pool = [str(random.random()) for _ in range(count)]
    return [{'unseen': pool, 'seen': pool}]
''',
    }
    for name in ('adjust_bottle', 'click_bell'):
        files[f'envs/{name}.py'] = ('from ._base_task import FakeBaseTask\n'
                                    f'class {name}(FakeBaseTask): pass\n')
    for path, content in files.items():
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    configs = {
        '_eval_step_limit': {
            'adjust_bottle': 5
        },
        '_embodiment_config': {
            'aloha-agilex': {
                'file_path': 'assets/embodiments/aloha-agilex'
            }
        },
        '_camera_config': {
            'test_camera': {
                'h': 4,
                'w': 6
            }
        },
    }
    for setting in ('clean', 'randomized'):
        configs['demo_' + setting] = dict(
            embodiment=['aloha-agilex'],
            clear_cache_freq=5,
            camera={'head_camera_type': 'test_camera'},
            render_freq=10,
            eval_video_log=True,
            eval_video_save_dir='must-not-be-used',
            domain_randomization={'random_light': setting == 'randomized'})
    for name, value in configs.items():
        path = root / 'task_config' / (name + '.yml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value))
    embodiment = root / 'assets/embodiments/aloha-agilex'
    embodiment.mkdir(parents=True)
    (embodiment / 'config.yml').write_text('{}')

    class Engine:

        def create_scene(self, config=None):
            return 'original'

    sapien = SimpleNamespace(
        Engine=Engine,
        Scene=lambda systems: systems,
        physx=SimpleNamespace(
            set_scene_config=lambda c: None, PhysxCpuSystem=lambda: 'cpu'),
        render=SimpleNamespace(
            RenderSystem=lambda device: device, clear_cache=lambda: None))
    monkeypatch.setitem(sys.modules, 'sapien', sapien)
    monkeypatch.setitem(
        sys.modules, 'warp',
        SimpleNamespace(
            config=SimpleNamespace(kernel_cache_dir='original-warp')))
    monkeypatch.setitem(
        sys.modules, 'torch',
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5')
    env = RoboTwinEnvironment(robotwin_root=str(root))
    env.configure_simulator_device('5')
    try:
        yield env
    finally:
        env.close()
        for name in list(sys.modules):
            if name == 'envs' or name.startswith('envs.') or name == (
                    'generate_episode_instructions'):
                sys.modules.pop(name)


def reset(env, case=0, name='clean/adjust_bottle'):
    return env.reset(
        TaskSpec(name), 0, environment_seed=123, environment_case_index=case)


@pytest.mark.parametrize('suite,count,first', [('all', 100, 0),
                                               ('clean', 50, 0),
                                               ('random', 50, 50)])
def test_tasks(suite, count, first):
    tasks = RoboTwinEnvironment(task_suite=suite).tasks()
    assert len(tasks) == count
    assert [t.metadata['task_index']
            for t in tasks] == list(range(first, first + count))
    assert tasks[0].task_id.endswith('/adjust_bottle')
    assert not {'suite', 'benchmark', 'task_suite_name'
                } & tasks[0].metadata.keys()


@pytest.mark.parametrize('kwargs', [{
    'task_suite': 'typo'
}, {
    'instruction_type': 'typo'
}, {
    'robotwin_seed': True
}, {
    'robotwin_seed': -1
}, {
    'instruction_pool_size': 0
}, {
    'max_consecutive_errors': 0
}])
def test_invalid_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RoboTwinEnvironment(**kwargs)


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv('ROBOTWIN_ROOT', raising=False)
    assert RoboTwinEnvironment().robotwin_root == (
        Path(__file__).resolve().parents[3] / 'RoboTwin')
    monkeypatch.setenv('ROBOTWIN_ROOT', str(tmp_path / 'env'))
    assert RoboTwinEnvironment().robotwin_root == tmp_path / 'env'
    env = RoboTwinEnvironment(robotwin_root=str(tmp_path / 'explicit'))
    assert env.robotwin_root == tmp_path / 'explicit'
    with pytest.raises(FileNotFoundError, match='install_robotwin'):
        env.setup()


def test_reset_order_seed_instruction_and_args(runtime):
    runtime.setup()
    obs, info = reset(runtime, 3)
    native = runtime._env
    assert [c[0] for c in native.calls
            ] == ['setup', 'play', 'close', 'setup', 'instruction']
    assert info['robotwin_seed'] == 800003
    assert info['environment_case_index'] == 3
    assert obs['task_description'] == info['task_description']
    for call in (native.calls[0], native.calls[3]):
        args = call[1]
        assert args['seed'] == 800003 and args['is_test']
        assert args['render_freq'] == 0 and args['eval_mode']
        assert not args['eval_video_log']
        assert 'eval_video_save_dir' not in args
        assert args['head_camera_w'] == 6
    assert reset(runtime, 3)[1]['task_description'] == info['task_description']
    with pytest.raises(RuntimeError, match='seed block'):
        reset(runtime, SEED_BLOCK)
    with pytest.raises(TypeError):
        reset(runtime, True)
    with pytest.raises(ValueError):
        reset(runtime, name='other')


@pytest.mark.parametrize('failure', ['unstable', 'plan', 'check', 'error'])
def test_expert_rejection(runtime, failure):
    runtime.setup()
    base = importlib.import_module('envs._base_task').FakeBaseTask
    base.failure = failure
    with pytest.raises(EnvironmentCaseRejected):
        reset(runtime)
    assert not runtime._active
    assert any(c[0] == 'close' for c in runtime._env.calls)


def test_error_limit_and_recovery(runtime):
    runtime.max_consecutive_errors = 2
    runtime.setup()
    base = importlib.import_module('envs._base_task').FakeBaseTask
    base.failure = 'error'
    with pytest.raises(EnvironmentCaseRejected):
        reset(runtime)
    base.failure = None
    reset(runtime, 1)
    assert runtime._consecutive_errors == 0
    base.failure = 'error'
    with pytest.raises(EnvironmentCaseRejected):
        reset(runtime, 2)
    with pytest.raises(RuntimeError, match='broken expert') as error:
        reset(runtime, 3)
    assert isinstance(error.value.__cause__, ValueError)


def test_lazy_observation_and_render(runtime):
    runtime.setup()
    obs, _ = reset(runtime)
    assert len(obs) == 6 and 'qpos' in obs and list(obs)
    assert obs['task_description']
    assert runtime._env.obs_calls == 0
    first = runtime.render()  # The initial frame is also the first prediction.
    for key in CAMERA_KEYS:
        assert obs[key].dtype == np.uint8
    assert obs['qpos'].shape == (14, )
    assert obs['qpos'] is obs['states']
    assert runtime._env.obs_calls == 1
    assert runtime.render() is first
    state = np.random.get_state()
    for _ in range(2):
        transition = runtime.step(np.zeros(14))
        runtime.render()  # Reuse native now_obs between prediction chunks.
    assert runtime._env.obs_calls == 1
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    with pytest.raises(RuntimeError, match='stale'):
        _ = obs['qpos']
    _ = transition.observation['qpos']
    assert runtime._env.obs_calls == 2
    assert runtime.step(np.zeros(14)).success


def test_step_validation_and_truncation(runtime):
    runtime.setup()
    reset(runtime)
    runtime._env.success_at = 100
    for value in (np.zeros(13), np.zeros((1, 14)), np.full(14, np.nan)):
        with pytest.raises(ValueError):
            runtime.step(value)
    for _ in range(runtime._env.step_lim):
        result = runtime.step(np.zeros(14))
    assert result.truncated and not result.terminated and not result.success


def test_cache_cadence_switch_and_close(runtime):
    cwd, paths = Path.cwd(), list(sys.path)
    scene = sys.modules['sapien'].Engine.create_scene
    runtime.setup()
    assert sys.modules['sapien'].Engine().create_scene({}) == ['cpu', 'cuda:0']
    temp = Path(runtime._runtime_dir.name)
    for i in range(5):
        reset(runtime, i)
    native = runtime._env
    reset(runtime, name='random/click_bell')
    closes = [
        c[1]['clear_cache'] for c in native.calls
        if c[0] == 'close' and 'clear_cache' in c[1]
    ]
    assert closes == [False, False, False, True, False]
    runtime.close()
    runtime.close()
    assert not temp.exists()
    assert Path.cwd() == cwd and sys.path == paths
    assert sys.modules['sapien'].Engine.create_scene is scene
    assert sys.modules['warp'].config.kernel_cache_dir == 'original-warp'
    assert 'print' not in vars(importlib.import_module('envs._base_task'))


def test_setup_failure_restores_process(runtime, monkeypatch):
    cwd, paths = Path.cwd(), list(sys.path)
    scene = sys.modules['sapien'].Engine.create_scene
    monkeypatch.setattr(runtime, '_build_args', lambda _: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        runtime.setup()
    assert Path.cwd() == cwd and sys.path == paths
    assert sys.modules['sapien'].Engine.create_scene is scene
    assert runtime._runtime_dir is None


def test_task_switch_releases_previous_instance(runtime, monkeypatch):
    runtime.setup()
    reset(runtime)
    # Native planners can own cycles. The adapter must collect them before
    # allocating the replacement, rather than waiting for a later GC pass.
    runtime._env.cycle = runtime._env
    previous = weakref.ref(runtime._env)
    reset(runtime, 1)
    assert runtime._env is previous()  # Consecutive cases reuse the task.
    cache_releases = []

    def empty_cache():
        assert previous() is None
        cache_releases.append(True)

    monkeypatch.setattr(sys.modules['torch'].cuda, 'is_available',
                        lambda: True)
    monkeypatch.setattr(
        sys.modules['torch'].cuda, 'empty_cache', empty_cache, raising=False)
    replacement = importlib.import_module('envs.click_bell').click_bell
    setup_demo = replacement.setup_demo

    def setup_after_release(self, **kwargs):
        assert previous() is None
        assert cache_releases
        return setup_demo(self, **kwargs)

    monkeypatch.setattr(replacement, 'setup_demo', setup_after_release)
    reset(runtime, name='random/click_bell')
    assert previous() is None
    current = weakref.ref(runtime._env)
    runtime._env.cycle = runtime._env
    runtime.close()
    assert current() is None


def test_partial_planner_initialization_is_not_reused(runtime, monkeypatch):
    runtime.setup()
    base = importlib.import_module('envs._base_task').FakeBaseTask
    setup_demo = base.setup_demo
    partial = []

    def fail_once(self, **kwargs):
        if not partial:
            self.robot = SimpleNamespace(left_planner=object())
            partial.append(weakref.ref(self))
            raise RuntimeError('planner allocation failed')
        assert not hasattr(self, 'robot')
        assert self is not partial[0]()
        return setup_demo(self, **kwargs)

    monkeypatch.setattr(base, 'setup_demo', fail_once)
    with pytest.raises(EnvironmentCaseRejected):
        reset(runtime)
    assert runtime._env is None
    assert runtime._task_name is None
    reset(runtime, 1)
    assert runtime._active


def test_execute_episode(runtime):
    runtime.setup()
    job = build_episode_jobs(
        tasks=[runtime.tasks()[0]],
        episodes_per_task=1,
        base_seed=7,
        run_session_id='robotwin')[0]
    policy = MockPolicyClient(action_dim=14, chunk_size=10)
    result = execute_episode(runtime, policy, job, 2000, 10, True)
    assert result.success and result.steps == 3 and result.model_calls == 1
    runtime._env.failure = 'unstable'
    with pytest.raises(EnvironmentCaseRejected):
        execute_episode(runtime, policy, job, 2000, 10, True)


@pytest.mark.parametrize('save_video', [False, True])
def test_episode_observes_only_at_chunk_boundaries(runtime, monkeypatch,
                                                   tmp_path, save_video):
    from fluxthemis.engines.runners import episode
    from fluxthemis.engines.runners.video import _safe_name

    class Policy(MockPolicyClient):

        def predict_action(self, observation, context):
            # Match the transport's membership checks followed by data reads.
            assert all(key in observation for key in CAMERA_KEYS)
            for key in (*CAMERA_KEYS, 'qpos', 'states'):
                _ = observation[key]
            return super().predict_action(observation, context)

    class Recorder:

        def __init__(self, *args, **kwargs):
            pass

        def start(self, task_id, episode_index):
            assert '/' not in _safe_name(task_id)

        def capture(self, environment):
            assert environment.render().shape == (4, 6, 3)

        def finish(self, success):
            return []

    monkeypatch.setattr(episode, 'EpisodeVideoRecorder', Recorder)
    runtime.setup()
    base = importlib.import_module('envs._base_task').FakeBaseTask
    base.success_at, base.limit = 12, 20
    job = build_episode_jobs(
        tasks=[runtime.tasks()[0]],
        episodes_per_task=1,
        base_seed=7,
        run_session_id='test')[0]
    result = execute_episode(
        runtime,
        Policy(action_dim=14, chunk_size=50),
        job,
        2000,
        10,
        True,
        save_videos=save_video,
        video_dir=tmp_path)
    assert result.steps == 12 and result.model_calls == 2
    assert runtime._env.obs_calls == 2


def test_formal_reset_failure_propagates_and_closes(runtime, monkeypatch):
    runtime.setup()
    base = importlib.import_module('envs._base_task').FakeBaseTask
    setup = base.setup_demo

    def fail_second(self, **kwargs):
        setup(self, **kwargs)
        if len([c for c in self.calls if c[0] == 'setup']) == 2:
            raise ValueError('formal setup failed')

    monkeypatch.setattr(base, 'setup_demo', fail_second)
    with pytest.raises(ValueError, match='formal setup'):
        reset(runtime)
    assert not runtime._active
    assert runtime._env is None


def test_runtime_environment_and_log_restoration(runtime, monkeypatch, capsys):
    monkeypatch.delenv('VK_ICD_FILENAMES', raising=False)
    monkeypatch.setenv('WARP_CACHE_PATH', 'prior-cache')
    monkeypatch.setattr(Path, 'glob', lambda *args: iter(()))
    runtime.setup()
    import os
    manifest = Path(os.environ['VK_ICD_FILENAMES'])
    assert manifest.is_file()
    base = importlib.import_module('envs._base_task')
    base.print('step: quiet', end='\r')
    base.print('visible')
    assert capsys.readouterr().out == 'visible\n'
    runtime.close()
    assert 'VK_ICD_FILENAMES' not in os.environ
    assert os.environ['WARP_CACHE_PATH'] == 'prior-cache'


def test_separate_embodiments_and_device_mapping(runtime, monkeypatch):
    runtime.setup()
    config = runtime.robotwin_root / 'task_config/demo_clean.yml'
    data = yaml.safe_load(config.read_text())
    data['embodiment'] = ['aloha-agilex', 'aloha-agilex', 0.6]
    config.write_text(yaml.safe_dump(data))
    args = runtime._build_args('clean')
    assert not args['dual_arm_embodied'] and args['embodiment_dis'] == 0.6
    data['embodiment'] = []
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='embodiment'):
        runtime._build_args('clean')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5')
    assert runtime._device_id == 1
    runtime.configure_simulator_device('GPU-uuid')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-uuid')
    assert runtime._device_id == 0
