"""RoboTwin 2.0 adapter; simulator dependencies are imported only in setup."""

from __future__ import annotations

import builtins
import copy
import gc
import importlib
import json
import logging
import os
import random
import sys
import tempfile
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import yaml

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment, EnvironmentCaseRejected

ROBOTWIN_TASKS = (
    'adjust_bottle', 'beat_block_hammer', 'blocks_ranking_rgb',
    'blocks_ranking_size', 'click_alarmclock', 'click_bell', 'dump_bin_bigbin',
    'grab_roller', 'handover_block', 'handover_mic', 'hanging_mug', 'lift_pot',
    'move_can_pot', 'move_pillbottle_pad', 'move_playingcard_away',
    'move_stapler_pad', 'open_laptop', 'open_microwave',
    'pick_diverse_bottles', 'pick_dual_bottles', 'place_a2b_left',
    'place_a2b_right', 'place_bread_basket', 'place_bread_skillet',
    'place_burger_fries', 'place_can_basket', 'place_cans_plasticbox',
    'place_container_plate', 'place_dual_shoes', 'place_empty_cup',
    'place_fan', 'place_mouse_pad', 'place_object_basket',
    'place_object_scale', 'place_object_stand', 'place_phone_stand',
    'place_shoe', 'press_stapler', 'put_bottles_dustbin', 'put_object_cabinet',
    'rotate_qrcode', 'scan_object', 'shake_bottle',
    'shake_bottle_horizontally', 'stack_blocks_three', 'stack_blocks_two',
    'stack_bowls_three', 'stack_bowls_two', 'stamp_seal', 'turn_switch')
ROBOTWIN_SETTINGS = {'clean': 'demo_clean', 'random': 'demo_randomized'}
ROBOTWIN_SUITES = {
    'clean': ('clean', ),
    'random': ('random', ),
    'all': ('clean', 'random')
}
CAMERA_KEYS = {
    'cam_high': 'head_camera',
    'cam_left_wrist': 'left_camera',
    'cam_right_wrist': 'right_camera',
}
SEED_BLOCK = 100000
ACTION_DIM = 14
_LOG = logging.getLogger(__name__)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def _quiet_print(*args, **kwargs):
    if (args and isinstance(args[0], str) and args[0].startswith('step:')
            and kwargs.get('end') == '\r'):
        return
    builtins.print(*args, **kwargs)


def _yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


class _RoboTwinObservation(Mapping):
    """Read one simulator observation per action chunk, never per step."""

    KEYS = (*CAMERA_KEYS, 'qpos', 'states', 'task_description')

    def __init__(self, owner, generation):
        self.owner = owner
        self.generation = generation

    def __iter__(self):
        return iter(self.KEYS)

    def __len__(self):
        return len(self.KEYS)

    def __contains__(self, key):
        return key in self.KEYS

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        if self.generation != self.owner._obs_generation:
            raise RuntimeError('stale RoboTwin observation')
        if key == 'task_description':
            return self.owner._instruction
        return self.owner._observation()[key]


@ENVIRONMENTS.register_module()
class RoboTwinEnvironment(BaseEnvironment):
    """Evaluate expert-validated cases with RoboTwin's native action loop."""

    def __init__(self,
                 task_suite='all',
                 robotwin_root=None,
                 robotwin_seed=7,
                 instruction_type='unseen',
                 instruction_pool_size=100,
                 max_consecutive_errors=20,
                 quiet_step_logs=True):
        if task_suite not in ROBOTWIN_SUITES:
            raise ValueError('task_suite must be clean, random, or all')
        if instruction_type not in ('seen', 'unseen'):
            raise ValueError('instruction_type must be seen or unseen')
        if not isinstance(quiet_step_logs, bool):
            raise TypeError('quiet_step_logs must be a boolean')
        self.task_suite = task_suite
        self.robotwin_root = Path(
            robotwin_root or os.environ.get('ROBOTWIN_ROOT')
            or Path(__file__).resolve().parents[4] /
            'RoboTwin').expanduser().resolve()
        self.robotwin_seed = _integer(robotwin_seed, 'robotwin_seed')
        self.instruction_type = instruction_type
        self.instruction_pool_size = _integer(instruction_pool_size,
                                              'instruction_pool_size', 1)
        self.max_consecutive_errors = _integer(max_consecutive_errors,
                                               'max_consecutive_errors', 1)
        self.quiet_step_logs = quiet_step_logs
        self.worker_id = 'local'
        self.device_id = '0'
        self.save_videos = False
        self.video_dir = None
        self._ready = False
        self._active = False
        self._env = None
        self._task_name = None
        self._base_args = {}
        self._accepted_episodes = 0
        self._consecutive_errors = 0
        self._obs_generation = 0
        self._obs_cache = None
        self._cwd = None
        self._sys_path = None
        self._runtime_dir = None
        self._restore_env = {}
        self._warp = None
        self._sapien = None
        self._base_task_module = None

    def tasks(self):
        return tuple(
            TaskSpec(
                task_id=f'{setting}/{task}',
                description=f'{task} ({setting})',
                metadata=dict(
                    robotwin_setting=setting,
                    robotwin_task=task,
                    robotwin_task_config=ROBOTWIN_SETTINGS[setting],
                    task_index=(0 if setting == 'clean' else 50) + index))
            for setting in ROBOTWIN_SUITES[self.task_suite]
            for index, task in enumerate(ROBOTWIN_TASKS))

    def configure_worker(self, worker_id, run_id):
        self.worker_id = str(worker_id)

    def configure_simulator_device(self, device_id):
        self.device_id = str(device_id)

    def configure_video_recording(self, enabled, output_dir):
        # The shared recorder owns files; there is no native writer.
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be a boolean')
        self.save_videos = enabled
        self.video_dir = output_dir

    @property
    def _device_id(self):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        devices = [v.strip() for v in visible.split(',') if v.strip()]
        if devices and devices != ['-1']:
            if len(devices) == 1:
                return 0
            if self.device_id not in devices:
                raise ValueError(f'GPU {self.device_id} is not visible')
            return devices.index(self.device_id)
        return _integer(int(self.device_id), 'device_id')

    def _setenv(self, key, value):
        self._restore_env.setdefault(key, os.environ.get(key))
        os.environ[key] = value

    def setup(self):
        if self._ready:
            return
        required = ('envs/_base_task.py', 'task_config/demo_clean.yml',
                    'task_config/demo_randomized.yml',
                    'task_config/_eval_step_limit.yml',
                    'description/utils/generate_episode_instructions.py',
                    'assets/embodiments')
        missing = [
            p for p in required if not (self.robotwin_root / p).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f'Incomplete RoboTwin checkout {self.robotwin_root}: '
                f'{missing}.'
                ' Run bash scripts/install_robotwin.sh')
        try:
            self._setup_runtime()
            self._ready = True
        except BaseException:
            self.close()
            raise

    def _setup_runtime(self):
        root = self.robotwin_root
        self._runtime_dir = tempfile.TemporaryDirectory(
            prefix='fluxthemis-robotwin-')
        runtime = Path(self._runtime_dir.name)
        icd_dirs = ('/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d')
        if not os.environ.get('VK_ICD_FILENAMES') and not any(
                list(Path(p).glob('nvidia*.json')) for p in icd_dirs):
            manifest = runtime / 'nvidia_icd.json'
            manifest.write_text(
                json.dumps(
                    dict(
                        file_format_version='1.0.0',
                        ICD=dict(
                            library_path='libGLX_nvidia.so.0',
                            api_version='1.2.175'))),
                encoding='utf-8')
            self._setenv('VK_ICD_FILENAMES', str(manifest))
        cache = runtime / 'warp'
        cache.mkdir()
        self._setenv('WARP_CACHE_PATH', str(cache))
        try:
            warp = importlib.import_module('warp')
        except ImportError:
            pass
        else:
            self._warp_cache = warp.config.kernel_cache_dir
            self._warp = warp
            warp.config.kernel_cache_dir = str(cache)
        self._cwd = Path.cwd()
        self._sys_path = list(sys.path)
        paths = [
            str(root),
            str(root / 'policy'),
            str(root / 'description/utils')
        ]
        sys.path[:] = paths + [p for p in sys.path if p not in paths]
        os.chdir(root)
        try:
            torch = importlib.import_module('torch')
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.set_device(self._device_id)
        sapien = importlib.import_module('sapien')
        self._create_scene = sapien.Engine.create_scene
        self._sapien = sapien
        device = f'cuda:{self._device_id}'

        def create_scene(engine, config=None):
            if config is not None:
                sapien.physx.set_scene_config(config)
            return sapien.Scene([
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem(device)
            ])

        sapien.Engine.create_scene = create_scene
        importlib.import_module('envs')
        self._unstable_error = importlib.import_module(
            'envs.utils.create_actor').UnStableError
        self._descriptions = importlib.import_module(
            'generate_episode_instructions').generate_episode_descriptions
        base_task = importlib.import_module('envs._base_task')
        if self.quiet_step_logs:
            self._base_task_module = base_task
            self._original_print = base_task.__dict__.get('print')
            base_task.print = _quiet_print
        for setting in ROBOTWIN_SUITES[self.task_suite]:
            self._base_args[setting] = self._build_args(setting)

    def _build_args(self, setting):
        config_dir = self.robotwin_root / 'task_config'
        task_config = ROBOTWIN_SETTINGS[setting]
        args = _yaml(config_dir / f'{task_config}.yml')
        args.update(
            task_config=task_config,
            ckpt_setting='fluxthemis',
            policy_name='fluxthemis',
            eval_mode=True,
            render_freq=0,
            eval_video_log=False)
        args.pop('eval_video_save_dir', None)
        types = args['embodiment']
        if len(types) not in (1, 3):
            raise ValueError('embodiment must contain 1 or 3 items')
        embodiment_cfg = _yaml(config_dir / '_embodiment_config.yml')
        args['dual_arm_embodied'] = len(types) == 1
        if len(types) == 3:
            args['embodiment_dis'] = types[2]
        robot_types = (types[0], types[0] if len(types) == 1 else types[1])
        for side, kind in zip(('left', 'right'), robot_types):
            path = Path(embodiment_cfg[kind]['file_path'])
            if not path.is_absolute():
                path = self.robotwin_root / path
            args[f'{side}_robot_file'] = str(path)
            args[f'{side}_embodiment_config'] = _yaml(path / 'config.yml')
        cameras = _yaml(config_dir / '_camera_config.yml')
        camera = cameras[args['camera']['head_camera_type']]
        args['head_camera_h'], args['head_camera_w'] = camera['h'], camera['w']
        _integer(args['clear_cache_freq'], 'clear_cache_freq', 1)
        return args

    def _invalidate_observation(self):
        self._obs_generation += 1
        self._obs_cache = None

    def _reject(self, reason, seed):
        _LOG.warning('[RoboTwin] worker=%s rejected seed=%s: %s',
                     self.worker_id, seed, reason)
        raise EnvironmentCaseRejected(f'{reason}; robotwin_seed={seed}')

    def _close_expert(self):
        if self._env is None:
            return
        try:
            self._env.close_env()
        except Exception:
            _LOG.exception('RoboTwin expert cleanup failed')

    def _release_task(self):
        # Drop scene/planner owners before collecting cycles and cached GPU
        # allocations. Doing this before the next task avoids overlapping two
        # tasks' planners; empty_cache alone cannot free referenced tensors.
        self._env = None
        self._task_name = None
        self._invalidate_observation()
        gc.collect()
        if self._sapien is not None:
            self._sapien.render.clear_cache()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self, task, episode_index, *, environment_seed,
              environment_case_index):
        if not self._ready:
            raise RuntimeError('RoboTwinEnvironment.setup() was not called')
        canonical = {t.task_id: t for t in self.tasks()}.get(task.task_id)
        if canonical is None:
            raise ValueError(f'Unknown RoboTwin task: {task.task_id}')
        case = _integer(environment_case_index, 'environment_case_index')
        if case >= SEED_BLOCK:
            raise RuntimeError('RoboTwin seed block exhausted; inspect cases')
        self._finish_active_episode()
        self._invalidate_observation()
        setting = canonical.metadata['robotwin_setting']
        name = canonical.metadata['robotwin_task']
        if name != self._task_name:
            # Upstream close_env() leaves scene/planner references on the task.
            # Reuse consecutive episodes, but do not retain every visited task.
            self._release_task()
            module = importlib.import_module(f'envs.{name}')
            self._env = getattr(module, name)()
            self._task_name = name
        seed = SEED_BLOCK * (1 + self.robotwin_seed) + case

        def setup_demo():
            args = copy.deepcopy(self._base_args[setting])
            args['task_name'] = name
            try:
                self._env.setup_demo(
                    now_ep_num=0, seed=seed, is_test=True, **args)
            except BaseException:
                # A failed planner construction can leave robot present but
                # without left/right_planner. Never reuse that partial task.
                self._close_expert()
                self._release_task()
                raise

        try:
            setup_demo()
            expert_info = self._env.play_once()
            self._env.close_env()
        except self._unstable_error:
            self._close_expert()
            self._reject('unstable seed', seed)
        except Exception as exc:
            self._close_expert()
            self._consecutive_errors += 1
            _LOG.exception('RoboTwin expert error at seed %s', seed)
            if self._consecutive_errors >= self.max_consecutive_errors:
                raise RuntimeError(
                    f'RoboTwin expert failed {self._consecutive_errors} times '
                    f'at seed {seed}: {exc}') from exc
            self._reject('expert error', seed)
        if not (self._env.plan_success and self._env.check_success()):
            self._reject('expert plan failed', seed)
        self._consecutive_errors = 0
        try:
            setup_demo()
            random.seed(seed)
            descriptions = self._descriptions(name, [expert_info['info']],
                                              self.instruction_pool_size)
            self._instruction = str(
                np.random.choice(descriptions[0][self.instruction_type]))
            self._env.set_instruction(instruction=self._instruction)
        except BaseException:
            self._close_expert()
            raise
        self._active = True
        self._accepted_episodes += 1
        self._clear_cache_freq = self._base_args[setting]['clear_cache_freq']
        self._info = dict(
            task_description=self._instruction,
            robotwin_seed=seed,
            robotwin_setting=setting,
            robotwin_task=name,
            step_lim=self._env.step_lim,
            take_action_cnt=0,
            environment_case_index=case)
        print(
            f'[RoboTwin] worker={self.worker_id} task={task.task_id} '
            f'episode={episode_index} case={case} seed={seed} '
            f'step_lim={self._env.step_lim} instruction={self._instruction}',
            flush=True)
        return _RoboTwinObservation(self,
                                    self._obs_generation), dict(self._info)

    def step(self, action):
        if not self._active:
            raise RuntimeError('RoboTwin reset() must be called before step()')
        action = np.asarray(action)
        if action.shape != (ACTION_DIM, ) or not np.isfinite(action).all():
            raise ValueError('RoboTwin action must be finite and shaped (14,)')
        self._env.take_action(action, action_type='qpos')
        self._invalidate_observation()
        success = bool(self._env.eval_success)
        truncated = (not success
                     and self._env.take_action_cnt >= self._env.step_lim)
        self._info['take_action_cnt'] = self._env.take_action_cnt
        return StepResult(
            observation=_RoboTwinObservation(self, self._obs_generation),
            reward=float(success),
            terminated=success,
            truncated=truncated,
            success=success,
            info=dict(self._info))

    @staticmethod
    def _image(value):
        value = np.ascontiguousarray(value, dtype=np.uint8)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError('RoboTwin image must have shape HxWx3')
        return value

    def _observation(self):
        if not self._active:
            raise RuntimeError('RoboTwin has no active episode')
        if self._obs_cache is None:
            observation = self._env.get_obs()
            state = np.asarray(
                observation['joint_action']['vector'], dtype=np.float32)
            if state.shape != (ACTION_DIM, ) or not np.isfinite(state).all():
                raise ValueError(
                    'RoboTwin qpos must be finite and shaped (14,)')
            self._obs_cache = {
                key: self._image(observation['observation'][source]['rgb'])
                for key, source in CAMERA_KEYS.items()
            }
            self._obs_cache.update(qpos=state, states=state)
        return self._obs_cache

    def render(self):
        if not self._active:
            return None
        if self._obs_cache is not None:
            return self._obs_cache['cam_high']
        if self._env.now_obs:
            return self._image(
                self._env.now_obs['observation']['head_camera']['rgb'])
        return self._observation()['cam_high']

    def _finish_active_episode(self):
        if self._active:
            self._active = False
            self._invalidate_observation()
            try:
                self._env.close_env(
                    clear_cache=((self._accepted_episodes + 1) %
                                 self._clear_cache_freq == 0))
            except Exception:
                _LOG.exception('RoboTwin episode cleanup failed')

    def close(self):
        self._finish_active_episode()
        self._release_task()
        if self._sapien is not None:
            self._sapien.Engine.create_scene = self._create_scene
            self._sapien = None
        if self._base_task_module is not None:
            if self._original_print is None:
                self._base_task_module.__dict__.pop('print', None)
            else:
                self._base_task_module.print = self._original_print
            self._base_task_module = None
        if self._cwd is not None:
            os.chdir(self._cwd)
            self._cwd = None
        if self._sys_path is not None:
            sys.path[:] = self._sys_path
            self._sys_path = None
        if self._warp is not None:
            self._warp.config.kernel_cache_dir = self._warp_cache
            self._warp = None
        for key, value in self._restore_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._restore_env.clear()
        if self._runtime_dir is not None:
            self._runtime_dir.cleanup()
            self._runtime_dir = None
        self._base_args.clear()
        self._ready = False
