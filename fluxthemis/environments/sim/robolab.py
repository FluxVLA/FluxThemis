"""RoboLab DROID joint-position adapter with optional simulator imports.

RoboLab and Isaac Sim are loaded only by setup(), in application launch order.
Model preprocessing and action-chunk execution belong to the policy side and
the shared episode runner, respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment
from .robolab_metrics import collect_episode_metrics


class NonFiniteObservationError(ValueError):
    """Nonfinite simulator state, distinct from shape errors."""

    def __init__(self, field, lane, shape, invalid_indices):
        super().__init__(f'{field} contains nonfinite values in lane {lane}')
        self.details = dict(
            field=field,
            lane=int(lane),
            shape=list(shape),
            invalid_indices=invalid_indices)


def _load_prelaunch_runtime():
    try:
        # Required by RoboLab before importing IsaacLab / starting Kit.
        importlib.import_module('cv2')
        # The packaged IsaacLab bootstrap exposes this from-import API without
        # necessarily setting AppLauncher as an attribute of isaaclab.app.
        from isaaclab.app import AppLauncher
        return AppLauncher
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'RoboLabEnvironment requires the RoboLab Isaac Sim environment; '
            'see docs/robolab.md for the optional runtime requirements.'
        ) from exc


def _load_postlaunch_runtime():
    try:
        constants = importlib.import_module('robolab.constants')
        factory = importlib.import_module('robolab.core.environments.factory')
        runtime = importlib.import_module('robolab.core.environments.runtime')
        instructions = importlib.import_module('robolab.core.task.instruction')
        registration = importlib.import_module(
            'robolab.registrations.droid.auto_env_registrations_jointpos')
        torch = importlib.import_module('torch')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'RoboLabEnvironment requires an installed RoboLab checkout and '
            'its simulator dependencies; see docs/robolab.md.') from exc
    return SimpleNamespace(
        constants=constants,
        factory=factory,
        create_env=runtime.create_env,
        register=registration.auto_register_droid_envs,
        torch=torch,
        instruction_text=instructions.instruction_text,
        select_instruction_fields=instructions.select_instruction_fields)


def _launch_app(launcher, args, render_device_id, cpu_thread_limit=None):
    """Keep one worker on one GPU, allowing distinct CUDA/Vulkan ordinals."""
    if render_device_id is None and cpu_thread_limit is None:
        return launcher(args, multi_gpu=False)
    if not hasattr(launcher, '_create_app'):
        raise RuntimeError('Unsupported IsaacLab app creation interface')

    class RenderDeviceLauncher(launcher):

        def _create_app(self):
            # IsaacLab resolves active_gpu from the CUDA ordinal, overwriting
            # a constructor argument. Override it after that resolution and
            # before SimulationApp allocates graphics resources.
            config = getattr(self, '_sim_app_config', None)
            if (not isinstance(config, dict)
                    or not {'active_gpu', 'physics_gpu'} <= config.keys()):
                raise RuntimeError('Unsupported IsaacLab GPU configuration')
            if render_device_id is not None:
                config.update(
                    active_gpu=render_device_id,
                    multi_gpu=False,
                    max_gpu_count=1)
            if cpu_thread_limit is not None:
                # Apply the native limit to Kit, TBB and USD/OpenBLAS.
                config['limit_cpu_threads'] = cpu_thread_limit
            return super()._create_app()

    return RenderDeviceLauncher(args, multi_gpu=False)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    if value < 0:
        raise ValueError(f'{name} must be non-negative')
    return int(value)


def _gpu_token(value):
    return isinstance(value, str) and (value.isdecimal() or re.fullmatch(
        r'GPU-[A-Za-z0-9-]+', value))


def _numpy(value):
    """Detach native tensors without a module-level Torch dependency."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value, name):
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f'{name} must describe one native environment')
    return array.reshape(-1)[0].item()


@ENVIRONMENTS.register_module()
class RoboLabEnvironment(BaseEnvironment):
    """DROID adapter using the official scalar lifecycle.

    A worker owns one Isaac application. Consecutive episodes of the same task
    reuse the native environment; changing tasks closes and recreates it.
    ``simulator_seed`` applies at creation, while ``environment_seed`` and
    ``environment_case_index`` identify scheduled episodes, not reset layouts.
    """

    def __init__(self,
                 task_name: str | Sequence[str],
                 robolab_root: str | os.PathLike | None = None,
                 output_dir: str | os.PathLike = 'work_dirs/robolab',
                 simulator_seed: int = 0,
                 device_id: int = 0,
                 render_device_id: int | None = None,
                 kit_args: str = '',
                 cpu_thread_limit: int | None = None,
                 headless: bool = True,
                 num_envs: int = 1,
                 camera_keys: Mapping[str, str] | None = None,
                 state_keys: Sequence[str] = ('arm_joint_pos', 'gripper_pos'),
                 renderer: str = 'realtime',
                 rendering_mode: str | None = None,
                 render_device_map: Mapping[str, int] | None = None,
                 kit_cache_root: str | os.PathLike | None = None) -> None:
        names = [task_name] if isinstance(task_name, str) else list(task_name)
        if not names or any(not isinstance(n, str) or not n.strip()
                            for n in names):
            raise ValueError('task_name must contain nonempty task names')
        if len(set(names)) != len(names) or ('all' in names
                                             and len(names) > 1):
            raise ValueError('task_name cannot mix all or duplicate names')
        self.task_names = tuple(names)
        self.simulator_seed = _nonnegative_integer(simulator_seed,
                                                   'simulator_seed')
        self.device_id = _nonnegative_integer(device_id, 'device_id')
        self.render_device_id = (None if render_device_id is None
                                 else _nonnegative_integer(
                                     render_device_id, 'render_device_id'))
        if render_device_map is not None:
            if render_device_id is not None:
                raise ValueError('Use render_device_map or render_device_id, '
                                 'not both')
            if not isinstance(render_device_map,
                              Mapping) or not render_device_map:
                raise ValueError(
                    'render_device_map must be a nonempty mapping')
            if any(not _gpu_token(key) for key in render_device_map):
                raise ValueError('render_device_map keys must be GPU index '
                                 'strings or UUIDs')
            render_device_map = {
                key: _nonnegative_integer(value, 'render_device_map value')
                for key, value in render_device_map.items()
            }
        self.render_device_map = render_device_map
        self._assigned_device = None
        if not isinstance(kit_args, str):
            raise TypeError('kit_args must be a string')
        self.kit_args = kit_args
        self.kit_cache_root = (None if kit_cache_root is None else
                               Path(kit_cache_root).expanduser().resolve())
        if self.kit_cache_root is not None:
            # The pinned IsaacLab launcher splits kit_args on whitespace.
            if any(char.isspace() for char in str(self.kit_cache_root)):
                raise ValueError('kit_cache_root must not contain whitespace')
            managed = {'--portable-root', '--/app/tokens/omni_global_logs'}
            if any(
                    arg.split('=', 1)[0] in managed
                    for arg in kit_args.split()):
                raise ValueError(
                    'kit_cache_root conflicts with manual Kit paths')
        self._cache_environment = {}
        self._kit_cache_dir = None
        self.cpu_thread_limit = (None if cpu_thread_limit is None
                                 else _nonnegative_integer(
                                     cpu_thread_limit, 'cpu_thread_limit'))
        if self.cpu_thread_limit == 0:
            raise ValueError('cpu_thread_limit must be positive')
        self.num_envs = _nonnegative_integer(num_envs, 'num_envs')
        if self.num_envs != 1:
            raise ValueError('num_envs must be 1')
        if not isinstance(headless, bool):
            raise TypeError('headless must be a boolean')
        self.headless = headless
        self.robolab_root = robolab_root
        self.output_dir = Path(output_dir).expanduser()
        self.camera_keys = dict(
            camera_keys if camera_keys is not None else {
                'over_shoulder_left_camera': 'over_shoulder_left_camera',
                'wrist_cam': 'wrist_cam',
            })
        if (not self.camera_keys or any(not isinstance(key, str) or not key
                                        for pair in self.camera_keys.items()
                                        for key in pair)
                or {'states', 'task_description'} & self.camera_keys.keys()):
            raise ValueError('camera_keys must map image names to native keys')
        self.state_keys = tuple(state_keys)
        if (len(self.state_keys) != 2
                or any(not isinstance(key, str) or not key
                       for key in self.state_keys)
                or self.state_keys[0] == self.state_keys[1]):
            raise ValueError('state_keys must name the arm and gripper fields')
        if renderer not in ('realtime', 'pathtracing'):
            raise ValueError('renderer must be realtime or pathtracing')
        if rendering_mode not in (None, 'performance', 'balanced', 'quality'):
            raise ValueError('unsupported rendering_mode')
        self.renderer = renderer
        self.rendering_mode = rendering_mode
        self.worker_id = 'standalone'
        self.run_id = 'run'
        self._runtime = None
        self._app = None
        self._added_path = None
        self._tasks = ()
        self._task_ids = set()
        self._env = None
        self._active_task_id = None
        self._last_observation = None
        self._episode_active = False
        self._instance = 0
        self._instance_episode = 0
        self._worker_dir = None
        self._episode_info = {}

    def configure_worker(self, worker_id: str, run_id: str) -> None:
        self._require_unstarted()
        self.worker_id = str(worker_id)
        self.run_id = str(run_id)

    def configure_simulator_device(self, device_id: str) -> None:
        self._require_unstarted()
        token = str(device_id)
        if not _gpu_token(token):
            raise ValueError('device_id must be a GPU index or UUID')
        if (self.render_device_map is not None
                and token not in self.render_device_map):
            raise ValueError(
                f'Assigned GPU {token!r} is absent from render_device_map')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            devices = visible.split(',')
            if token not in devices or len(set(devices)) != len(devices):
                raise ValueError('Assigned GPU is not uniquely present in '
                                 'CUDA_VISIBLE_DEVICES')
            # The supervisor isolates its assigned physical index/UUID before
            # this callback. CUDA numbers devices within that visible list.
            self.device_id = devices.index(token)
        elif token.isdecimal():
            self.device_id = int(token)
        else:
            raise ValueError('GPU UUID assignment requires '
                             'CUDA_VISIBLE_DEVICES')
        if self.render_device_map is not None:
            self.render_device_id = self.render_device_map[token]
        self._assigned_device = token

    def _prepare_kit_runtime(self):
        """Isolate writable Kit paths before importing its optional runtime."""
        if self.kit_cache_root is None:
            return self.kit_args
        self.kit_cache_root.mkdir(parents=True, exist_ok=True)
        prefix = re.sub(r'[^A-Za-z0-9_.-]', '_',
                        f'{self.run_id}_{self.worker_id}')[:100]
        root = Path(
            tempfile.mkdtemp(prefix=prefix + '_', dir=self.kit_cache_root))
        self._kit_cache_dir = root
        paths = {
            'XDG_CACHE_HOME': 'cache',
            'XDG_CONFIG_HOME': 'config',
            'XDG_DATA_HOME': 'data',
            'OPTIX_CACHE_PATH': 'optix',
            '__GL_SHADER_DISK_CACHE_PATH': 'shader',
        }
        for key, name in paths.items():
            path = root / name
            path.mkdir()
            self._cache_environment[key] = (os.environ.get(key), str(path))
            os.environ[key] = str(path)
        portable, logs = root / 'kit-portable', root / 'logs'
        portable.mkdir()
        logs.mkdir()
        return (f'{self.kit_args} --portable-root {portable} '
                f'--/app/tokens/omni_global_logs={logs}').strip()

    def _restore_cache_environment(self):
        for key, (previous, applied) in self._cache_environment.items():
            if os.environ.get(key) != applied:
                continue
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._cache_environment.clear()

    def setup(self) -> None:
        if self._runtime is not None:
            return
        try:
            if (self.render_device_map is not None
                    and self._assigned_device is None):
                raise ValueError(
                    'render_device_map requires a worker GPU assignment')
            kit_args = self._prepare_kit_runtime()
            root = self.robolab_root or os.environ.get('ROBOLAB_ROOT')
            if root is not None:
                root = str(Path(root).expanduser().resolve())
                if not (Path(root) / 'robolab').is_dir():
                    raise FileNotFoundError(
                        f'Invalid RoboLab checkout: {root}')
                if root not in sys.path:
                    sys.path.insert(0, root)
                    self._added_path = root
            launcher = _load_prelaunch_runtime()
            parser = argparse.ArgumentParser(add_help=False)
            launcher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless = self.headless
            args.enable_cameras = True
            args.device = f'cuda:{self.device_id}'
            args.kit_args = kit_args
            self._app = _launch_app(launcher, args, self.render_device_id,
                                    self.cpu_thread_limit).app
            runtime = _load_postlaunch_runtime()
            selection = (None if self.task_names == ('all', ) else list(
                self.task_names))
            runtime.register(
                task_dirs=['benchmark'],
                task=selection,
                randomize_background=False)
            factory = runtime.factory.get_global_env_factory()
            tasks = []
            for name in runtime.factory.get_envs(task=selection):
                details = factory.get_env_info(name)
                if details is None:
                    raise ValueError(f'RoboLab has no metadata for {name}')
                # Official generated configs assign instruction in
                # __post_init__, so read it from a config instance.
                task_config = details['config_class']()
                fields = runtime.select_instruction_fields(
                    task_config.instruction, 'default')
                tasks.append(
                    TaskSpec(
                        task_id=name,
                        description=runtime.instruction_text(fields),
                        metadata=dict(
                            benchmark='robolab',
                            task_name=details['task_name'],
                            robot='droid',
                            action_mode='jointpos')))
            if not tasks:
                raise ValueError('No RoboLab tasks matched task_name')
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = re.sub(r'[^A-Za-z0-9_.-]', '_',
                            f'{self.run_id}_{self.worker_id}')[:100]
            self._worker_dir = Path(
                tempfile.mkdtemp(
                    prefix=prefix + '_', dir=self.output_dir.resolve()))
            self._tasks = tuple(tasks)
            self._task_ids = {task.task_id for task in tasks}
            self._runtime = runtime
        except Exception:
            logging.getLogger(__name__).exception('RoboLab setup failed')
            self.close()
            raise

    def tasks(self) -> tuple[TaskSpec, ...]:
        self._require_setup()
        return self._tasks

    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[dict, dict]:
        self._require_setup()
        if not isinstance(task, TaskSpec):
            raise TypeError('task must be a TaskSpec')
        if task.task_id not in self._task_ids:
            raise KeyError(f'Unknown RoboLab task id: {task.task_id}')
        episode_index = _nonnegative_integer(episode_index, 'episode_index')
        if environment_case_index is not None:
            environment_case_index = _nonnegative_integer(
                environment_case_index, 'environment_case_index')
        if environment_seed is not None:
            environment_seed = _nonnegative_integer(environment_seed,
                                                    'environment_seed')
        self._episode_active = False
        self._last_observation = None
        try:
            if task.task_id != self._active_task_id:
                self._activate_task(task.task_id)
            # Official runner clears between runs; run_episode resets twice.
            # Neither reset reseeds or treats a case index as a layout index.
            with self._runtime.torch.inference_mode():
                self._env.reset_eval_state()
                self._env.reset()
                raw, _ = self._env.reset()
            recorder = self._env.recorder_manager
            if recorder is not None and hasattr(recorder, 'set_hdf5_file'):
                recorder.set_hdf5_file(f'run_{self._instance_episode}.hdf5')
                recorder.set_episode_index(0, env_ids=[0])
            observation = self._observation(raw)
            self._steps = 0
            self._episode_info = dict(
                benchmark='robolab',
                task_id=task.task_id,
                episode_index=episode_index,
                environment_seed=environment_seed,
                environment_case_index=environment_case_index,
                simulator_seed=self.simulator_seed,
                environment_instance=self._instance,
                instance_episode=self._instance_episode,
                native_hdf5_path=str(self._native_dir /
                                     f'run_{self._instance_episode}.hdf5'),
                initial_observation_sha256={
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in observation.items()
                    if isinstance(value, np.ndarray)
                },
                native_output_dir=str(self._native_dir),
                native_max_episode_steps=self._max_steps)
            self._instance_episode += 1
            self._last_observation = observation
            self._episode_active = True
            return observation, dict(self._episode_info)
        except Exception:
            self.close()
            raise

    def _activate_task(self, task_id):
        self._close_native()
        self._instance += 1
        # IDs remain in metadata; a numeric directory also avoids unsafe paths.
        self._native_dir = self._worker_dir / f'env_{self._instance:04d}'
        self._runtime.constants.set_output_dir(str(self._native_dir))
        self._env, cfg = self._runtime.create_env(
            task_id,
            device=f'cuda:{self.device_id}',
            seed=self.simulator_seed,
            num_envs=self.num_envs,
            instruction_type='default',
            renderer=self.renderer,
            rendering_mode=self.rendering_mode)
        if self._env.num_envs != self.num_envs:
            raise ValueError('RoboLab native num_envs differs from num_envs')
        if self._env.action_manager.total_action_dim != 8:
            raise ValueError('DROID jointpos must expose 8 action values')
        self._max_steps = _nonnegative_integer(self._env.max_episode_length,
                                               'max_episode_length')
        self._control_dt_s = float(cfg.sim.dt) * cfg.decimation
        if self._max_steps == 0:
            raise ValueError('max_episode_length must be positive')
        self._instruction = self._runtime.instruction_text(
            self._runtime.select_instruction_fields(cfg.instruction,
                                                    'default'))
        self._active_task_id = task_id
        self._instance_episode = 0

    def _observation(self, raw):
        if not isinstance(raw, Mapping):
            raise TypeError('RoboLab observation must be a mapping')
        flat = {}
        for output, source in self.camera_keys.items():
            images = raw['image_obs'][source]
            if (len(images.shape) != 4 or images.shape[0] != self.num_envs
                    or images.shape[-1] != 3 or min(images.shape[1:3]) == 0):
                raise ValueError(f'{source} must be [N,H,W,3] RGB uint8')
            image = _numpy(images[0])
            if image.dtype != np.uint8:
                raise ValueError(f'{source} must be [N,H,W,3] RGB uint8')
            flat[output] = image.copy()
        parts = []
        for key, width in zip(self.state_keys, (7, 1)):
            values = raw['proprio_obs'][key]
            if tuple(values.shape) != (self.num_envs, width):
                raise ValueError(f'{key} must be finite numeric [N,{width}]')
            part = _numpy(values[0])
            if part.dtype.kind not in 'fiu':
                raise ValueError(f'{key} must be finite numeric [N,{width}]')
            if not np.isfinite(part).all():
                raise NonFiniteObservationError(
                    key, 0, values.shape,
                    np.argwhere(~np.isfinite(part)).tolist())
            parts.append(part)
        flat['states'] = np.concatenate(parts).copy()
        flat['task_description'] = self._instruction
        return flat

    def step(self, action: Any) -> StepResult:
        self._require_setup()
        if not self._episode_active:
            raise RuntimeError(
                'Call reset(): episode has not started or ended')
        array = _numpy(action)
        if array.shape != (8, ):
            raise ValueError(
                f'DROID action must have shape [8], got {array.shape}')
        # Mixed Python bool/float lists would silently coerce to float.
        if (array.dtype.kind not in 'fiu'
                or isinstance(action, (list, tuple)) and any(
                    isinstance(value, (bool, np.bool_)) for value in action)):
            raise TypeError('DROID actions must be real numbers, not booleans')
        if not np.isfinite(array).all():
            raise ValueError('DROID action contains NaN or infinity')
        with np.errstate(over='ignore'):
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError('DROID action is outside the FP32 range')
        native_action = self._runtime.torch.as_tensor(
            array[None],
            dtype=self._runtime.torch.float32,
            device=self._env.device)
        # Simulator / contract exceptions require reset before more steps.
        self._episode_active = False
        with self._runtime.torch.inference_mode():
            raw, reward, term, trunc, _ = self._env.step(native_action)
        self._steps += 1
        observation = self._observation(raw)
        frozen = bool(self._env.all_terminated)
        results = self._env.get_env_results()
        if (len(results) != 1 or results[0].get('env_id') != 0):
            raise RuntimeError('RoboLab returned invalid single-env results')
        native_result = dict(results[0])
        success = native_result.get('success')
        if frozen:
            if (not isinstance(success, (bool, np.bool_))
                    or not isinstance(native_result.get('step'), Integral)):
                raise RuntimeError('Frozen RoboLab env has no final result')
            success = bool(success)
        elif success is not None or native_result.get('step') is not None:
            raise RuntimeError('Unfrozen RoboLab env has an unexpected result')
        elif self._steps >= self._max_steps:
            native_step = _single(self._env.episode_length_buf,
                                  'episode_length_buf')
            raise RuntimeError(
                'RoboLab reached the native outer-loop limit without a final '
                f'result: steps={self._steps}, result={native_result}, '
                f'native_episode_step={native_step}, '
                f'raw_terminated={_single(term, "terminated")}, '
                f'raw_truncated={_single(trunc, "truncated")}')
        reward_value = float(_single(reward, 'reward'))
        if not np.isfinite(reward_value):
            raise ValueError('RoboLab reward is not finite')
        info = dict(self._episode_info)
        info.update(
            wrapper_steps=self._steps,
            native_episode_step=int(
                _single(self._env.episode_length_buf, 'episode_length_buf')),
            raw_terminated=bool(_single(term, 'terminated')),
            raw_truncated=bool(_single(trunc, 'truncated')),
            native_frozen=frozen,
            native_result=native_result)
        score = None
        if frozen:
            try:
                score, metrics_info = collect_episode_metrics(
                    info['native_hdf5_path'], self._control_dt_s, success,
                    int(native_result['step']))
            except Exception as exc:
                raise RuntimeError(
                    f'RoboLab terminal metrics failed: native_result='
                    f'{native_result}, hdf5={info["native_hdf5_path"]}: '
                    f'{type(exc).__name__}: {exc}') from exc
            info.update(metrics_info)
        self._last_observation = observation
        self._episode_active = not frozen
        return StepResult(
            observation=observation,
            reward=reward_value,
            terminated=frozen and success is True,
            truncated=frozen and success is False,
            success=success,
            score=score,
            info=info)

    def render(self):
        if self._last_observation is None:
            return None
        return self._last_observation[next(iter(self.camera_keys))].copy()

    def _close_native(self):
        env, self._env = self._env, None
        self._active_task_id = None
        self._last_observation = None
        self._episode_active = False
        if env is not None:
            env.close()

    def close(self) -> None:
        try:
            self._close_native()
        finally:
            app, self._app = self._app, None
            self._runtime = None
            self._tasks = ()
            self._task_ids = set()
            try:
                if app is not None:
                    app.close()
            finally:
                if self._added_path is not None:
                    if self._added_path in sys.path:
                        sys.path.remove(self._added_path)
                    self._added_path = None
                self._restore_cache_environment()

    def _require_setup(self):
        if self._runtime is None:
            raise RuntimeError('RoboLabEnvironment.setup() was not called')

    def _require_unstarted(self):
        if self._app is not None:
            raise RuntimeError('Configure the RoboLab worker before setup()')
