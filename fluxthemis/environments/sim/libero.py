from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from ctypes.util import find_library
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from mmengine.utils import digit_version

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment

ROBOSUITE_MINIMUM_VERSION = '1.5.0'
ROBOSUITE_MAXIMUM_VERSION = '1.5.2'
LIBERO_ACTION_DIM = 7
LIBERO_DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
LIBERO_MAX_STEPS = {
    'libero_spatial': 220,
    'libero_object': 280,
    'libero_goal': 300,
    'libero_10': 520,
    'libero_90': 400,
}
NVIDIA_EGL_VENDOR_JSON = """{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
"""


@dataclass(frozen=True)
class _LiberoRuntime:
    benchmark: Any
    libero: Any
    envs: Any


@ENVIRONMENTS.register_module()
class LiberoEnvironment(BaseEnvironment):
    """LIBERO simulation adapter matching FluxVLA evaluation semantics.

    The adapter returns untouched LIBERO observations. Image rotation, model
    resizing, state conversion and action denormalization remain owned by the
    FluxVLA server-side dataset and transforms.
    """

    def __init__(self,
                 task_suite_name: str,
                 task_order_index: int = 0,
                 resolution: int = 256,
                 controller: str = 'OSC_POSE',
                 simulator_seed: int = 0,
                 num_steps_wait: int = 10,
                 env_kwargs: Mapping[str, Any] | None = None) -> None:
        if task_suite_name not in LIBERO_MAX_STEPS:
            raise ValueError(
                f'Unsupported LIBERO task suite {task_suite_name!r}; '
                f'expected one of {sorted(LIBERO_MAX_STEPS)}')
        self.task_suite_name = task_suite_name
        self.task_order_index = self._nonnegative_integer(
            task_order_index, 'task_order_index')
        if self.task_suite_name == 'libero_90' and self.task_order_index != 0:
            raise ValueError('libero_90 only supports task_order_index=0')
        self.resolution = self._positive_integer(resolution, 'resolution')
        if not isinstance(controller, str) or not controller:
            raise ValueError('controller must be a non-empty string')
        self.controller = controller
        if isinstance(simulator_seed,
                      bool) or not isinstance(simulator_seed, Integral):
            raise TypeError('simulator_seed must be an integer')
        self.simulator_seed = int(simulator_seed)
        self.num_steps_wait = self._nonnegative_integer(
            num_steps_wait, 'num_steps_wait')

        self.env_kwargs = dict(env_kwargs or {})
        reserved = {
            'bddl_file_name',
            'camera_heights',
            'camera_widths',
            'controller',
        }
        conflicts = sorted(reserved.intersection(self.env_kwargs))
        if conflicts:
            raise ValueError(
                f'env_kwargs cannot override managed fields: {conflicts}')

        self._runtime: _LiberoRuntime | None = None
        self._suite = None
        self._tasks: tuple[TaskSpec, ...] = ()
        self._task_indices: dict[str, int] = {}
        self._env = None
        self._active_task_index: int | None = None
        self._initial_states = None
        self._last_observation: dict[str, Any] | None = None
        self._episode_done = False
        self._warmup_success = False

    @property
    def recommended_max_episode_steps(self) -> int:
        return LIBERO_MAX_STEPS[self.task_suite_name]

    def setup(self) -> None:
        if self._suite is not None:
            return
        runtime = _load_libero_runtime()
        benchmark_dict = runtime.benchmark.get_benchmark_dict()
        if self.task_suite_name not in benchmark_dict:
            raise KeyError(
                f'LIBERO benchmark {self.task_suite_name!r} is unavailable; '
                f'available suites: {sorted(benchmark_dict)}')
        suite = benchmark_dict[self.task_suite_name](
            task_order_index=self.task_order_index)
        tasks = []
        for task_index in range(int(suite.n_tasks)):
            task = suite.get_task(task_index)
            tasks.append(
                TaskSpec(
                    task_id=str(task_index),
                    description=str(task.language),
                    metadata={
                        'benchmark':
                        'libero',
                        'task_suite_name':
                        self.task_suite_name,
                        'task_index':
                        task_index,
                        'task_name':
                        str(task.name),
                        'recommended_max_episode_steps':
                        self.recommended_max_episode_steps,
                    },
                ))
        if not tasks:
            raise ValueError(
                f'LIBERO suite {self.task_suite_name!r} contains no tasks')

        self._runtime = runtime
        self._suite = suite
        self._tasks = tuple(tasks)
        self._task_indices = {
            task.task_id: int(task.metadata['task_index'])
            for task in self._tasks
        }

    def tasks(self) -> tuple[TaskSpec, ...]:
        self._require_setup()
        return self._tasks

    def reset(
            self, task: TaskSpec, episode_index: int, *,
            environment_seed: int | None,
            environment_case_index: int | None) -> tuple[dict[str, Any], dict]:
        self._require_setup()
        if not isinstance(task, TaskSpec):
            raise TypeError('task must be a TaskSpec')
        episode_index = self._nonnegative_integer(episode_index,
                                                  'episode_index')
        environment_case_index = self._nonnegative_integer(
            environment_case_index, 'environment_case_index')
        if task.task_id not in self._task_indices:
            raise KeyError(
                f'Unknown LIBERO task id {task.task_id!r}; available ids: '
                f'{list(self._task_indices)}')
        task_index = self._task_indices[task.task_id]

        if task_index != self._active_task_index:
            self._activate_task(task_index)

        self._env.reset()
        init_state_index = environment_case_index % len(self._initial_states)
        observation = self._env.set_init_state(
            self._initial_states[init_state_index])

        warmup_steps = 0
        warmup_reward = 0.0
        warmup_info: Mapping[str, Any] = {}
        self._warmup_success = False
        for _ in range(self.num_steps_wait):
            observation, reward, done, info = self._env.step(
                list(LIBERO_DUMMY_ACTION))
            if not isinstance(info, Mapping):
                raise TypeError('LIBERO step info must be a mapping')
            warmup_steps += 1
            warmup_reward += float(reward)
            warmup_info = info
            if done:
                self._warmup_success = True
                break

        self._last_observation = self._validate_observation(observation)
        self._episode_done = self._warmup_success
        return self._last_observation, {
            'benchmark': 'libero',
            'task_suite_name': self.task_suite_name,
            'task_index': task_index,
            'task_name': task.metadata.get('task_name'),
            'environment_seed': environment_seed,
            'environment_case_index': environment_case_index,
            'simulator_seed': self.simulator_seed,
            'init_state_index': init_state_index,
            'warmup_steps': warmup_steps,
            'warmup_reward': warmup_reward,
            'warmup_success': self._warmup_success,
            'warmup_info': dict(warmup_info),
        }

    def step(self, action: Any) -> StepResult:
        self._require_episode()
        action_array = np.asarray(action)
        if (action_array.ndim != 1
                or action_array.shape[0] != LIBERO_ACTION_DIM):
            raise ValueError(
                f'LIBERO action must have shape [{LIBERO_ACTION_DIM}], got '
                f'{action_array.shape}')
        if not np.issubdtype(action_array.dtype, np.number):
            raise TypeError('LIBERO action must be numeric')
        try:
            action_array = np.asarray(action_array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError('LIBERO action must be numeric') from exc
        if not np.isfinite(action_array).all():
            raise ValueError('LIBERO action contains NaN or infinity')

        if self._episode_done:
            if not self._warmup_success:
                raise RuntimeError('LIBERO episode has already terminated')
            self._warmup_success = False
            return StepResult(
                observation=self._last_observation,
                reward=0.0,
                terminated=True,
                success=True,
                info={
                    'benchmark': 'libero',
                    'success_during_warmup': True,
                },
            )

        observation, reward, done, info = self._env.step(action_array.tolist())
        if not isinstance(info, Mapping):
            raise TypeError('LIBERO step info must be a mapping')
        self._last_observation = self._validate_observation(observation)
        self._episode_done = bool(done)
        result_info = dict(info)
        result_info.update({
            'benchmark': 'libero',
            'task_suite_name': self.task_suite_name,
            'task_index': self._active_task_index,
        })
        return StepResult(
            observation=self._last_observation,
            reward=float(reward),
            terminated=bool(done),
            truncated=False,
            success=bool(done),
            info=result_info,
        )

    def render(self) -> Any | None:
        if self._last_observation is None:
            return None
        image = self._last_observation.get('agentview_image')
        if image is None:
            return None
        return np.asarray(image).copy()

    def close(self) -> None:
        self._close_active_env()
        self._runtime = None
        self._suite = None
        self._tasks = ()
        self._task_indices = {}

    def _activate_task(self, task_index: int) -> None:
        self._close_active_env()
        task = self._suite.get_task(task_index)
        initial_states = self._suite.get_task_init_states(task_index)
        if len(initial_states) == 0:
            raise ValueError(f'LIBERO task {task_index} has no initial states')

        bddl_path = os.path.join(
            self._runtime.libero.get_libero_path('bddl_files'),
            task.problem_folder,
            task.bddl_file,
        )
        kwargs = {
            'bddl_file_name': bddl_path,
            'camera_heights': self.resolution,
            'camera_widths': self.resolution,
            'controller': self.controller,
        }
        kwargs.update(self.env_kwargs)
        env = self._runtime.envs.OffScreenRenderEnv(**kwargs)
        try:
            env.seed(self.simulator_seed)
        except Exception:
            env.close()
            raise

        self._env = env
        self._active_task_index = task_index
        self._initial_states = initial_states
        self._last_observation = None
        self._episode_done = False
        self._warmup_success = False

    def _close_active_env(self) -> None:
        env = self._env
        self._env = None
        self._active_task_index = None
        self._initial_states = None
        self._last_observation = None
        self._episode_done = False
        self._warmup_success = False
        if env is not None:
            env.close()

    def _require_setup(self) -> None:
        if self._suite is None or self._runtime is None:
            raise RuntimeError('LiberoEnvironment.setup() was not called')

    def _require_episode(self) -> None:
        self._require_setup()
        if self._env is None or self._last_observation is None:
            raise RuntimeError(
                'LiberoEnvironment.reset() must be called before step()')

    @staticmethod
    def _validate_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError('LIBERO observation must be a mapping')
        return dict(observation)

    @staticmethod
    def _nonnegative_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        value = int(value)
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
        return value

    @classmethod
    def _positive_integer(cls, value: Any, name: str) -> int:
        value = cls._nonnegative_integer(value, name)
        if value == 0:
            raise ValueError(f'{name} must be positive')
        return value


def configure_mujoco_egl_defaults() -> None:
    """Match FluxVLA's NVIDIA EGL defaults before importing MuJoCo."""
    if os.environ.get('MUJOCO_GL') in {'osmesa', 'glfw', 'glx'}:
        return
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    if os.environ.get('MUJOCO_GL') != 'egl':
        return
    if os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES'):
        return
    if not _has_nvidia_egl_library():
        return
    vendor_file = _ensure_nvidia_egl_vendor_file()
    if vendor_file is not None:
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = vendor_file


def _load_libero_runtime() -> _LiberoRuntime:
    os.environ.setdefault(
        'LIBERO_CONFIG_PATH',
        str(Path(sys.prefix) / 'etc' / 'fluxthemis' / 'libero'),
    )
    configure_mujoco_egl_defaults()
    try:
        robosuite = importlib.import_module('robosuite')
        libero = importlib.import_module('libero.libero')
        benchmark = importlib.import_module('libero.libero.benchmark')
        envs = importlib.import_module('libero.libero.envs')
        _patch_libero_collections_compatibility()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'LiberoEnvironment requires the pinned LIBERO simulation stack. '
            'Install it with `bash scripts/install_libero.sh`.') from exc

    version = getattr(robosuite, '__version__', '')
    if not (digit_version(version) >= digit_version(ROBOSUITE_MINIMUM_VERSION)
            and digit_version(version)
            < digit_version(ROBOSUITE_MAXIMUM_VERSION)):
        raise RuntimeError(
            f'Robosuite=={version or "unknown"} is incompatible; install the '
            'pinned FluxVLA fork with `bash scripts/install_libero.sh`.')
    if not hasattr(robosuite, 'load_controller_config'):
        raise RuntimeError(
            'Robosuite is missing the legacy LIBERO compatibility API; '
            'install the pinned e293cc3 FluxVLA fork.')
    return _LiberoRuntime(benchmark=benchmark, libero=libero, envs=envs)


def _patch_libero_collections_compatibility() -> None:
    """Replace LIBERO's removed ``collections.Iterable`` reference."""
    utils = importlib.import_module('libero.libero.envs.utils')
    collections_module = getattr(utils, 'collections', None)
    if collections_module is not None and not hasattr(collections_module,
                                                      'Iterable'):
        utils.collections = importlib.import_module('collections.abc')


def _has_nvidia_egl_library() -> bool:
    if find_library('EGL_nvidia'):
        return True
    return any(
        os.path.exists(path) for path in (
            '/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0',
            '/usr/lib64/libEGL_nvidia.so.0',
            '/usr/lib/aarch64-linux-gnu/libEGL_nvidia.so.0',
        ))


def _ensure_nvidia_egl_vendor_file() -> str | None:
    configured = os.environ.get('FLUXTHEMIS_EGL_VENDOR_FILE')
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        Path(sys.prefix) / 'etc' / 'fluxthemis' / 'egl_vendor.d' /
        '10_nvidia.json',
    ])
    for path in candidates:
        try:
            if path.is_file() and 'libEGL_nvidia.so.0' in path.read_text():
                return path.as_posix()
        except OSError:
            continue

    output = (
        Path(configured).expanduser() if configured else Path(sys.prefix) /
        'etc' / 'fluxthemis' / 'egl_vendor.d' / '10_nvidia.json')
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(NVIDIA_EGL_VENDOR_JSON)
        return output.as_posix()
    except OSError:
        return None
