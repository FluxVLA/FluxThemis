from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment
from .libero import configure_mujoco_egl_defaults

ROBOCASA_ACTION_DIM = 29
ROBOCASA_PROMPT_KEY = 'annotation.human.coarse_action'
ROBOCASA_DEFAULT_RENDER_KEY = 'video.ego_view_pad_res256_freq20'
ROBOCASA_ACTION_ORDERS = {
    # FluxVLA converted parquet order.
    'fluxvla': {
        'action.left_arm': (0, 7),
        'action.left_hand': (7, 13),
        'action.right_arm': (13, 20),
        'action.right_hand': (20, 26),
        'action.waist': (26, 29),
    },
    # Official GR00T N1.5 fourier_gr1_arms_waist order.
    'n15': {
        'action.left_arm': (0, 7),
        'action.right_arm': (7, 14),
        'action.left_hand': (14, 20),
        'action.right_hand': (20, 26),
        'action.waist': (26, 29),
    },
}


@dataclass(frozen=True)
class _RoboCasaRuntime:
    gymnasium: Any
    robocasa: Any
    gym_utils: Any


@ENVIRONMENTS.register_module()
class RoboCasaEnvironment(BaseEnvironment):
    """RoboCasa GR1 adapter matching FluxVLA evaluation semantics.

    Observations and episode language remain untouched. FluxVLA owns image
    processing, state construction, prompt tokenization, model inference, and
    action denormalization. This adapter only splits one denormalized 29D
    action into the dictionary consumed by the RoboCasa Gymnasium wrapper.
    """

    def __init__(self,
                 task_list: Sequence[str],
                 action_order: str = 'fluxvla',
                 deterministic_env: bool = True,
                 prompt_key: str = ROBOCASA_PROMPT_KEY,
                 render_key: str = ROBOCASA_DEFAULT_RENDER_KEY,
                 env_kwargs: Mapping[str, Any] | None = None) -> None:
        if isinstance(task_list,
                      (str, bytes)) or not isinstance(task_list, Sequence):
            raise TypeError('task_list must be a sequence of environment ids')
        env_names = tuple(
            self._nonempty_string(value, 'task_list entry')
            for value in task_list)
        if not env_names:
            raise ValueError('task_list cannot be empty')
        if len(env_names) != len(set(env_names)):
            raise ValueError(
                'task_list cannot contain duplicate environment ids')
        if action_order not in ROBOCASA_ACTION_ORDERS:
            raise ValueError(
                f'Unsupported action_order {action_order!r}; expected one of '
                f'{sorted(ROBOCASA_ACTION_ORDERS)}')
        if not isinstance(deterministic_env, bool):
            raise TypeError('deterministic_env must be a bool')
        self.prompt_key = self._nonempty_string(prompt_key, 'prompt_key')
        self.render_key = self._nonempty_string(render_key, 'render_key')
        self.env_kwargs = dict(env_kwargs or {})
        if 'seed' in self.env_kwargs:
            raise ValueError('env_kwargs cannot override managed field seed')

        self.task_list = env_names
        self.action_order = action_order
        self.action_keys = ROBOCASA_ACTION_ORDERS[action_order]
        self.deterministic_env = deterministic_env
        self._runtime: _RoboCasaRuntime | None = None
        self._tasks: tuple[TaskSpec, ...] = ()
        self._task_indices: dict[str, int] = {}
        self._env = None
        self._active_task_index: int | None = None
        self._active_prompt: str | None = None
        self._last_observation: dict[str, Any] | None = None
        self._episode_done = False

    def setup(self) -> None:
        if self._runtime is not None:
            return
        runtime = _load_robocasa_runtime()
        tasks = []
        for task_index, env_name in enumerate(self.task_list):
            group = robocasa_group_name(env_name)
            tasks.append(
                TaskSpec(
                    task_id=str(task_index),
                    description=env_name,
                    metadata={
                        'benchmark': 'robocasa',
                        'task_suite_name': 'robocasa',
                        'task_index': task_index,
                        'task_name': env_name,
                        'env_name': env_name,
                        'group': group,
                        'prompt_source': self.prompt_key,
                        'action_order': self.action_order,
                    },
                ))
        self._runtime = runtime
        self._tasks = tuple(tasks)
        self._task_indices = {
            task.task_id: int(task.metadata['task_index'])
            for task in self._tasks
        }

    def tasks(self) -> tuple[TaskSpec, ...]:
        self._require_setup()
        return self._tasks

    def reset(self, task: TaskSpec, episode_index: int,
              seed: int) -> tuple[dict[str, Any], dict]:
        self._require_setup()
        if not isinstance(task, TaskSpec):
            raise TypeError('task must be a TaskSpec')
        episode_index = self._nonnegative_integer(episode_index,
                                                  'episode_index')
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError('seed must be an integer')
        if task.task_id not in self._task_indices:
            raise KeyError(
                f'Unknown RoboCasa task id {task.task_id!r}; available ids: '
                f'{list(self._task_indices)}')

        task_index = self._task_indices[task.task_id]
        env_name = self.task_list[task_index]
        episode_seed = int(seed) % np.iinfo(np.uint32).max
        self._close_active_env()
        make_kwargs = dict(self.env_kwargs)
        if self.deterministic_env:
            make_kwargs['seed'] = episode_seed
        env = self._runtime.gymnasium.make(env_name, **make_kwargs)
        try:
            if self.deterministic_env:
                observation, info = env.reset(seed=episode_seed)
            else:
                observation, info = env.reset()
            if not isinstance(info, Mapping):
                raise TypeError('RoboCasa reset info must be a mapping')
            observation = self._validate_observation(observation)
            prompt = observation.get(self.prompt_key)
            prompt = self._nonempty_string(prompt, self.prompt_key)
        except BaseException:
            env.close()
            raise

        self._env = env
        self._active_task_index = task_index
        self._active_prompt = prompt
        self._last_observation = observation
        self._episode_done = False
        reset_info = dict(info)
        reset_info.update({
            'benchmark': 'robocasa',
            'task_suite_name': 'robocasa',
            'task_index': task_index,
            'task_name': env_name,
            'env_name': env_name,
            'group': task.metadata['group'],
            'episode_index': episode_index,
            'episode_seed': episode_seed,
            'task_description': prompt,
            'action_order': self.action_order,
        })
        return observation, reset_info

    def step(self, action: Any) -> StepResult:
        self._require_episode()
        if self._episode_done:
            raise RuntimeError('RoboCasa episode has already terminated')
        action_array = np.asarray(action)
        if action_array.ndim != 1 or action_array.shape[0] != \
                ROBOCASA_ACTION_DIM:
            raise ValueError(
                f'RoboCasa action must have shape [{ROBOCASA_ACTION_DIM}], '
                f'got {action_array.shape}')
        if (not np.issubdtype(action_array.dtype, np.number)
                or np.issubdtype(action_array.dtype, np.bool_)):
            raise TypeError('RoboCasa action must be numeric')
        action_array = np.asarray(action_array, dtype=np.float32)
        if not np.isfinite(action_array).all():
            raise ValueError('RoboCasa action contains NaN or infinity')

        action_dict = {
            key: action_array[start:end].copy()
            for key, (start, end) in self.action_keys.items()
        }
        observation, reward, terminated, truncated, info = self._env.step(
            action_dict)
        if not isinstance(info, Mapping):
            raise TypeError('RoboCasa step info must be a mapping')
        observation = self._validate_observation(observation)
        # RoboCasa samples one instruction at reset. Match the native runner
        # by freezing that exact string (including the unlocked_waist prefix)
        # for the entire episode.
        prompt = self._active_prompt
        self._last_observation = observation
        self._episode_done = bool(terminated) or bool(truncated)
        result_info = dict(info)
        success = bool(result_info.get('success', False))
        result_info.update({
            'benchmark':
            'robocasa',
            'task_suite_name':
            'robocasa',
            'task_index':
            self._active_task_index,
            'task_name':
            self.task_list[self._active_task_index],
            'env_name':
            self.task_list[self._active_task_index],
            'group':
            robocasa_group_name(self.task_list[self._active_task_index]),
            'task_description':
            prompt,
            'action_order':
            self.action_order,
        })
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            info=result_info,
        )

    def render(self) -> Any | None:
        if self._last_observation is None:
            return None
        image = self._last_observation.get(self.render_key)
        if image is None:
            return None
        return np.asarray(image).copy()

    def close(self) -> None:
        self._close_active_env()
        self._runtime = None
        self._tasks = ()
        self._task_indices = {}

    def _close_active_env(self) -> None:
        env = self._env
        self._env = None
        self._active_task_index = None
        self._active_prompt = None
        self._last_observation = None
        self._episode_done = False
        if env is not None:
            env.close()

    def _require_setup(self) -> None:
        if self._runtime is None:
            raise RuntimeError('RoboCasaEnvironment.setup() was not called')

    def _require_episode(self) -> None:
        self._require_setup()
        if self._env is None or self._last_observation is None:
            raise RuntimeError(
                'RoboCasaEnvironment.reset() must be called before step()')

    @staticmethod
    def _validate_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError('RoboCasa observation must be a mapping')
        return dict(observation)

    @staticmethod
    def _nonempty_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{name} must be a non-empty string')
        return value

    @staticmethod
    def _nonnegative_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        value = int(value)
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
        return value


def robocasa_group_name(env_name: str) -> str:
    """Return the native FluxVLA RoboCasa reporting group."""
    task_name = str(env_name).split('/')[-1]
    if 'ToCabinet' in task_name:
        return 'Cabinet'
    if 'ToDrawer' in task_name:
        return 'Drawer'
    if 'ToMicrowave' in task_name:
        return 'Microwave'
    return 'Generalization'


def _load_robocasa_runtime() -> _RoboCasaRuntime:
    configure_mujoco_egl_defaults()
    try:
        gymnasium = importlib.import_module('gymnasium')
        robocasa = importlib.import_module('robocasa')
        # Importing gym_utils creates all GR1 Gymnasium registrations.
        gym_utils = importlib.import_module('robocasa.utils.gym_utils')
        getattr(gym_utils, 'GrootRoboCasaEnv')
    except (ImportError, AttributeError) as exc:
        raise ModuleNotFoundError(
            'RoboCasaEnvironment requires the dedicated RoboCasa runtime. '
            'Install it with `bash scripts/install_robocasa.sh`.') from exc
    return _RoboCasaRuntime(
        gymnasium=gymnasium,
        robocasa=robocasa,
        gym_utils=gym_utils,
    )
