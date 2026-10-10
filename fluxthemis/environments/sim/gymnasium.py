from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment


@ENVIRONMENTS.register_module()
class GymnasiumEnvironment(BaseEnvironment):
    """Adapter for Gymnasium environments with explicit success mapping."""

    def __init__(self,
                 env_id: str,
                 tasks: Sequence[str | Mapping[str, Any]] | None = None,
                 make_kwargs: Mapping[str, Any] | None = None,
                 render_mode: str | None = None,
                 success_key: str | None = None,
                 terminated_is_success: bool = False,
                 squeeze_scalar_action: bool = False,
                 scalar_action_type: str | None = None,
                 task_option_key: str | None = None) -> None:
        self.env_id = str(env_id)
        task_values = tasks or [dict(task_id=self.env_id)]
        self._tasks = [TaskSpec.from_value(value) for value in task_values]
        self.make_kwargs = dict(make_kwargs or {})
        if render_mode is not None:
            self.make_kwargs.setdefault('render_mode', render_mode)
        self.success_key = success_key
        self.terminated_is_success = bool(terminated_is_success)
        self.squeeze_scalar_action = bool(squeeze_scalar_action)
        if scalar_action_type not in (None, 'float', 'int'):
            raise ValueError(
                "scalar_action_type must be None, 'float', or 'int'")
        self.scalar_action_type = scalar_action_type
        self.task_option_key = task_option_key
        self.env = None

    def setup(self) -> None:
        try:
            import gymnasium
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'GymnasiumEnvironment requires simulation dependencies. '
                'Install with `pip install -e .`.') from exc
        self.env = gymnasium.make(self.env_id, **self.make_kwargs)

    def tasks(self) -> Sequence[TaskSpec]:
        return tuple(self._tasks)

    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[Any, dict]:
        self._require_setup()
        options = dict(task.metadata.get('reset_options', {}))
        if self.task_option_key is not None:
            options.setdefault(self.task_option_key, task.task_id)
        observation, info = self.env.reset(
            seed=environment_seed,
            options=options or None,
        )
        if not isinstance(info, Mapping):
            raise TypeError('Gymnasium reset info must be a mapping')
        reset_info = dict(info)
        reset_info.update({
            'environment_seed': environment_seed,
            'environment_case_index': environment_case_index,
        })
        return observation, reset_info

    def step(self, action: Any) -> StepResult:
        self._require_setup()
        action_value = action
        if self.squeeze_scalar_action:
            action_array = np.asarray(action)
            if action_array.size != 1:
                raise ValueError(
                    'squeeze_scalar_action requires exactly one action value')
            action_value = action_array.reshape(-1)[0].item()
            if self.scalar_action_type == 'int':
                action_value = int(action_value)
            elif self.scalar_action_type == 'float':
                action_value = float(action_value)
        observation, reward, terminated, truncated, info = self.env.step(
            action_value)
        if not isinstance(info, Mapping):
            raise TypeError('Gymnasium step info must be a mapping')
        success = self._read_success(info, terminated)
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            info=dict(info),
        )

    def render(self) -> Any | None:
        self._require_setup()
        return self.env.render()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None

    def _read_success(self, info: Mapping[str, Any],
                      terminated: bool) -> bool | None:
        if self.success_key is not None:
            if self.success_key not in info:
                raise KeyError(
                    f'Success key {self.success_key!r} is absent from info')
            return bool(info[self.success_key])
        if self.terminated_is_success:
            return bool(terminated)
        return None

    def _require_setup(self) -> None:
        if self.env is None:
            raise RuntimeError('GymnasiumEnvironment.setup() was not called')
