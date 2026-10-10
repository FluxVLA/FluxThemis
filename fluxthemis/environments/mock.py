from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from .base import BaseEnvironment


@ENVIRONMENTS.register_module()
class MockEnvironment(BaseEnvironment):
    """Deterministic environment used for smoke tests and CI."""

    def __init__(self,
                 tasks: Sequence[str | Mapping[str, Any]] | None = None,
                 success_after: int | None = 3,
                 max_internal_steps: int = 100) -> None:
        task_values = tasks or [dict(task_id='mock', description='mock task')]
        self._tasks = [TaskSpec.from_value(value) for value in task_values]
        self.success_after = success_after
        self.max_internal_steps = int(max_internal_steps)
        if self.success_after is not None and self.success_after < 1:
            raise ValueError('success_after must be positive or None')
        if self.max_internal_steps < 1:
            raise ValueError('max_internal_steps must be positive')
        self.current_task: TaskSpec | None = None
        self.step_count = 0
        self.received_actions: list[np.ndarray] = []
        self.is_setup = False

    def setup(self) -> None:
        self.is_setup = True

    def tasks(self) -> Sequence[TaskSpec]:
        return tuple(self._tasks)

    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[dict, dict]:
        if not self.is_setup:
            raise RuntimeError('MockEnvironment.setup() must be called first')
        self.current_task = task
        self.step_count = 0
        observation = self._observation()
        info = {
            'task_id': task.task_id,
            'episode_index': episode_index,
            'environment_seed': environment_seed,
            'environment_case_index': environment_case_index,
        }
        return observation, info

    def step(self, action: Any) -> StepResult:
        if self.current_task is None:
            raise RuntimeError('reset() must be called before step()')
        action_array = np.asarray(action, dtype=np.float32)
        self.received_actions.append(action_array.copy())
        self.step_count += 1
        success = (
            self.success_after is not None
            and self.step_count >= self.success_after)
        truncated = self.step_count >= self.max_internal_steps and not success
        return StepResult(
            observation=self._observation(),
            reward=float(success),
            terminated=bool(success),
            truncated=bool(truncated),
            success=bool(success),
            info={
                'task_id': self.current_task.task_id,
                'step': self.step_count,
            },
        )

    def close(self) -> None:
        self.current_task = None
        self.is_setup = False

    def _observation(self) -> dict:
        return {
            'state':
            np.asarray([self.step_count], dtype=np.float32),
            'task_description':
            (self.current_task.description if self.current_task else ''),
        }
