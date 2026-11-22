from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from fluxthemis.structures import (CaseRejection, EnvironmentReset,
                                   EpisodeInterruption, StepResult, TaskSpec)


@dataclass(frozen=True)
class EnvironmentResetRequest:
    """Episode identity and initial condition for one environment slot."""

    task: TaskSpec
    episode_index: int
    environment_seed: int | None
    environment_case_index: int | None


class BaseEnvironment(ABC):
    """Slot-based contract implemented by simulation and robot adapters."""

    supports_native_video_recording = False
    requires_single_task = False

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of slots owned by this environment."""

    def configure_worker(self, worker_id: str, run_id: str) -> None:
        del worker_id, run_id

    def configure_simulator_device(self, device_id: str) -> None:
        del device_id

    def configure_video_recording(self, enabled: bool,
                                  output_dir: str | None) -> None:
        del enabled, output_dir

    def setup(self) -> None:
        """Allocate simulator or hardware resources."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        """Return the tasks available for evaluation."""

    def task_capacity(self, task: TaskSpec) -> int:
        """Return this task's assignment capacity after setup."""
        return self.num_envs

    @abstractmethod
    def reset_batch(
        self, requests: list[EnvironmentResetRequest | None]
    ) -> list[EnvironmentReset | CaseRejection | EpisodeInterruption | None]:
        """Reset slots, returning observations, rejections or interruptions.

        Both lists have num_envs entries in slot order. None marks an unused
        slot, which must remain inactive and return None.
        """

    @abstractmethod
    def step_batch(
        self, actions: list[Any | None]
    ) -> list[StepResult | CaseRejection | EpisodeInterruption | None]:
        """Advance active slots once, retaining their order.

        None marks an inactive slot; freeze or isolate it and return None.
        Never auto-reset a slot. Return CaseRejection for invalid cases or
        EpisodeInterruption to retry an attempt with the same initial case.
        """

    def render(self, slot: int) -> Any | None:
        return None

    @abstractmethod
    def finalize_episode(self, slot: int, *, success: bool,
                         termination_reason: str) -> dict:
        """Finalize a naturally or externally ended episode in this slot.

        Called for natural completion, PolicyStop and max_episode_steps.
        Only perform cleanup: do not advance simulation or change the episode
        outcome. Return additional information, such as video_paths.

        When termination_reason is 'case_rejected' or 'interrupted', discard
        this slot's recordings and return an empty dictionary.

        Multi-scene adapters must freeze this slot here. Subsequent actions
        for the slot will be None.
        """

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""


class SingleEnvironmentAdapter(BaseEnvironment):
    """Bridge legacy scalar reset/step implementations to one batch slot.

    Scalar resets may return their existing (observation, info) tuple or an
    EnvironmentReset. Forward CaseRejection and EpisodeInterruption without
    conversion. All callers use the batch methods.
    """

    num_envs = 1

    @abstractmethod
    def reset(
        self, task: TaskSpec, episode_index: int, *,
        environment_seed: int | None, environment_case_index: int | None
    ) -> (tuple[Any, dict] | EnvironmentReset | CaseRejection |
          EpisodeInterruption):
        """Reset the legacy scalar environment."""

    @abstractmethod
    def step(self,
             action: Any) -> StepResult | CaseRejection | EpisodeInterruption:
        """Step the legacy scalar environment."""

    def reset_batch(
        self, requests: list[EnvironmentResetRequest | None]
    ) -> list[EnvironmentReset | CaseRejection | EpisodeInterruption | None]:
        if len(requests) != 1:
            raise ValueError(
                'SingleEnvironmentAdapter expects exactly one slot')
        request = requests[0]
        if request is None:
            return [None]
        result = self.reset(
            request.task,
            request.episode_index,
            environment_seed=request.environment_seed,
            environment_case_index=request.environment_case_index)
        result_types = (EnvironmentReset, CaseRejection, EpisodeInterruption)
        if not isinstance(result, result_types):
            result = EnvironmentReset(*result)
        return [result]

    def step_batch(
        self, actions: list[Any | None]
    ) -> list[StepResult | CaseRejection | EpisodeInterruption | None]:
        if len(actions) != 1:
            raise ValueError(
                'SingleEnvironmentAdapter expects exactly one slot')
        action = actions[0]
        return [None if action is None else self.step(action)]

    def render(self, slot: int) -> Any | None:
        self._require_slot(slot)
        return self._render()

    def _render(self) -> Any | None:
        return None

    def finalize_episode(self, slot: int, *, success: bool,
                         termination_reason: str) -> dict:
        self._require_slot(slot)
        return {}

    @staticmethod
    def _require_slot(slot: int) -> None:
        if type(slot) is not int or slot != 0:
            raise ValueError('SingleEnvironmentAdapter only owns slot 0')
