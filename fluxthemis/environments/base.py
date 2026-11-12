from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from fluxthemis.structures import StepResult, TaskSpec


class EnvironmentCaseRejected(RuntimeError):
    """Signal rejected initial cases, optionally identifying batch slots."""

    def __init__(self, message: str, *, slots: Sequence[int] | None = None):
        super().__init__(message)
        self.slots = None if slots is None else tuple(slots)
        if self.slots is not None and (not self.slots or len(set(
                self.slots)) != len(self.slots) or any(
                    type(slot) is not int or slot < 0 for slot in self.slots)):
            raise ValueError(
                'Rejected slots must be distinct nonnegative integers')


@dataclass(frozen=True)
class EnvironmentReset:
    """Episode identity and initial condition for one environment slot."""

    task: TaskSpec
    episode_index: int
    environment_seed: int | None
    environment_case_index: int | None


class BaseEnvironment(ABC):
    """Contract implemented by simulation and real-robot adapters."""

    supports_native_video_recording = False
    num_envs = 1

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

    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[Any, dict]:
        """Reset one episode and return ``(observation, info)``."""
        raise NotImplementedError

    def step(self, action: Any) -> StepResult:
        """Execute one environment-unit action."""
        raise NotImplementedError

    def reset_batch(
        self, requests: Sequence[EnvironmentReset | None]
    ) -> Sequence[tuple[Any, dict] | None]:
        """Reset every slot, in order, without implicit episode restarts.

        None marks an unused slot; return None without starting an episode.
        Native multi-environment adapters override this and ``step_batch``.
        The default bridges existing scalar adapters to the same runner.
        """
        if self.num_envs != 1 or len(requests) != 1:
            raise NotImplementedError('Adapter must implement reset_batch')
        request = requests[0]
        if request is None:
            return [None]
        return [
            self.reset(
                request.task,
                request.episode_index,
                environment_seed=request.environment_seed,
                environment_case_index=request.environment_case_index)
        ]

    def step_batch(
            self,
            actions: Sequence[Any | None]) -> Sequence[StepResult | None]:
        """Advance active slots once, retaining their order.

        ``None`` marks an inactive slot. Adapters must freeze or isolate it
        from active episodes and return None for it. Never auto-reset a slot;
        terminal observations and metrics belong to the finishing episode.
        """
        if self.num_envs != 1 or len(actions) != 1:
            raise NotImplementedError('Adapter must implement step_batch')
        return [None if actions[0] is None else self.step(actions[0])]

    def render(self) -> Any | None:
        return None

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
