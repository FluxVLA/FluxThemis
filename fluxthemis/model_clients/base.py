from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxthemis.structures import ActionChunk, PolicyStop, TaskSpec


class BasePolicyClient(ABC):
    """Policy boundary used by simulator workers.

    Implementations must ensure that preprocessing, inference and action
    denormalization happen locally or in their remote policy service. Returned
    actions must already be in environment units.
    """

    def setup(self) -> None:
        """Allocate model or network resources."""

    def reset(self, task: TaskSpec, episode_index: int,
              policy_seed: int) -> None:
        """Reset episode-local model state."""

    def observe(self, observation: Any) -> None:
        """Record post-step observations, including within action chunks."""

    @abstractmethod
    def set_episode_id(self, episode_id: str) -> None:
        """Set the exact id used by subsequent episode inference requests."""

    @abstractmethod
    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk | PolicyStop:
        """Return native float32 ``[T, A]`` actions or stop this episode.

        PolicyStop is an expected outcome, not a crashed worker; task success
        remains determined by environment transitions, never by the policy.
        """

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
